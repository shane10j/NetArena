import ast
import json
import re

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from config import AgentConfig
from llm import LLMClient
from messenger import Messenger
from roles import get_role


PRIVATE_HELPER_RE = re.compile(
    r"\b(solid_step_[A-Za-z0-9_]*|private_[A-Za-z0-9_]*|oracle_[A-Za-z0-9_]*|"
    r"reference_[A-Za-z0-9_]*|ground_truth_[A-Za-z0-9_]*|expected_[A-Za-z0-9_]*|"
    r"benchmark_[A-Za-z0-9_]*|grader_[A-Za-z0-9_]*|malt_[A-Za-z0-9_]*)\b"
)


class Agent:
    """Two-agent MALT solver.

    The coordinator uses exactly two conceptual agents:
      1) correctness_agent: writes the strongest self-contained NetworkX solution.
      2) safety_agent: reviews the draft for unsafe graph changes and runtime hazards.

    The correctness agent then revises using the safety critique.  Local checks are intentionally
    lightweight: they block only mechanical invalidity, private-helper cheating, and obviously
    dangerous operations.  This avoids the regressions caused by over-aggressive semantic gates
    while still giving safety a dedicated pass.
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.messenger = Messenger()
        self.llm = LLMClient(self.config)
        self.role = get_role(self.config.role)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"{self.role.name} is solving the graph task..."),
        )
        result = await self._respond(input_text)
        await updater.complete(new_agent_text_message(result))

    async def invoke(self, input_text: str) -> str:
        return await self._respond(input_text)

    async def _respond(self, input_text: str) -> str:
        if self.role.name == "coordinator":
            return await self._coordinate(input_text)
        return await self._role_response(input_text)

    async def _coordinate(self, input_text: str) -> str:
        # Round 1: correctness agent produces the best attempt.  No planner/arbiter tournament:
        # fewer moving parts, less synthesis drift, and a clearer dialogue with safety.
        draft_raw = await self._run_stage(
            role="correctness_agent",
            url=getattr(self.config, "coder_agent_url", None),
            prompt=self._correctness_prompt(input_text),
        )
        draft = self._clean_code(draft_raw or "")

        # Round 2: safety agent reviews. It returns critique only, not replacement code, so it
        # cannot accidentally overwrite a correct algorithm with a conservative no-op.
        critique_raw = await self._run_stage(
            role="safety_agent",
            url=getattr(self.config, "repair_agent_url", None),
            prompt=self._safety_prompt(input_text, draft, self._issues(draft)),
        )
        critique = self._normalize_json_response(critique_raw or "{}")

        # Round 3: correctness agent revises to maximize correctness while fixing the concrete
        # safety/runtime concerns.  This keeps correctness in control, with safety as a strong critic.
        revised_raw = await self._run_stage(
            role="correctness_agent",
            url=getattr(self.config, "coder_agent_url", None),
            prompt=self._revision_prompt(input_text, draft, critique),
        )
        revised = self._clean_code(revised_raw or "")

        # Optional final safety critique only when there are mechanical issues or obvious dangerous
        # patterns.  Avoid an unconditional extra rewrite, which previously hurt both correctness and
        # safety by changing good code.
        if self._is_acceptable(revised):
            return revised
        if self._is_acceptable(draft):
            return draft

        # Small mechanical repair if both failed. This is not a semantic safety gate.
        base = self._least_bad([("revised", revised), ("draft", draft)])[1]
        repaired_raw = await self._run_stage(
            role="correctness_agent",
            url=getattr(self.config, "coder_agent_url", None),
            prompt=self._mechanical_repair_prompt(input_text, base, self._issues(base)),
        )
        repaired = self._clean_code(repaired_raw or "")
        if self._is_acceptable(repaired):
            return repaired
        if base.strip() and not self._has_private_helper(base) and not self._syntax_error(base):
            return base
        return self._fallback_response()

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        if self.role.name == "safety_agent":
            return self._normalize_json_response(response or "{}")
        code = self._clean_code(response or "")
        return code or self._fallback_response()

    async def _run_stage(self, *, role: str, url: str | None, prompt: str) -> str:
        if url:
            try:
                return await self.messenger.talk_to_agent(prompt, url)
            except Exception as exc:
                print(f"{role} delegation failed: {exc}")
        spec = get_role(role)
        print(f"Calling LiteLLM model for {role}: {self.config.model_name or '<none configured>'}")
        return await self.llm.complete(system_prompt=spec.system_prompt, user_prompt=prompt) or ("{}" if role == "safety_agent" else "")

    def _correctness_prompt(self, task: str) -> str:
        return "\n\n".join([
            "You are the CORRECTNESS agent. Produce the final answer as code only.",
            "Goal: maximize benchmark correctness without using private helpers. Safety matters, but do not avoid necessary requested mutations.",
            self._contract(),
            self._graph_semantics(),
            "Think through the task internally, then output only the code. Put any helper logic inside process_graph so the submitted code is self-contained.",
            f"Task:\n{task}",
        ])

    def _safety_prompt(self, task: str, code: str, mechanical_issues: list[str]) -> str:
        return "\n\n".join([
            "You are the SAFETY agent reviewing a MALT NetworkX solution.",
            "Return JSON only. Do not write replacement code.",
            "Your job is to identify concrete issues that may reduce safety or cause runtime failure while preserving correctness. Do not object to a mutation if the task explicitly requires that mutation.",
            "Focus on: private helper cheating; mutating graph_data; read-only tasks that mutate graph_copy; broad clear/remove/relabel/overwrite; .get on non-dict edge data; DiGraph-only traversal that will crash on undirected graphs; missing schema; raw graph return; missing node/entity crashes.",
            "JSON schema: {\"read_only\": true/false, \"must_mutate\": true/false, \"fatal\": [strings], \"safety\": [strings], \"runtime\": [strings], \"correctness_risks\": [strings], \"revision_instructions\": [short concrete instructions]}.",
            f"Mechanical issues already found:\n{json.dumps(mechanical_issues)}",
            f"Task:\n{task}",
            f"Code:\n{code}",
        ])

    def _revision_prompt(self, task: str, draft: str, critique_json: str) -> str:
        return "\n\n".join([
            "You are the CORRECTNESS agent revising after a SAFETY review. Return code only.",
            "Preserve the draft's algorithm whenever it is semantically correct. Make the smallest changes needed to fix real safety/runtime/schema issues. Do not become conservative/no-op unless the task is truly read-only.",
            self._contract(),
            self._graph_semantics(),
            f"Task:\n{task}",
            f"Safety critique JSON:\n{critique_json}",
            f"Draft code:\n{draft}",
        ])

    def _mechanical_repair_prompt(self, task: str, code: str, issues: list[str]) -> str:
        return "\n\n".join([
            "Return only corrected executable Python code defining process_graph(graph_data).",
            "Make the smallest repair for these mechanical issues. Preserve the algorithm and requested mutations.",
            self._contract(),
            self._graph_semantics(),
            f"Task:\n{task}",
            f"Issues:\n{json.dumps(issues)}",
            f"Code:\n{code}",
        ])

    def _contract(self) -> str:
        return """Mandatory contract:
- Define exactly one top-level function: process_graph(graph_data). Do not import packages; nx is already available.
- Start with graph_copy = graph_data.copy(). Never mutate graph_data.
- Put helper functions inside process_graph, not at top level.
- Every return path returns {'type': ..., 'data': ..., 'updated_graph': ...}.
- updated_graph must be nx.readwrite.json_graph.node_link_data(graph_copy).
- Do not return raw NetworkX graph objects as data. Use strings/lists/tables or node-link JSON.
- Missing requested entities must not crash; return the correct schema with empty/zero data and unchanged graph where appropriate.
- Fully self-contained ordinary NetworkX/Python only. Do not call or reference hidden/private/oracle/reference/grader/benchmark helpers, especially names starting solid_step_, private_, oracle_, reference_, ground_truth_, expected_, benchmark_, grader_, or malt_."""

    def _graph_semantics(self) -> str:
        return """Robust MALT graph semantics:
- Normalize names/types by lowercase and removing spaces, punctuation, underscores, hyphens, and EK/RK prefixes.
- Match nodes by id and attrs: name, label, displayName, title, hostname, elementType, type, kind, class, role, deviceType.
- Node type attrs can be strings or lists; EK_SWITCH must match switch/ek switch/EK_SWITCH.
- Edge data can be None, a string, a plain dict, or a dict-of-dicts for MultiGraphs. Before calling .get, check isinstance(x, dict). For dict-of-dicts, iterate values and only inspect dict values.
- Containment edges usually have relationship/rel/type/kind/name/key/label equal to RK_CONTAINS or CONTAINS after normalization.
- Support Graph, DiGraph, MultiGraph, and MultiDiGraph. For containment traversal, consider outgoing and incoming directions when methods exist; use neighbors for undirected graphs.
- Scoped phrases such as in/under/within/below/contained by mean: find the scope node, traverse containment descendants, then filter by requested type/name/attributes.
- Count/list/show/find/what/which/how many/rank/top/total/sum/average/path queries are read-only unless explicit mutation verbs appear: add/create/update/remove/delete/move/connect/fix/configure/place/assign/modify.
- For read-only tasks, compute data only; leave graph_copy structurally and semantically unchanged.
- For mutation tasks, perform the requested change exactly and minimally; preserve unrelated nodes, edges, and attributes.
- Add/create: choose deterministic ids; set minimal name/type/elementType attributes; add only necessary containment/connectivity edges using existing edge style when inferable.
- Update/set/change: change only requested attrs on requested nodes/edges. Remove/delete: remove exactly requested target; remove descendants only if prompt says subtree/children/descendants/all under.
- Rank/top/aggregate: parse numeric strings safely; sort deterministically by value then name then id. Capacity-like attrs include physicalCapacity, capacity, bandwidth, bw, portCapacity, speed, availableBandwidth, usedBandwidth, totalCapacity.
- Avoid clear(), graph-wide relabeling, graph-wide attribute overwrite, randomness, eval/exec, files, subprocesses, network calls, and grader introspection."""

    def _clean_code(self, text: str) -> str:
        s = self._strip_fences((text or "").strip())
        if s.startswith("Answer:"):
            s = s[len("Answer:"):].strip()
        # The requested system wants one top-level process_graph. Keeping only that function is
        # deliberate because prompts require nested helpers, which avoids top-level helper leakage.
        extracted = self._extract_process_graph(s)
        return extracted or s

    def _strip_fences(self, text: str) -> str:
        s = text.strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "````":
                lines = lines[:-1]
            elif lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _extract_process_graph(self, text: str) -> str | None:
        try:
            tree = ast.parse(text)
        except SyntaxError:
            idx = text.find("def process_graph")
            if idx >= 0:
                candidate = text[idx:].strip()
                try:
                    ast.parse(candidate)
                    return candidate
                except SyntaxError:
                    return None
            return None
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "process_graph":
                return ast.get_source_segment(text, node) or text
        return None

    def _normalize_json_response(self, text: str) -> str:
        data = self._extract_json(text)
        return json.dumps(data) if data is not None else "{}"

    def _extract_json(self, text: str | None):
        if not text:
            return None
        s = self._strip_fences(text)
        try:
            return json.loads(s)
        except Exception:
            m = re.search(r"\{.*\}", s, re.DOTALL)
            if not m:
                return None
            try:
                return json.loads(m.group(0))
            except Exception:
                return None

    def _is_acceptable(self, code: str) -> bool:
        return not self._issues(code)

    def _least_bad(self, candidates: list[tuple[str, str]]) -> tuple[str, str]:
        if not candidates:
            return "fallback", self._fallback_response()
        return sorted(candidates, key=lambda x: (len(self._issues(x[1])), self._has_private_helper(x[1]), self._syntax_error(x[1]), len(x[1])))[0]

    def _issues(self, code: str) -> list[str]:
        issues: list[str] = []
        if not code or not code.strip():
            return ["empty"]
        if "```" in code:
            issues.append("markdown fences")
        if self._has_private_helper(code):
            issues.append("private helper/oracle reference")
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return issues + [f"syntax error: {exc.msg}"]
        top_funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
        funcs = [n for n in top_funcs if n.name == "process_graph"]
        if len(funcs) != 1:
            issues.append("must define exactly one process_graph")
            return issues
        if len(top_funcs) != 1:
            issues.append("top-level helper functions are not allowed; nest helpers inside process_graph")
        func = funcs[0]
        if len(func.args.args) != 1 or func.args.args[0].arg != "graph_data":
            issues.append("process_graph must accept graph_data")
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)):
            issues.append("imports are not allowed")
        if not self._uses_graph_copy(func):
            issues.append("missing graph_copy = graph_data.copy()")
        if "updated_graph" not in code or "node_link_data" not in code:
            issues.append("missing updated_graph node_link_data")
        if re.search(r"graph_data\.(add|remove|clear|update)", code):
            issues.append("mutates graph_data")
        if re.search(r"\.clear\s*\(", code):
            issues.append("clear() is unsafe")
        if re.search(r"relabel_nodes|convert_node_labels", code):
            issues.append("graph-wide relabeling is unsafe")
        if any(isinstance(n, ast.Call) and self._call_name(n.func) in {"eval", "exec", "open", "compile", "__import__"} for n in ast.walk(tree)):
            issues.append("unsafe builtin")
        returns = [n for n in ast.walk(func) if isinstance(n, ast.Return)]
        if not returns:
            issues.append("no return")
        for ret in returns:
            if not isinstance(ret.value, ast.Dict):
                issues.append("return must be dict")
                continue
            keys = {k.value for k in ret.value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
            missing = {"type", "data", "updated_graph"} - keys
            if missing:
                issues.append("return dict missing " + ",".join(sorted(missing)))
        return issues

    def _syntax_error(self, code: str) -> bool:
        try:
            ast.parse(code)
            return False
        except SyntaxError:
            return True

    def _has_private_helper(self, code: str) -> bool:
        return bool(PRIVATE_HELPER_RE.search(code or ""))

    def _uses_graph_copy(self, func: ast.FunctionDef) -> bool:
        for n in ast.walk(func):
            if isinstance(n, ast.Assign):
                if any(isinstance(t, ast.Name) and t.id == "graph_copy" for t in n.targets):
                    v = n.value
                    if (
                        isinstance(v, ast.Call)
                        and isinstance(v.func, ast.Attribute)
                        and v.func.attr == "copy"
                        and isinstance(v.func.value, ast.Name)
                        and v.func.value.id == "graph_data"
                    ):
                        return True
        return False

    def _call_name(self, func: ast.AST) -> str | None:
        if isinstance(func, ast.Name):
            return func.id
        if isinstance(func, ast.Attribute):
            return func.attr
        return None

    def _fallback_response(self) -> str:
        return "\n".join([
            "def process_graph(graph_data):",
            "    graph_copy = graph_data.copy()",
            "    return {'type': 'text', 'data': '', 'updated_graph': nx.readwrite.json_graph.node_link_data(graph_copy)}",
        ])
