import ast
import asyncio
import json
import re
from dataclasses import dataclass

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState
from a2a.utils import get_message_text, new_agent_text_message

from config import AgentConfig
from llm import LLMClient
from messenger import Messenger
from roles import get_role


PLAN_SCHEMA = """{
  "task_summary": "one sentence",
  "read_only": true,
  "must_mutate": false,
  "expected_return_type": "text | list | table | graph",
  "entities": {
    "names": ["explicit node names"],
    "types": ["explicit EK_* or natural types"],
    "attributes": ["explicit attributes"],
    "scopes": ["parents / containers / query roots"]
  },
  "operations": ["ordered atomic state/action operations"],
  "safety_invariants": ["preserve unrelated nodes, edges, attributes"]
}"""


@dataclass(frozen=True)
class Candidate:
    label: str
    code: str
    issues: tuple[str, ...]
    hazards: tuple[str, ...]
    score: float


class Agent:
    """Correctness-first MALT agent with lightweight safety discipline.

    The observed peak came from diversity + an arbiter. Regressions came from
    aggressive safety gating, over-selection, and gratuitous repairs that either
    rejected correct programs or rewrote them into weaker code. This version
    returns to the peak-producing topology: plan/direct in parallel, two planned
    specialists in parallel, then one synthesis arbiter. Safety is handled mainly
    by prompt discipline and only truly mechanical contract checks; the coordinator
    does not second-guess graph semantics with brittle heuristics.
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
        # Round 1: get a compact semantic plan while also producing an
        # independent direct implementation. The direct candidate often wins
        # when the plan is slightly wrong.
        plan_task = asyncio.create_task(
            self._run_stage(
                role="task_analyst",
                url=getattr(self.config, "planner_agent_url", None),
                prompt=self._planner_prompt(input_text),
            )
        )
        direct_task = asyncio.create_task(
            self._run_stage(
                role="graph_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, "{}", "direct"),
            )
        )
        plan_raw, direct_raw = await asyncio.gather(plan_task, direct_task)
        plan = self._normalize_json_response(plan_raw or "{}")

        # Round 2: two complementary planned solvers. These are parallel to keep
        # latency close to the previous high-performing version.
        semantic_task = asyncio.create_task(
            self._run_stage(
                role="semantic_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, plan, "semantic"),
            )
        )
        invariant_task = asyncio.create_task(
            self._run_stage(
                role="invariant_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, plan, "invariant"),
            )
        )
        semantic_raw, invariant_raw = await asyncio.gather(semantic_task, invariant_task)

        candidates = [
            self._candidate("direct", direct_raw or "", input_text, plan),
            self._candidate("semantic", semantic_raw or "", input_text, plan),
            self._candidate("invariant", invariant_raw or "", input_text, plan),
        ]

        # If a candidate is already excellent and the task is a simple read-only
        # query, avoid arbiter drift. This is deliberately narrow: broad routing
        # through a local selector caused regressions.
        if self._very_simple_read_only(input_text, plan):
            clean = [c for c in candidates if not c.issues and not c.hazards]
            if clean:
                clean.sort(key=lambda c: (-c.score, len(c.code)))
                return clean[0].code

        # Round 3: one correctness-oriented arbiter. It sees only compact issue
        # metadata and candidate code; it is told to copy the best candidate unless
        # a small, obvious merge is needed. This is the component that previously
        # raised correctness, so keep it instead of brittle hard gating.
        arbiter_raw = await self._run_stage(
            role="arbiter",
            url=None,
            prompt=self._arbiter_prompt(input_text, plan, candidates),
        )
        arbiter = self._candidate("arbiter", arbiter_raw or "", input_text, plan)
        if not arbiter.issues and not self._has_hard_hazard(arbiter.hazards):
            return arbiter.code

        # One minimal repair only for mechanical problems. Do not repair clean
        # originals just because they may be semantically imperfect.
        if arbiter.code.strip() and not self._fatal(arbiter.issues):
            repaired_raw = await self._run_stage(
                role="repair_agent",
                url=getattr(self.config, "repair_agent_url", None),
                prompt=self._repair_prompt(input_text, plan, arbiter),
            )
            repaired = self._candidate("repaired", repaired_raw or "", input_text, plan)
            if not repaired.issues and not self._has_hard_hazard(repaired.hazards):
                return repaired.code
            candidates.append(repaired)

        candidates.append(arbiter)
        return self._best_candidate(candidates).code

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        if self.role.name in {"planner", "task_analyst", "critic"}:
            return self._normalize_json_response(response or "{}")
        code = self._normalize_code(response or "")
        code = self._extract_process_graph_source(code) or code
        return code or self._fallback_response()

    async def _run_stage(self, *, role: str, url: str | None, prompt: str) -> str:
        if url:
            try:
                return await self.messenger.talk_to_agent(prompt, url)
            except Exception as exc:
                print(f"{role} delegation failed: {exc}")
        role_spec = get_role(role)
        print(f"Calling LiteLLM model for {role}: {self.config.model_name or '<none configured>'}")
        response = await self.llm.complete(
            system_prompt=role_spec.system_prompt,
            user_prompt=prompt,
        )
        return response or ("{}" if role in {"planner", "task_analyst", "critic"} else "")

    def _planner_prompt(self, input_text: str) -> str:
        return "\n\n".join([
            "Plan this MALT/NetArena NetworkX task. Return JSON only; no markdown, no prose, no code.",
            "Be literal: extract only entities and operations stated in the task. Do not invent ids.",
            "Classify read_only=false only when the requested final answer requires changing the graph state.",
            "Schema:",
            PLAN_SCHEMA,
            f"Task:\n{input_text}",
        ])

    def _solver_prompt(self, input_text: str, plan: str, style: str) -> str:
        style_text = {
            "direct": (
                "Solve directly from the user task. Use robust first-principles NetworkX code. "
                "Do not overfit to the planner; this candidate should be independently correct."
            ),
            "semantic": (
                "Prioritize hidden-test semantic correctness: identify exact names, types, attributes, "
                "containment scopes, expected return type, and required state transition."
            ),
            "invariant": (
                "Prioritize correctness under safety invariants: perform the required operation but keep "
                "all unrelated nodes, edges, and attributes bit-for-bit unchanged."
            ),
        }[style]
        return "\n\n".join([
            "Return ONLY executable Python code defining process_graph(graph_data).",
            self._contract_block(),
            self._semantics_block(),
            style_text,
            "Use local helper functions when helpful, but keep the implementation focused and deterministic.",
            "Important: many hidden tasks are simple. Prefer exact, small logic over broad generic rewriting.",
            f"Planner JSON:\n{plan}",
            f"Original task:\n{input_text}",
        ])

    def _arbiter_prompt(self, input_text: str, plan: str, candidates: list[Candidate]) -> str:
        ordered = sorted(candidates, key=lambda c: (-c.score, len(c.issues), len(c.hazards)))
        payload = []
        for i, c in enumerate(ordered):
            payload.append({
                "index": i,
                "label": c.label,
                "score": round(c.score, 2),
                "contract_issues": list(c.issues),
                "safety_hazards": list(c.hazards),
                "code": c.code[:12000],
            })
        return "\n\n".join([
            "You are the final MALT arbiter. Return ONLY executable Python code defining process_graph(graph_data).",
            self._contract_block(),
            self._semantics_block(),
            "Decision policy:",
            "1. Prefer the candidate most likely to pass hidden correctness.",
            "2. Prefer copying a strong candidate unchanged over rewriting it.",
            "3. Only synthesize a small merge when a candidate has correct task logic but another has a better helper/return schema.",
            "4. Preserve safety: read-only tasks leave graph_copy unchanged; mutation tasks change only explicit targets.",
            "5. Never introduce broad deletes, clear(), relabeling, randomness, imports, subprocesses, files, or network calls.",
            "6. Ensure every return is JSON-serializable and includes updated_graph node-link JSON.",
            f"Planner JSON:\n{plan}",
            f"Task:\n{input_text}",
            "Candidate implementations:",
            json.dumps(payload),
        ])

    def _repair_prompt(self, input_text: str, plan: str, candidate: Candidate) -> str:
        return "\n\n".join([
            "Repair the selected implementation with the smallest mechanical patch. Return code only.",
            "Do not rewrite its algorithm unless needed to fix syntax, contract, serialization, or an explicit hard safety hazard.",
            self._contract_block(),
            self._semantics_block(),
            f"Task:\n{input_text}",
            f"Planner JSON:\n{plan}",
            "Contract issues:",
            json.dumps(list(candidate.issues)),
            "Safety hazards:",
            json.dumps(list(candidate.hazards)),
            "Candidate code:",
            candidate.code,
        ])

    def _contract_block(self) -> str:
        return """Mandatory output contract:
- Define exactly one public function: process_graph(graph_data).
- Assume nx is already available; do not import anything.
- The first graph operation must preserve input state: graph_copy = graph_data.copy().
- Never mutate graph_data directly.
- Every return path returns {'type': ..., 'data': ..., 'updated_graph': ...}.
- updated_graph is nx.readwrite.json_graph.node_link_data(graph_copy).
- If returning graph data in data, serialize it using node_link_data; never return a raw NetworkX graph object.
- Missing entities must not raise; return a safe empty/unchanged result with the same schema."""

    def _semantics_block(self) -> str:
        return """MALT graph semantics:
- Match nodes by id string and attrs: name, label, displayName, hostname, elementType, type, kind, class, role.
- Normalize strings by lowercasing and stripping punctuation, spaces, hyphens, underscores, and the EK/RK prefixes for type/relationship matching.
- Containment edges often use RK_CONTAINS or CONTAINS in relationship, rel, type, kind, name, or key. Support Graph, DiGraph, MultiGraph, and MultiDiGraph.
- For under/in/within/contained-by scopes, traverse containment descendants from the scoped parent, then filter by requested type/name/attributes.
- Deterministic output order: human name/label first, then node id string.
- Read-only questions such as count/list/show/find/what/which/how many/rank/top/total/sum/average/path must compute only and leave graph_copy unchanged.
- Mutation questions use explicit verbs such as add/create/update/set/change/remove/delete/move/connect/disconnect/attach/detach/fix/repair/configure/assign/place/allocate/modify. Mutate only the named or scoped targets.
- Add/create: generate deterministic unique ids, infer elementType/type and containment edge style from similar siblings when possible, and add only required nodes/edges.
- Update/set/change: modify only requested attributes on explicit target nodes/edges.
- Remove/delete: remove only explicit targets; remove descendants only if asked for subtree/children/descendants/all under.
- Capacity/bandwidth numeric attrs include physicalCapacity, capacity, bandwidth, bw, portCapacity, speed, availableBandwidth, usedBandwidth; parse numeric strings safely.
- Preserve unrelated nodes, edges, and attributes exactly."""

    def _candidate(self, label: str, raw: str, input_text: str, plan: str) -> Candidate:
        code = self._normalize_code(raw or "")
        code = self._extract_process_graph_source(code) or code
        issues = tuple(self._static_issues(code))
        hazards = tuple() if self._fatal(issues) else tuple(self._hazards(code, input_text, plan))
        score = self._score(code, issues, hazards, input_text, plan)
        return Candidate(label, code, issues, hazards, score)

    def _score(self, code: str, issues: tuple[str, ...], hazards: tuple[str, ...], input_text: str, plan: str) -> float:
        if not code.strip():
            return -9999.0
        score = 100.0
        for issue in issues:
            score -= 100 if self._fatal([issue]) else 25
        for hazard in hazards:
            score -= 80 if hazard.startswith("hard") else 8
        # Useful but lightweight heuristics; avoid over-penalizing longer robust code.
        for needle, bonus in {
            "node_link_data": 8,
            "graph_copy = graph_data.copy()": 8,
            ".nodes(data=True)": 4,
            "lower()": 3,
            "sorted": 3,
            "RK_CONTAINS": 4,
            "CONTAINS": 3,
            "is_multigraph": 3,
            "successors": 2,
            "predecessors": 2,
            "def norm": 2,
            "def normalize": 2,
        }.items():
            if needle in code:
                score += bonus
        if self._looks_read_only(input_text, plan) and not self._mutates_graph_copy(code):
            score += 6
        line_count = len(code.splitlines())
        if 25 <= line_count <= 240:
            score += 3
        elif line_count > 320:
            score -= min(20, (line_count - 320) / 12)
        return score

    def _best_candidate(self, candidates: list[Candidate]) -> Candidate:
        def key(c: Candidate) -> tuple:
            return (
                self._fatal(c.issues),
                self._has_hard_hazard(c.hazards),
                len(c.issues),
                len(c.hazards),
                -c.score,
                len(c.code),
            )
        viable = sorted(candidates, key=key)
        return viable[0] if viable else Candidate("fallback", self._fallback_response(), tuple(), tuple(), -9999)

    def _very_simple_read_only(self, input_text: str, plan: str) -> bool:
        # Only bypass the arbiter for unmistakable simple queries where synthesis
        # tends to introduce unnecessary mutations. Do not classify rank/top/path
        # as simple because those often need careful semantics.
        if not self._looks_read_only(input_text, plan):
            return False
        t = input_text.lower()
        if re.search(r"\b(add|create|delete|remove|update|set|change|move|connect|disconnect|fix|repair|configure|assign|place|allocate|modify)\b", t):
            return False
        return bool(re.search(r"\b(count|how many|list|show|find|what|which|get|display)\b", t)) and not bool(re.search(r"\b(rank|top|bottom|path|shortest|longest|total|sum|average)\b", t))

    def _looks_read_only(self, input_text: str, plan: str) -> bool:
        data = self._extract_json(plan)
        if isinstance(data, dict):
            if data.get("must_mutate") is True:
                return False
            if data.get("read_only") is True:
                return True
        t = input_text.lower()
        if re.search(r"\b(add|create|insert|delete|remove|update|set|change|rename|move|connect|disconnect|attach|detach|fix|repair|configure|assign|place|allocate|modify)\b", t):
            return False
        return bool(re.search(r"\b(count|how many|list|show|find|what|which|rank|top|bottom|total|sum|average|path|shortest|longest|get|display|return)\b", t))

    def _hazards(self, code: str, input_text: str, plan: str) -> list[str]:
        hazards: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return hazards
        func = self._process_graph_node(tree)
        if func is None:
            return hazards
        read_only = self._looks_read_only(input_text, plan)
        if read_only and self._mutates_graph_copy(code, func):
            hazards.append("soft read-only mutation")
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                root = self._root_name(node.func.value)
                attr = node.func.attr
                if root == "graph_copy" and attr in {"clear", "clear_edges"}:
                    hazards.append("hard broad clear")
                if root == "graph_copy" and attr in {"remove_nodes_from", "remove_edges_from"} and node.args:
                    arg = ast.unparse(node.args[0]) if hasattr(ast, "unparse") else ""
                    if "graph_copy.nodes" in arg or "graph_copy.edges" in arg or arg.strip().startswith("list(graph_copy"):
                        hazards.append("hard broad removal")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    target_txt = ast.unparse(target) if hasattr(ast, "unparse") else ""
                    if target_txt == "graph_copy" and not self._is_graph_copy_assignment(node):
                        hazards.append("hard replaces graph_copy")
        if "nx.relabel" in code or "convert_node_labels_to_integers" in code:
            hazards.append("hard broad relabel")
        if "random." in code or "uuid" in code or "subprocess" in code or "open(" in code:
            hazards.append("hard nondeterministic/external side effect")
        return hazards

    def _has_hard_hazard(self, hazards: tuple[str, ...] | list[str]) -> bool:
        return any(h.startswith("hard") for h in hazards)

    def _fatal(self, issues: tuple[str, ...] | list[str]) -> bool:
        return any(i.startswith("SyntaxError") or "Missing process_graph" in i or "Empty code" in i for i in issues)

    def _mutates_graph_copy(self, code: str, func: ast.FunctionDef | None = None) -> bool:
        if func is None:
            try:
                tree = ast.parse(code)
                func = self._process_graph_node(tree)
            except SyntaxError:
                return False
        if func is None:
            return False
        mutating_methods = {
            "add_node", "add_nodes_from", "add_edge", "add_edges_from", "remove_node", "remove_nodes_from",
            "remove_edge", "remove_edges_from", "clear", "clear_edges", "update", "set_node_attributes",
            "set_edge_attributes",
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                root = self._root_name(node.func.value)
                if attr in mutating_methods and (root == "graph_copy" or (attr.startswith("set_") and node.args and self._name_is(node.args[0], "graph_copy"))):
                    return True
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    txt = ast.unparse(target) if hasattr(ast, "unparse") else ""
                    if txt.startswith("graph_copy.nodes") or txt.startswith("graph_copy.edges") or txt.startswith("graph_copy["):
                        return True
        return False

    def _static_issues(self, code: str) -> list[str]:
        stripped = (code or "").strip()
        issues: list[str] = []
        if not stripped:
            return ["Empty code response."]
        if "```" in stripped:
            issues.append("Code contains Markdown fences.")
        if re.search(r"(^|\n)\s*(Here is|This code|Explanation:|Answer:)\b", stripped):
            issues.append("Code contains prose outside the function.")
        try:
            tree = ast.parse(stripped)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc.msg} at line {exc.lineno}."]
        process_graph = self._process_graph_node(tree)
        if process_graph is None:
            return ["Missing process_graph(graph_data)."]
        if len(process_graph.args.args) != 1 or process_graph.args.args[0].arg != "graph_data":
            issues.append("process_graph must accept exactly one argument named graph_data.")
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            issues.append("Generated code should not import packages; nx is already available.")
        if not self._uses_graph_copy(process_graph):
            issues.append("Missing graph_copy = graph_data.copy().")
        if self._mutates_graph_data(process_graph):
            issues.append("Mutates graph_data directly instead of graph_copy.")
        if "updated_graph" not in stripped:
            issues.append("Missing updated_graph in return schema.")
        if "node_link_data" not in stripped:
            issues.append("Missing node-link JSON serialization.")
        if re.search(r"return\s+graph_copy\b", stripped):
            issues.append("Returns raw graph_copy instead of a result dictionary.")
        if re.search(r"['\"]data['\"]\s*:\s*graph_copy\b", stripped):
            issues.append("Returns raw NetworkX graph object in data.")
        returns = self._process_graph_returns(process_graph)
        if not returns:
            issues.append("process_graph has no return path.")
        for ret in returns:
            issues.extend(self._return_schema_issues(ret.value))
        return issues

    def _process_graph_node(self, tree: ast.AST) -> ast.FunctionDef | None:
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == "process_graph":
                return node
        return None

    def _process_graph_returns(self, process_graph: ast.FunctionDef) -> list[ast.Return]:
        class Collector(ast.NodeVisitor):
            def __init__(self) -> None:
                self.returns: list[ast.Return] = []
            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                if node is process_graph:
                    self.generic_visit(node)
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                return
            def visit_Lambda(self, node: ast.Lambda) -> None:
                return
            def visit_Return(self, node: ast.Return) -> None:
                self.returns.append(node)
        collector = Collector()
        collector.visit(process_graph)
        return collector.returns

    def _uses_graph_copy(self, process_graph: ast.FunctionDef) -> bool:
        for node in ast.walk(process_graph):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "graph_copy" for t in node.targets):
                continue
            if self._is_graph_copy_assignment(node):
                return True
        return False

    def _is_graph_copy_assignment(self, node: ast.Assign) -> bool:
        v = node.value
        return (
            isinstance(v, ast.Call)
            and isinstance(v.func, ast.Attribute)
            and v.func.attr == "copy"
            and isinstance(v.func.value, ast.Name)
            and v.func.value.id == "graph_data"
        )

    def _mutates_graph_data(self, process_graph: ast.FunctionDef) -> bool:
        mutating = {
            "add_node", "add_nodes_from", "add_edge", "add_edges_from", "remove_node",
            "remove_nodes_from", "remove_edge", "remove_edges_from", "clear", "clear_edges", "update",
        }
        for node in ast.walk(process_graph):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                if node.func.attr in mutating and self._root_name(node.func.value) == "graph_data":
                    return True
            if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    txt = ast.unparse(target) if hasattr(ast, "unparse") else ""
                    if txt.startswith("graph_data[") or txt.startswith("graph_data.nodes") or txt.startswith("graph_data.edges"):
                        return True
        return False

    def _return_schema_issues(self, value: ast.AST | None) -> list[str]:
        if not isinstance(value, ast.Dict):
            return ["Return paths must return an explicit dictionary with type, data, and updated_graph."]
        keys = {k.value for k in value.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
        missing = sorted({"type", "data", "updated_graph"} - keys)
        issues = []
        if missing:
            issues.append(f"Return dictionary missing required key(s): {', '.join(missing)}.")
        for key, item in zip(value.keys, value.values):
            if isinstance(key, ast.Constant) and key.value == "data" and isinstance(item, ast.Name) and item.id == "graph_copy":
                issues.append("Return dictionary uses raw graph_copy as data.")
        return issues

    def _root_name(self, node: ast.AST) -> str | None:
        cur = node
        while isinstance(cur, ast.Attribute):
            cur = cur.value
        if isinstance(cur, ast.Subscript):
            cur = cur.value
            while isinstance(cur, ast.Attribute):
                cur = cur.value
        if isinstance(cur, ast.Name):
            return cur.id
        return None

    def _name_is(self, node: ast.AST, name: str) -> bool:
        return isinstance(node, ast.Name) and node.id == name

    def _extract_json(self, text: str | None) -> object | None:
        if not text:
            return None
        normalized = self._strip_fences(text)
        try:
            return json.loads(normalized)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", normalized, re.DOTALL)
            if not match:
                return None
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None

    def _normalize_json_response(self, text: str) -> str:
        data = self._extract_json(text)
        if data is None:
            return "{}"
        return json.dumps(data, separators=(",", ":"))

    def _normalize_code(self, text: str) -> str:
        stripped = (text or "").strip()
        for prefix in ("Answer:", "Code:", "Python:"):
            if stripped.startswith(prefix):
                stripped = stripped[len(prefix):].strip()
        return self._strip_fences(stripped)

    def _strip_fences(self, text: str) -> str:
        stripped = (text or "").strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].lstrip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _extract_process_graph_source(self, text: str) -> str | None:
        stripped = self._strip_fences(text)
        try:
            tree = ast.parse(stripped)
        except SyntaxError:
            match = re.search(r"(^|\n)(def\s+process_graph\s*\(\s*graph_data\s*\)\s*:\n.*)", stripped, re.DOTALL)
            if not match:
                return None
            candidate = match.group(2).rstrip()
            try:
                ast.parse(candidate)
                return candidate
            except SyntaxError:
                return None
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == "process_graph":
                lines = stripped.splitlines()
                if hasattr(node, "end_lineno") and node.end_lineno is not None:
                    return "\n".join(lines[node.lineno - 1: node.end_lineno]).strip()
        return None

    def _fallback_response(self) -> str:
        return "\n".join([
            "def process_graph(graph_data):",
            "    graph_copy = graph_data.copy()",
            "    graph_json = nx.readwrite.json_graph.node_link_data(graph_copy)",
            "    return {'type': 'graph', 'data': graph_json, 'updated_graph': graph_json}",
        ])
