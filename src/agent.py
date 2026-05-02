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
  "read_only": true,
  "must_mutate": false,
  "task_summary": "one sentence",
  "targets": {"names": [], "types": [], "attributes": [], "scopes": []},
  "operations": ["ordered atomic graph operations"],
  "expected_return": {"type": "text|list|table|graph", "shape": "description"},
  "safety_invariants": ["preserve unrelated graph state"]
}"""


@dataclass(frozen=True)
class Candidate:
    label: str
    code: str
    static_issues: tuple[str, ...]
    safety_notes: tuple[str, ...]
    score: float


class Agent:
    """Diverse MALT agent optimized from observed regressions.

    The best observed correctness came from diverse generation plus an arbiter. The
    later regressions came from over-hard safety gates and too many repairs, which
    filtered or rewrote semantically strong candidates. This version keeps the
    correctness-producing diversity, makes safety a selector preference rather than
    a brittle hard gate, and repairs only the selected code if it has concrete
    contract/safety problems.
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
            new_agent_text_message(f"{self.role.name} is generating benchmark code..."),
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
        # Planner + direct solver in parallel keeps latency down and gives one
        # candidate that is not over-conditioned by a potentially imperfect plan.
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

        specs = [
            ("direct", direct_raw or ""),
        ]

        # Three complementary candidates recover the correctness gains seen in the
        # 0.857 run, but without the expensive repair-all loop that hurt safety.
        jobs = [
            self._run_stage(
                role="graph_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, plan, "general"),
            ),
            self._run_stage(
                role="semantic_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, plan, "semantic"),
            ),
            self._run_stage(
                role="invariant_programmer",
                url=getattr(self.config, "coder_agent_url", None),
                prompt=self._solver_prompt(input_text, plan, "invariant"),
            ),
        ]
        generated = await asyncio.gather(*jobs)
        specs.extend((label, raw or "") for label, raw in zip(["general", "semantic", "invariant"], generated))

        read_only_hint = self._read_only_hint(input_text, plan)
        candidates = [self._make_candidate(label, raw, input_text, read_only_hint) for label, raw in specs]

        # First let the LLM choose among existing candidates. This preserves the
        # semantic boost from the arbiter while avoiding unsafe synthesized rewrites.
        selected = await self._select_candidate(input_text, plan, candidates, read_only_hint)

        # If the selected code is already valid, return it. Safety notes are soft
        # unless they indicate concrete high-risk mutation; the hidden safety metric
        # often rewards exact correct state, so over-filtering hurts both metrics.
        if not selected.static_issues and not self._has_hard_safety_issue(selected.safety_notes):
            return selected.code

        # Repair only the selected candidate, minimally. This restores contract
        # compliance without rewriting every candidate and increasing variance.
        repaired_raw = await self._run_stage(
            role="repair_agent",
            url=getattr(self.config, "repair_agent_url", None),
            prompt=self._repair_prompt(input_text, plan, selected),
        )
        repaired = self._make_candidate("selected_repaired", repaired_raw or "", input_text, read_only_hint)
        if not repaired.static_issues and not self._has_hard_safety_issue(repaired.safety_notes):
            return repaired.code

        # Deterministic fallback selection: prefer executable, then high semantic
        # score, then fewest safety concerns. This is intentionally not a hard gate.
        viable = [c for c in candidates + [repaired] if not self._fatal_issues(c.static_issues)]
        if viable:
            viable.sort(key=lambda c: (self._has_hard_safety_issue(c.safety_notes), len(c.static_issues), -c.score, len(c.safety_notes)))
            return viable[0].code
        return self._fallback_response()

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(system_prompt=self.role.system_prompt, user_prompt=input_text)
        if self.role.name in {"planner", "task_analyst", "critic", "arbiter"}:
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
        response = await self.llm.complete(system_prompt=role_spec.system_prompt, user_prompt=prompt)
        return response or ("{}" if role in {"planner", "task_analyst", "critic", "arbiter"} else "")

    def _planner_prompt(self, input_text: str) -> str:
        return "\n\n".join([
            "Plan this MALT NetworkX task as a NetArena state/action episode.",
            "Return only JSON. Do not write code. Do not include markdown or prose.",
            "Do not invent ids or facts. Classify read_only vs must_mutate carefully.",
            "Schema:",
            PLAN_SCHEMA,
            f"Task:\n{input_text}",
        ])

    def _solver_prompt(self, input_text: str, plan: str, style: str) -> str:
        style_lines = {
            "direct": [
                "Solve directly from the user task. Use the plan only if it is empty-safe; do not depend on it.",
                "Favor a robust complete implementation with helpers.",
            ],
            "general": [
                "Use the plan to implement the expected graph operations exactly.",
                "Balance hidden-test correctness with minimal safe mutation.",
            ],
            "semantic": [
                "Prioritize semantic correctness: exact entities, containment traversal, type normalization, and expected return shape.",
                "Do not skip necessary mutations when the prompt explicitly asks for them.",
            ],
            "invariant": [
                "Prioritize safety and invariants: preserve unrelated graph state and avoid broad graph operations.",
                "Still perform the exact requested mutation if the task explicitly requires one.",
            ],
        }[style]
        return "\n\n".join([
            "Return only executable Python code defining process_graph(graph_data). No imports, markdown, or explanation.",
            self._contract_block(),
            self._malt_rules_block(),
            *style_lines,
            "Implementation advice: define local helper functions for normalize, node label/type matching, containment edge detection/traversal, numeric parsing, deterministic sort, and graph serialization when useful.",
            "Use graph_copy = graph_data.copy() and never mutate graph_data.",
            f"Planner JSON:\n{plan}",
            f"Original task:\n{input_text}",
        ])

    async def _select_candidate(self, input_text: str, plan: str, candidates: list[Candidate], read_only_hint: bool) -> Candidate:
        ordered = sorted(candidates, key=lambda c: c.score, reverse=True)
        summary = []
        for i, c in enumerate(ordered):
            summary.append({
                "index": i,
                "label": c.label,
                "score": round(c.score, 2),
                "static_issues": list(c.static_issues),
                "safety_notes": list(c.safety_notes),
                "code": c.code[:9000],
            })
        prompt = "\n\n".join([
            "Choose the best existing implementation. Return JSON only: {\"best_index\": <int>, \"reason\": \"short\"}.",
            "Prioritize hidden benchmark correctness, then safety/minimal mutation, then contract compliance. Do not synthesize code.",
            f"Read-only hint: {read_only_hint}",
            f"Task:\n{input_text}",
            f"Planner JSON:\n{plan}",
            "Candidates in ranked order:",
            json.dumps(summary),
        ])
        response = await self._run_stage(role="critic", url=None, prompt=prompt)
        data = self._extract_json(response)
        if isinstance(data, dict):
            try:
                idx = int(data.get("best_index"))
                if 0 <= idx < len(ordered):
                    return ordered[idx]
            except Exception:
                pass
        return ordered[0]

    def _repair_prompt(self, input_text: str, plan: str, candidate: Candidate) -> str:
        return "\n\n".join([
            "Repair this selected MALT implementation with the smallest necessary patch. Return only code.",
            "Do not rewrite a sound algorithm. Fix only concrete issues listed below.",
            self._contract_block(),
            self._malt_rules_block(),
            f"Task:\n{input_text}",
            f"Planner JSON:\n{plan}",
            "Static issues:",
            json.dumps(list(candidate.static_issues)),
            "Safety notes:",
            json.dumps(list(candidate.safety_notes)),
            "Candidate code:",
            candidate.code,
        ])

    def _contract_block(self) -> str:
        return """Mandatory contract:
- Define process_graph(graph_data) only.
- Assume nx is available; do not import anything.
- Start by copying: graph_copy = graph_data.copy().
- Never mutate graph_data directly.
- Return {'type': ..., 'data': ..., 'updated_graph': ...} on every path.
- updated_graph must be nx.readwrite.json_graph.node_link_data(graph_copy).
- Never return a raw NetworkX graph object in data; graph data must be node-link JSON.
- Missing requested entities should not raise exceptions."""

    def _malt_rules_block(self) -> str:
        return """MALT graph rules:
- Match by node id and attrs: name, label, displayName, hostname, elementType, type, kind, class, role.
- Normalize type strings by lowercasing and removing ek/punctuation/spaces/underscores/hyphens.
- Containment edges are usually RK_CONTAINS/CONTAINS in relationship/rel/type/kind/name/key; support Graph/DiGraph/MultiGraph.
- For scoped queries, traverse containment descendants and filter by requested type/name/attrs.
- Read-only tasks (count/list/show/find/what/which/how many/rank/top/total/sum/average/path) should not mutate graph_copy.
- Mutation tasks require explicit change verbs; mutate only explicit target/scope and preserve unrelated state.
- Add/create deterministic unique ids and minimal inferred attrs; update only requested attrs; delete only requested targets.
- Sort deterministic outputs by human name then id. Ensure all data is JSON-serializable.
- Avoid clear(), graph-wide relabeling, graph-wide attr updates, randomness, external files, subprocesses, and network calls."""

    def _make_candidate(self, label: str, raw: str, input_text: str, read_only_hint: bool) -> Candidate:
        code = self._normalize_code(raw or "")
        code = self._extract_process_graph_source(code) or code
        static = tuple(self._static_issues(code))
        safety = tuple() if self._fatal_issues(static) else tuple(self._safety_notes(code, read_only_hint))
        score = self._score(code, static, safety, input_text, read_only_hint)
        return Candidate(label, code, static, safety, score)

    def _score(self, code: str, static: tuple[str, ...], safety: tuple[str, ...], input_text: str, read_only_hint: bool) -> float:
        if not code.strip():
            return -10000.0
        score = 100.0
        for issue in static:
            score -= 90 if self._fatal_issues([issue]) else 28
        for note in safety:
            score -= 35 if self._hard_note(note) else 10
        bonuses = {
            "node_link_data": 8,
            "graph_copy = graph_data.copy()": 8,
            ".nodes(data=True)": 5,
            "lower()": 3,
            "sorted": 4,
            "RK_CONTAINS": 5,
            "CONTAINS": 3,
            "is_multigraph": 3,
            "successors": 2,
            "predecessors": 2,
            "def norm": 2,
            "def normalize": 2,
        }
        for needle, bonus in bonuses.items():
            if needle in code:
                score += bonus
        if read_only_hint and not self._mutates_graph_copy(code):
            score += 12
        if not read_only_hint and self._mutates_graph_copy(code):
            # Small bonus only: many mutation prompts need this, but the critic still decides.
            score += 3
        lines = len(code.splitlines())
        if 35 <= lines <= 210:
            score += 3
        elif lines > 260:
            score -= min(25, (lines - 260) / 8)
        if "except Exception" in code or "except:" in code:
            score -= 5
        return score

    def _read_only_hint(self, input_text: str, plan: str) -> bool:
        data = self._extract_json(plan)
        if isinstance(data, dict):
            if data.get("must_mutate") is True:
                return False
            if data.get("read_only") is True:
                return True
        t = input_text.lower()
        mutation = r"\b(add|create|insert|delete|remove|update|set|change|rename|move|connect|disconnect|attach|detach|fix|repair|configure|assign|place|allocate|modify)\b"
        query = r"\b(count|how many|list|show|find|what|which|rank|top|bottom|most|least|total|sum|average|path|shortest|longest|return|get|display)\b"
        if re.search(mutation, t):
            return False
        return bool(re.search(query, t))

    def _safety_notes(self, code: str, read_only_hint: bool) -> list[str]:
        notes: list[str] = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return notes
        func = self._process_graph_node(tree)
        if func is None:
            return notes
        if read_only_hint and self._mutates_graph_copy(code, func):
            notes.append("read-only-hint mutation")
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                root = self._root_name(node.func.value)
                attr = node.func.attr
                if root == "graph_copy" and attr in {"clear", "clear_edges"}:
                    notes.append("hard broad clear")
                if root == "graph_copy" and attr in {"remove_nodes_from", "remove_edges_from"} and node.args:
                    arg = ast.unparse(node.args[0]) if hasattr(ast, "unparse") else ""
                    if "graph_copy.nodes" in arg or "graph_copy.edges" in arg or arg.strip().startswith("list(graph_copy"):
                        notes.append("hard broad removal")
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    txt = ast.unparse(target) if hasattr(ast, "unparse") else ""
                    if txt == "graph_copy" and not self._is_graph_copy_assignment(node):
                        notes.append("hard replaces graph_copy")
        if "nx.relabel" in code or "convert_node_labels_to_integers" in code:
            notes.append("hard broad relabel")
        return notes

    def _has_hard_safety_issue(self, notes: tuple[str, ...] | list[str]) -> bool:
        return any(self._hard_note(n) for n in notes)

    def _hard_note(self, note: str) -> bool:
        return note.startswith("hard")

    def _mutates_graph_copy(self, code: str, func: ast.FunctionDef | None = None) -> bool:
        if func is None:
            try:
                tree = ast.parse(code)
                func = self._process_graph_node(tree)
            except SyntaxError:
                return False
        if func is None:
            return False
        mutating = {
            "add_node", "add_nodes_from", "add_edge", "add_edges_from",
            "remove_node", "remove_nodes_from", "remove_edge", "remove_edges_from",
            "clear", "clear_edges", "update", "set_node_attributes", "set_edge_attributes",
        }
        for node in ast.walk(func):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                root = self._root_name(node.func.value)
                if attr in mutating and (root == "graph_copy" or (attr.startswith("set_") and node.args and self._name_is(node.args[0], "graph_copy"))):
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
        for return_node in returns:
            issues.extend(self._return_schema_issues(return_node.value))
        return issues

    def _fatal_issues(self, issues: tuple[str, ...] | list[str]) -> bool:
        return any(issue.startswith("SyntaxError") or "Missing process_graph" in issue or "Empty code" in issue for issue in issues)

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
