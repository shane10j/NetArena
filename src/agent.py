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


SUPPORTED_OPS = {
    "ADD_CHILD",
    "REMOVE_SUBTREE",
    "COUNT_CHILDREN",
    "LIST_CHILDREN",
    "RANK",
    "UPDATE_ATTR",
    "QUERY_ATTR",
}

MUTATION_OPS = {"ADD_CHILD", "REMOVE_SUBTREE", "UPDATE_ATTR"}

OP_DSL_SCHEMA = """{
  "ops": [
    {
      "op": "ADD_CHILD | REMOVE_SUBTREE | COUNT_CHILDREN | LIST_CHILDREN | RANK | UPDATE_ATTR | QUERY_ATTR",
      "target_name": "string or null",
      "target_type": "string or null",
      "parent_name": "string or null",
      "child_type_filter": "EK_PORT | EK_PACKET_SWITCH | null",
      "attribute_name": "string or null",
      "attribute_value": "string | number | boolean | null",
      "sort_key": "physical_capacity_bps | string | null",
      "sort_order": "ascending | descending | null",
      "scope": "direct_children | descendants | subtree",
      "edge_type": "RK_CONTAINS",
      "return_type": "text | list | graph",
      "query_after_mutation": true,
      "requires_mutation": true,
      "must_preserve": ["all unrelated nodes", "all unrelated edges", "all attributes"]
    }
  ],
  "final_answer_contract": {
    "type": "text | list | graph",
    "data_shape": "string | list[str] | list[tuple[str, number]] | node_link_json",
    "include_updated_graph": true
  }
}"""


MALT_PRIMITIVE_EXAMPLES = """
Canonical implementation intent:
- ADD_CHILD: find parent by attrs.get("name"), add a node with name/type, add parent -> child RK_CONTAINS edge.
- REMOVE_SUBTREE: find target by attrs.get("name"), remove it and all RK_CONTAINS descendants.
- COUNT_CHILDREN: after prior mutations, count nodes in the requested scope matching child_type_filter.
- LIST_CHILDREN: after prior mutations, list node names in the requested scope.
- RANK: after prior mutations, rank scoped nodes by sort_key; for physical_capacity_bps, sum contained port capacity.
- UPDATE_ATTR: update only the requested target node attribute.
- QUERY_ATTR: return the requested attribute without mutating the graph.
Every output must be {'type': ..., 'data': ..., 'updated_graph': nx.readwrite.json_graph.node_link_data(graph_copy)}.
"""


class Agent:
    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig.from_env()
        self.messenger = Messenger()
        self.llm = LLMClient(self.config)
        self.role = get_role(self.config.role)

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)
        await updater.update_status(
            TaskState.working,
            new_agent_text_message(f"{self.role.name} is preparing a response..."),
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
        plan_text = await self._run_stage(
            role="planner",
            url=self.config.planner_agent_url,
            prompt=self._planner_prompt(input_text),
        )
        plan_data = self._extract_json(plan_text)
        plan_issues = self._plan_issues(plan_data)
        if plan_issues:
            repaired_plan = await self._run_stage(
                role="planner",
                url=self.config.planner_agent_url,
                prompt=self._plan_repair_prompt(input_text, plan_text, plan_issues),
            )
            repaired_plan_data = self._extract_json(repaired_plan)
            repaired_plan_issues = self._plan_issues(repaired_plan_data)
            if repaired_plan_issues:
                return self._fallback_response(input_text)
            plan_data = repaired_plan_data

        if not isinstance(plan_data, dict):
            return self._fallback_response(input_text)

        if self._all_ops_supported(plan_data):
            code = self._template_code(plan_data)
        else:
            code = await self._run_stage(
                role="graph_programmer",
                url=self.config.coder_agent_url,
                prompt=self._coder_prompt(input_text, plan_data),
            )
            code = self._normalize_code(code)

        issues = self._deterministic_issues(code, plan_data)
        if not issues:
            return code

        feedback = json.dumps({"pass": False, "issues": issues}, indent=2)
        repaired = await self._run_stage(
            role="repair_agent",
            url=self.config.repair_agent_url,
            prompt=self._repair_prompt(input_text, plan_data, code, feedback),
        )
        repaired = self._normalize_code(repaired)

        repaired_issues = self._deterministic_issues(repaired, plan_data)
        if not repaired_issues:
            return repaired

        return self._fallback_response(input_text)

    async def _role_response(self, input_text: str) -> str:
        response = await self.llm.complete(
            system_prompt=self.role.system_prompt,
            user_prompt=input_text,
        )
        if self.role.name == "planner":
            return self._normalize_json_response(response or "{}")
        return self._normalize_code(response or self._fallback_response(input_text))

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
        return response or ("{}" if role == "planner" else self._fallback_response(prompt))

    def _planner_prompt(self, input_text: str) -> str:
        return "\n\n".join(
            [
                "Extract a semantic operation DSL for this MALT task.",
                "Return only JSON. Do not include Markdown or prose.",
                "Use this exact schema and preserve operation order:",
                OP_DSL_SCHEMA,
                "Required defaults: edge_type is RK_CONTAINS, scope is direct_children unless the task says descendants/subtree, sort_order is descending for ranking unless specified.",
                "For compositional tasks, each mutation must appear before the query that depends on it.",
                f"Task:\n{input_text}",
            ]
        )

    def _plan_repair_prompt(self, input_text: str, plan_text: str, issues: list[str]) -> str:
        return "\n\n".join(
            [
                "Repair this MALT operation DSL. Return only corrected JSON.",
                "Do not write code. Fill required fields from the task when possible.",
                "Schema:",
                OP_DSL_SCHEMA,
                f"Task:\n{input_text}",
                f"Invalid plan:\n{plan_text}",
                f"Validation issues:\n{json.dumps(issues, indent=2)}",
            ]
        )

    def _coder_prompt(self, input_text: str, plan_data: dict) -> str:
        return "\n\n".join(
            [
                "Generate plain NetworkX code for this MALT task because the DSL has an unsupported operation.",
                "Return only executable Python code defining process_graph(graph_data).",
                "Implement the exact ordered ops from the plan. Do not infer new operations.",
                "Always copy with graph_copy = graph_data.copy().",
                "Find nodes by attrs.get('name'), preserve unrelated attrs, and return type/data/updated_graph on every path.",
                MALT_PRIMITIVE_EXAMPLES,
                f"Original task:\n{input_text}",
                f"Plan:\n{json.dumps(plan_data, indent=2)}",
            ]
        )

    def _repair_prompt(self, input_text: str, plan_data: dict, code: str, feedback: str) -> str:
        return "\n\n".join(
            [
                "Patch this code using the exact failing deterministic checks.",
                "Return only corrected executable Python code.",
                "Keep the exact plan semantics and do not add extra operations.",
                f"Task:\n{input_text}",
                f"Plan:\n{json.dumps(plan_data, indent=2)}",
                f"Current code:\n{code}",
                f"Failing checks:\n{feedback}",
            ]
        )

    def _plan_issues(self, plan_data: object | None) -> list[str]:
        if not isinstance(plan_data, dict):
            return ["Plan is not a JSON object."]

        issues: list[str] = []
        ops = plan_data.get("ops")
        contract = plan_data.get("final_answer_contract")
        if not isinstance(ops, list) or not ops:
            issues.append("Plan must include a non-empty ops list.")
            return issues
        if not isinstance(contract, dict):
            issues.append("Plan must include final_answer_contract.")
        else:
            if contract.get("type") not in {"text", "list", "graph"}:
                issues.append("final_answer_contract.type must be text, list, or graph.")
            if contract.get("include_updated_graph") is not True:
                issues.append("final_answer_contract.include_updated_graph must be true.")

        for index, op in enumerate(ops):
            prefix = f"ops[{index}]"
            if not isinstance(op, dict):
                issues.append(f"{prefix} must be an object.")
                continue
            op_name = op.get("op")
            if not self._value(op_name):
                issues.append(f"{prefix}.op is missing.")
                continue
            if op.get("return_type") not in {"text", "list", "graph"}:
                issues.append(f"{prefix}.return_type must be text, list, or graph.")
            if op.get("edge_type") not in {None, "RK_CONTAINS"}:
                issues.append(f"{prefix}.edge_type must be RK_CONTAINS.")
            if op.get("scope") not in {None, "direct_children", "descendants", "subtree"}:
                issues.append(f"{prefix}.scope must be direct_children, descendants, or subtree.")
            if op_name not in SUPPORTED_OPS:
                continue

            if op_name == "ADD_CHILD":
                self._require(op, prefix, issues, "target_name", "target_type", "parent_name")
            elif op_name == "REMOVE_SUBTREE":
                self._require(op, prefix, issues, "target_name")
            elif op_name == "COUNT_CHILDREN":
                if not self._value(op.get("parent_name")) and not self._value(op.get("target_name")):
                    issues.append(f"{prefix} requires parent_name or target_name.")
                self._require(op, prefix, issues, "child_type_filter")
            elif op_name == "LIST_CHILDREN":
                if not self._value(op.get("parent_name")) and not self._value(op.get("target_name")):
                    issues.append(f"{prefix} requires parent_name or target_name.")
            elif op_name == "RANK":
                if not self._value(op.get("parent_name")) and not self._value(op.get("target_name")):
                    issues.append(f"{prefix} requires parent_name or target_name.")
                self._require(op, prefix, issues, "sort_key", "sort_order")
            elif op_name == "UPDATE_ATTR":
                self._require(op, prefix, issues, "target_name", "attribute_name", "attribute_value")
            elif op_name == "QUERY_ATTR":
                self._require(op, prefix, issues, "target_name", "attribute_name")

        return issues

    def _require(self, op: dict, prefix: str, issues: list[str], *fields: str) -> None:
        for field in fields:
            if not self._value(op.get(field)):
                issues.append(f"{prefix} requires {field}.")

    def _value(self, value: object) -> bool:
        return value is not None and value != ""

    def _all_ops_supported(self, plan_data: dict) -> bool:
        return all(isinstance(op, dict) and op.get("op") in SUPPORTED_OPS for op in plan_data.get("ops", []))

    def _template_code(self, plan_data: dict) -> str:
        plan_literal = repr(plan_data)
        return f'''def process_graph(graph_data):
    plan = {plan_literal}
    graph_copy = graph_data.copy()

    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _has_type(attrs, type_name):
        if not type_name:
            return True
        return type_name in _as_list(attrs.get("type"))

    def _edge_matches(attrs, edge_type):
        return (edge_type or "RK_CONTAINS") in _as_list(attrs.get("type"))

    def _serialize(G):
        return nx.readwrite.json_graph.node_link_data(G)

    def _find_node_by_name(G, name):
        if name is None:
            return None
        for node_id, attrs in G.nodes(data=True):
            if attrs.get("name") == name:
                return node_id
        return None

    def _contained_descendants(G, root_id, edge_type="RK_CONTAINS"):
        if root_id is None or root_id not in G:
            return []
        seen = set()
        ordered = []
        stack = [root_id]
        while stack:
            current = stack.pop()
            for _, child, attrs in G.out_edges(current, data=True):
                if child in seen or not _edge_matches(attrs, edge_type):
                    continue
                seen.add(child)
                ordered.append(child)
                stack.append(child)
        return ordered

    def _scoped_nodes(G, root_id, scope, type_filter=None, edge_type="RK_CONTAINS"):
        if root_id is None or root_id not in G:
            return []
        if scope == "descendants":
            nodes = _contained_descendants(G, root_id, edge_type)
        elif scope == "subtree":
            nodes = [root_id] + _contained_descendants(G, root_id, edge_type)
        else:
            nodes = [child for _, child, attrs in G.out_edges(root_id, data=True) if _edge_matches(attrs, edge_type)]
        return [node_id for node_id in nodes if node_id in G and _has_type(G.nodes[node_id], type_filter)]

    def _node_name(G, node_id):
        return G.nodes[node_id].get("name", node_id)

    def _capacity(G, node_id):
        if node_id is None or node_id not in G:
            return 0.0
        if _has_type(G.nodes[node_id], "EK_PORT"):
            return float(G.nodes[node_id].get("physical_capacity_bps", 0) or 0)
        total = 0.0
        for descendant in _contained_descendants(G, node_id):
            if descendant in G and _has_type(G.nodes[descendant], "EK_PORT"):
                total += float(G.nodes[descendant].get("physical_capacity_bps", 0) or 0)
        return total

    def _root_for_op(op):
        return _find_node_by_name(graph_copy, op.get("parent_name") or op.get("target_name"))

    contract = plan.get("final_answer_contract", {{}})
    result_type = contract.get("type") or "graph"
    result_data = "" if result_type == "text" else []

    for op in plan.get("ops", []):
        op_name = op.get("op")
        edge_type = op.get("edge_type") or "RK_CONTAINS"
        scope = op.get("scope") or "direct_children"

        if op_name == "ADD_CHILD":
            parent_id = _find_node_by_name(graph_copy, op.get("parent_name"))
            existing_id = _find_node_by_name(graph_copy, op.get("target_name"))
            if parent_id is not None and existing_id is None:
                new_id = op.get("target_name")
                graph_copy.add_node(new_id, name=op.get("target_name"), type=[op.get("target_type")])
                graph_copy.add_edge(parent_id, new_id, type=[edge_type])
            result_type = op.get("return_type") or result_type

        elif op_name == "REMOVE_SUBTREE":
            target_id = _find_node_by_name(graph_copy, op.get("target_name"))
            if target_id is not None:
                graph_copy.remove_nodes_from([target_id] + _contained_descendants(graph_copy, target_id, edge_type))
            result_type = op.get("return_type") or result_type

        elif op_name == "UPDATE_ATTR":
            target_id = _find_node_by_name(graph_copy, op.get("target_name"))
            if target_id is not None:
                graph_copy.nodes[target_id][op.get("attribute_name")] = op.get("attribute_value")
            result_type = op.get("return_type") or result_type

        elif op_name == "COUNT_CHILDREN":
            root_id = _root_for_op(op)
            result_data = str(len(_scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)))
            result_type = "text"

        elif op_name == "LIST_CHILDREN":
            root_id = _root_for_op(op)
            nodes = _scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)
            result_data = [_node_name(graph_copy, node_id) for node_id in nodes]
            result_type = "list"

        elif op_name == "RANK":
            root_id = _root_for_op(op)
            nodes = _scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)
            sort_key = op.get("sort_key") or "physical_capacity_bps"
            rows = []
            for node_id in nodes:
                if sort_key == "physical_capacity_bps":
                    value = _capacity(graph_copy, node_id)
                else:
                    value = graph_copy.nodes[node_id].get(sort_key, 0) or 0
                rows.append((_node_name(graph_copy, node_id), value))
            rows.sort(key=lambda item: item[1], reverse=(op.get("sort_order") != "ascending"))
            result_data = rows
            result_type = "list"

        elif op_name == "QUERY_ATTR":
            target_id = _find_node_by_name(graph_copy, op.get("target_name"))
            result_data = "" if target_id is None else str(graph_copy.nodes[target_id].get(op.get("attribute_name"), ""))
            result_type = op.get("return_type") or "text"

    graph_json = _serialize(graph_copy)
    if result_type == "graph":
        result_data = graph_json
    return {{"type": result_type, "data": result_data, "updated_graph": graph_json}}
'''

    def _deterministic_issues(self, code: str, plan_data: dict) -> list[str]:
        issues = self._static_issues(code)
        if issues:
            return issues
        if not self._all_ops_supported(plan_data):
            return self._execution_shape_issues(code, plan_data)
        return self._semantic_issues(code, plan_data)

    def _static_issues(self, code: str) -> list[str]:
        issues = []
        stripped = code.strip()
        if "```" in stripped:
            issues.append("Code contains Markdown fences.")
        try:
            tree = ast.parse(stripped)
        except SyntaxError as exc:
            return [f"SyntaxError: {exc.msg} at line {exc.lineno}."]

        process_graph = self._process_graph_node(tree)
        if process_graph is None:
            issues.append("Missing process_graph(graph_data).")
            return issues
        if len(process_graph.args.args) != 1 or process_graph.args.args[0].arg != "graph_data":
            issues.append("process_graph must accept exactly one argument named graph_data.")
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in ast.walk(tree)):
            issues.append("Generated code should not import packages.")
        if not self._uses_graph_copy(process_graph):
            issues.append("Missing graph_copy = graph_data.copy().")
        if "updated_graph" not in stripped:
            issues.append("Missing updated_graph in return schema.")
        if "node_link_data" not in stripped:
            issues.append("Missing node-link JSON serialization.")
        if re.search(r"graph_data\\.(add|remove|clear|update)", stripped):
            issues.append("Mutates graph_data directly instead of graph_copy.")
        if re.search(r"return\\s+graph_copy\\b", stripped):
            issues.append("Returns raw graph_copy instead of a result dictionary.")
        if re.search(r"['\\\"]data['\\\"]\\s*:\\s*graph_copy\\b", stripped):
            issues.append("Returns raw NetworkX graph object in data.")

        returns = self._process_graph_returns(process_graph)
        if not returns:
            issues.append("process_graph has no return path.")
        for return_node in returns:
            issues.extend(self._return_schema_issues(return_node.value))
        return issues

    def _semantic_issues(self, code: str, plan_data: dict) -> list[str]:
        try:
            import networkx as nx
        except ImportError:
            return ["networkx is unavailable for local semantic execution."]

        namespace: dict[str, object] = {}
        safe_builtins = {
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }
        try:
            exec(code, {"nx": nx, "__builtins__": safe_builtins}, namespace)
        except Exception as exc:
            return [f"Code fails during definition: {type(exc).__name__}: {exc}."]

        process_graph = namespace.get("process_graph")
        if not callable(process_graph):
            return ["process_graph is not callable after execution."]

        issues: list[str] = []
        for index, graph in enumerate(self._semantic_graphs(plan_data, nx), start=1):
            expected = self._reference_result(plan_data, graph, nx)
            try:
                actual = process_graph(graph.copy())
            except Exception as exc:
                issues.append(f"Semantic graph {index} raised {type(exc).__name__}: {exc}.")
                continue
            issues.extend(self._compare_result(actual, expected, index, nx))
        return issues

    def _execution_shape_issues(self, code: str, plan_data: dict) -> list[str]:
        try:
            import networkx as nx
        except ImportError:
            return ["networkx is unavailable for local execution."]

        namespace: dict[str, object] = {}
        safe_builtins = {
            "all": all,
            "any": any,
            "bool": bool,
            "dict": dict,
            "enumerate": enumerate,
            "Exception": Exception,
            "float": float,
            "int": int,
            "isinstance": isinstance,
            "len": len,
            "list": list,
            "max": max,
            "min": min,
            "range": range,
            "set": set,
            "sorted": sorted,
            "str": str,
            "sum": sum,
            "tuple": tuple,
        }
        try:
            exec(code, {"nx": nx, "__builtins__": safe_builtins}, namespace)
        except Exception as exc:
            return [f"Code fails during definition: {type(exc).__name__}: {exc}."]

        process_graph = namespace.get("process_graph")
        if not callable(process_graph):
            return ["process_graph is not callable after execution."]

        issues: list[str] = []
        for index, graph in enumerate(self._semantic_graphs(plan_data, nx), start=1):
            try:
                result = process_graph(graph.copy())
            except Exception as exc:
                issues.append(f"Execution graph {index} raised {type(exc).__name__}: {exc}.")
                continue
            issues.extend(self._result_shape_issues(result, index))
        return issues

    def _result_shape_issues(self, result: object, index: int) -> list[str]:
        if not isinstance(result, dict):
            return [f"Execution graph {index} returned {type(result).__name__}, expected dict."]
        issues: list[str] = []
        required = {"type", "data", "updated_graph"}
        missing = sorted(required - set(result.keys()))
        if missing:
            issues.append(f"Execution graph {index} missing result key(s): {', '.join(missing)}.")
            return issues
        if result.get("type") not in {"text", "list", "table", "graph"}:
            issues.append(f"Execution graph {index} returned invalid type field: {result.get('type')!r}.")
        updated_graph = result.get("updated_graph")
        if not isinstance(updated_graph, dict) or "nodes" not in updated_graph:
            issues.append(f"Execution graph {index} updated_graph is not node-link JSON.")
        if result.get("type") == "text" and not isinstance(result.get("data"), str):
            issues.append(f"Execution graph {index} text data is not a string.")
        if result.get("type") == "graph":
            data = result.get("data")
            if not isinstance(data, dict) or "nodes" not in data:
                issues.append(f"Execution graph {index} graph data is not node-link JSON.")
        return issues

    def _compare_result(self, actual: object, expected: dict, index: int, nx_module: object) -> list[str]:
        if not isinstance(actual, dict):
            return [f"Semantic graph {index} returned {type(actual).__name__}, expected dict."]

        issues: list[str] = []
        required = {"type", "data", "updated_graph"}
        missing = sorted(required - set(actual.keys()))
        if missing:
            issues.append(f"Semantic graph {index} missing result key(s): {', '.join(missing)}.")
            return issues

        if actual.get("type") != expected.get("type"):
            issues.append(f"Semantic graph {index} type mismatch: {actual.get('type')!r} != {expected.get('type')!r}.")
        if self._normalize_data(actual.get("data")) != self._normalize_data(expected.get("data")):
            issues.append(f"Semantic graph {index} data mismatch.")

        actual_graph = self._graph_from_node_link(actual.get("updated_graph"), nx_module)
        expected_graph = self._graph_from_node_link(expected.get("updated_graph"), nx_module)
        if actual_graph is None:
            issues.append(f"Semantic graph {index} updated_graph is not valid node-link JSON.")
        elif expected_graph is not None:
            if self._node_signature(actual_graph) != self._node_signature(expected_graph):
                issues.append(f"Semantic graph {index} updated graph nodes/attributes mismatch.")
            if self._edge_signature(actual_graph) != self._edge_signature(expected_graph):
                issues.append(f"Semantic graph {index} updated graph edges/attributes mismatch.")

        if actual.get("type") == "graph":
            data_graph = self._graph_from_node_link(actual.get("data"), nx_module)
            if data_graph is None:
                issues.append(f"Semantic graph {index} graph data is not valid node-link JSON.")
        if actual.get("type") == "text" and not isinstance(actual.get("data"), str):
            issues.append(f"Semantic graph {index} text data is not a string.")
        return issues

    def _reference_result(self, plan_data: dict, graph: object, nx_module: object) -> dict:
        graph_copy = graph.copy()
        contract = plan_data.get("final_answer_contract", {})
        result_type = contract.get("type") or "graph"
        result_data: object = "" if result_type == "text" else []

        for op in plan_data.get("ops", []):
            op_name = op.get("op")
            edge_type = op.get("edge_type") or "RK_CONTAINS"
            scope = op.get("scope") or "direct_children"
            if op_name == "ADD_CHILD":
                parent_id = self._find_node_by_name(graph_copy, op.get("parent_name"))
                existing_id = self._find_node_by_name(graph_copy, op.get("target_name"))
                if parent_id is not None and existing_id is None:
                    new_id = op.get("target_name")
                    graph_copy.add_node(new_id, name=op.get("target_name"), type=[op.get("target_type")])
                    graph_copy.add_edge(parent_id, new_id, type=[edge_type])
                result_type = op.get("return_type") or result_type
            elif op_name == "REMOVE_SUBTREE":
                target_id = self._find_node_by_name(graph_copy, op.get("target_name"))
                if target_id is not None:
                    graph_copy.remove_nodes_from([target_id] + self._contained_descendants(graph_copy, target_id, edge_type))
                result_type = op.get("return_type") or result_type
            elif op_name == "UPDATE_ATTR":
                target_id = self._find_node_by_name(graph_copy, op.get("target_name"))
                if target_id is not None:
                    graph_copy.nodes[target_id][op.get("attribute_name")] = op.get("attribute_value")
                result_type = op.get("return_type") or result_type
            elif op_name == "COUNT_CHILDREN":
                root_id = self._root_for_op(graph_copy, op)
                nodes = self._scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)
                result_data = str(len(nodes))
                result_type = "text"
            elif op_name == "LIST_CHILDREN":
                root_id = self._root_for_op(graph_copy, op)
                nodes = self._scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)
                result_data = [self._node_name(graph_copy, node_id) for node_id in nodes]
                result_type = "list"
            elif op_name == "RANK":
                root_id = self._root_for_op(graph_copy, op)
                nodes = self._scoped_nodes(graph_copy, root_id, scope, op.get("child_type_filter"), edge_type)
                sort_key = op.get("sort_key") or "physical_capacity_bps"
                rows = []
                for node_id in nodes:
                    if sort_key == "physical_capacity_bps":
                        value = self._capacity(graph_copy, node_id)
                    else:
                        value = graph_copy.nodes[node_id].get(sort_key, 0) or 0
                    rows.append((self._node_name(graph_copy, node_id), value))
                rows.sort(key=lambda item: item[1], reverse=(op.get("sort_order") != "ascending"))
                result_data = rows
                result_type = "list"
            elif op_name == "QUERY_ATTR":
                target_id = self._find_node_by_name(graph_copy, op.get("target_name"))
                result_data = "" if target_id is None else str(graph_copy.nodes[target_id].get(op.get("attribute_name"), ""))
                result_type = op.get("return_type") or "text"

        graph_json = nx_module.readwrite.json_graph.node_link_data(graph_copy)
        if result_type == "graph":
            result_data = graph_json
        return {"type": result_type, "data": result_data, "updated_graph": graph_json}

    def _semantic_graphs(self, plan_data: dict, nx_module: object) -> list[object]:
        names = self._names_from_plan(plan_data)

        def add_base(include_names: bool) -> object:
            G = nx_module.DiGraph()
            parent = names["parents"][0] if include_names and names["parents"] else "smoke.parent"
            target = names["targets"][0] if include_names and names["targets"] else "smoke.target"
            sibling = f"{parent}.sibling"
            port_a = f"{target}.p1"
            port_b = f"{target}.p2"
            parent_port = f"{parent}.p0"
            G.add_node(parent, name=parent, type=["EK_AGG_BLOCK"], preserved="parent")
            G.add_node(target, name=target, type=["EK_PACKET_SWITCH"], preserved="target", status="old")
            G.add_node(sibling, name=sibling, type=["EK_PACKET_SWITCH"], preserved="sibling")
            G.add_node(parent_port, name=parent_port, type=["EK_PORT"], physical_capacity_bps=500.0)
            G.add_node(port_a, name=port_a, type=["EK_PORT"], physical_capacity_bps=1000.0)
            G.add_node(port_b, name=port_b, type=["EK_PORT"], physical_capacity_bps=2000.0)
            G.add_edge(parent, target, type=["RK_CONTAINS"], preserved="edge")
            G.add_edge(parent, sibling, type=["RK_CONTAINS"])
            G.add_edge(parent, parent_port, type=["RK_CONTAINS"])
            G.add_edge(target, port_a, type=["RK_CONTAINS"])
            G.add_edge(target, port_b, type=["RK_CONTAINS"])
            for extra_parent in names["parents"][1:]:
                if include_names and extra_parent not in G:
                    G.add_node(extra_parent, name=extra_parent, type=["EK_AGG_BLOCK"])
                    G.add_edge(extra_parent, parent, type=["RK_CONTAINS"])
            for extra_target in names["targets"][1:]:
                if include_names and extra_target not in G:
                    G.add_node(extra_target, name=extra_target, type=["EK_PACKET_SWITCH"], status="old")
                    G.add_edge(parent, extra_target, type=["RK_CONTAINS"])
            return G

        return [add_base(True), add_base(False)]

    def _names_from_plan(self, plan_data: dict) -> dict[str, list[str]]:
        names = {"parents": [], "targets": []}
        for op in plan_data.get("ops", []):
            if not isinstance(op, dict):
                continue
            parent_name = op.get("parent_name")
            target_name = op.get("target_name")
            if isinstance(parent_name, str) and parent_name and parent_name not in names["parents"]:
                names["parents"].append(parent_name)
            if isinstance(target_name, str) and target_name and op.get("op") != "ADD_CHILD" and target_name not in names["targets"]:
                names["targets"].append(target_name)
        return names

    def _process_graph_node(self, tree: ast.AST) -> ast.FunctionDef | None:
        for node in getattr(tree, "body", []):
            if isinstance(node, ast.FunctionDef) and node.name == "process_graph":
                return node
        return None

    def _process_graph_returns(self, process_graph: ast.FunctionDef) -> list[ast.Return]:
        class ReturnCollector(ast.NodeVisitor):
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

        collector = ReturnCollector()
        collector.visit(process_graph)
        return collector.returns

    def _uses_graph_copy(self, process_graph: ast.FunctionDef) -> bool:
        for node in ast.walk(process_graph):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(target, ast.Name) and target.id == "graph_copy" for target in node.targets):
                continue
            value = node.value
            if (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Attribute)
                and value.func.attr == "copy"
                and isinstance(value.func.value, ast.Name)
                and value.func.value.id == "graph_data"
            ):
                return True
        return False

    def _return_schema_issues(self, value: ast.AST | None) -> list[str]:
        if not isinstance(value, ast.Dict):
            return ["Return paths must return an explicit dictionary with type, data, and updated_graph."]
        keys = {key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
        missing = sorted({"type", "data", "updated_graph"} - keys)
        issues = []
        if missing:
            issues.append(f"Return dictionary missing required key(s): {', '.join(missing)}.")
        for key, item in zip(value.keys, value.values):
            if isinstance(key, ast.Constant) and key.value == "data" and isinstance(item, ast.Name) and item.id == "graph_copy":
                issues.append("Return dictionary uses raw graph_copy as data.")
        return issues

    def _root_for_op(self, graph: object, op: dict) -> object | None:
        return self._find_node_by_name(graph, op.get("parent_name") or op.get("target_name"))

    def _find_node_by_name(self, graph: object, name: str | None) -> object | None:
        if name is None:
            return None
        for node_id, attrs in graph.nodes(data=True):
            if attrs.get("name") == name:
                return node_id
        return None

    def _as_list(self, value: object) -> list:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    def _has_type(self, attrs: dict, type_name: str | None) -> bool:
        return not type_name or type_name in self._as_list(attrs.get("type"))

    def _edge_matches(self, attrs: dict, edge_type: str = "RK_CONTAINS") -> bool:
        return edge_type in self._as_list(attrs.get("type"))

    def _contained_descendants(self, graph: object, root_id: object, edge_type: str = "RK_CONTAINS") -> list:
        if root_id is None or root_id not in graph:
            return []
        seen = set()
        ordered = []
        stack = [root_id]
        while stack:
            current = stack.pop()
            for _, child, attrs in graph.out_edges(current, data=True):
                if child in seen or not self._edge_matches(attrs, edge_type):
                    continue
                seen.add(child)
                ordered.append(child)
                stack.append(child)
        return ordered

    def _scoped_nodes(self, graph: object, root_id: object | None, scope: str, type_filter: str | None, edge_type: str) -> list:
        if root_id is None or root_id not in graph:
            return []
        if scope == "descendants":
            nodes = self._contained_descendants(graph, root_id, edge_type)
        elif scope == "subtree":
            nodes = [root_id] + self._contained_descendants(graph, root_id, edge_type)
        else:
            nodes = [child for _, child, attrs in graph.out_edges(root_id, data=True) if self._edge_matches(attrs, edge_type)]
        return [node_id for node_id in nodes if node_id in graph and self._has_type(graph.nodes[node_id], type_filter)]

    def _capacity(self, graph: object, node_id: object) -> float:
        if node_id is None or node_id not in graph:
            return 0.0
        if self._has_type(graph.nodes[node_id], "EK_PORT"):
            return float(graph.nodes[node_id].get("physical_capacity_bps", 0) or 0)
        total = 0.0
        for descendant in self._contained_descendants(graph, node_id):
            if descendant in graph and self._has_type(graph.nodes[descendant], "EK_PORT"):
                total += float(graph.nodes[descendant].get("physical_capacity_bps", 0) or 0)
        return total

    def _node_name(self, graph: object, node_id: object) -> object:
        return graph.nodes[node_id].get("name", node_id)

    def _graph_from_node_link(self, data: object, nx_module: object) -> object | None:
        if not isinstance(data, dict) or "nodes" not in data:
            return None
        try:
            return nx_module.readwrite.json_graph.node_link_graph(data)
        except Exception:
            return None

    def _node_signature(self, graph: object) -> dict:
        return {attrs.get("name", node_id): self._freeze(attrs) for node_id, attrs in graph.nodes(data=True)}

    def _edge_signature(self, graph: object) -> list:
        edges = []
        for source, target, attrs in graph.edges(data=True):
            source_name = graph.nodes[source].get("name", source)
            target_name = graph.nodes[target].get("name", target)
            edges.append((source_name, target_name, self._freeze(attrs)))
        return sorted(edges, key=repr)

    def _normalize_data(self, value: object) -> object:
        if isinstance(value, tuple):
            return [self._normalize_data(item) for item in value]
        if isinstance(value, list):
            return [self._normalize_data(item) for item in value]
        if isinstance(value, dict):
            return {key: self._normalize_data(item) for key, item in value.items()}
        return value

    def _freeze(self, value: object) -> object:
        if isinstance(value, dict):
            return tuple(sorted((key, self._freeze(item)) for key, item in value.items()))
        if isinstance(value, list):
            return tuple(self._freeze(item) for item in value)
        return value

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
        return json.dumps(data)

    def _normalize_code(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("Answer:"):
            stripped = stripped[len("Answer:") :].strip()
        return self._strip_fences(stripped)

    def _strip_fences(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            stripped = "\n".join(lines).strip()
        return stripped

    def _fallback_response(self, input_text: str) -> str:
        return "\n".join(
            [
                "def process_graph(graph_data):",
                "    graph_copy = graph_data.copy()",
                "    def serialize(G):",
                "        return nx.readwrite.json_graph.node_link_data(G)",
                "    graph_json = serialize(graph_copy)",
                "    return {'type': 'graph', 'data': graph_json, 'updated_graph': graph_json}",
            ]
        )
