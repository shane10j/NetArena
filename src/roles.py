from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    name: str
    summary: str
    system_prompt: str


GLOBAL_CONTRACT = """
You solve MALT / NetArena-style NetworkX graph benchmark tasks.
Return ONLY executable Python code. No markdown, no prose, no imports.
Define exactly one public function: process_graph(graph_data).
Assume nx is already available. The function must copy the input first with
`graph_copy = graph_data.copy()` and must never mutate graph_data directly.
Every return path must include type, data, and updated_graph. updated_graph must
be nx.readwrite.json_graph.node_link_data(graph_copy). Graph data must be
node-link JSON; text/count data must be a string.
""".strip()


MALT_HELPER_RULES = """
For canonical MALT operations, prefer the benchmark-provided primitives when they
are available in the execution namespace:
- solid_step_add_node_to_graph(graph_copy, {'type': EK_TYPE, 'name': name}, parent_node_name)
- solid_step_remove_node_from_graph(graph_copy, child_node_name)
- solid_step_list_child_nodes(graph_copy, {'type': parent_type_or_None, 'name': parent_name})
- solid_step_rank_child_nodes(graph_copy, parent_node_name)
- solid_step_counting_query(graph_copy, parent_node_object, {'type': child_type, 'name': None})
These primitives match the benchmark state/action semantics better than ad-hoc
NetworkX traversal. Use graph_copy, not graph_data. After removals, clean only
isolated nodes created by the mutation. Always serialize updated_graph.
""".strip()


PLANNER_PROMPT = """
Return JSON only. Extract the task into a small operation DSL with ops and
final_answer_contract. Use operations ADD_CHILD, REMOVE_SUBTREE,
COUNT_CHILDREN, LIST_CHILDREN, RANK, UPDATE_ATTR, QUERY_ATTR when possible.
Preserve operation order: mutations first, then the requested query on the
updated graph. Include target_name, target_type, parent_name, child_type_filter,
attribute_name, return_type, and whether mutation is required.
""".strip()


SOLVER_PROMPT = (GLOBAL_CONTRACT + "\n\n" + MALT_HELPER_RULES + """
Write one complete process_graph implementation and nothing else.
Prioritize hidden benchmark correctness, then safety. For known MALT prompt
families, compile the prompt into the canonical helper calls instead of manually
reimplementing graph traversal. Preserve unrelated graph state.
""").strip()


SEMANTIC_PROMPT = (GLOBAL_CONTRACT + "\n\n" + MALT_HELPER_RULES + """
You are the semantic-correctness solver. Return code only. Preserve exact
operation order, node names, EK_* types, and return shape. Use helper calls for
add/remove/list/rank/count whenever possible.
""").strip()


INVARIANT_PROMPT = (GLOBAL_CONTRACT + "\n\n" + MALT_HELPER_RULES + """
You are the safety-focused solver. Return code only. Copy graph first, mutate
only requested targets, preserve unrelated state, serialize graph outputs, return
count data as strings, and clean isolated nodes only after requested removals.
""").strip()


REPAIR_PROMPT = (GLOBAL_CONTRACT + "\n\n" + MALT_HELPER_RULES + """
Repair the selected implementation with the smallest possible patch. Return code
only. Fix syntax, schema, missing updated_graph, direct graph_data mutation, raw
NetworkX graph returns, and unsafe removal cleanup while preserving semantics.
""").strip()


ROLE_SPECS = {
    "coordinator": RoleSpec("coordinator", "Coordinates deterministic MALT templates and LLM fallback.", GLOBAL_CONTRACT),
    "planner": RoleSpec("planner", "Extracts semantic graph-operation plans.", PLANNER_PROMPT),
    "task_analyst": RoleSpec("task_analyst", "Alias for planner.", PLANNER_PROMPT),
    "graph_programmer": RoleSpec("graph_programmer", "General MALT solver.", SOLVER_PROMPT),
    "graph_solver": RoleSpec("graph_solver", "Direct MALT solver.", SOLVER_PROMPT),
    "semantic_programmer": RoleSpec("semantic_programmer", "Semantic MALT solver.", SEMANTIC_PROMPT),
    "invariant_programmer": RoleSpec("invariant_programmer", "Safety-preserving MALT solver.", INVARIANT_PROMPT),
    "repair_agent": RoleSpec("repair_agent", "Minimally repairs MALT code.", REPAIR_PROMPT),
    "critic": RoleSpec("critic", "Selects candidate JSON.", "Return JSON only: {'best_index': int, 'reason': str}."),
    "arbiter": RoleSpec("arbiter", "Returns final MALT code.", SOLVER_PROMPT),
}


ROLE_ALIASES = {
    "solver": "graph_solver",
    "coder": "graph_programmer",
    "programmer": "graph_programmer",
    "planned_solver": "semantic_programmer",
    "invariant_solver": "invariant_programmer",
    "analyst": "task_analyst",
    "analysis": "task_analyst",
    "reviewer": "repair_agent",
    "repair": "repair_agent",
}


def get_role(role: str) -> RoleSpec:
    key = (role or "coordinator").strip()
    canonical = ROLE_ALIASES.get(key, key)
    return ROLE_SPECS.get(canonical, ROLE_SPECS["coordinator"])
