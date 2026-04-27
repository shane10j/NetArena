from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    name: str
    summary: str
    system_prompt: str


ROLE_SPECS = {
    "coordinator": RoleSpec(
        name="coordinator",
        summary="Orchestrates the three-stage MALT graph-code pipeline.",
        system_prompt=(
            "You coordinate a MALT NetworkX pipeline. The final benchmark response must be only "
            "executable Python code defining process_graph(graph_data)."
        ),
    ),
    "planner": RoleSpec(
        name="planner",
        summary="Extracts semantic graph-operation DSL plans.",
        system_prompt=(
            "You are the planning stage for MALT NetworkX tasks. Return only JSON, no Markdown and "
            "no prose. Extract a semantic operation DSL with keys ops and final_answer_contract. "
            "Each op must use ADD_CHILD, REMOVE_SUBTREE, COUNT_CHILDREN, LIST_CHILDREN, RANK, "
            "UPDATE_ATTR, or QUERY_ATTR when possible. Preserve operation order. Include target_name, "
            "target_type, parent_name, child_type_filter, attribute_name, attribute_value, sort_key, "
            "sort_order, scope, edge_type, return_type, query_after_mutation, requires_mutation, and "
            "must_preserve. Use null only when a field is genuinely irrelevant. Distinguish "
            "direct_children, descendants, and subtree. final_answer_contract must include type, "
            "data_shape, and include_updated_graph."
        ),
    ),
    "graph_programmer": RoleSpec(
        name="graph_programmer",
        summary="Writes fallback plain NetworkX code for unsupported DSL ops.",
        system_prompt=(
            "You write fallback benchmark answers as plain executable Python code. Return only code. "
            "Define process_graph(graph_data). Implement the exact ordered ops from the plan and do "
            "not infer new operations. Always create graph_copy = graph_data.copy(). Use attrs.get('name') "
            "for node lookup. Preserve unrelated nodes, edges, and attributes. Every return path must "
            "include type, data, and updated_graph. updated_graph must be node-link JSON. Text/count "
            "data must be a string; graph data must be node-link JSON."
        ),
    ),
    "repair_agent": RoleSpec(
        name="repair_agent",
        summary="Patches code against deterministic failing tests.",
        system_prompt=(
            "You repair MALT NetworkX code using deterministic failing checks. Return only corrected "
            "executable Python code, no Markdown and no explanation. Make the smallest change that "
            "fixes the listed failures while preserving the exact plan semantics. Keep graph_copy "
            "usage, name-attribute lookup, node-link JSON serialization, consistent type/data/updated_graph "
            "returns, and safe missing-node behavior."
        ),
    ),
}


def get_role(role: str) -> RoleSpec:
    return ROLE_SPECS.get(role, ROLE_SPECS["coordinator"])
