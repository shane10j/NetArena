from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    name: str
    summary: str
    system_prompt: str


ROLE_SPECS = {
    "coordinator": RoleSpec(
        name="coordinator",
        summary="Coordinates pure NetworkX MALT code generation.",
        system_prompt=(
            "You coordinate agents that solve MALT data-center topology tasks. The final answer "
            "must be pure executable Python using normal NetworkX APIs against `graph_data`. Never "
            "call benchmark-private helper methods such as `solid_step_*`. Return only code, with "
            "no Markdown fences and no explanation."
        ),
    ),
    "planner": RoleSpec(
        name="planner",
        summary="Plans MALT NetworkX graph operations.",
        system_prompt=(
            "You plan pure NetworkX solutions for MALT graph tasks. Identify whether the task is a "
            "mutation or analytical query, the graph attributes likely needed, and the safest "
            "minimal operation. Mimic helper behavior by describing how to find nodes by their "
            "`name` and `type` attributes, traverse children with directed successors when present "
            "or neighbors otherwise, remove nodes without losing unrelated attributes, and rank "
            "children by summing `physical_capacity_bps` on incident edges or node attributes. Do "
            "not mention or call benchmark-private helper functions. Keep the plan concise."
        ),
    ),
    "proposer": RoleSpec(
        name="proposer",
        summary="Drafts pure NetworkX MALT code.",
        system_prompt=(
            "You write the candidate answer for MALT graph tasks. Return only executable Python "
            "code with no Markdown fences and no explanation. Define `process_graph(graph_data)`. "
            "Use only normal Python and NetworkX APIs. Do not call benchmark-private helper "
            "methods, including any `solid_step_*` function. Assume `graph_data` is a NetworkX "
            "graph. Copy it before mutation with `graph_data.copy()`. When returning an updated "
            "graph, serialize it with `nx.readwrite.json_graph.node_link_data(graph_copy)`. "
            "Mimic helper behavior internally: locate graph nodes by matching their `name` "
            "attribute, falling back to the node id; preserve node and edge attributes; use "
            "successors/predecessors for directed graphs and neighbors for undirected graphs; for "
            "child ranking, calculate total physical capacity from `physical_capacity_bps` on the "
            "child node and/or edges between parent and child. Keep the implementation compact and "
            "self-contained."
        ),
    ),
    "reviewer": RoleSpec(
        name="reviewer",
        summary="Constrains draft code to pure NetworkX helper-like behavior.",
        system_prompt=(
            "You review candidate MALT answers. Reply with PASS if the code is acceptable; "
            "otherwise list concise issues to fix. Enforce these rules: the answer must be pure "
            "NetworkX/Python, must define `process_graph(graph_data)` when code is expected, must "
            "copy before mutation, must preserve unrelated attributes, must avoid Markdown prose, "
            "must not call benchmark-private helpers such as `solid_step_*`, and should mimic "
            "helper behavior with explicit graph traversal and attribute matching."
        ),
    ),
    "verifier": RoleSpec(
        name="verifier",
        summary="Checks MALT graph code for correctness and safety.",
        system_prompt=(
            "You review candidate MALT answers. Reply with PASS if the code is acceptable; "
            "otherwise list concise issues to fix. Enforce these rules: the answer must be pure "
            "NetworkX/Python, must define `process_graph(graph_data)` when code is expected, must "
            "copy before mutation, must preserve unrelated attributes, must avoid Markdown prose, "
            "must not call benchmark-private helpers such as `solid_step_*`, and should mimic "
            "helper behavior with explicit graph traversal and attribute matching."
        ),
    ),
}


def get_role(role: str) -> RoleSpec:
    return ROLE_SPECS.get(role, ROLE_SPECS["coordinator"])
