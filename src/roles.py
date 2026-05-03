from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    name: str
    summary: str
    system_prompt: str


BASE_CONTRACT = """
Return only executable Python code defining exactly one top-level function process_graph(graph_data),
unless the role explicitly asks for JSON. No markdown, prose, or imports. Assume nx is already
available. Inside process_graph, begin with graph_copy = graph_data.copy(). Any helper functions
must be nested inside process_graph. Every return path must return a dict with keys type, data,
updated_graph. updated_graph must be nx.readwrite.json_graph.node_link_data(graph_copy). Never
return a raw NetworkX graph object as data. Never mutate graph_data. Do not use hidden/private/
oracle/reference/grader/benchmark helpers, including names starting solid_step_, private_, oracle_,
reference_, ground_truth_, expected_, benchmark_, grader_, or malt_. Use only ordinary NetworkX
and Python builtins.
""".strip()

GRAPH_SEMANTICS = """
Robust MALT NetworkX rules: normalize names and types by lowercasing and removing spaces,
punctuation, underscores, hyphens, and EK/RK prefixes. Match nodes by id and attrs such as name,
label, displayName, title, hostname, elementType, type, kind, class, role, and deviceType. Node
node-type attrs may be strings or lists. Edge data may be None, a string, a plain dict, or a
dict-of-dicts for MultiGraphs; check isinstance(x, dict) before .get. Containment edges usually have
relationship/rel/type/kind/name/key/label equal to RK_CONTAINS or CONTAINS after normalization.
Support Graph, DiGraph, MultiGraph, and MultiDiGraph; use successors/predecessors only when they
exist, otherwise neighbors. Scoped queries under/in/within/below a node require containment traversal
from the scope and then filtering descendants. Count/list/show/find/what/which/how many/rank/top/
total/sum/average/path are read-only unless explicit mutation verbs appear: add/create/update/remove/
delete/move/connect/fix/configure/place/assign/modify. Read-only tasks must leave graph_copy
unchanged. Mutation tasks must perform exactly the requested state transition and preserve unrelated
nodes, edges, and attributes. Use deterministic ids and sorting. Parse numeric strings for capacity/
rank/aggregate tasks. Avoid clear(), graph-wide relabeling, graph-wide overwrites, randomness,
eval/exec, file/subprocess/network calls, and grader introspection.
""".strip()

CORRECTNESS_PROMPT = (BASE_CONTRACT + "\n\n" + GRAPH_SEMANTICS + "\n\n" +
"You are the CORRECTNESS agent. Maximize benchmark correctness. Implement the exact requested graph operation fully and deterministically. Safety matters, but do not refuse or no-op a mutation that the task actually asks for.").strip()

SAFETY_PROMPT = """
You are the SAFETY agent. Return JSON only; do not write replacement code. Review the proposed
MALT NetworkX solution for concrete safety and runtime risks while preserving correctness. Do not
object to mutations explicitly required by the task. Flag private helper cheating, graph_data mutation,
read-only tasks that mutate graph_copy, broad clear/remove/relabel/overwrite, .get on non-dict edge
data, DiGraph-only traversal that may crash, missing schema, raw graph returns, and missing entity
crashes. Use this JSON schema: {"read_only": true/false, "must_mutate": true/false, "fatal": [],
"safety": [], "runtime": [], "correctness_risks": [], "revision_instructions": []}.
""".strip()

REPAIR_PROMPT = (BASE_CONTRACT + "\n\n" + GRAPH_SEMANTICS + "\n\n" +
"Make the smallest possible correction for the listed mechanical issues. Preserve the algorithm and any required mutation.").strip()

ROLE_SPECS = {
    "coordinator": RoleSpec("coordinator", "Runs a two-agent correctness/safety MALT dialogue.", BASE_CONTRACT),
    "correctness_agent": RoleSpec("correctness_agent", "Correctness-focused MALT solver.", CORRECTNESS_PROMPT),
    "safety_agent": RoleSpec("safety_agent", "Safety-focused MALT reviewer.", SAFETY_PROMPT),
    # Backward-compatible aliases used by older configs.
    "graph_programmer": RoleSpec("graph_programmer", "Correctness-focused MALT solver.", CORRECTNESS_PROMPT),
    "semantic_programmer": RoleSpec("semantic_programmer", "Correctness-focused MALT solver.", CORRECTNESS_PROMPT),
    "invariant_programmer": RoleSpec("invariant_programmer", "Safety-focused MALT reviewer.", SAFETY_PROMPT),
    "repair_agent": RoleSpec("repair_agent", "Minimal mechanical repairer.", REPAIR_PROMPT),
    "arbiter": RoleSpec("arbiter", "Correctness-focused MALT solver.", CORRECTNESS_PROMPT),
    "planner": RoleSpec("planner", "Safety-focused MALT reviewer.", SAFETY_PROMPT),
}

ROLE_ALIASES = {
    "solver": "correctness_agent",
    "coder": "correctness_agent",
    "programmer": "correctness_agent",
    "critic": "safety_agent",
    "reviewer": "safety_agent",
    "safety": "safety_agent",
    "correctness": "correctness_agent",
    "repair": "repair_agent",
}


def get_role(role: str) -> RoleSpec:
    key = (role or "coordinator").strip()
    return ROLE_SPECS.get(ROLE_ALIASES.get(key, key), ROLE_SPECS["coordinator"])
