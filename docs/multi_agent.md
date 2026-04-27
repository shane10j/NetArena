# Multi-Agent Implementation Notes

The current agent uses a three-stage aligned pipeline instead of free-form agent discussion:

```text
Task
-> Planner
-> Template/Code Generator
-> Deterministic Semantic Verifier/Repair
-> Final code
```

## Roles

`planner` returns a semantic JSON operation DSL. It extracts an ordered `ops` list plus the
`final_answer_contract` so the generator gets concrete actions instead of vague intent.

`template/code generator` emits plain NetworkX code for the seven supported ops:
`ADD_CHILD`, `REMOVE_SUBTREE`, `COUNT_CHILDREN`, `LIST_CHILDREN`, `RANK`, `UPDATE_ATTR`, and
`QUERY_ATTR`. The `graph_programmer` LLM is only used when the DSL contains an unsupported op.

`deterministic semantic verifier` validates the DSL before coding, then checks generated code with
static checks and execution-based semantic tests on synthetic graphs.

`repair_agent` is called once when deterministic checks fail, and it receives the exact failing
static or semantic checks. If repair fails, the coordinator returns a conservative safe fallback.

## Deterministic Checks

The coordinator runs local checks for syntax, `process_graph` shape, Markdown fences, graph-copy
usage, consistent `type`/`data`/`updated_graph` returns, node-link JSON serialization, direct
`graph_data` mutation, raw NetworkX graph returns, and executable behavior on synthetic semantic
graphs.

The semantic tests compare generated code against an independent reference evaluator for:
add child, remove subtree, count/list/rank after mutation, update attr, query attr, direct vs
descendant scope, graph serialization, and preservation of unrelated nodes/edges/attributes.

## Key Safety Invariants

- Define `process_graph(graph_data)`.
- Copy before mutation with `graph_copy = graph_data.copy()`.
- Find nodes by `attrs.get("name")`, not assumed node IDs.
- Preserve unrelated node and edge attributes.
- Handle descendants safely when removing topology nodes.
- Avoid removing isolated nodes unless they were created by the requested mutation.
- Serialize updated graphs with `nx.readwrite.json_graph.node_link_data(graph_copy)`.
- Return the same schema on every path: `type`, `data`, and `updated_graph`.
- Return counts as text when the prompt asks for text.
- Do not return raw NetworkX graph objects.

## Running Locally

Build the image once:

```bash
docker build --platform linux/amd64 --no-cache -t purple_agent_shane:latest .
```

Then run the MALT compose stack with the multi-agent override:

```bash
cd netarena_leaderboard
docker compose -f docker-compose.yml -f docker-compose.multi-agent.yml down -v
docker compose -f docker-compose.yml -f docker-compose.multi-agent.yml up \
  --pull never \
  --timestamps \
  --no-color \
  --exit-code-from agentbeats-client \
  --abort-on-container-exit
```

The coordinator receives MALT traffic as `malt_operator`; the other role containers stay on the
same Docker network and are only called by the coordinator.
