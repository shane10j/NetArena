# Multi-Agent Implementation Notes

The current agent is intentionally simple and LLM-first:

```text
Task
-> Planner
-> Graph Programmer
-> Reviewer / Repairer
-> Final code
```

## Roles

`planner` reads the task and returns JSON with a short task summary, ordered graph operations,
mentioned entities, expected return shape, and safety notes. It does not write code.

`graph_programmer` writes the full `process_graph(graph_data)` implementation using ordinary
NetworkX operations from first principles.

`repair_agent` reviews the draft once. It either returns the same code or a minimally patched version
that fixes syntax, schema, graph-copy, serialization, or obvious graph-semantic issues.

The coordinator only performs lightweight formatting and static checks around the LLM calls.

## Code Contract

- Define `process_graph(graph_data)`.
- Copy before work with `graph_copy = graph_data.copy()`.
- Find nodes by `attrs.get("name")`, not assumed node IDs.
- Preserve unrelated node and edge attributes.
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

The coordinator receives MALT traffic as `malt_operator`; the planner, programmer, and repairer
containers stay on the same Docker network and are only called by the coordinator.
