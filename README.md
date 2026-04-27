# NetArena MALT Purple Agent

A basic [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) purple agent for the NetArena MALT
data-center planning benchmark.

## Project Structure

```
src/
├─ server.py      # Server setup, role selection, and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Planner/template generator/deterministic repair control flow
├─ agent_old.py   # Archived placeholder from an earlier experiment
├─ llm.py         # LiteLLM adapter
├─ roles.py       # Role prompts and metadata
├─ config.py      # Runtime config from env/Amber
└─ messenger.py   # A2A messaging utilities
manifests/
└─ purple-agent-component.json5 # Reusable Amber component manifest
docs/
└─ multi_agent.md # Typed multi-agent workflow guide
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber root manifest wiring the role containers
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Improve the MALT behavior** - Extend [`src/agent.py`](src/agent.py) and
   [`src/roles.py`](src/roles.py) with stronger NetworkX graph reasoning, code generation, and
   answer formatting.

2. **Configure model access** - The agent uses the LiteLLM adapter in [`src/llm.py`](src/llm.py).
   Set `MODEL_NAME` and `LITELLM_*` variables or bind the Amber `llm` slot during scenario
   orchestration.

3. **Set the image** - Replace the placeholder image in [`amber-manifest.json5`](amber-manifest.json5)
   or provide it as Amber config.

4. **Write benchmark-specific tests** - Add custom tests for your agent in [`tests/test_agent.py`](tests/test_agent.py).

## Running Locally

```bash
# Install dependencies
uv sync

# Run the server
uv run src/server.py
```

Run a specific role:

```bash
uv run src/server.py --role planner --port 9011
```

## Running with Docker

```bash
# Build the image
docker build -t my-agent .

# Run the container
docker run -p 9009:9009 my-agent
```

The default container role is `coordinator`. Use `--role planner`, `--role graph_programmer`, or
`--role repair_agent` to run the other LLM stages from the same image.

## Multi-Agent Layout

The current agent is a three-stage aligned implementation:

- `planner`: extracts a semantic JSON operation DSL.
- template/code generator: deterministically emits plain NetworkX code for supported ops; the
  `graph_programmer` LLM is used only for unsupported DSL operations.
- deterministic semantic verifier/repair: validates the plan, checks syntax/schema, executes
  semantic tests on synthetic graphs, and calls `repair_agent` once with exact failures.
- `coordinator`: orchestrates the stages and exposes the public A2A endpoint.

[`amber-manifest.json5`](amber-manifest.json5) is a root manifest that launches multiple component
instances from [`manifests/purple-agent-component.json5`](manifests/purple-agent-component.json5).
It binds the role A2A exports into the coordinator and exports only the coordinator's A2A endpoint
for the benchmark.

Useful validation commands once Amber is available:

```bash
amber docs manifest
amber check amber-manifest.json5
```

You can also run the same pattern manually with three local LLM-stage processes:

```bash
uv run src/server.py --role planner --port 9011 --model-name openai/gpt-4.1
uv run src/server.py --role graph_programmer --port 9012 --model-name openai/gpt-4.1
uv run src/server.py --role repair_agent --port 9014 --model-name openai/gpt-4.1
PLANNER_AGENT_URL=http://127.0.0.1:9011 \
CODER_AGENT_URL=http://127.0.0.1:9012 \
REPAIR_AGENT_URL=http://127.0.0.1:9014 \
uv run src/server.py --role coordinator --port 9009 --model-name openai/gpt-4.1
```

The same image can serve every role; only `--role` and the stage URLs change. See
[`docs/multi_agent.md`](docs/multi_agent.md) for the full local Docker workflow.

## MALT Leaderboard Submission

The [`netarena_leaderboard`](netarena_leaderboard) folder is the AgentBeats leaderboard repo for
NetArena. MALT is the data-center planning benchmark: the green agent sends text prompts describing
a NetworkX/MALT topology task, and the purple agent should answer with Python code or a direct
answer that satisfies the prompt. This agent uses LLM-generated NetworkX code, deterministic local
checks, and staged repair before returning final code.

The leaderboard has two ways to identify agents:

- `image = "..."` works for local testing only.
- `agentbeats_id = "..."` is required in GitHub Actions submissions.

### 1. Publish And Register The Agent

Build and publish a Docker image for this repo, or let this repo's GitHub Actions publish one to
GHCR after a push to `main` or a semver tag.

```bash
docker build -t ghcr.io/<your-user>/<your-repo>:latest .
docker push ghcr.io/<your-user>/<your-repo>:latest
```

Then register the image as a purple agent on [AgentBeats](https://agentbeats.dev/). After
registration, copy the purple agent ID from the agent page. The AgentBeats tutorial describes this
as the "Register Agent" flow and exposes a "Copy agent ID" button.

Make sure the Docker image is public or readable from the leaderboard workflow. If the image is in a
private GHCR package, add a `GHCR_TOKEN` secret in your leaderboard fork with permission to pull it.

### 2. Configure MALT

For local testing, edit [`netarena_leaderboard/malt_scenario.toml`](netarena_leaderboard/malt_scenario.toml)
to point at the image directly:

```toml
[[participants]]
image = "ghcr.io/<your-user>/<your-repo>:latest"
name = "malt_operator"
env = {
  OPENAI_API_KEY = "${OPENAI_API_KEY}",
  OPENAI_API_BASE = "https://api.tokenfactory.nebius.com/v1/",
  MODEL_NAME = "openai/gpt-4.1"
}
```

For an actual leaderboard submission, use the AgentBeats ID instead:

```toml
[[participants]]
agentbeats_id = "<your-purple-agent-id>"
name = "malt_operator"
env = {
  OPENAI_API_KEY = "${OPENAI_API_KEY}",
  OPENAI_API_BASE = "https://api.tokenfactory.nebius.com/v1/",
  MODEL_NAME = "openai/gpt-4.1"
}
```

The agent accepts both the reference server CLI flags (`--model-name`, `--api-key`,
`--api-base-url`, `--api-version`) and env vars: `MODEL_NAME`, `OPENAI_API_KEY`, `OPENAI_API_BASE`,
`AZURE_API_KEY`, `AZURE_API_BASE`, and the equivalent `LITELLM_*` names.

Keep the existing MALT green agent and `[config]` section unless you intentionally want to change the
evaluation size:

```toml
[green_agent]
agentbeats_id = "019ba416-0462-7cf2-86f0-bf85123df8a4"
env = { LOG_LEVEL = "INFO" }
```

### 3. Local Dry Run

Install the leaderboard helper dependencies, generate the assessment compose file, and run it:

```bash
cd netarena_leaderboard
python -m pip install tomli tomli-w pyyaml requests
python generate_compose.py --scenario malt_scenario.toml --app malt
cp .env.example .env
# Fill in .env values such as OPENAI_API_KEY before running.
docker compose up --timestamps --no-color --exit-code-from agentbeats-client --abort-on-container-exit
```

The generated files are ignored by the leaderboard repo: `docker-compose.yml`, `a2a-scenario.toml`,
`.env.example`, `.env`, and `output/`. Results appear under `netarena_leaderboard/output/`.

### 4. Submit To The Leaderboard

Fork the NetArena leaderboard repo, enable GitHub Actions in the fork, and add any referenced secrets
from `malt_scenario.toml` under **Settings -> Secrets and variables -> Actions**. Then commit and
push only the MALT scenario change in that fork:

```bash
cd netarena_leaderboard
git add malt_scenario.toml
git commit -m "Submit MALT benchmark"
git push
```

Pushing `malt_scenario.toml` triggers `.github/workflows/run-malt.yml`. The workflow resolves the
green agent and participant images, runs `agentbeats-client`, records provenance, copies the scenario
and result JSON into `submissions/` and `results/`, creates a submission branch, and prints a
"Submit your results" pull-request link in the GitHub Actions summary. Open that PR against the
leaderboard repo. Once it is merged, the AgentBeats leaderboard can pick up the new score.

## Testing

Run A2A conformance tests against your agent.

```bash
# Install test dependencies
uv sync --extra test

# Start your agent (uv or docker; see above)

# Run tests against your running agent URL
uv run pytest --agent-url http://localhost:9009
```

## Publishing

The repository includes a GitHub Actions workflow that automatically builds, tests, and publishes a Docker image of your agent to GitHub Container Registry.

If your agent needs API keys or other secrets, add them in Settings → Secrets and variables → Actions → Repository secrets. They'll be available as environment variables during CI tests.

- **Push to `main`** → publishes `latest` tag:
```
ghcr.io/<your-username>/<your-repo-name>:latest
```

- **Create a git tag** (e.g. `git tag v1.0.0 && git push origin v1.0.0`) → publishes version tags:
```
ghcr.io/<your-username>/<your-repo-name>:1.0.0
ghcr.io/<your-username>/<your-repo-name>:1
```

Once the workflow completes, find your Docker image in the Packages section (right sidebar of your repository). Configure the package visibility in package settings.

> **Note:** Organization repositories may need package write permissions enabled manually (Settings → Actions → General). Version tags must follow [semantic versioning](https://semver.org/) (e.g., `v1.0.0`).
