# AgentX K8s Purple Agent

A scaffolded [A2A (Agent-to-Agent)](https://a2a-protocol.org/latest/) agent for AgentX Phase 2
Kubernetes purple-team benchmarks.

## Project Structure

```
src/
├─ server.py      # Server setup, role selection, and agent card configuration
├─ executor.py    # A2A request handling
├─ agent.py       # Coordinator/planner/verifier control flow
├─ llm.py         # LiteLLM adapter
├─ roles.py       # Role prompts and metadata
├─ config.py      # Runtime config from env/Amber
└─ messenger.py   # A2A messaging utilities
manifests/
└─ purple-agent-component.json5 # Reusable Amber component manifest
tests/
└─ test_agent.py  # Agent tests
Dockerfile            # Docker configuration
pyproject.toml        # Python dependencies
amber-manifest.json5  # Amber root manifest wiring coordinator/planner/verifier
.github/
└─ workflows/
   └─ test-and-publish.yml # CI workflow
```

## Getting Started

1. **Implement the real K8s behaviors** - Extend [`src/agent.py`](src/agent.py) with the control
   flow, tools, and prompts you want to compete with.

2. **Configure model access** - The scaffold has a LiteLLM adapter in [`src/llm.py`](src/llm.py).
   Add `litellm` to the project when you are ready for live model calls, then set `MODEL_NAME` and
   `LITELLM_*` variables or bind the Amber `llm` slot during scenario orchestration.

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

The default container role is `coordinator`. Use `--role planner` or `--role verifier` to run helper
containers from the same image.

## Amber Layout

[`amber-manifest.json5`](amber-manifest.json5) is a root manifest that launches three component
instances from [`manifests/purple-agent-component.json5`](manifests/purple-agent-component.json5):
`coordinator`, `planner`, and `verifier`. It binds the planner/verifier A2A exports into the
coordinator and exports only the coordinator's A2A endpoint for the benchmark.

Useful validation commands once Amber is available:

```bash
amber docs manifest
amber check amber-manifest.json5
```

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
