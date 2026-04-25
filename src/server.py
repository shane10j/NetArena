import argparse
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)

from config import AgentConfig
from executor import Executor


def main():
    parser = argparse.ArgumentParser(description="Run the A2A agent.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind the server")
    parser.add_argument("--port", type=int, default=9009, help="Port to bind the server")
    parser.add_argument("--card-url", type=str, help="URL to advertise in the agent card")
    parser.add_argument(
        "--role",
        type=str,
        default="coordinator",
        choices=["coordinator", "planner", "verifier"],
        help="Role this container should serve.",
    )
    args = parser.parse_args()
    config = AgentConfig.from_env(role=args.role)

    skill = AgentSkill(
        id=f"k8s_purple_{config.role}",
        name=f"K8s Purple {config.role.title()}",
        description="Coordinates Kubernetes benchmark investigation, action planning, and verification.",
        tags=["kubernetes", "k8s", "purple-team", "agentx", config.role],
        examples=[
            "Investigate why a workload cannot reach an internal service.",
            "Plan safe Kubernetes remediation steps and verify the result.",
        ],
    )

    agent_card = AgentCard(
        name=f"AgentX K8s Purple Agent ({config.role})",
        description="A scaffolded multi-agent A2A purple agent for AgentX Kubernetes benchmarks.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version='1.0.0',
        default_input_modes=['text'],
        default_output_modes=['text'],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    request_handler = DefaultRequestHandler(
        agent_executor=Executor(config),
        task_store=InMemoryTaskStore(),
    )
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler,
    )
    uvicorn.run(server.build(), host=args.host, port=args.port)


if __name__ == '__main__':
    main()
