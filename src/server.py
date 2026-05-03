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
    parser.add_argument("--model-name", type=str, help="LiteLLM model name to serve")
    parser.add_argument("--api-key", type=str, help="LiteLLM/OpenAI-compatible API key")
    parser.add_argument("--api-base-url", type=str, help="LiteLLM/OpenAI-compatible base URL")
    parser.add_argument("--api-version", type=str, help="API version for Azure-compatible endpoints")
    parser.add_argument(
        "--role",
        type=str,
        default="coordinator",
        choices=["coordinator", "planner", "proposer", "reviewer", "verifier"],
        help="Role this container should serve.",
    )
    args = parser.parse_args()
    config = AgentConfig.from_env(
        role=args.role,
        model_name=args.model_name,
        litellm_api_key=args.api_key,
        litellm_api_base_url=args.api_base_url,
        litellm_api_version=args.api_version,
    )

    skill = AgentSkill(
        id=f"malt_purple_{config.role}",
        name=f"MALT Purple {config.role.title()}",
        description="Solves MALT data-center topology tasks with pure NetworkX code generation.",
        tags=["malt", "networkx", "data-center", "planning", "agentbeats", config.role],
        examples=[
            "Remove a named node from the graph and return the updated graph.",
            "Rank children of a topology node by bandwidth and return the answer.",
        ],
    )

    agent_card = AgentCard(
        name=f"NetArena MALT Purple Agent ({config.role})",
        description="A basic multi-agent A2A purple agent for the NetArena MALT benchmark.",
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
