import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Agent
from config import AgentConfig


def test_local_review_rejects_private_helpers():
    agent = Agent(AgentConfig(role="coordinator"))

    feedback = agent._local_review(
        "\n".join(
            [
                "def process_graph(graph_data):",
                "    graph_copy = graph_data.copy()",
                "    return solid_step_list_child_nodes(graph_copy, {'name': 'leaf1'})",
            ]
        )
    )

    assert feedback != "PASS"
    assert "pure NetworkX" in feedback


def test_fallback_is_pure_networkx_code():
    agent = Agent(AgentConfig(role="coordinator"))

    code = agent._fallback_response("Remove leaf1 from the graph. Return a graph.")

    assert "def process_graph(graph_data)" in code
    assert "graph_data.copy()" in code
    assert "node_link_data" in code
    assert "solid_step_" not in code


def test_reviewer_extracts_draft_before_static_review():
    agent = Agent(AgentConfig(role="reviewer"))
    good_draft = agent._fallback_response("Return the graph.")
    review_prompt = agent._build_review_prompt("Return the graph.", good_draft)

    assert "solid_step_*" in review_prompt
    assert agent._local_review(agent._extract_review_draft(review_prompt)) == "PASS"


@pytest.mark.asyncio
async def test_conformance_ping_does_not_call_llm():
    agent = Agent(AgentConfig(role="coordinator", model_name="openai/example"))

    assert await agent.invoke("Hello") == "Agent is ready."
    assert await agent.invoke("ready") == "Agent is ready."
