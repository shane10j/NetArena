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
    assert "import networkx" not in code


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


def test_deterministic_add_packet_switch_adds_ports_without_helpers():
    agent = Agent(AgentConfig(role="coordinator"))

    code = agent._deterministic_malt_response(
        "Add new node with name new_EK_PACKET_SWITCH_16 type EK_PACKET_SWITCH, to ju1.a3.m2. Return a graph."
    )

    assert code is not None
    assert "solid_step_" not in code
    assert "range(1, 17)" in code
    assert "EK_PORT" in code
    compile(code, "<generated>", "exec")


def test_deterministic_remove_list_removes_descendants():
    agent = Agent(AgentConfig(role="coordinator"))

    code = agent._deterministic_malt_response(
        "Remove ju1.a4.m3.s2c3 from the graph. List direct child nodes of ju1.a4.m3 in the updated graph. Return a list of child nodes name."
    )

    assert code is not None
    assert "solid_step_" not in code
    assert "to_remove" in code
    assert "child_names" in code
    compile(code, "<generated>", "exec")
