import asyncio
import pytest
from unittest.mock import AsyncMock, patch
from app.core import Agent, Orchestrator


@pytest.mark.asyncio
async def test_agent_process_subtask():
    agent = Agent(agent_id="test_agent", port=1234)
    result = await agent.process_subtask("test subtask")
    assert "test_agent processed: test subtask" == result
    assert result in agent.memory_context


@pytest.mark.asyncio
async def test_orchestrator_decompose_task():
    orchestrator = Orchestrator(port=8000, agents=[])
    subtasks = orchestrator.decompose_task("Main task")
    assert len(subtasks) == 2
    assert subtasks[0] == "Part 1 of Main task"
    assert subtasks[1] == "Part 2 of Main task"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_orchestrator_delegate_subtask(mock_post):
    # Mock HTTP response from agent
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "jsonrpc": "2.0",
        "result": "mocked result",
        "id": 1,
    }
    mock_post.return_value = mock_response

    agent = Agent(agent_id="agent1", port=8001)
    orchestrator = Orchestrator(port=8000, agents=[agent])
    result = await orchestrator.delegate_subtask(agent, "subtask", request_id=1)
    assert result == "mocked result"


@pytest.mark.asyncio
@patch("httpx.AsyncClient.post")
async def test_orchestrator_run_task(mock_post):
    # Mock HTTP responses for both agents
    def side_effect(url, json, timeout):
        class MockResp:
            def __init__(self, result):
                self.status_code = 200
                self._result = result

            def raise_for_status(self):
                pass

            def json(self):
                return {"jsonrpc": "2.0", "result": self._result, "id": json["id"]}

        if "8001" in url:
            return MockResp("agent1 processed subtask")
        elif "8002" in url:
            return MockResp("agent2 processed subtask")
        else:
            raise ValueError("Unknown URL")

    mock_post.side_effect = side_effect

    agent1 = Agent(agent_id="agent1", port=8001)
    agent2 = Agent(agent_id="agent2", port=8002)
    orchestrator = Orchestrator(port=8000, agents=[agent1, agent2])

    result = await orchestrator.run_task("Test task")
    assert "summary" in result
    assert "agent1 processed subtask" in result["summary"]
    assert "agent2 processed subtask" in result["summary"]
