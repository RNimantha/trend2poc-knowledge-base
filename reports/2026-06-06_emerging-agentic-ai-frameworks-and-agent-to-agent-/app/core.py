import asyncio
import httpx
import json
from typing import Any, Dict, Optional, List


class JSONRPCError(Exception):
    pass


def create_jsonrpc_request(method: str, params: Any, request_id: int) -> Dict[str, Any]:
    """Create a JSON-RPC 2.0 request dictionary."""
    return {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": request_id,
    }


def parse_jsonrpc_response(response_json: Dict[str, Any]) -> Any:
    """Parse JSON-RPC response, raise if error."""
    if "error" in response_json:
        raise JSONRPCError(response_json["error"])
    if "result" not in response_json:
        raise JSONRPCError("Missing 'result' in response")
    return response_json["result"]


class Agent:
    def __init__(self, agent_id: str, port: int) -> None:
        self.agent_id = agent_id
        self.port = port
        self.memory_context: List[str] = []  # ephemeral in-memory context

    async def process_subtask(self, subtask: str) -> str:
        """Simulate processing a subtask and store in memory context."""
        # For demonstration, just echo with agent id and subtask
        result = f"{self.agent_id} processed: {subtask}"
        self.memory_context.append(result)
        await asyncio.sleep(0.1)  # simulate async work
        return result


class Orchestrator:
    def __init__(self, port: int, agents: List[Agent]) -> None:
        self.port = port
        self.agents = agents

    def decompose_task(self, task: str) -> List[str]:
        """Decompose a high-level task into subtasks for agents."""
        # For simplicity, split task into two subtasks
        return [f"Part 1 of {task}", f"Part 2 of {task}"]

    async def delegate_subtask(self, agent: Agent, subtask: str, request_id: int) -> str:
        """Send JSON-RPC request to agent and get response with error handling."""
        url = f"http://localhost:{agent.port}/jsonrpc"
        request_payload = create_jsonrpc_request("process", subtask, request_id)
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(url, json=request_payload)
                response.raise_for_status()
                response_json = response.json()
                result = parse_jsonrpc_response(response_json)
                return result
        except (httpx.RequestError, httpx.HTTPStatusError, JSONRPCError, json.JSONDecodeError) as e:
            # Log error and return fallback response
            error_msg = f"Error communicating with agent {agent.agent_id}: {str(e)}"
            print(error_msg)
            return error_msg

    async def run(self, task: str) -> List[str]:
        subtasks = self.decompose_task(task)
        results = []
        tasks = []
        for i, (agent, subtask) in enumerate(zip(self.agents, subtasks)):
            tasks.append(self.delegate_subtask(agent, subtask, request_id=i+1))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Handle exceptions in results
        final_results = []
        for res in results:
            if isinstance(res, Exception):
                final_results.append(f"Exception during subtask processing: {str(res)}")
            else:
                final_results.append(res)
        return final_results
