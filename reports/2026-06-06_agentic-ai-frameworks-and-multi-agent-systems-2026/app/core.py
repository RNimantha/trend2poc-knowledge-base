import asyncio
import random
from typing import Any, Dict, List, Optional

from app.config import settings
import httpx


class Agent:
    """Base class for agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    async def send_message(self, recipient: 'Agent', message: Dict[str, Any]) -> None:
        await recipient.receive_message(message, sender=self)

    async def receive_message(self, message: Dict[str, Any], sender: 'Agent') -> None:
        raise NotImplementedError("Subclasses must implement receive_message")


class PlannerAgent(Agent):
    """Agent that decomposes a complex task into subtasks and collects results."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self._results: Dict[str, str] = {}
        self._pending_responses = 0
        self._condition = asyncio.Condition()

    async def plan_task(self, task_description: str) -> List[str]:
        if settings.use_mock:
            # Mock decomposition
            await asyncio.sleep(0.1)  # simulate latency
            subtasks = [
                "Book venue",
                "Arrange catering",
                "Send invitations",
            ]
            return subtasks
        else:
            # Real LLM call to decompose task
            prompt = f"Decompose the following task into a list of subtasks:\n{task_description}\nSubtasks:"  # simple prompt
            subtasks_text = await call_openai_chat(prompt)
            subtasks = [line.strip('- ').strip() for line in subtasks_text.splitlines() if line.strip()]
            return subtasks

    async def receive_message(self, message: Dict[str, Any], sender: Agent) -> None:
        # Handle worker responses
        subtask = message.get("subtask")
        result = message.get("result")
        if subtask and result:
            async with self._condition:
                self._results[subtask] = result
                self._pending_responses -= 1
                if self._pending_responses <= 0:
                    self._condition.notify_all()

    async def wait_for_all_results(self) -> Dict[str, str]:
        async with self._condition:
            while self._pending_responses > 0:
                await self._condition.wait()
            return dict(self._results)

    def increment_pending_responses(self, count: int = 1) -> None:
        # Safely increment pending responses count
        self._pending_responses += count


class WorkerAgent(Agent):
    """Agent that executes a subtask."""

    def __init__(self, name: str) -> None:
        super().__init__(name)

    async def execute_subtask(self, subtask: str) -> str:
        if settings.use_mock:
            await asyncio.sleep(random.uniform(0.1, 0.3))  # simulate work
            return f"{subtask} completed successfully."
        else:
            prompt = f"Perform the following subtask and provide a concise completion message:\n{subtask}\nResult:"
            result = await call_openai_chat(prompt)
            return result.strip()

    async def receive_message(self, message: Dict[str, Any], sender: Agent) -> None:
        # Expecting a subtask assignment
        subtask = message.get("subtask")
        if not subtask:
            return
        result = await self.execute_subtask(subtask)
        # Send result back to sender (PlannerAgent)
        response_message = {"subtask": subtask, "result": result}
        await self.send_message(sender, response_message)


async def call_openai_chat(prompt: str) -> str:
    """Call OpenAI chat completion API with the given prompt."""
    import os

    api_key = settings.openai_api_key
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    json_data = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.7,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            response = await client.post(url, headers=headers, json=json_data)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except (httpx.HTTPError, KeyError) as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")


async def run_multi_agent_workflow(task_description: str) -> Dict[str, str]:
    """Orchestrate the multi-agent collaboration to complete the complex task."""
    planner = PlannerAgent("PlannerAgent")

    # Step 1: Planner decomposes the task
    subtasks = await planner.plan_task(task_description)

    # Step 2: Create WorkerAgents for each subtask
    workers = [WorkerAgent(f"WorkerAgent-{i+1}") for i in range(len(subtasks))]

    # Initialize pending responses count safely
    planner.increment_pending_responses(len(subtasks))

    # Send subtasks to workers
    send_tasks = [planner.send_message(worker, {"subtask": subtask}) for worker, subtask in zip(workers, subtasks)]
    await asyncio.gather(*send_tasks)

    # Wait for all worker responses
    results = await planner.wait_for_all_results()

    return results
