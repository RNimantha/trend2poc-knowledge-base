import asyncio
import json
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn
from app.config import get_settings
from app.core import Agent, Orchestrator

logging.basicConfig(level=logging.INFO)


async def start_agent(agent: Agent) -> None:
    app = FastAPI()

    @app.post("/jsonrpc")
    async def jsonrpc_endpoint(request: Request) -> JSONResponse:
        try:
            data = await request.json()
            method = data.get("method")
            params = data.get("params")
            request_id = data.get("id")
            if method != "process" or not isinstance(params, str):
                return JSONResponse(
                    status_code=400,
                    content={"jsonrpc": "2.0", "error": "Invalid method or params", "id": request_id},
                )
            result = await agent.process_subtask(params)
            return JSONResponse({"jsonrpc": "2.0", "result": result, "id": request_id})
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"jsonrpc": "2.0", "error": str(e), "id": None},
            )

    config = uvicorn.Config(app, host="127.0.0.1", port=agent.port, log_level="info")
    server = uvicorn.Server(config)
    logging.info(f"Starting {agent.agent_id} on port {agent.port}")
    await server.serve()


async def start_orchestrator(orchestrator: Orchestrator, sample_task: str) -> None:
    # Orchestrator does not expose HTTP endpoints in this POC; it runs the workflow
    logging.info(f"Orchestrator received task: {sample_task}")
    logging.info("Delegating subtasks to agents...")
    result = await orchestrator.run_task(sample_task)
    logging.info(f"Combined result: {result}")


async def main() -> None:
    settings = get_settings()

    agent1 = Agent(agent_id="agent1", port=settings.AGENT1_PORT)
    agent2 = Agent(agent_id="agent2", port=settings.AGENT2_PORT)
    orchestrator = Orchestrator(port=settings.ORCHESTRATOR_PORT, agents=[agent1, agent2])

    # Load sample task from file
    with open("examples/sample_input.json", "r", encoding="utf-8") as f:
        sample_input = json.load(f)
    sample_task = sample_input.get("task", "Default task")

    # Run agents and orchestrator concurrently
    # Agents run FastAPI servers
    # Orchestrator runs the task delegation

    agent1_task = asyncio.create_task(start_agent(agent1))
    agent2_task = asyncio.create_task(start_agent(agent2))

    # Wait a short moment for agents to start
    await asyncio.sleep(1)

    await start_orchestrator(orchestrator, sample_task)

    # Shutdown agents after orchestrator finishes
    agent1_task.cancel()
    agent2_task.cancel()

    # Wait for agents to shutdown gracefully
    try:
        await asyncio.gather(agent1_task, agent2_task)
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logging.error(f"Fatal error: {e}")
