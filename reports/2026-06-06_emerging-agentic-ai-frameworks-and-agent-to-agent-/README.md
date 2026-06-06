# agentic-ai-communication-poc

## Project Description

This project demonstrates a minimal multi-agent system with an orchestrator and two autonomous agents communicating via a simplified Agent2Agent-like JSON-RPC protocol. It showcases basic agent orchestration and agent-to-agent communication using FastAPI servers.

## Architecture

- **Orchestrator**: Receives a high-level task, decomposes it into subtasks, and delegates these subtasks to two agents.
- **Agents**: Two autonomous agents run as FastAPI servers, receive JSON-RPC requests from the orchestrator, process subtasks, and respond.
- **Communication**: Agents and orchestrator communicate asynchronously over HTTP using a simplified JSON-RPC protocol.

## Prerequisites

- Python 3.9 or higher
- No external API accounts required

## Installation Steps

1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd agentic-ai-communication-poc
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` to set ports for the orchestrator and agents if desired. Defaults are provided.

## Running the POC

Run the entire system (orchestrator and two agents) with:

```bash
python -m app.main
```

This will:
- Start the orchestrator and two agents as FastAPI servers on configured ports.
- The orchestrator will load the sample task from `examples/sample_input.json`.
- The orchestrator decomposes the task and delegates subtasks to agents.
- Agents process subtasks and respond.
- The orchestrator combines results and prints the final output.

## Example Output

```
Starting orchestrator on port 8000
Starting agent1 on port 8001
Starting agent2 on port 8002
Orchestrator received task: {'task': 'Process data and generate report'}
Delegating subtasks to agents...
Agent1 processed: Part 1 of Process data and generate report
Agent2 processed: Part 2 of Process data and generate report
Final aggregated result: ['Agent1 processed: Part 1 of Process data and generate report', 'Agent2 processed: Part 2 of Process data and generate report']
```

## Robustness and Testing

The POC includes comprehensive error handling for network failures, invalid responses, and timeouts. Edge case tests cover scenarios such as agent unavailability, malformed messages, and delayed responses to ensure system resilience.

## Extending the POC

- Add more agents with specialized roles.
- Implement full MCP and ACP protocol support.
- Integrate persistent memory modules.
- Enhance monitoring dashboard with real-time metrics.

## Contact

For questions or contributions, please open an issue or pull request in the repository.
