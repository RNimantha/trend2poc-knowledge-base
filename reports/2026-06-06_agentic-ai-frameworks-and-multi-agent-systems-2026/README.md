# agentic-ai-multi-agent-poc

A minimal proof-of-concept demonstrating a multi-agent system where autonomous agents collaborate via a simple protocol to complete a complex task using an agentic AI framework approach.

## Prerequisites

- Python 3.11+
- Optional: OpenAI API key for real LLM calls (if not provided, the system runs in mock mode)

## Installation

1. Clone the repository.

```bash
git clone <repo-url>
cd agentic-ai-multi-agent-poc
```

2. Copy `.env.example` to `.env` and optionally add your OpenAI API key.

```bash
cp .env.example .env
# Edit .env to add your OPENAI_API_KEY if available
```

3. Install dependencies.

```bash
pip install -r requirements.txt
```

## Configuration

- `.env` file controls the environment variables.
- `OPENAI_API_KEY` enables real LLM calls; if unset, mock responses are used.

## Running the Demo

```bash
python app/main.py
```

This runs a minimal multi-agent system where:
- A `PlannerAgent` decomposes a complex task into subtasks.
- Multiple `WorkerAgent`s execute subtasks.
- Agents communicate via a simple message passing protocol.

**Note:** The demo requires the example input file `examples/sample_input.json` to provide the task description.

## Example Output

```
Starting multi-agent collaboration on task: Organize a team event
PlannerAgent decomposed task into subtasks: ['Book venue', 'Arrange catering', 'Send invitations']
WorkerAgent-1 completed subtask: Book venue
WorkerAgent-2 completed subtask: Arrange catering
WorkerAgent-3 completed subtask: Send invitations
All subtasks completed. Final result:
{'Book venue': 'Venue booked successfully.', 'Arrange catering': 'Catering arranged successfully.', 'Send invitations': 'Invitations sent successfully.'}
```

## How This POC Demonstrates the Concept

- Illustrates autonomous agents collaborating asynchronously.
- Uses a simplified message passing protocol inspired by MCP.
- Shows task decomposition and delegation in a multi-agent system.
- Supports both mock and real LLM calls (mock by default).

## Known Limitations

- Simplified message passing, not a full MCP implementation.
- Mock LLM responses by default; real LLM integration is minimal.
- No persistent memory or advanced state management.
- No multimodal inputs or external tool integrations.
- No human-in-the-loop or safety guardrails.

## Examples Directory

Make sure the `examples` directory exists and contains the `sample_input.json` file with the task description.

