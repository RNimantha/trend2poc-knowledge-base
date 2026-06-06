import asyncio
import json
import sys
from pathlib import Path

from app.config import settings
from app.core import run_multi_agent_workflow


async def main() -> None:
    example_input_path = Path(__file__).parent.parent / "examples" / "sample_input.json"

    try:
        with example_input_path.open("r", encoding="utf-8") as f:
            input_data = json.load(f)
    except FileNotFoundError:
        print(f"Example input file not found: {example_input_path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Failed to parse example input JSON: {e}")
        sys.exit(1)

    task_description = input_data.get("task_description")
    if not task_description:
        print("Input JSON must contain 'task_description' field.")
        sys.exit(1)

    print(f"Starting multi-agent collaboration on task: {task_description}")

    try:
        results = await run_multi_agent_workflow(task_description)
    except Exception as e:
        print(f"Error during multi-agent workflow: {e}")
        sys.exit(1)

    print("All subtasks completed. Final result:")
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
