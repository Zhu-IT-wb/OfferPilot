import argparse
import asyncio
import json
import sys

from app.mcp.client import PersistentMCPClient


async def run(query: str, owner_id: str, top_k: int) -> None:
    client = PersistentMCPClient()
    try:
        result = await client.call_tool(
            "search_knowledge",
            {
                "query": query,
                "owner_id": owner_id,
                "top_k": top_k,
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        await client.close()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(
        description="Run an end-to-end search through the local Career Knowledge MCP server."
    )
    parser.add_argument("query")
    parser.add_argument("--owner-id", default="api:local_user")
    parser.add_argument("--top-k", type=int, default=5)
    arguments = parser.parse_args()
    asyncio.run(run(arguments.query, arguments.owner_id, arguments.top_k))


if __name__ == "__main__":
    main()
