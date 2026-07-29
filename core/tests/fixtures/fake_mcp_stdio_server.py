#!/usr/bin/env python3
"""Minimal fake MCP stdio server for `stdio_transport.rs` tests.

Speaks the same newline-delimited JSON-RPC shape the real transport
expects: one JSON object per line on stdin, one JSON-RPC response per line
on stdout. Implements just enough of `initialize`/`tools/list`/`tools/call`
to exercise the Rust client for real, without needing an actual MCP SDK
installed anywhere in this environment.

    python3 fake_mcp_stdio_server.py [--exit-immediately]
"""

import json
import sys


def respond(id_, result=None, error=None):
    message = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        message["error"] = {"message": error}
    else:
        message["result"] = result
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def handle(request: dict) -> None:
    method = request.get("method")
    id_ = request.get("id")
    params = request.get("params") or {}

    if method == "initialize":
        respond(id_, {"serverInfo": {"name": "fake-mcp-server"}})
    elif method == "tools/list":
        respond(
            id_,
            {
                "tools": [
                    {
                        "name": "add",
                        "description": "Adds two numbers.",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "a": {"type": "number"},
                                "b": {"type": "number"},
                            },
                            "required": ["a", "b"],
                        },
                    }
                ]
            },
        )
    elif method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments") or {}
        if name == "add":
            total = arguments.get("a", 0) + arguments.get("b", 0)
            respond(id_, {"content": [{"type": "text", "text": str(total)}], "isError": False})
        else:
            respond(id_, error=f"unknown tool: {name}")
    else:
        respond(id_, error=f"unknown method: {method}")


def main() -> None:
    if "--exit-immediately" in sys.argv[1:]:
        return

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        handle(request)


if __name__ == "__main__":
    main()
