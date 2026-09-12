#!/usr/bin/env python3
"""Minimal MCP server over streamable HTTP, for the environment placement test.

Speaks JSON-RPC 2.0 over POST per the MCP streamable HTTP transport, answering
each request with one JSON body and each notification with 202. Exposes one
tool: echo(message) -> message. Usage: mcp_http_echo_server.py <port>
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

SERVER_INFO = {"name": "test-http-echo-server", "version": "0.1.0"}

TOOL = {
    "name": "echo",
    "description": "Echo back the message",
    "inputSchema": {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    },
}


def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")
    params = req.get("params", {})
    if req_id is None:
        return None
    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
            },
        }
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [TOOL]}}
    if method == "tools/call":
        if params.get("name") == "echo":
            msg = params.get("arguments", {}).get("message", "")
            if msg.startswith("__env:") and msg.endswith("__"):
                msg = os.environ.get(msg[len("__env:") : -len("__")], "")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"content": [{"type": "text", "text": msg}]},
            }
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "content": [{"type": "text", "text": "unknown tool"}],
                "isError": True,
            },
        }
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": -32601, "message": f"Method not found: {method}"},
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format, *args):  # noqa: A002 - quiet
        return

    def do_GET(self):  # noqa: N802
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_DELETE(self):  # noqa: N802
        self.send_response(405)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        try:
            req = json.loads(body or b"{}")
        except json.JSONDecodeError:
            self.send_response(400)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        resp = handle_request(req)
        if resp is None:
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        payload = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    port = int(sys.argv[1])
    server = HTTPServer(("127.0.0.1", port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
