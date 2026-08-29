#!/bin/bash
# Wrapper script to run the sterile-nu-bbn MCP server
# Ensures correct working directory and Python environment
cd "$(dirname "$0")"
exec .venv2/bin/python -m mcp_server "$@"
