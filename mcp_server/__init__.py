"""MCP server entry point for sterile-nu-bbn-server."""
from __future__ import annotations

import argparse
import asyncio

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from tools.sterile_tools import (
    describe_sterile_nu_tools,
    predict_from_particle_params,
    check_sbl_anomaly,
    predict_xray_signal,
    scan_constraints,
    plot_exclusion,
    plot_bbn_vs_neff,
)

TOOLS = {
    "describe_sterile_nu_tools": describe_sterile_nu_tools,
    "predict_from_particle_params": predict_from_particle_params,
    "check_sbl_anomaly": check_sbl_anomaly,
    "predict_xray_signal": predict_xray_signal,
    "scan_constraints": scan_constraints,
    "plot_exclusion": plot_exclusion,
    "plot_bbn_vs_neff": plot_bbn_vs_neff,
}


def build_server() -> Server:
    server = Server("sterile-nu-bbn-server")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        import inspect
        out = []
        for name, fn in TOOLS.items():
            sig = inspect.signature(fn)
            props = {}
            for pname, p in sig.parameters.items():
                if p.annotation == float:
                    props[pname] = {"type": "number"}
                elif p.annotation == int:
                    props[pname] = {"type": "integer"}
                elif p.annotation == str:
                    props[pname] = {"type": "string"}
                else:
                    props[pname] = {"type": "string"}
            out.append(
                Tool(
                    name=name,
                    description=(fn.__doc__ or "").strip().split("\n")[0],
                    inputSchema={
                        "type": "object",
                        "properties": props,
                    },
                )
            )
        return out

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        import json
        fn = TOOLS.get(name)
        if fn is None:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
        result = fn(**arguments)
        return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]

    return server


async def main():
    server = build_server()
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
