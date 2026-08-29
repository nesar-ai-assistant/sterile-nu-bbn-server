"""Allow running as `python -m mcp_server`."""
from mcp_server import mcp

mcp.run(transport="stdio")
