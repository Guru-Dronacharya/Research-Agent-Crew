"""MCP Server for Filesystem Operations.

Exposes Model Context Protocol (MCP) tools for saving research reports,
reading existing research notes, and listing generated reports in a safe workspace directory.
"""

import os
import json
from pathlib import Path
from mcp.server.mcpserver import MCPServer

fs_mcp_server = MCPServer("filesystem_server")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_path(filename: str, subfolder: str = "reports") -> Path:
    """Validate and resolve target file path, preventing directory traversal."""
    if ".." in filename or ".." in subfolder:
        raise ValueError("Directory traversal attempt detected.")
    if filename.startswith("/") or filename.startswith("\\"):
        raise ValueError("Absolute path traversal attempt detected.")
    clean_name = os.path.basename(filename)
    if not clean_name:
        clean_name = "untitled_report.md"
    target_dir = BASE_DIR / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    resolved = (target_dir / clean_name).resolve()
    if not str(resolved).startswith(str(BASE_DIR)):
        raise ValueError("Directory traversal attempt detected.")
    return resolved


@fs_mcp_server.tool()
def save_report(filename: str, content: str, folder: str = "reports") -> str:
    """Save a research report or structured analysis to the filesystem.

    Args:
        filename: Name of the file (e.g. 'langgraph_vs_autogen.md' or 'summary.json').
        content: Text or JSON content to write.
        folder: Subdirectory within project (default 'reports').

    Returns:
        JSON string confirming file write, path, and size in bytes.
    """
    try:
        path = _safe_path(filename, folder)
        path.write_text(content, encoding="utf-8")
        return json.dumps({
            "status": "success",
            "filename": path.name,
            "filepath": str(path),
            "bytes_written": len(content.encode("utf-8")),
            "message": f"Successfully persisted report to {path.name}"
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to save report: {str(e)}"
        })


@fs_mcp_server.tool()
def read_report(filename: str, folder: str = "reports") -> str:
    """Read an existing saved research report from disk.

    Args:
        filename: Name of the file to read.
        folder: Subdirectory within project (default 'reports').

    Returns:
        JSON string containing the content or error.
    """
    try:
        path = _safe_path(filename, folder)
        if not path.exists():
            return json.dumps({
                "status": "not_found",
                "message": f"File '{filename}' does not exist in {folder}"
            })
        content = path.read_text(encoding="utf-8")
        return json.dumps({
            "status": "success",
            "filename": path.name,
            "content": content
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to read report: {str(e)}"
        })


@fs_mcp_server.tool()
def list_reports(folder: str = "reports") -> str:
    """List all saved research reports and summaries in the reports directory.

    Args:
        folder: Subdirectory to inspect (default 'reports').

    Returns:
        JSON string listing files with size and modification timestamp.
    """
    try:
        target_dir = BASE_DIR / folder
        if not target_dir.exists():
            return json.dumps({"status": "success", "reports": []})

        files = []
        for p in target_dir.iterdir():
            if p.is_file():
                files.append({
                    "name": p.name,
                    "path": str(p),
                    "size_bytes": p.stat().st_size,
                    "modified": p.stat().st_mtime
                })
        return json.dumps({
            "status": "success",
            "count": len(files),
            "reports": files
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to list reports: {str(e)}"
        })


if __name__ == "__main__":
    fs_mcp_server.run("stdio")
