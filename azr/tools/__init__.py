"""
Tools package: exposes Python execution and web tools.
"""

from .python_tool import run_code, PythonResult
from .web import WebTool, WebSearchResult

__all__ = ["run_code", "PythonResult", "WebTool", "WebSearchResult"]
