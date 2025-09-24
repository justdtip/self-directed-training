from azr.sandbox import ToolSandbox
from azr.tools.web import WebTool


def test_sandbox_turn_limit():
    sb = ToolSandbox(timeout_s=1, memory_mb=64, max_tool_turns=2)
    sb.check()
    sb.check()
    try:
        sb.check()
        assert False, "Expected limit error"
    except RuntimeError:
        pass


def test_web_fetch_example():
    tool = WebTool(max_bytes=1024, user_agent="AZR-Research/1.0 (+no-bots)", timeout_s=3)
    text = tool.fetch("https://example.com")
    assert isinstance(text, str)
    assert len(text) <= 1024
    assert "Example Domain" in text

