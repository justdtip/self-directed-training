import json

import tool_loop


def test_extract_last_json_line_balanced():
    blob = "noise\n" + json.dumps({"tool_call": {"name": "python.run", "arguments": {"code": "print(1)"}}})
    obj = tool_loop._extract_last_json_line(blob)
    assert obj is not None
    assert obj["tool_call"]["name"] == "python.run"


def test_tool_dispatch_unknown():
    result = tool_loop.tool_dispatch("unknown.tool", {})
    assert "error" in result
