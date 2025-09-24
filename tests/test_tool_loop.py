import json

from azr.tool_loop import ToolDispatcher, extract_last_json_obj


def test_extract_last_json_obj_balanced():
    blob = "noise\n" + json.dumps({"tool_call": {"name": "python.run", "arguments": {"code": "print(1)"}}})
    obj = extract_last_json_obj(blob)
    assert obj is not None
    assert obj["tool_call"]["name"] == "python.run"


def test_tool_dispatch_unknown():
    dispatcher = ToolDispatcher()
    result = dispatcher.call("unknown.tool", {})
    assert "error" in result
