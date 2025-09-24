import json


def test_schema_tools():
    with open("/opt/azr/tools/schema.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    names = [tool["function"]["name"] for tool in data["tools"]]
    assert set(names) == {"python.run", "web.search", "web.fetch"}
