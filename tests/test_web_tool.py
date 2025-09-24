import pytest

from tools import web_tool


@pytest.mark.network
def test_search_ddg():
    out = web_tool.search_ddg("site:huggingface.co deepcogito cogito", 2)
    assert out["query"]
    assert isinstance(out["results"], list)
    assert len(out["results"]) <= 2


@pytest.mark.network
def test_fetch_url_text():
    res = web_tool.fetch_url("https://huggingface.co", max_bytes=200_000, timeout=10)
    assert res["status"] == 200
    assert isinstance(res["text"], str)
    assert len(res["text"]) > 0
