"""Quick smoke test for sandbox and tools."""

from __future__ import annotations

from tool_harness import run_python
from tools.web_tool import search_ddg


def main() -> None:
    ok = run_python("print(2+2)")
    assert ok["returncode"] == 0 and ok["stdout"] == "4\n"

    timeout = run_python("import time; time.sleep(5)", timeout_s=1)
    assert timeout["returncode"] == 124 and "TIMEOUT" in timeout["stderr"]

    results = search_ddg("site:huggingface.co deepcogito cogito", 2)
    assert isinstance(results, dict)
    assert len(results.get("results", [])) <= 2


if __name__ == "__main__":  # pragma: no cover - script entry
    main()
