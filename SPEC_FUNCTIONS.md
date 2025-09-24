# SPEC_FUNCTIONS.md — AZR Tool-Calling + Python-Exec + Rewards + Training

## Table of contents

1. [Global constraints & glossary](#global-constraints--glossary)
2. [/usr/local/bin/sbx_py (bash)](#usrlocalbinsbx_py-bash)
3. [/opt/azr/tool_harness.py](#optazrtool_harnesspy)

   * `_limit_resources`
   * `run_python`
4. [/opt/azr/tools/web_tool.py](#optazrtoolsweb_toolpy)

   * `search_ddg`
   * `fetch_url`
5. [/opt/azr/tools/schema.json](#optazrtoolsschemajson)
6. [/opt/azr/tool_loop.py](#optazrtool_looppy)

   * `_load_tools`
   * `_extract_last_json_line`
   * `tool_dispatch`
   * `roll_with_tools`
7. [/opt/azr/rewards.py](#optazrrewardspy)

   * `_last_python_block`
   * `score_code_tests`
   * `style_penalty`
   * `timeout_penalty`
   * `blended_reward`
8. [/opt/azr/rl_train_gspo.py](#optazrrl_train_gspopy)

   * `build_prompt`
   * `format_for_model`
   * `reward_fn`
   * `main`
9. [/opt/azr/collect_rollouts.py](#optazrcollect_rolloutspy)

   * `main`
10. [/opt/azr/sft_train.py](#optazrsft_trainpy)

* `build_pairs`
* `main`

11. [Acceptance tests (pytest)](#acceptance-tests-pytest)

---

## Global constraints & glossary

**Constraints (MUST):**

* Functions **must not** rely on global mutable state except where explicitly specified.
* All public functions must include **type hints** and **docstrings** matching this spec.
* No `print` side-effects in library code; return values carry information.
* **Time limits**: operations that may block (web, subprocess) must expose a timeout control and adhere to it.
* **Security**: untrusted code only executes via `nsjail` or rlimit fallback; no network inside the sandbox; filesystem write scope restricted to `/tmp`.
* **Determinism**: generation seeds are set in training scripts; reward code must be pure w.r.t. inputs (no random numbers).

**Glossary:**
*Tool* = callable function exposed to the model. *Transcript* = list of assistant turns (text) produced across tool loop iterations. *Tool log* = structured record of calls and results. *Tests* = Python `assert` strings appended to candidate code for verification. *GSPO* = sequence-level importance ratio updates via GRPO config.

---

## /usr/local/bin/sbx_py (bash)

**Purpose:** Launch Python 3 inside an **nsjail** with strong isolation.

**Invocation:**
`/usr/local/bin/sbx_py <CODE_PATH> [MEM_MB=256] [TIME_S=2]`

**Behavior (MUST):**

* Use `nsjail -Mo` with:

  * `--user 65534 --group 65534` (nobody)
  * `--iface_no_lo` (no loopback; no network)
  * `--disable_proc`
  * `--rlimit_as MEM_MB*1024*1024`
  * `--rlimit_cpu TIME_S`, `--time_limit TIME_S`
  * `--rlimit_nofile 256`, `--rlimit_nproc 128`, `--rlimit_stack 64MiB`
  * `--chroot /`, `--cwd /tmp`
  * `--bindmount_ro /usr`, `--bindmount_ro /lib`, `--bindmount_ro /lib64` (ignore `/lib64` on distros lacking it)
  * `--bindmount_rw /tmp`
  * Exec `/usr/bin/python3 "$CODE_PATH"`.
* Exit code MUST mirror the child’s exit code; if nsjail fails, exit non-zero with stderr message.

**Error cases:**

* Missing nsjail → the wrapper may not exist; library will detect absence and fall back (see `run_python`).
* If mounts are missing, the wrapper may adjust by skipping the absent read-only bind (acceptable).

**Acceptance:** Example calls:

* `sbx_py /tmp/ok.py 256 2` returns 0 on success;
* long-running code must be killed and return non-zero within `TIME_S+1s`.

---

## /opt/azr/tool_harness.py

### `_limit_resources(memory_mb: int, cpu_s: int) -> Callable[[], None]`

**Purpose:** Pre-exec hook to enforce rlimits when **nsjail** is not used.

**Behavior (MUST):**

* Set `RLIMIT_AS = memory_mb * 1024 * 1024` (soft=hard).
* Set `RLIMIT_CPU = cpu_s` (soft=hard).
* Set `RLIMIT_NPROC = 128`, `RLIMIT_NOFILE = 256`.
* Call `os.setsid()` to create a new session so the parent can kill the process group on timeout.
* **No return value**, no prints; used as `preexec_fn`.

**Failure modes:** If `resource.setrlimit` fails (e.g., permissions), raise the underlying `OSError`.

---

### `run_python(code: str, timeout_s: int = 2, memory_mb: int = 256) -> dict`

**Purpose:** Execute **untrusted** Python code with isolation and resource limits.

**Inputs (MUST):**

* `code`: Python source text. MAY include multiple statements and functions.
* `timeout_s`: hard wall-clock limit (>0).
* `memory_mb`: memory limit in megabytes (≥64).

**Algorithm (MUST):**

1. **Persist code** to a unique file in `/tmp` (suffix `.py`).
2. If both `nsjail` and `/usr/local/bin/sbx_py` are present, execute:
   `sbx_py <path> memory_mb timeout_s` with an outer timeout of `timeout_s + 1` seconds.
   Else run `sys.executable <path>` with `preexec_fn=_limit_resources(...)` and timeout `timeout_s`.
3. **Capture** `stdout`, `stderr`, and **return code`.
4. On Python or wrapper timeout (`subprocess.TimeoutExpired`):

   * Attempt `os.killpg(0, SIGKILL)` to kill the process group;
   * Return `{"stdout": "", "stderr": "TIMEOUT", "returncode": 124}`.
5. **Always** delete the temp file.

**Outputs (MUST):**

```
{
  "stdout": "<captured STDOUT str>",
  "stderr": "<captured STDERR str>",
  "returncode": <int>  # 0 if success, 124 if timeout, else Python exit code
}
```

**Invariants (MUST):**

* No network access during code execution (enforced via nsjail `--iface_no_lo`; the fallback relies on platform network policies and is intended only when nsjail is unavailable).
* No writes outside `/tmp`.
* Function is **pure** w.r.t. inputs and OS state (except temp file).

**Complexity:** O(|code|) build time + process runtime ≤ `timeout_s`.

**Acceptance:** See tests `test_tool_harness.py` (below).

---

## /opt/azr/tools/web_tool.py

### `search_ddg(query: str, count: int = 5) -> dict`

**Purpose:** Lightweight web search for research (outside the sandbox).

**Behavior (MUST):**

* Use `duckduckgo_search.DDGS().text(...)` with `max_results=count`.
* Compose output:

```
{
  "query": "<query>",
  "results": [
    {"title": str, "href": str, "body": str},  # length may be truncated by library
    ...
  ]
}
```

**Constraints:**

* `count` in [1, 10]; **clamp** silently if out of range.
* Must not raise on no results; return empty `results` list.
* Timeouts are governed by the DDGS library; retries are optional, but do **not** silently swallow exceptions other than network errors (raise `requests`/library exceptions).

---

### `fetch_url(url: str, max_bytes: int = 800000, timeout: int = 15) -> dict`

**Purpose:** Fetch a URL and return **readable text**.

**Behavior (MUST):**

1. `GET` with headers `User-Agent: UA` (as defined), `timeout=timeout`.
2. Truncate content to `max_bytes`.
3. Parse with `BeautifulSoup(..., "html.parser")`, extract `.get_text(" ", strip=True)`.
4. If `readability.Document` is available, attempt readability summary; fall back if it errors.
5. Return:

```
{
  "url": url,
  "status": <HTTP status code>,
  "text": "<readable text, truncated to max_bytes>"
}
```

**Constraints:**

* Raise `requests.HTTPError` for non-2xx unless explicitly handled; test expects exception on 404.
* **Do not** follow file downloads (e.g., pdf, zip) into binary parsing; returning best-effort text is acceptable.

---

## /opt/azr/tools/schema.json

**Purpose:** Canonical **tool contract** the model sees.

**Requirements (MUST):**

* Includes **three** functions with exact names: `python.run`, `web.search`, `web.fetch`.
* Every function object includes `name`, `description`, and a JSON Schema `parameters` object with types and `required` list.
* This file is **read-only** at runtime; clients must not mutate it.

**Acceptance:** `tool_loop._load_tools()` must read this file and produce a dict with a key `"tools"` whose value is a list of 3 items.

---

## /opt/azr/tool_loop.py

### `_load_tools() -> dict`

**Purpose:** Load and cache the tool schema.

**Behavior (MUST):**

* Read `/opt/azr/tools/schema.json` once; cache in a module-level variable.
* On file read or JSON parse error: raise the underlying exception.

**Idempotency:** Subsequent calls return the cached dict (same object identity acceptable).

---

### `_extract_last_json_line(text: str) -> dict | None`

**Purpose:** Extract the **last valid JSON object** from a mixed text (often containing narration).

**Algorithm (MUST):**

1. Scan the string **backwards** to find a **balanced** JSON object:

   * Maintain `depth = 0`; iterate chars from end to start;
   * Increment `depth` on `}`; decrement on `{`;
   * When `depth` transitions to 0 after encountering `{`, you have a candidate slice `[start_idx: end_idx+1]`.
   * Attempt `json.loads` on the candidate; if it loads, **return** the dict.
2. If none load, return `None`.

**Constraints:**

* Must ignore unmatched braces inside code blocks or strings (the balancing handles this well enough).
* Regex may be used as a fallback, but the **primary** method is the balanced scan (to avoid false positives).

**Complexity:** O(n).

---

### `tool_dispatch(name: str, args: dict) -> dict`

**Purpose:** Route a tool call to the correct implementation.

**Behavior (MUST):**

* Supported names: `"python.run"`, `"web.search"`, `"web.fetch"`.
* Validate minimal arguments:

  * `"python.run"` requires `"code": str`; optional `"timeout_s": int`, `"memory_mb": int`.
  * `"web.search"` requires `"query": str`; optional `"count": int`.
  * `"web.fetch"` requires `"url": str`; optional `"max_bytes": int`.
* On unknown tool **return** `{"error": "unknown tool <name>"}` (do not raise).
* **No logging or prints**; return values carry info.

---

### `roll_with_tools(model, tokenizer, system: str, user: str, max_turns: int = 6) -> tuple[str, list, list]`

**Purpose:** Run a **multi-turn** tool session until the model emits a final answer or `max_turns` is reached.

**Protocol (MUST):**

* Construct an initial instruction that embeds the `TOOLS_SPEC` (from `_load_tools`) and clearly instructs the model to output **one** of:

  * `{"tool_call": {"name": "<tool>", "arguments": { ... }}}`
  * `{"final_answer": "..."}`
* For each turn `t=1..max_turns`:

  1. Prepare the prompt using `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
  2. Generate with `max_new_tokens=512`, `do_sample=True`, `temperature≈0.7`, `top_p≈0.95`.
  3. Decode; append raw text to `transcript`.
  4. Parse `obj = _extract_last_json_line(text)`.
  5. If `obj` contains `"final_answer"`, **return** `(obj["final_answer"], tool_log, transcript)`.
  6. If it contains `"tool_call"`, extract `name`, `arguments` dict; call `tool_dispatch`; append to `tool_log` as:
     `{"call":{"name": name, "args": arguments}, "result": <tool-result>}`.
     Add a new message: `{"role":"tool","content": json.dumps({"name": name, "result": <result>})}`.
  7. Otherwise (unknown JSON or no JSON), **return** `(text.strip(), tool_log, transcript)` as a graceful fallback.
* If `max_turns` reached, return the **last** assistant text with collected `tool_log`, `transcript`.

**Invariants:**

* Each turn either terminates or performs **at most one** tool call.
* The function is deterministic for fixed seeds and model.

**Outputs:** `(final_answer: str, tool_log: list[dict], transcript: list[str])`.

---

## /opt/azr/rewards.py

### `_last_python_block(text: str) -> str | None`

**Purpose:** Extract the **last fenced** Python code block from the model output.

**Behavior (MUST):**

* Match backtick blocks of the form `python … ` or generic `…`; return the **inner** content of the **last** match; return `None` if none found.
* Must use `re.DOTALL` and be **case-insensitive** for the language tag.

---

### `score_code_tests(model_output: str, tests: list[str], timeout_s=2, memory_mb=256) -> tuple[float, dict]`

**Purpose:** Execute tests against code found in `model_output` and compute **pass rate**.

**Behavior (MUST):**

1. If `tests` is empty:

   * Return `(0.1, {"passes": 0, "total": 0})` **only if** `model_output` is non-empty; else `(0.0, {"passes":0,"total":0})`.
2. Extract code via `_last_python_block`. If `None`, return `(0.0, {"passes":0,"total":len(tests),"reason":"no-code-block"})`.
3. For each test string `t` in `tests`:

   * Compose `block = code + "\n\n" + t`.
   * Execute via `run_python(block, timeout_s, memory_mb)`.
   * A test **passes** iff `returncode == 0` **and** `"AssertionError"` **not** in `stderr`.
   * Count passes; do **not** short-circuit on failures.
4. Return `(passes / total, {"passes": passes, "total": total})`.

**Constraints:**

* No mutation of `tests`; no I/O beyond the sandbox execution.

---

### `style_penalty(model_output: str) -> float`

**Purpose:** Reward helpful formatting.

**Behavior (MUST):**

* If output contains `"final answer"` (case-insensitive) **or** contains a JSON object with `"final_answer"` key, return `+0.05`; else `0.0`.

---

### `timeout_penalty(stderr: str) -> float`

**Purpose:** Penalize timeouts.

**Behavior (MUST):**

* Return `-0.05` if substring `"TIMEOUT"` appears in `stderr`; else `0.0`.

---

### `blended_reward(model_output: str, tests: list[str], extra: dict | None = None) -> tuple[float, dict]`

**Purpose:** Combine test pass rate with minor style/timeout adjustments.

**Behavior (MUST):**

* Extract `timeout_s` and `memory_mb` from `extra` if provided; default to 2s and 256MB.
* Call `score_code_tests(...)` to get `base` and `stats`.
* Compute `bonus = style_penalty(model_output) + timeout_penalty(extra.get("stderr",""))`.
* `score = clamp(base + bonus, 0.0, 1.0)`.
* Return `(score, {"base": base, "bonus": bonus, **stats})`.

---

## /opt/azr/rl_train_gspo.py

### `build_prompt(task: dict) -> list[dict]`

**Purpose:** Convert a dataset row into a **chat** message list.

**Input (MUST):**
`task = {"prompt": str, "tests": list[str], "timeout_s": int?, "memory_mb": int?}`

**Behavior (MUST):**

* Return a **two-turn** conversation:

  * system: **fixed** instruction that encourages code + final answer line
  * user: the `task["prompt"]` string

**Output example:**

```
[
  {"role":"system","content":"You are a careful engineer... 'Final answer: ...'."},
  {"role":"user","content": task["prompt"]}
]
```

---

### `format_for_model(messages: list[dict], tokenizer) -> str`

**Purpose:** Render chat messages into a prompt string using the model’s template.

**Behavior (MUST):**

* Call `tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)` and return the string.

---

### `reward_fn(batch_prompts: list[str], policy_outputs: list[str], metadata: list[dict]) -> list[float]`

**Purpose:** Compute per-sample rewards for GRPO/GSPO.

**Behavior (MUST):**

* For each triple `(out, meta)`:

  * Compute `score, _ = blended_reward(out, meta.get("tests", []), {"timeout_s": meta.get("timeout_s",2), "memory_mb": meta.get("memory_mb",256)})`.
  * Append `score`.
* Return the list of scores, **length equal** to `len(policy_outputs)`.

**Constraints:**

* **Pure** function; no prints; no randomness.
* Must not mutate `policy_outputs` or `metadata`.

---

### `main() -> None`

**Purpose:** End-to-end RL training loop with **GSPO** updates.

**Behavior (MUST):**

1. Load config from `/opt/azr/azr.params.yaml` via `opt_azr_params.PARAMS`.
2. Compute `max_seq = max_prompt_length + max_completion_length`.
3. Load model via `FastLanguageModel.from_pretrained(model_id, max_seq_length=max_seq, load_in_4bit=(quantization=="4bit"))`.
4. Apply LoRA adapters from params.
5. Build `GRPOConfig` with:

   * `importance_sampling_level="sequence"`,
   * `clip_range_ratio` from params,
   * `use_vllm=True`, `bf16` per params,
   * sampling params (`temperature`, `top_p`) per params.
6. Load dataset from `/opt/azr/data/train.jsonl` with `datasets.load_dataset(..., data_files={...})["train"]`.
7. Wrap dataset in a class that returns **prompt strings** and **metadata** (tests, limits).
8. Instantiate `GRPOTrainer(model, tokenizer, args=cfg, train_dataset=wrapped, reward_funcs=[reward_fn])`.
9. Call `trainer.train()`.
10. Exit 0 on success; propagate exceptions on errors.

**Acceptance:** Training should begin and produce logs; does not need to reach convergence in tests.

---

## /opt/azr/collect_rollouts.py

### `main(out_path="/opt/azr/data/rollouts.jsonl", max_samples=100) -> None`

**Purpose:** Collect tool-augmented transcripts for evaluation or SFT.

**Behavior (MUST):**

1. Load model/tokenizer using params; `max_seq` consistent with RL.
2. Load training dataset as in RL; iterate up to `max_samples`.
3. For each item:

   * Call `roll_with_tools(model, tok, system_prompt, item["prompt"], max_turns=PARAMS["azr"]["sandbox"]["max_tool_turns"])`.
   * Write JSONL lines with keys: `prompt`, `final_answer`, `tool_log`, `transcript`, `tests`.
4. Ensure the output directory exists; **overwrite** any existing file.

**Constraints:**

* Must not crash on tool errors; tool results are recorded as returned.

---

## /opt/azr/sft_train.py

### `build_pairs(example: dict) -> dict`

**Purpose:** Map rollout example to a **single-turn** supervised example.

**Behavior (MUST):**

* Build messages with:

  * system: generic instruction about tools and final answer
  * user: `example["prompt"]`
  * assistant: `example["final_answer"]`
* **Optionally** include `transcript` (excluding final) as assistant turns **before** the final answer.
* Return `{"text": messages_list}`.

---

### `main() -> None`

**Purpose:** Light SFT on collected rollouts.

**Behavior (MUST):**

1. Load params; load base model + LoRA (like RL but shorter `max_seq` OK).
2. Load `/opt/azr/data/rollouts.jsonl`; map via `build_pairs`.
3. Tokenize with `tokenizer.apply_chat_template(..., add_generation_prompt=False)`, `max_length=4096`, truncation/padding.
4. Configure `TrainingArguments` (1 epoch, bf16, small batch with gradient accumulation).
5. Train with `Trainer`.
6. Save to `/opt/azr/runs/sft`.

---

## /opt/azr/run_smoke.py

**Purpose:** Quick programmatic smoke test.

**Behavior (MUST):**

* Call `run_python("print(2+2)")` → expect `returncode==0` and `'4\n'` in stdout.
* Call `run_python("import time; time.sleep(5)", timeout_s=1)` → expect `returncode==124` and `"TIMEOUT"` in stderr.
* Call `search_ddg("site:huggingface.co deepcogito cogito", 2)` → expect dict with at most 2 results.

---

# Acceptance tests (pytest)

Copy the following into `tests/` and run with the venv active. These tests enforce full implementations.

> **Note:** Web tests are marked as `network` and can be skipped (`-m "not network"`) if needed.

```python
# tests/test_tool_harness.py
import os, re, json, time
from pathlib import Path
from importlib import import_module

tool = import_module("tool_harness")

def test_run_python_ok(tmp_path):
    res = tool.run_python("print('hello')", timeout_s=2, memory_mb=128)
    assert isinstance(res, dict)
    assert res["returncode"] == 0
    assert res["stdout"].strip() == "hello"
    assert "TIMEOUT" not in res["stderr"]

def test_run_python_timeout():
    res = tool.run_python("import time; time.sleep(3)", timeout_s=1, memory_mb=128)
    assert res["returncode"] == 124
    assert "TIMEOUT" in res["stderr"]

def test_run_python_memlimit():
    # Try to allocate > memory_mb to trigger failure (implementation-dependent)
    code = "x = 'a'* (50*1024*1024); print(len(x))"
    res = tool.run_python(code, timeout_s=2, memory_mb=32)
    assert res["returncode"] != 0 or "MemoryError" in (res["stderr"] or "")
```

```python
# tests/test_web_tool.py
import pytest
from importlib import import_module
web = import_module("tools.web_tool")

@pytest.mark.network
def test_search_ddg():
    out = web.search_ddg("site:huggingface.co deepcogito cogito", 2)
    assert out["query"]
    assert isinstance(out["results"], list)
    assert len(out["results"]) <= 2

@pytest.mark.network
def test_fetch_url_text():
    res = web.fetch_url("https://huggingface.co", max_bytes=200_000, timeout=10)
    assert res["status"] == 200
    assert isinstance(res["text"], str)
    assert len(res["text"]) > 0
```

```python
# tests/test_tool_loop.py
import json, re
from importlib import import_module

loop = import_module("tool_loop")

def test_extract_last_json_line_balanced():
    text = "irrelevant\n{'not': 'json'}\nMore\n" + json.dumps({"final_answer":"OK"})
    obj = loop._extract_last_json_line(text)
    assert obj and obj.get("final_answer") == "OK"

def test_tool_dispatch_unknown():
    out = loop.tool_dispatch("unknown.tool", {})
    assert "error" in out
```

```python
# tests/test_rewards.py
from importlib import import_module
rw = import_module("rewards")

def test_last_python_block_and_score():
    out = """
Here is code:
```python
def fib(n):
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return a
print(fib(10))
````

Final answer: 55
"""
    score, stats = rw.score_code_tests(out, ["assert fib(10)==55"], timeout_s=2, memory_mb=128)
    assert score == 1.0 and stats["passes"] == 1 and stats["total"] == 1

def test_style_and_timeout_penalties():
    assert rw.style_penalty("Final answer: 42") > 0.0
    assert rw.timeout_penalty("...TIMEOUT...") < 0.0

def test_blended_reward_clamped():
    s, meta = rw.blended_reward("", [], None)
    assert 0.0 <= s <= 1.0
```

```python
# tests/test_schema.py
import json
def test_schema_tools():
    obj = json.load(open("/opt/azr/tools/schema.json"))
    names = [t["function"]["name"] for t in obj["tools"]]
    assert set(names) == {"python.run","web.search","web.fetch"}
```

---

# Quality gates & coding standards (enforced by CI)

* **mypy**: type hints must pass `mypy --ignore-missing-imports /opt/azr/*.py /opt/azr/tools/*.py`.
* **flake8**: no style errors (line length ≤ 120).
* **pytest**: all tests above must pass (`-m "not network"` acceptable for offline CI; but `tool_harness` and `rewards` must always pass).

---

## Notes on extendability (non-blocking)

* If you later add a **self-play** reward, ensure the new function is pure and returns a numeric score in [0,1] and attach to `GRPOTrainer` with a weighted average to maintain the range.
* You can upgrade `roll_with_tools` to support **multiple tool calls per turn** by switching to a loop over parsed JSON objects; if you do, update tests accordingly and add an explicit `max_tool_calls_per_turn` guard.

---

**Bottom line:** With these specs + tests, any “placeholder” or partial implementation will **fail** deterministically. The only way to pass is to fully implement each function according to the detailed requirements above.

