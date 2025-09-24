import pytest

from azr.rewards import (
    blended_reward,
    extract_last_code_block,
    score_code_tests,
    style_penalty,
    timeout_penalty,
)


def test_extract_last_code_block_picks_last():
    text = """
Intro
```python
print('first')
```
More text
```python
print('second')
```
"""
    block = extract_last_code_block(text)
    assert block is not None and "second" in block


def test_score_code_tests_success_and_failure():
    code = """
```python
def fib(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
"""
    tests = ["assert fib(5) == 5", "assert fib(6) == 8"]
    score, stats = score_code_tests(code, tests, timeout_s=2, memory_mb=128)
    assert score == 1.0
    assert stats == {"passes": 2, "total": 2}

    score, stats = score_code_tests("no code here", tests)
    assert score == 0.0
    assert stats["passes"] == 0
    assert stats["total"] == 2


def test_style_and_timeout_penalties():
    assert style_penalty("Final answer: 42") == pytest.approx(0.05)
    assert style_penalty('{"final_answer": "ok"}') == pytest.approx(0.05)
    assert timeout_penalty("TIMEOUT") == pytest.approx(-0.05)
    assert timeout_penalty("") == 0.0


def test_blended_reward_combines_components():
    text = """
```python
x = 21 * 2
print(x)
```
Final answer: 42
"""
    score, stats = blended_reward(text, ["assert x == 42"], extra={"stderr": ""})
    assert 0.0 <= score <= 1.0
    assert pytest.approx(stats["base"], rel=1e-6) == 1.0
    assert stats["bonus"] >= 0.05
