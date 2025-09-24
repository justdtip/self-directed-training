import rewards as rw


def test_last_python_block_and_score():
    snippet = """
Here is code:
```python
def fib(n):
    a,b=0,1
    for _ in range(n):
        a,b=b,a+b
    return a
print(fib(10))
```

Final answer: 55
"""
    score, stats = rw.score_code_tests(snippet, ["assert fib(10)==55"], timeout_s=2, memory_mb=128)
    assert score == 1.0
    assert stats["passes"] == 1
    assert stats["total"] == 1


def test_style_and_timeout_penalties():
    assert rw.style_penalty("Final answer: 42") > 0.0
    assert rw.timeout_penalty("...TIMEOUT...") < 0.0


def test_blended_reward_clamped():
    val, meta = rw.blended_reward("", [], None)
    assert 0.0 <= val <= 1.0
