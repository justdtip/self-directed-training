from selfplay_manager import SelfPlayManager, score_selfplay_pair


def test_score_selfplay_pair_prefers_higher_pass_rate():
    policy = """
```python
def foo():
    return 42
```
"""
    opponent = "print('no code')"
    score = score_selfplay_pair(policy, opponent, ["assert foo()==42"], timeout_s=2, memory_mb=128)
    assert score == 1.0


def test_score_selfplay_pair_tie_breaker_code_length():
    policy = """
```python
def foo():
    return 1
```
"""
    opp = """
```python
def foo():
    return 1  # opponent with comment
```
"""
    score = score_selfplay_pair(policy, opp, ["assert foo()==1"], timeout_s=2, memory_mb=128)
    assert score == 2.0 / 3.0


def test_selfplay_manager_disabled_returns_ties():
    mgr = SelfPlayManager(tokenizer=None, max_seq=2048, enabled=False)
    scores = mgr.compute_selfplay_scores(["prompt"], ["out"], [{"tests": []}])
    assert scores == [0.5]
