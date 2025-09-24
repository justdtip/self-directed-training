import pytest

from azr.config import AzrModelCfg
from azr.selfplay_manager import SelfPlayManager


class DummyBatch(dict):
    def __init__(self, prompt: str) -> None:
        super().__init__(prompt=prompt)

    def to(self, device: str):
        self["device"] = device
        return self


class DummyTokenizer:
    def __call__(self, text: str, return_tensors: str = "pt"):
        return DummyBatch(text)

    def decode(self, output, skip_special_tokens: bool = True) -> str:
        return output


class DummyModel:
    def __init__(self) -> None:
        self.device = None
        self.generated = []

    def to(self, device: str):
        self.device = device
        return self

    def eval(self):
        pass

    def generate(self, **kwargs):
        prompt = kwargs.get("prompt", "")
        self.generated.append(prompt)
        return [f"{prompt}::opponent"]


@pytest.fixture
def manager(monkeypatch):
    cfg = AzrModelCfg(model_id="stub", lora_r=4, lora_alpha=8, quantization="4bit")
    monkeypatch.setattr("azr.selfplay_manager.load_tokenizer", lambda model_id: DummyTokenizer())
    monkeypatch.setattr("azr.selfplay_manager.setup_model", lambda cfg, **kwargs: DummyModel())
    return SelfPlayManager(cfg, opponent_device="cpu")


def test_generate_opponent_uses_stub(monkeypatch):
    cfg = AzrModelCfg(model_id="stub", lora_r=4, lora_alpha=8, quantization="4bit")
    tokenizer = DummyTokenizer()
    model = DummyModel()
    monkeypatch.setattr("azr.selfplay_manager.load_tokenizer", lambda model_id: tokenizer)
    monkeypatch.setattr("azr.selfplay_manager.setup_model", lambda cfg, **kwargs: model)
    mgr = SelfPlayManager(cfg, opponent_device="cpu")

    outputs = mgr.generate_opponent(["prompt"], max_tokens=5)
    assert outputs == ["prompt::opponent"]
    assert model.device == "cpu"


def test_compute_scores_prefers_higher_pass_rate(manager):
    policy = """\n```python\ndef foo():\n    return 42\n```\n"""
    opponent = "print('no code')"
    scores = manager.compute_scores([policy], [opponent], [["assert foo()==42"]])
    assert scores == [1.0]


def test_compute_scores_tie_breaker_code_length(manager):
    policy = """\n```python\ndef foo():\n    return 1\n```\n"""
    opponent = """\n```python\ndef foo():\n    return 1  # opponent with comment\n```\n"""
    scores = manager.compute_scores([policy], [opponent], [["assert foo()==1"]])
    assert scores == [pytest.approx(2.0 / 3.0)]


def test_compute_scores_identical_code_is_tie(manager):
    snippet = """\n```python\ndef foo():\n    return 5\n```\n"""
    scores = manager.compute_scores([snippet], [snippet], [["assert foo()==5"]])
    assert scores == [0.5]
