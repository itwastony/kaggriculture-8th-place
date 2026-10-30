"""Tests of tools/strip_init.py on tiny synthetic checkpoints (skipped when torch is not installed)."""

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_t_strip_init", ROOT / "tools" / "strip_init.py")
si = importlib.util.module_from_spec(spec)
spec.loader.exec_module(si)


def _save(path, model, **extra):
    torch.save({"model": model, **extra}, path)
    return str(path)


def _run(monkeypatch, src, like, out):
    monkeypatch.setattr(sys, "argv", ["strip_init.py", "--src", src, "--like", like, "--out", str(out)])
    si.main()


@pytest.fixture
def ckpts(tmp_path):
    w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    src = _save(
        tmp_path / "appo.pt",
        {"enc.w": w, "head.b": torch.ones(4), "venc.w": torch.zeros(2), "venc.b": torch.zeros(1)},
        d=8,
        layers=2,
        step=123,
        opt={"lr": 1.0},
    )
    like = _save(tmp_path / "bc.pt", {"enc.w": torch.zeros(2, 3), "head.b": torch.zeros(4)}, d=8, layers=2)
    return tmp_path, src, like, w


def test_keeps_reference_keys(ckpts, monkeypatch, capsys):
    tmp, src, like, w = ckpts
    out = tmp / "init.pt"
    _run(monkeypatch, src, like, out)
    assert capsys.readouterr().out.strip() == "{'kept': 2, 'dropped': 2}"
    c = torch.load(out, map_location="cpu", weights_only=False)
    assert set(c) == {"model", "d", "layers", "step", "src"}
    assert (c["d"], c["layers"], c["step"], c["src"]) == (8, 2, 0, src)
    assert set(c["model"]) == {"enc.w", "head.b"}
    assert torch.equal(c["model"]["enc.w"], w)  # values come from --src, not from --like
    with pytest.raises(FileExistsError):  # an existing --out is never overwritten
        _run(monkeypatch, src, like, out)


def test_rejects_missing_or_reshaped_keys(ckpts, monkeypatch):
    tmp, src, _, _ = ckpts
    like = _save(tmp / "bc2.pt", {"enc.w": torch.zeros(2, 3), "extra": torch.zeros(1)})
    with pytest.raises(AssertionError):
        _run(monkeypatch, src, like, tmp / "a.pt")
    like = _save(tmp / "bc3.pt", {"enc.w": torch.zeros(3, 2)})
    with pytest.raises(AssertionError, match="enc.w"):
        _run(monkeypatch, src, like, tmp / "b.pt")
    assert not (tmp / "a.pt").exists() and not (tmp / "b.pt").exists()
