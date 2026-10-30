"""Tests of tools/fetch_league_bots.py and of the league bot records it reads.

1. agents/sources.json is consistent with itself, with the two shipped bots (sha256 prefixes) and with the league of
   the final run (configs/league_d7.json).
2. The offline install path (--from-dir) on synthetic notebook outputs, never calling the Kaggle CLI: plain main.py,
   main.py inside a submission archive, a changed notebook (skipped, or installed with --accept-any), --status.
"""

import ast
import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fbl = _load("_t_fetch_league_bots", "tools/fetch_league_bots.py")
SOURCES = json.loads((ROOT / "agents" / "sources.json").read_text())


# ------------------------------------------------------------------ records


def test_sources_json_is_consistent():
    folders = [e["folder"] for e in SOURCES]
    assert len(folders) == len(set(folders))
    for e in SOURCES:
        assert e["status"] in {"public", "not_released", "unavailable", "shipped"}, e["folder"]
        pref = e["main_sha256_prefix"]
        assert len(pref) == 8 and all(c in "0123456789abcdef" for c in pref), e["folder"]
        assert e["league_weight"] > 0
        if e["status"] == "public":
            assert e.get("kernel") or e.get("dataset"), e["folder"]


def test_shipped_bots_match_their_recorded_hash():
    shipped = [e for e in SOURCES if e["status"] == "shipped"]
    assert {e["folder"] for e in shipped} == {"caretaker", "pass_agent"}
    for e in shipped:
        assert fbl.sha256(ROOT / "agents" / e["folder"] / "main.py").startswith(e["main_sha256_prefix"])


def test_league_config_lists_exactly_the_recorded_bots():
    league = json.loads((ROOT / "configs" / "league_d7.json").read_text())
    code = {name.split("/")[1]: w for name, w in league if name.startswith("agents/")}
    assert all(name == f"agents/{name.split('/')[1]}/main.py" for name, _ in league if name.startswith("agents/"))
    assert code == {e["folder"]: e["league_weight"] for e in SOURCES}


# ------------------------------------------------------------------ offline install


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _tar(files):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


MAIN_A = b"def agent(obs):\n    return {}\n"
MAIN_B = b"def agent(obs, conf):\n    return {'farmer': ['PASS']}\n"
MAIN_NEW = b"def agent(obs):\n    return None\n"


@pytest.fixture
def setup(tmp_path, monkeypatch):
    """Synthetic sources.json + notebook outputs in DIR/<owner>__<slug>/output/; the Kaggle CLI must not be called."""
    agents = tmp_path / "agents"
    agents.mkdir()
    src = tmp_path / "outputs"
    entries = [
        {"folder": "bot_plain", "status": "public", "kernel": "o/plain", "main_sha256_prefix": _sha(MAIN_A)[:8]},
        {"folder": "bot_tar", "status": "public", "kernel": "o/tar", "main_sha256_prefix": _sha(MAIN_B)[:8]},
        {"folder": "bot_changed", "status": "public", "kernel": "o/changed", "main_sha256_prefix": "00000000"},
        {"folder": "bot_missing", "status": "public", "dataset": "o/gone", "main_sha256_prefix": "11111111"},
        {"folder": "bot_private", "status": "not_released", "main_sha256_prefix": "22222222"},
    ]
    for e in entries:
        e["league_weight"] = 1
    (agents / "sources.json").write_text(json.dumps(entries))

    out = src / "o__plain" / "output"
    out.mkdir(parents=True)
    (out / "main.py").write_bytes(MAIN_A)
    (out / "notes.py").write_bytes(b"x = 1\n")
    (out / "bundle.tar.gz").write_bytes(_tar({"main.py": MAIN_NEW}))  # an archive whose main.py does not match

    out = src / "o__tar" / "output"
    out.mkdir(parents=True)
    (out / "submission.tar.gz").write_bytes(_tar({"main.py": MAIN_B, "lib/helper.py": b"Y = 2\n"}))

    out = src / "o__changed" / "output"
    out.mkdir(parents=True)
    with zipfile.ZipFile(out / "sub.zip", "w") as z:
        z.writestr("main.py", MAIN_NEW)

    def no_kaggle(*args):
        raise AssertionError(f"Kaggle CLI called: {args}")

    monkeypatch.setattr(fbl, "AGENTS", str(agents))
    monkeypatch.setattr(fbl, "kaggle", no_kaggle)
    dest, cache = tmp_path / "dest", tmp_path / "cache"
    return {"agents": agents, "src": src, "dest": dest, "cache": cache}


def _main(monkeypatch, capsys, s, *extra):
    argv = ["fetch_league_bots.py", "--from-dir", str(s["src"]), "--dest", str(s["dest"]), "--cache", str(s["cache"])]
    monkeypatch.setattr(sys, "argv", argv + ["--sleep", "0", *extra])
    fbl.main()
    out = capsys.readouterr().out
    summary = next(line for line in out.splitlines() if line.startswith("Summary:"))
    return ast.literal_eval(summary.split(":", 1)[1].strip()), out


def test_install_from_dir(setup, monkeypatch, capsys):
    s = setup
    res, out = _main(monkeypatch, capsys, s)
    assert res == {"match": 2, "changed": 1, "changed_installed": 0, "failed": 1, "skipped": 0}
    plain = s["dest"] / "bot_plain"
    assert (plain / "main.py").read_bytes() == MAIN_A
    assert sorted(os.listdir(plain)) == ["_source_bundle.tar.gz", "main.py"]  # the plain file + the single root archive
    tar = s["dest"] / "bot_tar"
    assert (tar / "main.py").read_bytes() == MAIN_B
    assert (tar / "lib" / "helper.py").read_bytes() == b"Y = 2\n"
    assert (tar / "_source_bundle.tar.gz").exists()
    assert not (s["dest"] / "bot_changed").exists() and not (s["dest"] / "bot_private").exists()
    assert "changed: bot_changed" in out and "failed: bot_missing" in out

    res, _ = _main(monkeypatch, capsys, s)  # installed bots are skipped without --force
    assert res["skipped"] == 2 and res["match"] == 0

    res, _ = _main(monkeypatch, capsys, s, "--only", "bot_changed", "--accept-any")
    assert res == {"match": 0, "changed": 0, "changed_installed": 1, "failed": 0, "skipped": 0}
    assert (s["dest"] / "bot_changed" / "main.py").read_bytes() == MAIN_NEW


def test_status(setup, monkeypatch, capsys):
    s = setup
    _main(monkeypatch, capsys, s, "--only", "bot_plain")
    (s["dest"] / "bot_tar").mkdir()
    (s["dest"] / "bot_tar" / "main.py").write_bytes(b"other\n")
    monkeypatch.setattr(sys, "argv", ["fetch_league_bots.py", "--status", "--dest", str(s["dest"])])
    fbl.main()
    lines = {line.split()[0]: line.split(None, 2)[2] for line in capsys.readouterr().out.splitlines()}
    assert lines == {
        "bot_plain": "match",
        "bot_tar": "installed, differs",
        "bot_changed": "missing",
        "bot_missing": "missing",
        "bot_private": "missing",
    }


def test_candidates_prefer_main_py_from_archives(tmp_path):
    (tmp_path / "a.py").write_text("")
    (tmp_path / "main.py").write_text("")
    (tmp_path / "pkg.tar.gz").write_bytes(_tar({"main.py": b"", "util.py": b""}))
    archives = fbl.unpack_archives(str(tmp_path))
    assert archives == {str(tmp_path / "pkg.tar.gz") + "__x": str(tmp_path / "pkg.tar.gz")}
    names = [os.path.relpath(p, tmp_path) for p in fbl.candidates(str(tmp_path))]
    assert names[:2] == [os.path.join("pkg.tar.gz__x", "main.py"), "main.py"]
    assert set(names[2:]) == {"a.py", os.path.join("pkg.tar.gz__x", "util.py")}
