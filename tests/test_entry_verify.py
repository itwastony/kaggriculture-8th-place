"""The code comparison of entry_points/verify_submission.py (check 3): what it ignores and what it must catch."""

import ast
import importlib

verify = importlib.import_module("verify_submission")  # entry_points/ is on sys.path (conftest.py)
same = verify.same_program

BASE = """
from rl_0123abcd import act as A


def agent(obs, config):
    # a comment
    x = obs["step"] + 1
    return A.execute(obs, x, "ok")
"""

TSELL = """
from rl_0123abcd import act as A
_TSELL = False
_TPROD = ("MILK",)


def _last_step(obs):
    return obs["step"] >= 716


def agent(obs, config):
    x = obs["step"] + 1
    a = A.execute(obs, x, "ok")
    if _TSELL and _last_step(obs):
        a["market"] = []
    return a
"""

TSELL_BASE = """
from rl_0123abcd import act as A


def agent(obs, config):
    x = obs["step"] + 1
    a = A.execute(obs, x, "ok")
    return a
"""


def test_identical_and_formatting_only():
    assert same(BASE, BASE) == []
    reformatted = BASE.replace("# a comment", "# another comment").replace('obs["step"] + 1', "(obs['step'] + 1)")
    assert same(BASE, reformatted) == []


def test_package_hash_is_ignored():
    assert same(BASE, BASE.replace("rl_0123abcd", "rl_fedcba98")) == []


def test_cyrillic_strings_are_ignored_other_strings_are_not():
    ru = BASE.replace('"ok"', '"готово"')  # Russian text in the shipped archive
    assert same(ru, BASE) == [] and same(BASE, ru) == []
    assert same(BASE, BASE.replace('"ok"', '"no"'))
    assert same(BASE, BASE.replace('obs["step"]', 'obs["stop"]'))


def test_code_changes_are_detected():
    assert same(BASE, BASE.replace("+ 1", "+ 2"))
    assert same(BASE, BASE.replace("+ 1", "- 1"))
    assert same(BASE, BASE.replace("    return A.execute", "    x = x\n    return A.execute"))
    assert same(BASE, BASE.replace("config", "cfg"))


def test_disabled_terminal_sell_is_removed_with_a_note():
    notes = []
    assert same(TSELL_BASE, TSELL, notes) == []
    assert len(notes) == 1 and "_TSELL = False" in notes[0]
    # only when the re-export (b) has the option and the archive (a) does not
    assert same(TSELL, TSELL_BASE)


def test_enabled_terminal_sell_is_a_difference():
    assert same(TSELL_BASE, TSELL.replace("_TSELL = False", "_TSELL = True")) == ["_TSELL is True, not False"]
    assert same(TSELL_BASE, TSELL.replace("_TSELL = False", "_TSELL = bool(1)")) == [
        "_TSELL is 'not a constant', not False"
    ]


def test_strip_tsell_keeps_everything_else():
    tree = ast.parse(TSELL)
    assert verify.strip_tsell(tree) is False
    assert ast.dump(tree) == ast.dump(ast.parse(TSELL_BASE))


def test_pkg_dir(tmp_path):
    for d in ("rl_xyz", "rl_0123abcd", "weights.npz"):
        (tmp_path / d).mkdir()
    assert verify.pkg_dir(str(tmp_path)) == str(tmp_path / "rl_0123abcd")
