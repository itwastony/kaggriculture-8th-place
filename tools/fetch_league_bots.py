"""Download the public league bots listed in agents/sources.json into agents/<folder>/.

The d7 league (configs/league_d7.json) uses 101 code bots. Two are shipped (caretaker, pass_agent); 77 are public
Kaggle notebooks (or a public dataset). This script downloads each notebook's output with the Kaggle CLI, looks for a
main.py whose sha256 starts with the recorded prefix (as a file or inside a .tar.gz / .zip of the output) and installs
it the way the league expects:

  - found inside an archive: the archive is unpacked into agents/<folder>/ and kept as _source_bundle.tar.gz
    (some bots import helper modules from it at run time);
  - found only as a plain file: it is copied as main.py, and a single .tar.gz in the output root is kept as
    _source_bundle.tar.gz.

Notebooks change: Kaggle serves the output of the latest version, so a bot may no longer match the file we trained
against. Such bots are reported as "changed" and skipped; --accept-any installs the newest main.py of the output
instead (a different opponent than in the original run). Missing bots are handled by
`entry_points/train_appo.py --league-available-only`.

Usage:
  python tools/fetch_league_bots.py                    # all public bots that are not installed yet
  python tools/fetch_league_bots.py --only pub0923_tetsutani_dp meta2_salemali
  python tools/fetch_league_bots.py --from-dir DIR     # offline: copy DIR/<owner>__<slug>/output/, no downloads
  python tools/fetch_league_bots.py --status           # what is installed, and which main.py matches the record

Downloads are cached in runs/league_bots_cache/<owner>__<slug>/ (a non-empty cache folder is reused). Installing a
bot replaces its agents/<folder>/ entirely. Requires `pip install kaggle==2.2.4` and a Kaggle API token for downloads.
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AGENTS = os.path.join(ROOT, "agents")
CACHE = os.path.join(ROOT, "runs", "league_bots_cache")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def kaggle(*args):
    """Run the Kaggle CLI; returns (exit code, stdout + stderr)."""
    exe = shutil.which("kaggle") or os.path.join(os.path.dirname(sys.executable), "kaggle")
    r = subprocess.run([exe, *args], capture_output=True, text=True)
    return r.returncode, (r.stdout + r.stderr).strip()


def download(entry, from_dir, sleep, cache):
    """Returns a directory with the raw output of the notebook (or dataset), or None.

    The cache folder is reused when it is not empty; with from_dir, DIR/<owner>__<slug>/output/ is copied into it."""
    key = (entry.get("kernel") or entry["dataset"]).replace("/", "__")
    d = os.path.join(cache, key)
    if os.path.isdir(d) and os.listdir(d):
        return d
    if from_dir:
        src = os.path.join(from_dir, key, "output")
        if not os.path.isdir(src):
            print(f"    not found: {src}")
            return None
        shutil.copytree(src, d, dirs_exist_ok=True)
        return d
    os.makedirs(d, exist_ok=True)
    if entry.get("kernel"):
        rc, msg = kaggle("kernels", "output", entry["kernel"], "-p", d, "-o", "-q")
    else:
        rc, msg = kaggle("datasets", "download", entry["dataset"], "-p", d, "--unzip", "-q")
    time.sleep(sleep)
    if rc != 0 or not os.listdir(d):
        print(f"    download failed: {msg[-300:]}")
        return None
    return d


def unpack_archives(d):
    """Unpacks every .tar.gz / .tgz / .zip of d next to it (into <name>__x/). Returns {unpacked dir: archive}."""
    out = {}
    for dirpath, _, files in os.walk(d):
        if "__x" in dirpath:
            continue
        for name in files:
            p = os.path.join(dirpath, name)
            x = p + "__x"
            try:
                if name.endswith((".tar.gz", ".tgz")):
                    if not os.path.isdir(x):
                        with tarfile.open(p) as t:
                            try:
                                t.extractall(x, filter="data")
                            except TypeError:  # Python < 3.11.4
                                t.extractall(x)
                elif name.endswith(".zip"):
                    if not os.path.isdir(x):
                        with zipfile.ZipFile(p) as z:
                            z.extractall(x)
                else:
                    continue
            except (tarfile.TarError, zipfile.BadZipFile, OSError) as e:
                print(f"    cannot unpack {name}: {e}")
                continue
            out[x] = p
    return out


def candidates(d):
    """All .py files under d: main.py inside an unpacked archive first, then other main.py, then the rest."""
    py = []
    for dirpath, _, files in os.walk(d):
        if "__pycache__" in dirpath:
            continue
        py += [os.path.join(dirpath, f) for f in files if f.endswith(".py")]
    return sorted(py, key=lambda p: (os.path.basename(p) != "main.py", "__x" not in p, -os.path.getmtime(p)))


def install(folder, main, archives, raw, dest):
    """Install the candidate file main (found under raw) as dest/<folder>/main.py, replacing that folder.

    archives = {unpacked dir: archive} from unpack_archives. A main.py inside an unpacked archive brings its whole
    directory (and the archive as _source_bundle.tar.gz); otherwise the file is copied alone, plus the single
    .tar.gz / .tgz of the output root as _source_bundle.tar.gz when there is exactly one."""
    dst = os.path.join(dest, folder)
    if os.path.isdir(dst):
        shutil.rmtree(dst)
    os.makedirs(dst)
    home = next((x for x in archives if main.startswith(x + os.sep)), None)
    if home is not None and os.path.basename(main) == "main.py":  # from a submission archive: its content is the bot
        shutil.copytree(os.path.dirname(main), dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"))
        if archives[home].endswith((".tar.gz", ".tgz")):
            shutil.copy2(archives[home], os.path.join(dst, "_source_bundle.tar.gz"))
    else:  # a plain file (or one file of a collection): copy it alone; a single root archive is kept as the bundle
        shutil.copy2(main, os.path.join(dst, os.path.basename(main)))
        tars = [os.path.join(raw, n) for n in os.listdir(raw) if n.endswith((".tar.gz", ".tgz"))]
        if len(tars) == 1:
            shutil.copy2(tars[0], os.path.join(dst, "_source_bundle.tar.gz"))
    if os.path.basename(main) != "main.py":
        shutil.move(os.path.join(dst, os.path.basename(main)), os.path.join(dst, "main.py"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help="folders to fetch (default: all public bots)")
    ap.add_argument("--from-dir", help="offline source: DIR/<owner>__<slug>/output/")
    ap.add_argument("--accept-any", action="store_true", help="install the newest main.py when no file matches")
    ap.add_argument("--force", action="store_true", help="re-install bots that are already installed")
    ap.add_argument("--sleep", type=float, default=2.0, help="seconds between Kaggle requests")
    ap.add_argument("--status", action="store_true", help="only report what is installed")
    ap.add_argument("--dest", default=AGENTS, help="where to install (default: agents/)")
    ap.add_argument("--cache", default=CACHE, help="download cache (default: runs/league_bots_cache/)")
    a = ap.parse_args()

    entries = json.load(open(os.path.join(AGENTS, "sources.json")))
    if a.status:
        for e in entries:
            p = os.path.join(a.dest, e["folder"], "main.py")
            state = (
                "missing"
                if not os.path.exists(p)
                else ("match" if sha256(p).startswith(e["main_sha256_prefix"]) else "installed, differs")
            )
            print(f"{e['folder']:32s} {e['status']:13s} {state}")
        return

    todo = [e for e in entries if e["status"] == "public" and (not a.only or e["folder"] in a.only)]
    res = {"match": [], "changed": [], "changed_installed": [], "failed": [], "skipped": []}
    for e in todo:
        folder, pref = e["folder"], e["main_sha256_prefix"]
        have = os.path.join(a.dest, folder, "main.py")
        if os.path.exists(have) and not a.force:
            res["skipped"].append(folder)
            continue
        print(f"{folder}  <- {e.get('kernel') or e.get('dataset')}")
        raw = None
        for key in ("kernel", "dataset"):
            if e.get(key):
                raw = download({key: e[key]}, a.from_dir, a.sleep, a.cache)
                if raw:
                    archives = unpack_archives(raw)
                    hit = next((p for p in candidates(raw) if sha256(p).startswith(pref)), None)
                    if hit:
                        install(folder, hit, archives, raw, a.dest)
                        res["match"].append(folder)
                        print(f"    installed, main.py sha256 {pref}… matches the record")
                        break
        else:
            if raw is None:
                res["failed"].append(folder)
                continue
            mains = [p for p in candidates(raw) if os.path.basename(p) == "main.py"]
            if a.accept_any and mains:
                install(folder, mains[0], unpack_archives(raw), raw, a.dest)
                res["changed_installed"].append(folder)
                print(f"    no file matches {pref}…; installed the newest main.py (a different version)")
            else:
                res["changed"].append(folder)
                print(f"    no file matches {pref}… (the notebook has changed); skipped")
    print("\nSummary:", {k: len(v) for k, v in res.items()})
    for k in ("changed", "changed_installed", "failed"):
        if res[k]:
            print(f"  {k}: {' '.join(res[k])}")


if __name__ == "__main__":
    main()
