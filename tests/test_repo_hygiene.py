"""
Repo-level invariants: nothing leaks, nothing is silently excluded, docs match code.

Two real bugs motivated this file:
  * `.gitignore` rule `*_token*` matched `centroid_erasure/visual_tokens.py`
  * `.gitignore` rule `data/` matched `centroid_erasure/data/`
Both would have shipped a broken package with no error anywhere.
"""

import ast
import os
import re
import subprocess

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def git(*args):
    return subprocess.run(
        ["git", "-C", REPO, *args], capture_output=True, text=True
    ).stdout


def _in_git_work_tree():
    """A source tarball or `git archive` export is a valid way to consume this
    repo, and git-dependent checks cannot run there. Skip rather than fail."""
    r = subprocess.run(
        ["git", "-C", REPO, "rev-parse", "--is-inside-work-tree"],
        capture_output=True, text=True,
    )
    return r.returncode == 0 and r.stdout.strip() == "true"


needs_git = pytest.mark.skipif(
    not _in_git_work_tree(), reason="not a git work tree (source export)"
)


def py_files():
    out = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".venv", "build"}]
        out += [os.path.join(root, f) for f in files if f.endswith(".py")]
    return out


# ── nothing is silently excluded ──

MUST_SHIP = [
    "centroid_erasure/__init__.py",
    "centroid_erasure/centroids.py",
    "centroid_erasure/hooks.py",
    "centroid_erasure/decoding.py",
    "centroid_erasure/models.py",
    "centroid_erasure/visual_tokens.py",
    "centroid_erasure/constants.py",
    "centroid_erasure/data/__init__.py",
    "centroid_erasure/data/blink.py",
    "centroid_erasure/data/coco.py",
    "main.py",
    "setup.py",
    "setup.sh",
    "requirements.txt",
    "README.md",
    "LICENSE",
    "pipeline/paper_sweep.py",
    "demo/run_demo.py",
    "centroids/qwen.npz",
]


@pytest.mark.parametrize("rel", MUST_SHIP)
def test_required_file_exists(rel):
    assert os.path.exists(os.path.join(REPO, rel)), f"{rel} is missing"


@needs_git
@pytest.mark.parametrize("rel", MUST_SHIP)
def test_required_file_is_not_gitignored(rel):
    rc = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", rel]).returncode
    assert rc != 0, f".gitignore excludes {rel}, which must ship"


# Synthetic paths only. These must stay content-free: naming real internal or
# third-party documents here would republish exactly what the rules exclude.
MUST_NOT_SHIP = [
    "results/run.json", "artifacts/x.safetensors", "_private/notes.md",
    "_working/draft.md", "notes/todo.md", "scan.pdf",
    "saved.mhtml", "a/per_sample.jsonl", "run.log", "main.tex",
    ".env", "api_key.txt", "stray_activations.npz", "photo.png",
    "checkpoints/model.safetensors", "archive.zip",
]


@needs_git
@pytest.mark.parametrize("rel", MUST_NOT_SHIP)
def test_sensitive_path_is_gitignored(rel):
    rc = subprocess.run(["git", "-C", REPO, "check-ignore", "-q", rel]).returncode
    assert rc == 0, f".gitignore does NOT exclude {rel}"


# ── nothing leaks ──

# Reviewer identifiers from the review thread. Assembled from
# fragments so this file does not itself contain a literal match, which would
# make the scanner below flag its own pattern list.
_REVIEWERS = ["gy" + "FC", "VY" + "F5", "S2" + "yC", "3A" + "cd"]

# Third-party and internal document identifiers. A .gitignore rule that NAMES
# the confidential thing it excludes republishes it; that happened once and is
# why these scanners no longer exempt .gitignore.
_THIRD_PARTY = ["Core" + "SemDB", "BL" + "_SQL"]
_INTERNAL = ["CAMERA_" + "READY", "Z_" + "COLM", "REBUT" + "TAL", "final_" + "responses",
             "REVIEWER_" + "COVERAGE", "CLAIMS_" + "RIGHTSIZING"]

FORBIDDEN = {
    **{f"reviewer id {r}": r"\b" + r + r"\b" for r in _REVIEWERS},
    **{f"third-party submission {t}": t for t in _THIRD_PARTY},
    **{f"internal doc {d}": d for d in _INTERNAL},
    "author home directory": r"/home/[a-z]+/",
    "openai key": r"sk-[A-Za-z0-9]{16}",
    "google api key": r"AIza[A-Za-z0-9_\-]{10}",
    "hf token": r"hf_[A-Za-z0-9]{16}",
}


@needs_git
@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN.items()))
def test_no_forbidden_content_in_tracked_files(label, pattern):
    tracked = git("ls-files").split()
    rx = re.compile(pattern)
    hits = []
    for rel in tracked:
        p = os.path.join(REPO, rel)
        try:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for i, line in enumerate(f, 1):
                    if rx.search(line):
                        hits.append(f"{rel}:{i}")
        except (OSError, UnicodeDecodeError):
            continue
    assert not hits, f"{label} found in: {hits[:5]}"


# ── the package actually imports ──


def test_every_intra_package_import_resolves():
    """Catches the exact class of bug that broke pipeline/paper_sweep.py:
    an import rewritten to a module path that does not exist."""
    bad = []
    for path in py_files():
        try:
            tree = ast.parse(open(path).read())
        except SyntaxError as e:
            bad.append(f"{path}: syntax error {e}")
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = node.names[0].name
            if not mod or not mod.startswith("centroid_erasure"):
                continue
            rel = mod.replace(".", os.sep)
            if not (
                os.path.exists(os.path.join(REPO, rel + ".py"))
                or os.path.isdir(os.path.join(REPO, rel))
            ):
                bad.append(f"{os.path.relpath(path, REPO)}: imports missing {mod}")
    assert not bad, "unresolvable imports: " + "; ".join(bad)


def test_no_imports_from_the_private_working_repo():
    """`src.*`, `experiments.*`, `experiments_A6000.*` do not exist in the release."""
    bad = []
    rx = re.compile(r"^\s*(from|import)\s+(src|experiments|experiments_A6000)\b", re.M)
    for path in py_files():
        for m in rx.finditer(open(path).read()):
            bad.append(f"{os.path.relpath(path, REPO)}: {m.group(0).strip()}")
    assert not bad, bad


def test_public_api_is_fully_resolvable():
    import centroid_erasure as ce

    missing = [n for n in ce.__all__ if not hasattr(ce, n)]
    assert not missing, f"__all__ names nothing: {missing}"


# ── docs match code ──


def _main_py_flags():
    import main as cli

    parser = cli.build_parser()
    known = {"--help", "-h"}
    for action in parser._actions:
        known.update(action.option_strings)
    for sub in parser._subparsers._group_actions[0].choices.values():
        for action in sub._actions:
            known.update(action.option_strings)
    return known


def _docs():
    for rel in ("README.md", "docs/EXTENDING.md", "docs/PROTOCOL.md", "demo/README.md"):
        yield rel, open(os.path.join(REPO, rel)).read()


FLAG_RX = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]+)")


def test_every_main_py_flag_shown_in_docs_actually_exists():
    """Only inspects invocations of main.py, so flags belonging to the shell
    script or to demo/run_demo.py are not falsely attributed to the CLI."""
    known = _main_py_flags()
    bad = []
    for rel, text in _docs():
        for line in text.splitlines():
            if "main.py" not in line:
                continue
            for flag in FLAG_RX.findall(line):
                if flag not in known:
                    bad.append(f"{rel}: '{flag}' in `{line.strip()}`")
    assert not bad, "docs show main.py flags that do not exist: " + "; ".join(bad)


def test_every_demo_flag_shown_in_docs_actually_exists():
    import argparse
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_demo", os.path.join(REPO, "demo", "run_demo.py")
    )
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    # Rebuild the demo's parser by calling main() with --help intercepted is
    # fragile; instead read its add_argument calls out of the source.
    src = open(os.path.join(REPO, "demo", "run_demo.py")).read()
    known = set(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', src)) | {"--help", "-h"}

    bad = []
    for rel, text in _docs():
        for line in text.splitlines():
            if "run_demo.py" not in line:
                continue
            for flag in FLAG_RX.findall(line):
                if flag not in known:
                    bad.append(f"{rel}: '{flag}' in `{line.strip()}`")
    assert not bad, "docs show demo flags that do not exist: " + "; ".join(bad)


def test_every_repro_script_flag_shown_in_docs_actually_exists():
    script = os.path.join(REPO, "scripts", "reproduce_core_result.sh")
    src = open(script).read()
    known = set(re.findall(r'"(--[a-z0-9-]+)"', src))
    bad = []
    for rel, text in _docs():
        for line in text.splitlines():
            if "reproduce_core_result.sh" not in line:
                continue
            for flag in FLAG_RX.findall(line):
                if flag not in known:
                    bad.append(f"{rel}: '{flag}' in `{line.strip()}`")
    assert not bad, "docs show script flags that do not exist: " + "; ".join(bad)


def test_readme_subcommands_exist():
    import main as cli

    choices = set(cli.build_parser()._subparsers._group_actions[0].choices)
    readme = open(os.path.join(REPO, "README.md")).read()
    for cmd in re.findall(r"python main\.py (\w+)", readme):
        assert cmd in choices, f"README documents unknown subcommand '{cmd}'"


def test_protocol_doc_matches_the_code_constants():
    from centroid_erasure import PAPER_PROTOCOL

    doc = open(os.path.join(REPO, "docs", "PROTOCOL.md")).read()
    assert f"K = {PAPER_PROTOCOL['k']}" in doc
    assert f"L{PAPER_PROTOCOL['text_layer']}" in doc
    assert f"L{PAPER_PROTOCOL['visual_layer']}" in doc
    assert str(PAPER_PROTOCOL["data_seed"]) in doc
    assert str(PAPER_PROTOCOL["kmeans_seed"]) in doc


def test_readme_documents_every_shipped_bank():
    import glob

    readme = open(os.path.join(REPO, "README.md")).read()
    for p in glob.glob(os.path.join(REPO, "centroids", "*.npz")):
        name = os.path.basename(p)[:-4]
        assert f"`{name}`" in readme, f"shipped bank '{name}' is undocumented"


# ── registry sanity ──


def test_registry_entries_are_well_formed():
    from centroid_erasure import MODEL_REGISTRY

    for name, cfg in MODEL_REGISTRY.items():
        assert "/" in cfg.model_id, f"{name}: model_id '{cfg.model_id}' is not org/repo"
        assert cfg.lm_layer_path, f"{name}: empty lm_layer_path"
        assert not (cfg.quant_4bit and cfg.quant_8bit), f"{name}: both quant modes set"


def test_every_shipped_bank_has_a_registry_entry():
    import glob

    from centroid_erasure import MODEL_REGISTRY

    for p in glob.glob(os.path.join(REPO, "centroids", "*.npz")):
        name = os.path.basename(p)[:-4]
        assert name in MODEL_REGISTRY, f"bank '{name}' has no MODEL_REGISTRY entry"


@pytest.mark.parametrize("label,pattern", sorted(FORBIDDEN.items()))
def test_no_forbidden_content_anywhere_in_the_tree(label, pattern):
    """Git-free equivalent of the tracked-files scan, so a source export or a
    tarball consumer is covered too."""
    rx = re.compile(pattern)
    hits = []
    for root, dirs, files in os.walk(REPO):
        dirs[:] = [
            d for d in dirs
            if d not in {".git", "__pycache__", ".venv", "build", "node_modules",
                         ".pytest_cache", ".mypy_cache", ".ruff_cache", "dist"}
            and not d.startswith("repro_")
            and not d.endswith(".egg-info")
        ]
        for name in files:
            if name.endswith((".npz", ".pdf", ".png", ".pyc")):
                continue
            path = os.path.join(root, name)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if rx.search(line):
                            hits.append(f"{os.path.relpath(path, REPO)}:{i}")
            except OSError:
                continue
    assert not hits, f"{label} found in: {hits[:5]}"


def test_readme_dataset_links_match_the_loaders():
    """A README pointing at a different dataset than the code loads sends
    people to the wrong data. Checked offline against the loader sources."""
    readme = open(os.path.join(REPO, "README.md")).read()
    data_dir = os.path.join(REPO, "centroid_erasure", "data")
    loader_ids = set()
    for name in os.listdir(data_dir):
        if not name.endswith(".py"):
            continue
        src = open(os.path.join(data_dir, name)).read()
        loader_ids.update(re.findall(r'"([\w.\-]+/[\w.\-]+)"', src))

    # Every HF dataset the README links must be one a loader actually uses.
    linked = set(re.findall(r"huggingface\.co/datasets/([\w.\-]+/[\w.\-]+)", readme))
    unknown = {d for d in linked if d not in loader_ids}
    assert not unknown, (
        f"README links datasets no loader references: {sorted(unknown)}. "
        f"Loaders use: {sorted(loader_ids)}"
    )


def test_reproduction_record_matches_the_shipped_reference():
    """docs/REPRODUCTION.md quotes published values. If a fixture is ever
    regenerated, the record must not silently keep quoting the old ones."""
    import json

    doc = open(os.path.join(REPO, "docs", "REPRODUCTION.md")).read()
    pub = json.load(
        open(os.path.join(REPO, "demo", "fixtures", "qwen_expected.json"))
    )["_summary"]
    assert f"{pub['mean_text_cost']:.4f}" in doc
    assert f"{pub['asymmetry_ratio']}x" in doc


def test_reproduce_script_uses_the_strict_bar_by_default():
    """The default path reads a byte-identical bank and fits nothing, so a
    loose tolerance there would hide a broken environment."""
    src = open(os.path.join(REPO, "scripts", "reproduce_core_result.sh")).read()
    assert "mean_tol = 0.010 if strict else 0.05" in src
    assert "two items' worth of accuracy" in src
    # The K-means caveat must not be used to excuse the shipped-bank path.
    tail = src[src.index("VERDICTS"):]
    assert "K-means backend is therefore" in tail and "irrelevant" in tail
