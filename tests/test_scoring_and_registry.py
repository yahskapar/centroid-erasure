"""
Scoring-path and registry invariants.

Every gap here was found by an audit and confirmed failing before the fix:
  * main.py scored over A-E while the published pipeline scores over A-D,
    making CLI numbers non-comparable to the shipped fixture
  * parse_mc_answer handled only A-D while ALL_LETTERS is A-H
  * three MODEL_REGISTRY keys had no visual-token finder, so the hook would
    fall back to a positional heuristic and silently degrade results
"""

import os
import re

import pytest
import torch

from centroid_erasure import MODEL_REGISTRY, find_visual_token_range
from centroid_erasure.constants import ALL_LETTERS, EVAL_TASKS, TEXT_COMPETES, TEXT_NEEDED
from centroid_erasure.data.utils import parse_mc_answer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── answer parsing ──


@pytest.mark.parametrize("letter", ALL_LETTERS)
def test_parse_mc_answer_covers_every_letter_the_scorer_can_emit(letter):
    assert parse_mc_answer(f"({letter})") == letter
    assert parse_mc_answer(letter) == letter
    assert parse_mc_answer(f"({letter}) some trailing text") == letter


def test_parse_mc_answer_returns_none_on_garbage():
    assert parse_mc_answer("") is None
    assert parse_mc_answer("no letter here") is None


# ── the scorer's letter set must match the published pipeline ──


def test_cli_scores_over_the_same_letters_as_the_published_pipeline():
    """The published pipeline uses ALL_LETTERS[:n_choices] with n_choices=4 for
    BLINK. Scoring over a wider set can change an argmax and silently shift
    every reported number."""
    sweep = open(os.path.join(REPO, "pipeline", "paper_sweep.py")).read()
    assert "letters = ALL_LETTERS[:nc]" in sweep
    assert 'samples[0].get("_n_choices", 4)' in sweep

    main_src = open(os.path.join(REPO, "main.py")).read()
    assert "ALL_LETTERS[: args.n_choices]" in main_src, (
        "main.py must not use the wider default letter set"
    )
    assert re.search(r'default=4,\s*dest="n_choices"', main_src), (
        "main.py --n-choices must default to 4, matching the pipeline"
    )


def test_demo_scores_over_four_letters():
    demo = open(os.path.join(REPO, "demo", "run_demo.py")).read()
    assert "ALL_LETTERS[:4]" in demo


# ── registry ──


# Three registry keys deliberately have NO visual-token finder, matching the
# published run. Widening the dispatch would change which code path the
# InternVL2.5/3/3.5 longitudinal series takes, and that series is in the paper.
NO_FINDER_BY_DESIGN = {"qwen2_vl", "internvl3_8b", "internvl35_8b"}


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY))
def test_visual_token_dispatch_matches_the_published_set(name):
    ids = torch.zeros(1, 40, dtype=torch.long)
    hidden = torch.zeros(1, 40, 8)
    try:
        start, end = find_visual_token_range(name, ids, hidden, None)
    except ValueError as e:
        assert "No visual token finder" in str(e)
        assert name in NO_FINDER_BY_DESIGN, (
            f"{name} lost its finder; this changes results silently"
        )
        return
    except Exception:
        return  # processor-dependent paths need a real processor
    assert name not in NO_FINDER_BY_DESIGN, (
        f"{name} gained a finder. That changes its code path relative to the "
        "published run; see the note in centroid_erasure/visual_tokens.py."
    )
    assert 0 <= start <= end <= hidden.shape[1]


def test_missing_finder_error_is_loud_about_the_consequence():
    """A bare 'not supported' message invites people to ignore it. The message
    must say that callers fall back and degrade silently."""
    with pytest.raises(ValueError, match="positional heuristic"):
        find_visual_token_range("qwen2_vl", torch.zeros(1, 40, dtype=torch.long),
                                torch.zeros(1, 40, 8), None)


def test_registry_has_no_duplicate_model_ids():
    seen = {}
    for name, cfg in MODEL_REGISTRY.items():
        seen.setdefault(cfg.model_id, []).append(name)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    assert not dupes, f"duplicate model_ids: {dupes}"


def test_shipped_model_configs_use_the_pinned_transformers_decoder_paths():
    """Paper models should not depend on the broad ModuleList fallback.

    These paths were checked against metadata-only model instances under the
    pinned transformers==5.4.0 API. A stale path can otherwise resolve a vision
    stack instead of the language decoder after an architecture refactor.
    """
    expected = {
        "qwen": "model.language_model.layers",
        "qwen_3b": "model.language_model.layers",
        "qwen3": "model.language_model.layers",
        "qwen3_4b": "model.language_model.layers",
        "internvl": "model.language_model.layers",
        "llava_ov": "model.language_model.layers",
        "idefics3": "model.text_model.layers",
    }
    for name, path in expected.items():
        assert MODEL_REGISTRY[name].lm_layer_path == path
        assert re.fullmatch(r"[0-9a-f]{40}", MODEL_REGISTRY[name].revision)


# ── shared constants ──


def test_task_split_partitions_the_evaluated_tasks():
    assert set(TEXT_COMPETES) | set(TEXT_NEEDED) == set(EVAL_TASKS)
    assert not set(TEXT_COMPETES) & set(TEXT_NEEDED)
    assert len(TEXT_COMPETES) == len(TEXT_NEEDED) == 3


def test_main_and_constants_agree_on_the_task_split():
    import main as cli

    assert set(cli.PAPER_TASKS) == set(EVAL_TASKS)
    assert set(cli.TEXT_COMPETES) == set(TEXT_COMPETES)


def test_all_letters_covers_the_default_choice_set():
    from centroid_erasure.eval_mcqa import get_choice_token_ids
    import inspect

    default = inspect.getsource(get_choice_token_ids)
    for letter in re.findall(r'"([A-H])"', default):
        assert letter in ALL_LETTERS


# ── gold-answer parsing must be shared, not reimplemented inline ──


@pytest.mark.parametrize(
    "raw", ["(A)", "(B)", "(C)", "(D)", "A", "D", "(D) blue", "  (C)  ", "(a)"]
)
def test_cli_and_demo_gold_agree_with_the_shared_parser(raw):
    import main as cli

    assert cli._gold({"answer": raw}) == parse_mc_answer(raw)


def test_gold_handles_an_answer_with_trailing_text():
    """'(D) blue' must yield 'D'. An inline .strip('()').upper() gives 'D BLUE',
    which never matches a predicted letter and silently scores zero."""
    import main as cli

    assert cli._gold({"answer": "(D) blue"}) == "D"


def test_neither_cli_nor_demo_reimplements_the_parser_inline():
    for rel in ("main.py", "demo/run_demo.py"):
        src = open(os.path.join(REPO, rel)).read()
        assert '.strip("()").upper()' not in src, f"{rel} still parses gold inline"
        assert "parse_mc_answer" in src, f"{rel} does not use the shared parser"


@pytest.mark.parametrize("fn", ["load_model", "prepare_inputs"])
def test_every_registry_model_has_a_branch_in(fn):
    """A registry entry with no loader raises a bare ValueError at run time."""
    import ast

    tree = ast.parse(open(os.path.join(REPO, "centroid_erasure", "models.py")).read())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == fn)
    handled = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Compare):
            for c in sub.comparators:
                if isinstance(c, ast.Constant) and isinstance(c.value, str):
                    handled.add(c.value)
                elif isinstance(c, (ast.Tuple, ast.List)):
                    handled.update(
                        e.value for e in c.elts
                        if isinstance(e, ast.Constant) and isinstance(e.value, str)
                    )
    missing = sorted(set(MODEL_REGISTRY) - handled)
    assert not missing, f"{fn} has no branch for: {missing}"


# ── the --compare path ──


def test_compare_renders_and_passes_when_fed_the_published_numbers(capsys):
    """Feeding the published values back in must reproduce them exactly and
    report PASS on both checks. Guards the comparison arithmetic itself."""
    import json
    import sys

    sys.path.insert(0, REPO)
    import main as cli

    pub = json.load(
        open(os.path.join(REPO, "demo", "fixtures", "qwen_expected.json"))
    )["sufficiency"]
    res = {
        t: {
            "n": v.get("n", 0),
            "baseline": v.get("baseline", 0.0),
            "text_centroid_cost": v["text_centroid_cost"],
            "vis_centroid_cost": v["vis_centroid_cost"],
        }
        for t, v in pub.items()
    }
    mt = sum(r["text_centroid_cost"] for r in res.values()) / len(res)
    mv = sum(r["vis_centroid_cost"] for r in res.values()) / len(res)

    cli._compare_to_published("qwen", res, mt, mv, strict=True)
    out = capsys.readouterr().out
    assert "text cost exceeds visual cost      : PASS" in out
    assert "mean text cost within 0.010        : PASS" in out
    assert "every task within 2 items          : PASS" in out
    assert "REVIEW" not in out


def test_compare_flags_a_run_that_is_far_from_published(capsys):
    import sys

    sys.path.insert(0, REPO)
    import main as cli

    res = {"Forensic_Detection": {"text_centroid_cost": 0.0, "vis_centroid_cost": 0.0}}
    cli._compare_to_published("qwen", res, 0.0, 0.0, strict=True)
    out = capsys.readouterr().out
    assert "REVIEW" in out, "a wildly different run must not silently report PASS"


def test_strict_mode_catches_a_deviation_the_loose_bar_would_pass(capsys):
    """0.03 on the mean is roughly four items per task. On the shipped-bank
    path that is an environment problem, not sampling noise, and the old 0.05
    tolerance would have reported PASS."""
    import json
    import sys

    sys.path.insert(0, REPO)
    import main as cli

    pub = json.load(
        open(os.path.join(REPO, "demo", "fixtures", "qwen_expected.json"))
    )["sufficiency"]
    res = {
        t: {
            "n": v.get("n", 0),
            "text_centroid_cost": v["text_centroid_cost"] + 0.03,
            "vis_centroid_cost": v["vis_centroid_cost"],
        }
        for t, v in pub.items()
    }
    mt = sum(r["text_centroid_cost"] for r in res.values()) / len(res)
    mv = sum(r["vis_centroid_cost"] for r in res.values()) / len(res)

    cli._compare_to_published("qwen", res, mt, mv, strict=True)
    strict_out = capsys.readouterr().out
    assert "REVIEW" in strict_out
    assert "K-means backend is irrelevant here" in strict_out

    cli._compare_to_published("qwen", res, mt, mv, strict=False)
    loose_out = capsys.readouterr().out
    assert "within 0.050" in loose_out and "REVIEW" not in loose_out


def test_the_kmeans_caveat_is_not_used_to_excuse_the_shipped_bank_path():
    """The demo and the strict comparison fit nothing, so blaming the K-means
    backend there is false and hides real breakage."""
    demo_doc = open(os.path.join(REPO, "demo", "README.md")).read()
    demo_src = open(os.path.join(REPO, "demo", "run_demo.py")).read()
    for text, label in ((demo_doc, "demo/README.md"), (demo_src, "demo/run_demo.py")):
        assert "backend" not in text or "irrelevant" in text or "no part" in text, (
            f"{label} still blames the K-means backend on a path that fits nothing"
        )


def test_compare_handles_a_model_with_no_published_reference(capsys):
    import sys

    sys.path.insert(0, REPO)
    import main as cli

    cli._compare_to_published("not_a_model", {}, 0.1, 0.01)
    assert "no published reference" in capsys.readouterr().out
