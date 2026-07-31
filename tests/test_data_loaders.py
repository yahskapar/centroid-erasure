from centroid_erasure.data import blink, coco, cvbench, medblink, mmvp, pope
from centroid_erasure.data import scienceqa


class FakeImage:
    def __init__(self, name):
        self.name = name
        self.conversions = []

    def convert(self, mode):
        self.conversions.append(mode)
        return f"{self.name}:{mode}"


class FakeStream:
    def __init__(self, rows):
        self.rows = list(rows)
        self.shuffle_calls = []

    def __iter__(self):
        return iter(self.rows)

    def shuffle(self, *, seed, buffer_size):
        self.shuffle_calls.append((seed, buffer_size))
        return self


def test_blink_loader_pins_revision_normalizes_images_and_skips_failures(
    monkeypatch,
):
    calls = []
    first = FakeImage("first")
    second = FakeImage("second")

    def fake_load(repo, task, *, split, revision):
        calls.append((repo, task, split, revision))
        if task == "Unavailable":
            raise RuntimeError("offline")
        return [
            {"idx": "skip", "prompt": "none", "answer": "(A)"},
            {
                "idx": "one",
                "prompt": "choose",
                "answer": "(B)",
                "image_1": first,
                "image_2": second,
            },
            {
                "idx": "two",
                "prompt": "later",
                "answer": "(A)",
                "image_1": FakeImage("later"),
            },
        ]

    monkeypatch.setattr(blink, "hf_load", fake_load)
    result = blink.load_blink(
        tasks=["Counting", "Unavailable"], split="val", max_per_task=1
    )

    assert result == {
        "Counting": [
            {
                "idx": "one",
                "prompt": "choose",
                "images": ["first:RGB", "second:RGB"],
                "answer": "(B)",
            }
        ]
    }
    assert first.conversions == ["RGB"]
    assert second.conversions == ["RGB"]
    assert calls == [
        (
            "BLINK-Benchmark/BLINK",
            "Counting",
            "val",
            blink.BLINK_REVISION,
        ),
        (
            "BLINK-Benchmark/BLINK",
            "Unavailable",
            "val",
            blink.BLINK_REVISION,
        ),
    ]


def test_coco_loader_uses_explicit_fallback_provenance_and_seed(
    monkeypatch,
):
    calls = []
    streams = []
    rows = [
        {"image": FakeImage("one"), "caption": ["caption one"]},
        {"image": None, "caption": "skip"},
        {"image": FakeImage("two"), "sentences": {"raw": "caption two"}},
        {"image": FakeImage("three"), "text": "caption three"},
    ]

    def fake_load(repo, *, split, streaming, revision):
        calls.append((repo, split, streaming, revision))
        if repo == "detection-datasets/coco":
            raise RuntimeError("primary unavailable")
        stream = FakeStream(rows)
        streams.append(stream)
        return stream

    monkeypatch.setattr(coco, "hf_load", fake_load)
    result = coco.load_coco(max_samples=2, seed=9, allow_fallback=True)

    assert [sample["answer"] for sample in result] == [
        "caption one",
        "caption two",
    ]
    assert [sample["images"] for sample in result] == [
        ["one:RGB"],
        ["two:RGB"],
    ]
    for sample in result:
        assert sample["_source"] == "HuggingFaceM4/COCO"
        assert sample["_split"] == "val"
        assert sample["_revision"] is None
        assert sample["_shuffle_seed"] == 9
    assert calls[0] == (
        "detection-datasets/coco",
        "train",
        True,
        coco.COCO_REVISION,
    )
    assert calls[1:] == [
        ("HuggingFaceM4/COCO", "val", True, None),
        ("HuggingFaceM4/COCO", "val", True, None),
    ]
    assert streams[-1].shuffle_calls == [(9, coco.SHUFFLE_BUFFER)]


def test_coco_loader_fails_closed_when_fallback_is_disabled(monkeypatch):
    calls = []

    def unavailable(repo, **kwargs):
        calls.append((repo, kwargs))
        raise RuntimeError("offline")

    monkeypatch.setattr(coco, "hf_load", unavailable)
    assert coco.load_coco(max_samples=1, allow_fallback=False) == []
    assert [repo for repo, _ in calls] == ["detection-datasets/coco"]
    assert calls[0][1]["revision"] == coco.COCO_REVISION


def test_cvbench_loader_routes_categories_caps_each_and_pins_builtin_repo(
    monkeypatch,
):
    calls = []
    rows = [
        {
            "image": FakeImage("spatial-1"),
            "type": "2D spatial",
            "question": "left?",
            "choices": ["yes", "no"],
            "answer": "yes",
        },
        {
            "image": FakeImage("spatial-2"),
            "type": "spatial",
            "question": "right?",
        },
        {
            "image": FakeImage("depth"),
            "category": "3D depth",
            "prompt": "closer?",
            "answer": "A",
        },
        {
            "image": FakeImage("custom"),
            "category": "color",
            "question": "red?",
        },
        {"image": None, "type": "depth", "question": "skip"},
    ]

    def fake_load(repo, *, split, revision):
        calls.append((repo, split, revision))
        return rows

    monkeypatch.setattr(cvbench, "hf_load", fake_load)
    result = cvbench.load_cvbench(max_samples=1)

    assert calls == [
        (cvbench.CVBENCH_REPO, "test", cvbench.CVBENCH_REVISION)
    ]
    assert len(result["2d_spatial"]) == 1
    assert result["2d_spatial"][0]["image"] == "spatial-1:RGB"
    assert result["3d_depth"][0]["question"] == "closer?"
    assert result["color"][0]["category"] == "color"


def test_cvbench_and_medblink_return_empty_on_dataset_failure(monkeypatch):
    def unavailable(*_args, **_kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(cvbench, "hf_load", unavailable)
    monkeypatch.setattr(medblink, "hf_load", unavailable)

    assert cvbench.load_cvbench() == {}
    assert medblink.load_medblink() == {}


def test_medblink_formats_binary_and_sorted_multiple_choice_tasks(
    monkeypatch,
):
    rows = [
        {
            "task": "binary",
            "question_id": "b1",
            "question": "present?",
            "answer": " YES ",
            "image": FakeImage("b1"),
        },
        {
            "task": "binary",
            "question_id": "b2",
            "question": "absent?",
            "answer": "no",
            "image": FakeImage("b2"),
        },
        {
            "task": "color",
            "question_id": "c1",
            "question": "which color?",
            "answer": "red",
            "image": FakeImage("c1"),
        },
        {
            "task": "color",
            "question_id": "c2",
            "question": "which color?",
            "answer": "blue",
            "image": FakeImage("c2"),
        },
    ]
    calls = []

    def fake_load(repo, *, split, revision):
        calls.append((repo, split, revision))
        return rows

    monkeypatch.setattr(medblink, "hf_load", fake_load)
    result = medblink.load_medblink(max_per_task=1)

    assert calls == [
        (medblink.MEDBLINK_REPO, "val", medblink.MEDBLINK_REVISION)
    ]
    assert result["binary"] == [
        {
            "idx": "b1",
            "prompt": "present?\nAnswer:",
            "images": ["b1:RGB"],
            "answer": "yes",
        }
    ]
    color = result["color"][0]
    assert color["answer"] == "B"
    assert color["_n_choices"] == 2
    assert "(A) blue\n(B) red\nAnswer:" in color["prompt"]


def test_medblink_filters_tasks_and_skips_more_than_eight_options(
    monkeypatch,
):
    rows = [
        {
            "task": "too_many",
            "question": f"q{i}",
            "answer": f"answer-{i}",
            "image": FakeImage(str(i)),
        }
        for i in range(9)
    ]
    rows.append(
        {
            "task": "kept",
            "question": "q",
            "answer": "yes",
            "image": FakeImage("kept"),
        }
    )
    monkeypatch.setattr(medblink, "hf_load", lambda *_args, **_kwargs: rows)

    assert medblink.load_medblink(tasks=["too_many"]) == {}
    assert set(medblink.load_medblink(tasks=["kept"])) == {"kept"}


def test_pope_and_scienceqa_normalize_rows_and_use_pinned_revisions(
    monkeypatch,
):
    calls = []

    def pope_load(repo, *, split, revision):
        calls.append((repo, split, revision))
        return [
            {"image": None, "question": "skip", "answer": "yes"},
            {
                "image": FakeImage("pope"),
                "question": "Is there a cat?",
                "answer": " YES ",
                "category": "random",
            },
            {
                "image": FakeImage("later"),
                "question": "later",
                "answer": "no",
            },
        ]

    monkeypatch.setattr(pope, "hf_load", pope_load)
    pope_rows = pope.load_pope(max_samples=1)
    assert pope_rows == [
        {
            "question": "Is there a cat?",
            "answer": "yes",
            "image": "pope:RGB",
            "category": "random",
        }
    ]

    def science_load(repo, *, split, revision):
        calls.append((repo, split, revision))
        return [
            {"image": None, "question": "skip"},
            {
                "image": FakeImage("science"),
                "question": "Which?",
                "choices": ["x", "y"],
                "answer": "1",
                "hint": "look",
                "subject": "physics",
            },
        ]

    monkeypatch.setattr(scienceqa, "hf_load", science_load)
    science_rows = scienceqa.load_scienceqa_img(max_samples=1)
    assert science_rows == [
        {
            "question": "Which?",
            "choices": ["x", "y"],
            "answer": 1,
            "image": "science:RGB",
            "hint": "look",
            "subject": "physics",
        }
    ]
    assert calls == [
        (pope.POPE_REPO, "test", pope.POPE_REVISION),
        (
            scienceqa.SCIENCEQA_REPO,
            "test",
            scienceqa.SCIENCEQA_REVISION,
        ),
    ]


def test_mmvp_loader_and_pair_scoring_without_network(monkeypatch):
    import datasets

    class ImageDataset:
        def __init__(self):
            self.rows = [
                {"image": FakeImage("one")},
                {"image": FakeImage("two")},
                {"image": FakeImage("three")},
            ]

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            return self.rows[index]

    calls = []

    def fake_load(repo, *, split, revision):
        calls.append((repo, split, revision))
        return ImageDataset()

    questions = [
        {"prompt": "q1", "answer": "A", "options": ["x", "y"]},
        {"prompt": "q2", "answer": "B", "options": ["x", "y"]},
        {"prompt": "q3", "answer": "A", "options": ["x", "y"]},
    ]
    monkeypatch.setattr(datasets, "load_dataset", fake_load)
    monkeypatch.setattr(mmvp, "_load_questions_csv", lambda: questions)

    samples = mmvp.load_mmvp(max_samples=3)

    assert calls == [(mmvp.MMVP_REPO, "train", mmvp.MMVP_REVISION)]
    assert [sample["idx"] for sample in samples] == [
        "mmvp_1",
        "mmvp_2",
        "mmvp_3",
    ]
    assert [sample["pair_id"] for sample in samples] == [0, 0, 1]
    assert samples[0]["image"] == "one:RGB"
    assert samples[0]["images"] == ["one:RGB"]

    metrics = mmvp.get_pair_accuracy(
        [
            {"pair_id": 0, "correct": True},
            {"pair_id": 0, "correct": True},
            {"pair_id": 1, "correct": True},
            {"pair_id": 1, "correct": False},
        ]
    )
    assert metrics == {
        "n_images": 4,
        "n_pairs": 2,
        "image_acc": 0.75,
        "pair_acc": 0.5,
        "n_pair_correct": 1,
    }


def test_mmvp_parsing_helpers_and_csv_count_mismatch(monkeypatch):
    import datasets

    assert mmvp._parse_options("(a) Open wings (b) Closed wings") == [
        "Open wings",
        "Closed wings",
    ]
    assert mmvp._answer_to_letter("(B)") == "B"
    assert mmvp._answer_to_letter("choice c") == "CHOICE C"
    assert mmvp.get_pair_accuracy([]) == {
        "n_images": 0,
        "n_pairs": 0,
        "image_acc": 0,
        "pair_acc": 0,
        "n_pair_correct": 0,
    }

    class TwoImages:
        def __len__(self):
            return 2

    monkeypatch.setattr(
        datasets, "load_dataset", lambda *_args, **_kwargs: TwoImages()
    )
    monkeypatch.setattr(mmvp, "_load_questions_csv", lambda: [{"prompt": "q"}])
    assert mmvp.load_mmvp() == []


def test_mmvp_csv_loader_pins_revision_and_builds_letter_prompts(monkeypatch):
    import huggingface_hub
    import pandas as pd

    calls = []

    def fake_download(repo, filename, *, repo_type, revision):
        calls.append((repo, filename, repo_type, revision))
        return "/virtual/Questions.csv"

    class Frame:
        columns = ["Question", "Options", "Correct Answer"]

        def __len__(self):
            return 2

        def iterrows(self):
            rows = [
                {
                    "Question": "Open or closed?",
                    "Options": "(a) Open (b) Closed",
                    "Correct Answer": "(a)",
                },
                {
                    "Question": "Near or far?",
                    "Options": "(a) Near (b) Far",
                    "Correct Answer": "(B)",
                },
            ]
            return enumerate(rows)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(pd, "read_csv", lambda path: Frame())

    questions = mmvp._load_questions_csv()

    assert calls == [
        (
            mmvp.MMVP_REPO,
            "Questions.csv",
            "dataset",
            mmvp.MMVP_REVISION,
        )
    ]
    assert questions == [
        {
            "prompt": "Open or closed?\n(A) Open\n(B) Closed\nAnswer:",
            "answer": "A",
            "options": ["Open", "Closed"],
        },
        {
            "prompt": "Near or far?\n(A) Near\n(B) Far\nAnswer:",
            "answer": "B",
            "options": ["Near", "Far"],
        },
    ]
