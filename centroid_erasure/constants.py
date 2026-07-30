"""
Shared constants for the evaluation pipeline.

These were previously defined inside an exploratory experiment script in the
working repo. They are lifted here so the published pipeline in pipeline/ and
the library both depend on one definition.
"""

# Answer-letter tokens scored by the multiple-choice path. BLINK tasks have up
# to four options; the wider range covers benchmarks with more.
ALL_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]

# The six BLINK tasks reported in the paper.
EVAL_TASKS = [
    "Forensic_Detection",
    "Visual_Similarity",
    "Art_Style",
    "Counting",
    "Relative_Depth",
    "Spatial_Relation",
]

# The empirical split from the paper. TEXT-COMPETES tasks are the ones where
# erasing text structure helps once contrasted against; TEXT-NEEDED tasks are
# the ones where the text carries information the model actually requires.
TEXT_COMPETES = ("Forensic_Detection", "Visual_Similarity", "Art_Style")
TEXT_NEEDED = ("Counting", "Relative_Depth", "Spatial_Relation")


# The fixed generic prompt used to harvest the published COCO activations
# (pipeline/paper_sweep.py). It yields 16 post-image tokens per image under the
# pinned processors (32,000 text activations over 2,000 images). The resulting
# banks are applied to semantically different evaluation prompts, so this
# constant preserves the fitting protocol rather than asserting prompt identity.
HARVEST_PROMPT = "Describe what you see in this image.\nAnswer:"
