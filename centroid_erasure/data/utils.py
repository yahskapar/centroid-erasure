"""
Shared data utilities: image preprocessing and answer parsing.
"""

from PIL import Image
from typing import List, Optional


def concat_images_horizontal(
    images: List[Image.Image], border: int = 3
) -> Image.Image:
    """
    Concatenate multiple PIL images side-by-side with a gray border.

    Used by BLINK/MedBLINK where multiple images are presented in one prompt.
    """
    if len(images) == 1:
        return images[0]

    max_h = max(img.height for img in images)
    resized = []
    for img in images:
        if img.height != max_h:
            ratio = max_h / img.height
            new_w = int(img.width * ratio)
            img = img.resize((new_w, max_h), Image.LANCZOS)
        resized.append(img)

    total_w = sum(img.width for img in resized) + border * (len(resized) - 1)
    canvas = Image.new("RGB", (total_w, max_h), (200, 200, 200))
    x = 0
    for img in resized:
        canvas.paste(img, (x, 0))
        x += img.width + border
    return canvas


def parse_mc_answer(answer_str: str) -> Optional[str]:
    """
    Parse a multiple-choice answer string to a single letter.

    Covers every letter the scorer can emit (see constants.ALL_LETTERS),
    not just A-D: get_choice_token_ids and the published pipeline can both
    score wider option sets.

    Handles formats like: "(A)", "A", "A.", "(A) some text", etc.
    Returns None if parsing fails.
    """
    if not answer_str:
        return None
    s = answer_str.strip().upper()
    from ..constants import ALL_LETTERS

    for letter in ALL_LETTERS:
        if s.startswith(f"({letter})") or s.startswith(letter):
            return letter
    return None
