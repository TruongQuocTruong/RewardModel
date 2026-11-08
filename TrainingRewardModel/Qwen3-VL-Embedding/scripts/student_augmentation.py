"""Online augmentation of the Student input in Stage 2 (Unlabeled samples only).

The Teacher and the Inspector see the original sample; the Student sees the same
sample after one augmentation. Scenarios (confirmed 2026-11-04):

  masking      drop words / phrases that carry no edit content (articles, request
               openers, "in the image"-like phrases); quoted text, colours, numbers,
               positions, nouns and verbs are never touched
  injection    add a neutral prefix and / or suffix template (no new requirement)
  spatial      put another image (source image of an Unlabeled sample outside the
               selected set) left / right / above / below the source image, scale the
               composite back to the pixel count of the source image (aspect kept),
               and name the edited image's position in the instruction; the edited
               images A and B stay unchanged; the composite replaces the source image
               in all three sequences (A, B and source-only)
  flip         mirror the source image and both edited images, swap left <-> right in
               the instruction; not for text / number / rotation / mirror instructions
  swap         independent of the scenario above, with probability SWAP_PROBABILITY:
               the Student sees (B, A) and its target becomes (P_B, P_A, P_Tie)

Online: at training time every (sample, iteration) draws a new scenario and a new
instruction variant (seeded, so reruns repeat them); a variant that fails the automatic
checks falls back to the original instruction. build_augmentation_scenarios.py only
fixes which scenarios each sample may use (and its flipped instruction) and writes a
preview for reading; images are composed at training time.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCENARIOS = ("masking", "injection", "spatial", "flip")
SWAP_PROBABILITY = 0.5
POSITIONS = ("left", "right", "top", "bottom")

# ----- masking -------------------------------------------------------------------
MASKABLE_ARTICLES = ("the", "a", "an")
# Request openers that can be dropped without losing the edit (sentence start only).
MASKABLE_OPENERS = ("please", "help me", "i need you to", "can you")
# Phrases that only name the picture; "of the image" is excluded (it marks positions).
MASKABLE_IMAGE_PHRASES = (
    "in the image", "in this image", "from the image", "from this image",
    "to the image", "to this image", "in the picture", "in this picture",
    "from the picture", "from this picture", "to the picture", "from this photo",
    "in the photo", "in this photo",
)
MIN_WORDS_AFTER_MASKING = 3
# "Make this a market" / "Give it a makeover": the article after an object pronoun
# carries meaning ("this market" is a different request), so it is kept.
OBJECT_PRONOUNS = ("this", "that", "it", "him", "her", "them", "me", "us")
# "the skies found in this image": dropping the phrase leaves "found" dangling.
PARTICIPLES_BEFORE_IMAGE_PHRASE = ("found", "visible", "shown", "seen", "present", "depicted")

# ----- injection -----------------------------------------------------------------
INJECTION_PREFIXES = (
    "You are a professional photo editor. Apply the following edit:",
    "Editing request:",
    "Task for this photo:",
    "Please perform this image edit:",
    "Instruction for the editor:",
    "Edit the photo as described:",
    "Here is the edit to apply:",
    "Image editing task:",
    "The requested change is as follows:",
    "Carry out this edit on the photo:",
)
INJECTION_SUFFIXES = (
    "Follow the request exactly.",
    "Only perform the described edit.",
    "Apply the edit as described.",
    "That is the requested edit.",
    "Carry out the instruction above.",
    "Perform this edit on the given photo.",
    "This is the only change requested.",
    "Do the edit described above.",
)

# ----- spatial -------------------------------------------------------------------
SPATIAL_TEMPLATES = (
    "In the {position} image, {instruction_lower}",
    "{instruction} (edit only the {position} image)",
    "Only edit the {position} image: {instruction}",
)

# ----- flip ----------------------------------------------------------------------
TEXT_PATTERN = re.compile(
    r"[\"'“”‘’「」]|\d|[^\x00-\x7f]"
    r"|\b(word|words|text|texts|letter|letters|phrase|caption|title|font|number|numbers"
    r"|digit|digits|character|characters|logo|sign|signs|label|writing|written)\b",
    re.IGNORECASE,
)
DIRECTION_PATTERN = re.compile(
    r"\b(clockwise|counter-?clockwise|anti-?clockwise|mirror|mirrored|flip|flipped|reverse)\b",
    re.IGNORECASE,
)
LEFT_RIGHT_PATTERN = re.compile(r"\b(left|right)(most|ward|wards)?\b", re.IGNORECASE)

QUOTED_PATTERN = re.compile(
    r"\"[^\"]*\"|“[^”]*”|(?<![A-Za-z])‘.*?’(?![A-Za-z])|(?<![A-Za-z])'[^']*'(?![A-Za-z])"
)


def stable_random(*parts: Any) -> random.Random:
    """Random generator seeded by a hash of the parts (same across processes)."""
    digest = hashlib.sha256("\x1f".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def tidy(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    text = re.sub(r"^[,;:]\s*", "", text)
    text = re.sub(r",\s*([.?!])", r"\1", text)
    text = re.sub(r"\.(\s*\.)+", ".", text)
    return text


def capitalize_first(text: str) -> str:
    for index, char in enumerate(text):
        if char.isalpha():
            return text[:index] + char.upper() + text[index + 1:]
        if char not in " \"'“‘(":
            return text
    return text


def lower_first(text: str) -> str:
    """Lowercase the first word unless it is "I", an acronym or starts with a quote."""
    stripped = text.lstrip()
    if not stripped or not stripped[0].isalpha():
        return stripped
    first = re.match(r"[A-Za-z']+", stripped).group(0)
    if first in ("I", "I'm", "I'd", "I've", "I'll") or (len(first) > 1 and first.isupper()):
        return stripped
    return stripped[0].lower() + stripped[1:]


def split_protected(text: str) -> List[Tuple[str, bool]]:
    """Split into (piece, is_quoted) so quoted text is never changed."""
    pieces, position = [], 0
    for match in QUOTED_PATTERN.finditer(text):
        if match.start() > position:
            pieces.append((text[position:match.start()], False))
        pieces.append((match.group(0), True))
        position = match.end()
    if position < len(text):
        pieces.append((text[position:], False))
    return pieces


def masking_candidates(instruction: str) -> List[Tuple[int, int, str]]:
    """(start, end, kind) of every maskable span outside quoted text."""
    spans: List[Tuple[int, int, str]] = []
    offset = 0
    for piece, quoted in split_protected(instruction):
        if not quoted:
            for phrase in MASKABLE_IMAGE_PHRASES:
                for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", piece, re.IGNORECASE):
                    before = re.findall(r"[A-Za-z']+", instruction[: offset + match.start()])
                    if before and before[-1].lower() in PARTICIPLES_BEFORE_IMAGE_PHRASE:
                        continue
                    spans.append((offset + match.start(), offset + match.end(), "image_phrase"))
            for match in re.finditer(r"\b(the|a|an)\b", piece, re.IGNORECASE):
                start = offset + match.start()
                word = match.group(0)
                sentence_start = not instruction[:start].strip() or instruction[:start].rstrip()[-1] in ".!?"
                # A capitalized article inside a sentence may belong to a name ("The Beatles").
                if word[0].isupper() and not sentence_start:
                    continue
                before = re.findall(r"[A-Za-z']+", instruction[:start])
                if before and before[-1].lower() in OBJECT_PRONOUNS:
                    continue
                spans.append((start, offset + match.end(), "article"))
        offset += len(piece)
    head = re.match(r"\s*(" + "|".join(re.escape(o) for o in MASKABLE_OPENERS) + r")\b,?", instruction, re.IGNORECASE)
    if head:
        spans.append((head.start(1), head.end(), "opener"))
    # Drop article spans that lie inside a chosen phrase span.
    phrases = [s for s in spans if s[2] != "article"]
    spans = phrases + [
        s for s in spans if s[2] == "article"
        and not any(p[0] <= s[0] and s[1] <= p[1] for p in phrases)
    ]
    return sorted(spans)


def apply_masking(instruction: str, spans: Sequence[Tuple[int, int, str]]) -> str:
    out, position = [], 0
    for start, end, _ in sorted(spans):
        out.append(instruction[position:start])
        position = end
    out.append(instruction[position:])
    text = tidy("".join(out))
    # "Can you remove X? Only keep Y." without "can you" is a command: "?" becomes ".".
    if any(kind == "opener" and instruction[start:end].strip(" ,").lower() == "can you"
           for start, end, kind in spans):
        text = text.replace("?", ".", 1)
    # A masked article that started a later sentence leaves a lowercase sentence start.
    text = re.sub(r"([.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
    # Keep the original sentence case: capitalize only if the instruction started uppercase.
    first_alpha = next((char for char in instruction if char.isalpha()), "")
    return capitalize_first(text) if first_alpha.isupper() else text


def masking_variants(instruction: str, count: int, seed: Any) -> List[str]:
    spans = masking_candidates(instruction)
    if not spans:
        return []
    variants: List[str] = []
    rng = stable_random("masking", seed, instruction)
    for _ in range(count * 20):
        chosen = [span for span in spans if rng.random() < 0.5] or [rng.choice(spans)]
        text = apply_masking(instruction, chosen)
        if len(text.split()) < MIN_WORDS_AFTER_MASKING or text == tidy(instruction):
            continue
        if text not in variants:
            variants.append(text)
        if len(variants) == count:
            break
    return variants


def with_end_punctuation(instruction: str) -> str:
    text = instruction.strip()
    if text[-1:] in ".!?":
        return text
    # 'PARKWAY' or "no rear entry": a closing quote ends the sentence only if a
    # sentence mark stands right before it ("... are?").
    if text[-1:] in "\"”’'" and text[-2:-1] in ".!?":
        return text
    return text + "."


def injection_variants(instruction: str, count: int, seed: Any) -> List[str]:
    rng = stable_random("injection", seed, instruction)
    base = instruction.strip()
    variants: List[str] = []
    forms = ["prefix", "suffix", "both"]
    for index in range(count * 20):
        form = forms[index % 3] if index < 3 else rng.choice(forms)
        prefix = rng.choice(INJECTION_PREFIXES)
        suffix = rng.choice(INJECTION_SUFFIXES)
        if form == "prefix":
            text = f"{prefix} {base}"
        elif form == "suffix":
            text = f"{with_end_punctuation(base)} {suffix}"
        else:
            text = f"{prefix} {with_end_punctuation(base)} {suffix}"
        text = tidy(text)
        if text not in variants:
            variants.append(text)
        if len(variants) == count:
            break
    return variants


def spatial_variants(instruction: str) -> Dict[str, List[str]]:
    base = instruction.strip()
    return {
        position: [
            tidy(template.format(position=position, instruction=base, instruction_lower=lower_first(base)))
            for template in SPATIAL_TEMPLATES
        ]
        for position in POSITIONS
    }


def flip_exclusion(instruction: str) -> Optional[str]:
    if TEXT_PATTERN.search(instruction):
        return "text_number_or_quote"
    if DIRECTION_PATTERN.search(instruction):
        return "rotation_or_mirror"
    return None


def flip_instruction(instruction: str) -> str:
    def swap(match: re.Match) -> str:
        word, tail = match.group(1), match.group(2) or ""
        other = "right" if word.lower() == "left" else "left"
        if word[0].isupper():
            other = other.capitalize()
        if word.isupper():
            other = other.upper()
        return other + tail
    return LEFT_RIGHT_PATTERN.sub(swap, instruction)


# ----- training time ---------------------------------------------------------------
# ----- automatic checks (also run online; a failing variant falls back to the original) --
def content_words(text: str) -> Counter:
    """Words that masking must never remove (everything except maskable words)."""
    removable = set(MASKABLE_ARTICLES)
    for phrase in MASKABLE_IMAGE_PHRASES + MASKABLE_OPENERS:
        removable.update(phrase.split())
    return Counter(w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in removable)


def check_masking(original: str, variant: str) -> List[str]:
    problems = []
    if content_words(original) != content_words(variant):
        problems.append("content words changed")
    for quoted in QUOTED_PATTERN.findall(original):
        if quoted not in variant:
            problems.append(f"quoted text lost: {quoted}")
    if len(variant.split()) < MIN_WORDS_AFTER_MASKING:
        problems.append("too short")
    if variant == tidy(original):
        problems.append("unchanged")
    # Patterns found by reading the 1008 selected instructions (2026-11-04).
    for pronoun in OBJECT_PRONOUNS:
        for article in ("a", "an", "the"):
            if re.search(rf"\b{pronoun} {article}\b", original, re.I) and not re.search(
                rf"\b{pronoun} {article}\b", variant, re.I
            ):
                problems.append(f"article after '{pronoun}' removed")
    for word in PARTICIPLES_BEFORE_IMAGE_PHRASE:
        if re.search(rf"\b{word}\s*[.,!?]?$", variant, re.I) and not re.search(rf"\b{word}\s*[.,!?]?$", original, re.I):
            problems.append(f"dangling '{word}'")
    if re.search(r"[.!?]\s+[a-z]", variant) and not re.search(r"[.!?]\s+[a-z]", original):
        problems.append("lowercase sentence start")
    if ".." in variant and ".." not in original:
        problems.append("double period")
    return problems


def check_contains(original: str, variant: str) -> List[str]:
    core = tidy(original.strip())
    lowered = tidy(lower_first(original.strip()))
    if not any(part and part in variant for part in (core, lowered, core.rstrip("."), lowered.rstrip("."))):
        return ["original instruction not contained"]
    if re.search(r"['\"”’]\s+[A-Z]", variant) and not re.search(r"['\"”’]\s+[A-Z]", original) \
            and not re.search(r"[.!?]['\"”’]\s+[A-Z]", variant):
        return ["missing period after closing quote"]
    return []


def online_masking(instruction: str, rng: random.Random) -> Optional[str]:
    """One random subset of the maskable spans (each kept with probability 0.5)."""
    spans = masking_candidates(instruction)
    if not spans:
        return None
    chosen = [span for span in spans if rng.random() < 0.5] or [rng.choice(spans)]
    return apply_masking(instruction, chosen)


def online_injection(instruction: str, rng: random.Random) -> str:
    base = instruction.strip()
    form = rng.choice(("prefix", "suffix", "both"))
    prefix, suffix = rng.choice(INJECTION_PREFIXES), rng.choice(INJECTION_SUFFIXES)
    if form == "prefix":
        return tidy(f"{prefix} {base}")
    if form == "suffix":
        return tidy(f"{with_end_punctuation(base)} {suffix}")
    return tidy(f"{prefix} {with_end_punctuation(base)} {suffix}")


def choose_augmentation(
    scenario: Mapping[str, Any],
    sample_id: str,
    iteration: int,
    seed: int,
    distractor_count: int,
) -> Dict[str, Any]:
    """One augmentation of one sample in one iteration, generated online.

    The random generator is seeded by (seed, iteration, sample_id): every iteration
    draws a new scenario and a new instruction variant, and a rerun repeats them.
    A variant that fails the automatic checks falls back to the original instruction
    (``fallback`` names the failed checks).
    """
    rng = stable_random("choice", seed, iteration, sample_id)
    original = scenario["instruction"]
    name = rng.choice(list(scenario["eligible"]))
    choice: Dict[str, Any] = {"scenario": name, "swap": rng.random() < SWAP_PROBABILITY}
    problems: List[str] = []
    if name == "masking":
        variant = online_masking(original, rng)
        problems = ["no maskable span"] if variant is None else check_masking(original, variant)
    elif name == "injection":
        variant = online_injection(original, rng)
        problems = check_contains(original, variant)
    elif name == "spatial":
        position = rng.choice(POSITIONS)
        template = rng.choice(SPATIAL_TEMPLATES)
        variant = tidy(template.format(position=position, instruction=original.strip(),
                                       instruction_lower=lower_first(original.strip())))
        problems = check_contains(original, variant) + ([] if position in variant else ["position missing"])
        choice["position"] = position
        choice["distractor_index"] = rng.randrange(distractor_count)
    elif name == "flip":
        variant = scenario["flip"]
        if variant is None:
            raise ValueError(f"{sample_id}: flip is not eligible")
    else:
        raise ValueError(f"Unknown scenario {name}")
    if problems:
        choice["fallback"] = problems
        variant = original
    choice["instruction"] = variant
    return choice


def compose_spatial(source: Any, distractor: Any, position: str) -> Any:
    """Source image next to the distractor; the composite keeps the source pixel count."""
    from PIL import Image

    source = source.convert("RGB")
    distractor = distractor.convert("RGB")
    width, height = source.size
    if position in ("left", "right"):
        scaled = distractor.resize(
            (max(1, round(distractor.width * height / distractor.height)), height), Image.LANCZOS
        )
        canvas = Image.new("RGB", (width + scaled.width, height))
        first, second = (source, scaled) if position == "left" else (scaled, source)
        canvas.paste(first, (0, 0))
        canvas.paste(second, (first.width, 0))
    elif position in ("top", "bottom"):
        scaled = distractor.resize(
            (width, max(1, round(distractor.height * width / distractor.width))), Image.LANCZOS
        )
        canvas = Image.new("RGB", (width, height + scaled.height))
        first, second = (source, scaled) if position == "top" else (scaled, source)
        canvas.paste(first, (0, 0))
        canvas.paste(second, (0, first.height))
    else:
        raise ValueError(f"Unknown position {position}")
    factor = (width * height / (canvas.width * canvas.height)) ** 0.5
    return canvas.resize(
        (max(1, round(canvas.width * factor)), max(1, round(canvas.height * factor))), Image.LANCZOS
    )


def augmented_images(
    choice: Mapping[str, Any],
    source_path: str,
    image_a_path: str,
    image_b_path: str,
    distractor_path: Optional[str],
    work_dir: Path,
    tag: str,
) -> Tuple[str, str, str]:
    """Paths of (source, A, B) for the Student; new files only when images change."""
    from PIL import Image, ImageOps

    work_dir.mkdir(parents=True, exist_ok=True)
    if choice["scenario"] == "spatial":
        if distractor_path is None:
            raise ValueError("The spatial scenario needs a distractor image.")
        with Image.open(source_path) as source, Image.open(distractor_path) as distractor:
            composite = compose_spatial(source, distractor, choice["position"])
        path = work_dir / f"{tag}_source_{choice['position']}.png"
        composite.save(path)
        return str(path), image_a_path, image_b_path
    if choice["scenario"] == "flip":
        paths = []
        for name, original in (("source", source_path), ("a", image_a_path), ("b", image_b_path)):
            with Image.open(original) as image:
                mirrored = ImageOps.mirror(image.convert("RGB"))
            path = work_dir / f"{tag}_{name}_flip.png"
            mirrored.save(path)
            paths.append(str(path))
        return paths[0], paths[1], paths[2]
    return source_path, image_a_path, image_b_path


# =====================================================================================
# Combined strong augmentation (confirmed 2026-11-05)
# =====================================================================================
# Every (sample, iteration) draws, with its own seeded generator (always text + image):
#   1. one information-loss text operation (drop_action / mask_content) followed by one
#      noise text operation (typo / distractor_instruction); a sample with no loss
#      operation gets both noise operations:
#        drop_action     drop the request words before the first article / determiner /
#                        quote ("Make this look like a painting by Cezanne" ->
#                        "A painting by Cezanne")
#        mask_content    replace 30-50% of the content words by "___" (first word and
#                        quoted text kept, at least one content word kept)
#        typo            misspell 30-50% of the words of 4+ letters (quoted text and
#                        position words kept)
#        distractor_instruction  add the instruction of another sample, marked as not to
#                        be applied (with the spatial layout: the instruction of the image
#                        put next to the source image)
#   2. one layout among the eligible ones, never none: spatial (one distractor image next to
#      the source image, composite scaled back to the source pixel count) / flip
#   3. with probability DEGRADATION_PROBABILITY, one or two image degradations (JPEG, blur,
#      downscale, noise) with the same parameters on the source, A and B images (mild
#      levels for text edits)
#   4. A <-> B swap with SWAP_PROBABILITY
# Order: text operation -> flip words -> spatial wording -> distractor instruction;
# images: flip -> spatial composite -> degradation.

TEXT_OPERATIONS = ("drop_action", "mask_content", "typo", "distractor_instruction")
# Combined per sample (user, 2026-11-05): one of each group; the layout is never "none".
LOSS_TEXT_OPERATIONS = ("drop_action", "mask_content")
NOISE_TEXT_OPERATIONS = ("typo", "distractor_instruction")
LAYOUTS = ("none", "spatial", "flip")
# Brightness / contrast / grayscale are not used (2026-11-05): they change what aesthetic
# and realism judge, so the Teacher label of the original would no longer hold.
DEGRADATIONS = ("jpeg", "blur", "downscale", "noise")
DEGRADATION_PROBABILITY = 0.7
HANDWRITTEN_STYLES = ("indirect", "convoluted", "terse")
MASK_TOKEN = "___"
TEXT_FRACTION = (0.3, 0.5)

DROP_ACTION_START = re.compile(
    r"\b(a|an|the|all|some|every|each|his|her|their|its|my|your|our)\b|[\"“]|(?<![A-Za-z])['‘]", re.IGNORECASE
)
STOPWORDS = frozenset(
    "a an the and or but of to in on at by for with from into onto over under as is are was "
    "were be been being it its this that these those there here than then so too very can "
    "could would should will shall may might must do does did done have has had not no yes "
    "i me my we our you your he him his she her they them their what which who whom whose "
    "all some any each every one two make made look looks like such just also only more "
    "most less least".split()
)
POSITION_WORDS = frozenset(
    "left right top bottom upper lower above below middle center centre front back behind "
    "corner side leftmost rightmost".split()
)
DISTRACTOR_TEMPLATES = (
    "Ignore this unrelated request: \"{other}\". The actual edit: {instruction}",
    "{instruction} (Not this other request: \"{other}\")",
    "Unrelated note, do not apply: \"{other}\". Requested edit: {instruction}",
)
SPATIAL_DISTRACTOR_TEMPLATE = "{instruction} The {side} image is unrelated; ignore its request: \"{other}\"."
# drop_action only drops a request head: after its first word, the dropped words must all
# be function words or these ("I want to see this as", "Make this look like", "Put him on");
# "add kids playing in front of the two houses" keeps its content and is not eligible.
REQUEST_HEAD_WORDS = frozenset(
    "want wants wanted see seem seems appear appears need needs please help love like let "
    "let's i'd i'm us him them it me look looks".split()
)
# Edit verbs carry the meaning of the edit ("Remove the cat" -> "The cat" could read as
# adding it): a request head containing one of them is never dropped (user, 2026-11-05).
EDIT_VERBS = frozenset(
    "add adds adding remove removes removing delete deletes deleting erase replace replaces "
    "replacing change changes changing turn turns turning put puts putting insert inserts "
    "extract extracts convert converts transform transforms transfer transfers alter alters "
    "place places swap swaps give gives have has write writes bring brings get rid fill fills "
    "draw paint create generate include attach cut crop move take set color colour recolor "
    "rotate flip hide clear eliminate switch restyle redo reimagine render apply increase "
    "decrease reduce lower raise enlarge shrink zoom".split()
)
# The first dropped word must open a request ("Make this look like", "I want to see",
# "Let's see", "Show me", "Can you make"); any other first word is the edit itself.
REQUEST_HEAD_FIRST_WORDS = frozenset(
    "make makes let let's show i i'd i'm we please can could would imagine visualize see".split()
)
OPPOSITE_SIDE = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}

# Degradation levels: (normal, mild); mild is for instructions about text in the image.
DEGRADATION_LEVELS = {
    "jpeg": {"normal": (15, 30), "mild": (30, 45)},            # JPEG quality
    "blur": {"normal": (2.0, 3.0), "mild": (1.0, 1.5)},        # Gaussian radius (pixels)
    "downscale": {"normal": (0.3, 0.5), "mild": (0.5, 0.65)},  # side kept before upscaling back
    "noise": {"normal": (15.0, 25.0), "mild": (6.0, 12.0)},    # Gaussian sigma (0-255 scale)
}
TEXT_EDIT_PATTERN = re.compile(
    r"[\"“”‘’「」]|\b(word|words|text|texts|letter|letters|phrase|caption|title|font|number"
    r"|numbers|digit|digits|character|characters|logo|sign|signs|label|writing|written|says|say)\b",
    re.IGNORECASE,
)
QUALITY_PATTERN = re.compile(
    r"\b(quality|noise|noisy|grain|grainy|restore|restored|restoration|repair|jpeg|artifacts?"
    r"|compress\w*|denoise|enhance\w*|upscale\w*|hd|4k|resolution|pixel\w*|old photo|vintage)\b",
    re.IGNORECASE,
)
SHARPNESS_PATTERN = re.compile(
    r"\b(blur\w*|sharp\w*|focus\w*|bokeh|depth of field|clear|clearer|crisp\w*|detail\w*)\b",
    re.IGNORECASE,
)


def degradation_eligibility(instruction: str) -> Tuple[List[str], bool, Dict[str, str]]:
    """(eligible degradations, mild, excluded -> reason) of one instruction."""
    excluded: Dict[str, str] = {}
    if QUALITY_PATTERN.search(instruction):
        for name in ("jpeg", "noise", "downscale"):
            excluded[name] = "quality"
    if SHARPNESS_PATTERN.search(instruction):
        for name in ("blur", "downscale"):
            excluded.setdefault(name, "sharpness")
    eligible = [name for name in DEGRADATIONS if name not in excluded]
    return eligible, bool(TEXT_EDIT_PATTERN.search(instruction)), excluded


def word_spans(text: str) -> List[Tuple[int, int, str]]:
    """(start, end, word) of the alphabetic words outside quoted text."""
    spans, offset = [], 0
    for piece, quoted in split_protected(text):
        if not quoted:
            for match in re.finditer(r"[A-Za-z][A-Za-z'-]*", piece):
                spans.append((offset + match.start(), offset + match.end(), match.group(0)))
        offset += len(piece)
    return spans


def mask_candidates(instruction: str) -> List[Tuple[int, int, str]]:
    spans = word_spans(instruction)
    first = spans[0][0] if spans else -1
    return [s for s in spans if s[0] != first and len(s[2]) >= 3 and s[2].lower() not in STOPWORDS]


def typo_candidates(instruction: str) -> List[Tuple[int, int, str]]:
    return [s for s in word_spans(instruction)
            if len(s[2]) >= 4 and s[2].isalpha() and s[2].lower() not in POSITION_WORDS]


def replace_spans(text: str, replacements: Sequence[Tuple[int, int, str]]) -> str:
    out, position = [], 0
    for start, end, new in sorted(replacements):
        out.append(text[position:start])
        out.append(new)
        position = end
    out.append(text[position:])
    return "".join(out)


def drop_action(instruction: str) -> Optional[str]:
    """Drop the request words of the first sentence before its first article / quote."""
    text = instruction.strip()
    match = DROP_ACTION_START.search(text)
    if match is None or not re.search(r"[A-Za-z]", text[: match.start()]):
        return None
    if re.search(r"[.!?]", text[: match.start()]):
        return None  # the first article is not in the first sentence
    head = re.findall(r"[A-Za-z][A-Za-z']*", text[: match.start()])
    if not head or head[0].lower() not in REQUEST_HEAD_FIRST_WORDS:
        return None  # the first word is not a request opener: it is the edit itself
    if any(w.lower() in EDIT_VERBS for w in head):
        return None  # the request head holds the edit verb: dropping it changes the meaning
    if any(w.lower() in POSITION_WORDS for w in head) or any(
            w.lower() not in STOPWORDS and w.lower() not in REQUEST_HEAD_WORDS for w in head[1:]):
        return None  # content words before the article: dropping them loses the edit
    rest = text[match.start():]
    if len(re.findall(r"[A-Za-z0-9]+", rest)) < 2:
        return None
    return capitalize_first(rest) if text[:1].isupper() else rest


def mask_content(instruction: str, rng: random.Random) -> Optional[str]:
    candidates = mask_candidates(instruction)
    if len(candidates) < 2:
        return None
    count = max(1, min(len(candidates) - 1, round(len(candidates) * rng.uniform(*TEXT_FRACTION))))
    chosen = rng.sample(candidates, count)
    return replace_spans(instruction, [(s, e, MASK_TOKEN) for s, e, _ in chosen])


def misspell(word: str, rng: random.Random) -> str:
    inner = list(range(1, len(word) - 1))
    for _ in range(10):
        kind = rng.choice(("swap", "delete", "double", "replace"))
        letters = list(word)
        index = rng.choice(inner)
        if kind == "swap" and index + 1 < len(word) - 1:
            letters[index], letters[index + 1] = letters[index + 1], letters[index]
        elif kind == "delete":
            del letters[index]
        elif kind == "double":
            letters.insert(index, letters[index])
        else:
            letters[index] = rng.choice([c for c in "abcdefghijklmnopqrstuvwxyz" if c != letters[index].lower()])
        new = "".join(letters)
        if new.lower() != word.lower():
            return new
    return word[0] + word[2] + word[1] + word[3:]


def typo(instruction: str, rng: random.Random) -> Optional[str]:
    candidates = typo_candidates(instruction)
    if not candidates:
        return None
    count = max(1, round(len(candidates) * rng.uniform(*TEXT_FRACTION)))
    chosen = rng.sample(candidates, count)
    return replace_spans(instruction, [(s, e, misspell(w, rng)) for s, e, w in chosen])


def quoted_other(other: str) -> str:
    return tidy(other.strip().replace('"', "'").replace("“", "'").replace("”", "'")).rstrip(".")


def check_drop_action(original: str, variant: str) -> List[str]:
    problems = []
    if not tidy(original.strip()).lower().endswith(variant.strip().lower()):
        problems.append("not a suffix of the original")
    if len(re.findall(r"[A-Za-z0-9]+", variant)) < 2:
        problems.append("too short")
    for quoted in QUOTED_PATTERN.findall(original):
        if quoted not in variant and original.find(quoted) >= len(original) - len(variant):
            problems.append(f"quoted text lost: {quoted}")
    return problems


def check_mask_content(original: str, variant: str) -> List[str]:
    problems = []
    if MASK_TOKEN not in variant:
        problems.append("nothing masked")
    for quoted in QUOTED_PATTERN.findall(original):
        if quoted not in variant:
            problems.append(f"quoted text lost: {quoted}")
    if not mask_candidates(variant):
        problems.append("no content word left")
    if word_spans(original)[:1] and word_spans(variant)[:1] and word_spans(original)[0][2] != word_spans(variant)[0][2]:
        problems.append("first word changed")
    return problems


def check_typo(original: str, variant: str) -> List[str]:
    problems = []
    if variant == original:
        problems.append("unchanged")
    for quoted in QUOTED_PATTERN.findall(original):
        if quoted not in variant:
            problems.append(f"quoted text lost: {quoted}")
    if len(variant.split()) != len(original.split()):
        problems.append("word count changed")
    if [w.lower() for _, _, w in word_spans(original) if w.lower() in POSITION_WORDS] != \
            [w.lower() for _, _, w in word_spans(variant) if w.lower() in POSITION_WORDS]:
        problems.append("position word changed")
    return problems


TEXT_CHECKS = {"drop_action": check_drop_action, "mask_content": check_mask_content, "typo": check_typo}


def text_eligibility(instruction: str) -> List[str]:
    eligible = []
    if drop_action(instruction) is not None and not check_drop_action(instruction, drop_action(instruction)):
        eligible.append("drop_action")
    if len(mask_candidates(instruction)) >= 2:
        eligible.append("mask_content")
    if typo_candidates(instruction):
        eligible.append("typo")
    eligible.append("distractor_instruction")
    return eligible


def apply_text_operation(name: str, instruction: str, rng: random.Random) -> Tuple[str, List[str]]:
    """(variant, problems); the distractor instruction is added later (after the layout)."""
    if name == "distractor_instruction":
        return instruction, []
    variant = {"drop_action": lambda: drop_action(instruction),
               "mask_content": lambda: mask_content(instruction, rng),
               "typo": lambda: typo(instruction, rng)}[name]()
    if variant is None:
        return instruction, [f"{name} not applicable"]
    problems = TEXT_CHECKS[name](instruction, variant)
    return (instruction, problems) if problems else (tidy(variant), [])


def degradation_parameters(names: Sequence[str], mild: bool, rng: random.Random) -> List[Dict[str, Any]]:
    params = []
    for name in names:
        low, high = DEGRADATION_LEVELS[name]["mild" if mild else "normal"]
        if name == "jpeg":
            params.append({"name": name, "quality": rng.randint(int(low), int(high))})
        else:
            key = {"blur": "radius", "downscale": "scale", "noise": "sigma"}[name]
            params.append({"name": name, key: round(rng.uniform(low, high), 3)})
    return params


def choose_combined_augmentation(
    scenario: Mapping[str, Any],
    sample_id: str,
    iteration: int,
    seed: int,
    distractors: Sequence[Mapping[str, Any]],
    handwritten: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """One combined augmentation of one sample in one iteration (seeded, online).

    handwritten (instruction -> {indirect, convoluted, terse}): when given, the
    information-loss text operation is replaced by one hand-written rewrite (same
    meaning), drawn at the same random-number position; the noise operation, layout,
    degradation and swap are unchanged.
    """
    rng = stable_random("combined", seed, iteration, sample_id)
    original = scenario["instruction"].strip()
    # Text: one information-loss operation + one noise operation, always together.
    loss_pool = [n for n in LOSS_TEXT_OPERATIONS if n in scenario["text_eligible"]]
    noise_pool = [n for n in NOISE_TEXT_OPERATIONS if n in scenario["text_eligible"]]
    text_names = ([rng.choice(loss_pool)] if loss_pool else []) + (
        [rng.choice(noise_pool)] if loss_pool else list(noise_pool))
    if handwritten is None:
        return _finish_combined(scenario, rng, original, original, [], text_names, distractors)
    # Hand-written rewrite replaces the information-loss operation. The main generator
    # has drawn exactly as above, so layout / degradation / swap / noise operation are
    # the same as without hand-written rewrites; the style uses its own generator.
    style = stable_random("handwritten", seed, iteration, sample_id).choice(HANDWRITTEN_STYLES)
    noise_names = [n for n in text_names if n in NOISE_TEXT_OPERATIONS]
    return _finish_combined(scenario, rng, original, handwritten[original][style].strip(),
                            [f"handwritten_{style}"], noise_names, distractors,
                            shadow_text_names=text_names,
                            text_rng=stable_random("handwritten_text", seed, iteration, sample_id))


def _finish_combined(
    scenario: Mapping[str, Any],
    rng: random.Random,
    original: str,
    start: str,
    labels: List[str],
    text_names: List[str],
    distractors: Sequence[Mapping[str, Any]],
    shadow_text_names: Sequence[str] = (),
    text_rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Layout, degradation, swap and text operations of choose_combined_augmentation.

    shadow_text_names: text operations run on the original instruction with the main
    generator and discarded, so that the later draws (position, templates, distractors)
    are the same as without hand-written rewrites; the kept text operations then use
    text_rng.
    """
    # Image: the layout always changes (spatial or flip), plus degradation.
    layout = rng.choice([n for n in scenario["layout_eligible"] if n != "none"])
    degrade = bool(scenario["degradation_eligible"]) and rng.random() < DEGRADATION_PROBABILITY
    names = []
    if degrade:
        pool = list(scenario["degradation_eligible"])
        names = rng.sample(pool, min(len(pool), rng.choice((1, 2))))
    choice: Dict[str, Any] = {
        "scenario": "combined",
        "swap": rng.random() < SWAP_PROBABILITY,
        "text_operation": "+".join(labels + text_names),
        "layout": layout,
        "degradation": degradation_parameters(names, bool(scenario["degradation_mild"]), rng),
    }
    shadow = original
    for text_name in shadow_text_names:
        shadow, _ = apply_text_operation(text_name, shadow, rng)
    variant, problems = start, []
    for text_name in text_names:
        variant, found = apply_text_operation(text_name, variant, text_rng or rng)
        problems += found
    if layout == "flip":
        variant = flip_instruction(variant)
    if layout == "spatial":
        position = rng.choice(POSITIONS)
        template = rng.choice(SPATIAL_TEMPLATES)
        variant = tidy(template.format(position=position, instruction=variant,
                                       instruction_lower=lower_first(variant)))
        choice["position"] = position
        choice["distractor_index"] = rng.randrange(len(distractors))
    if "distractor_instruction" in text_names:
        if layout == "spatial":
            other_index = choice["distractor_index"]
            other = quoted_other(distractors[other_index]["instruction"])
            variant = tidy(SPATIAL_DISTRACTOR_TEMPLATE.format(
                instruction=with_end_punctuation(variant), side=OPPOSITE_SIDE[choice["position"]], other=other))
        else:
            other_index = rng.randrange(len(distractors))
            other = quoted_other(distractors[other_index]["instruction"])
            template = rng.choice(DISTRACTOR_TEMPLATES)
            variant = tidy(template.format(other=other, instruction=variant))
        choice["instruction_distractor_index"] = other_index
        if other.lower() == quoted_other(original).lower():
            problems.append("distractor instruction equals the instruction")
            variant = original
    if problems:
        choice["fallback"] = problems
    choice["instruction"] = variant
    return choice


def degrade_image(image: Any, params: Sequence[Mapping[str, Any]], rng_seed: str) -> Any:
    """Apply the degradations in order; the size never changes."""
    import io

    import numpy as np
    from PIL import Image, ImageFilter

    image = image.convert("RGB")
    size = image.size
    for step in params:
        name = step["name"]
        if name == "jpeg":
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG", quality=int(step["quality"]))
            buffer.seek(0)
            image = Image.open(buffer).convert("RGB")
        elif name == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=float(step["radius"])))
        elif name == "downscale":
            small = (max(1, round(size[0] * step["scale"])), max(1, round(size[1] * step["scale"])))
            image = image.resize(small, Image.BILINEAR).resize(size, Image.BILINEAR)
        elif name == "noise":
            generator = np.random.default_rng(int(hashlib.sha256(rng_seed.encode()).hexdigest()[:8], 16))
            array = np.asarray(image, dtype=np.float32)
            array = array + generator.normal(0.0, float(step["sigma"]), array.shape)
            image = Image.fromarray(np.clip(array, 0, 255).astype(np.uint8))
        else:
            raise ValueError(f"Unknown degradation {name}")
    return image


def combined_augmented_images(
    choice: Mapping[str, Any],
    source_path: str,
    image_a_path: str,
    image_b_path: str,
    distractor_path: Optional[str],
    work_dir: Path,
    tag: str,
) -> Tuple[str, str, str]:
    """Paths of (source, A, B) for the Student: flip -> spatial composite -> degradation."""
    from PIL import Image, ImageOps

    layout, params = choice["layout"], choice["degradation"]
    if layout == "none" and not params:
        return source_path, image_a_path, image_b_path
    work_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, original in (("source", source_path), ("a", image_a_path), ("b", image_b_path)):
        with Image.open(original) as opened:
            image = opened.convert("RGB")
        if layout == "flip":
            image = ImageOps.mirror(image)
        if layout == "spatial" and name == "source":
            if distractor_path is None:
                raise ValueError("The spatial layout needs a distractor image.")
            with Image.open(distractor_path) as distractor:
                image = compose_spatial(image, distractor, choice["position"])
        if params:
            # Same seed for the three images: same noise pattern where the sizes match.
            image = degrade_image(image, params, tag)
        path = work_dir / f"{tag}_{name}_combined.png"
        image.save(path)
        paths.append(str(path))
    return paths[0], paths[1], paths[2]


# =====================================================================================
# Part-wise augmentation (user, 2026-11-05): which part of the Student input makes the
# Student differ from the Teacher. All three use the combined scenarios file.
#   instruction_only   the text part of the hand-written combined augmentation (same
#                      rewrite style and noise operation draws); images unchanged
#   source_image_only  only the source image changes: spatial composite (0.5), small
#                      rotation (0.5), black rectangles (0.5), degradation (0.7), at least
#                      one; A and B unchanged; instruction only gets the position wording
#                      of the composite
#   extreme            90% of the instruction words dropped, one black rectangle over
#                      80-100% of the source; A and B unchanged
# Every mode keeps the A<->B swap of the combined augmentation (probability 0.5).
# =====================================================================================

#   blank              instruction replaced by " " and the whole source blacked out;
#                      A and B unchanged (user, 2026-11-05)
PART_MODES = ("instruction_only", "source_image_only", "extreme", "blank")
BLANK_INSTRUCTION = " "
SOURCE_OPERATION_PROBABILITY = 0.5      # spatial composite / rotation / black rectangles
ROTATION_DEGREES = (5.0, 15.0)          # absolute angle, sign drawn; corners filled black
MASK_AREA = (0.10, 0.30)                # total area of the black rectangles (fraction)
MASK_RECTANGLES = (1, 3)
EXTREME_WORDS_DROPPED = 0.9
EXTREME_BLACK_AREA = (0.80, 1.00)


def _swap_and_layout_of_combined(scenario, sample_id, iteration, seed, distractors):
    """Combined choice without hand-written rewrites: same swap / position / distractor
    draws as the hand-written combined runs (the main generator draws identically)."""
    return choose_combined_augmentation(scenario, sample_id, iteration, seed, distractors)


def _black_rectangle(rng: random.Random, area: float) -> Dict[str, float]:
    """Relative (x, y, width, height) of a rectangle of the given area fraction."""
    aspect = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
    width = min(1.0, math.sqrt(area * aspect))
    height = min(1.0, area / width)
    return {"x": round(rng.uniform(0.0, 1.0 - width), 4), "y": round(rng.uniform(0.0, 1.0 - height), 4),
            "width": round(width, 4), "height": round(height, 4)}


def choose_part_augmentation(
    mode: str,
    scenario: Mapping[str, Any],
    sample_id: str,
    iteration: int,
    seed: int,
    distractors: Sequence[Mapping[str, Any]],
    handwritten: Optional[Mapping[str, Mapping[str, str]]] = None,
) -> Dict[str, Any]:
    """One part-wise augmentation of one sample in one iteration (seeded, online)."""
    if mode not in PART_MODES:
        raise ValueError(f"Unknown part mode {mode}")
    original = scenario["instruction"].strip()
    combined = _swap_and_layout_of_combined(scenario, sample_id, iteration, seed, distractors)
    choice: Dict[str, Any] = {"scenario": mode, "swap": combined["swap"], "layout": "none",
                              "position": None, "degradation": [], "rotation": None, "mask": []}
    if mode == "instruction_only":
        if handwritten is None:
            raise ValueError("instruction_only needs the hand-written rewrites.")
        written = choose_combined_augmentation(scenario, sample_id, iteration, seed, distractors, handwritten)
        labels = written["text_operation"].split("+")
        style = labels[0][len("handwritten_"):]
        noise_names = [n for n in labels[1:] if n in NOISE_TEXT_OPERATIONS]
        text_rng = stable_random("handwritten_text", seed, iteration, sample_id)
        variant, problems = handwritten[original][style].strip(), []
        for name in noise_names:
            variant, found = apply_text_operation(name, variant, text_rng)
            problems += found
        if "distractor_instruction" in noise_names:
            own = stable_random("instruction_only", seed, iteration, sample_id)
            other_index = written.get("instruction_distractor_index")
            if other_index is None or written["layout"] == "spatial":
                other_index = own.randrange(len(distractors))
            other = quoted_other(distractors[other_index]["instruction"])
            variant = tidy(own.choice(DISTRACTOR_TEMPLATES).format(other=other, instruction=variant))
            choice["instruction_distractor_index"] = other_index
            if other.lower() == quoted_other(original).lower():
                problems.append("distractor instruction equals the instruction")
                variant = original
        choice["text_operation"] = written["text_operation"]
        choice["instruction"] = variant
        if problems:
            choice["fallback"] = problems
        return choice
    if mode == "blank":
        choice["text_operation"] = "blank_instruction"
        choice["instruction"] = BLANK_INSTRUCTION
        choice["mask"] = [{"x": 0.0, "y": 0.0, "width": 1.0, "height": 1.0}]
        choice["mask_area"] = 1.0
        return choice
    rng = stable_random(mode, seed, iteration, sample_id)
    if mode == "extreme":
        words = original.split()
        keep = max(1, round(len(words) * (1.0 - EXTREME_WORDS_DROPPED)))
        kept = sorted(rng.sample(range(len(words)), keep))
        choice["text_operation"] = f"drop_words_{EXTREME_WORDS_DROPPED:g}"
        choice["instruction"] = " ".join(words[i] for i in kept)
        area = rng.uniform(*EXTREME_BLACK_AREA)
        side = math.sqrt(area)  # same aspect as the image
        choice["mask"] = [{"x": round(rng.uniform(0.0, 1.0 - side), 4), "y": round(rng.uniform(0.0, 1.0 - side), 4),
                           "width": round(side, 4), "height": round(side, 4)}]
        choice["mask_area"] = round(area, 4)
        return choice
    # source_image_only
    spatial = rng.random() < SOURCE_OPERATION_PROBABILITY
    rotate = rng.random() < SOURCE_OPERATION_PROBABILITY
    mask = rng.random() < SOURCE_OPERATION_PROBABILITY
    degrade = bool(scenario["degradation_eligible"]) and rng.random() < DEGRADATION_PROBABILITY
    if not (spatial or rotate or mask or degrade):
        forced = rng.choice(("spatial", "rotation", "mask"))
        spatial, rotate, mask = forced == "spatial", forced == "rotation", forced == "mask"
    variant = original
    if rotate:
        choice["rotation"] = round(rng.uniform(*ROTATION_DEGREES) * rng.choice((-1, 1)), 2)
    if mask:
        count = rng.randint(*MASK_RECTANGLES)
        area = rng.uniform(*MASK_AREA)
        choice["mask"] = [_black_rectangle(rng, area / count) for _ in range(count)]
        choice["mask_area"] = round(area, 4)
    if degrade:
        pool = list(scenario["degradation_eligible"])
        names = rng.sample(pool, min(len(pool), rng.choice((1, 2))))
        choice["degradation"] = degradation_parameters(names, bool(scenario["degradation_mild"]), rng)
    if spatial:
        choice["layout"] = "spatial"
        if combined["layout"] == "spatial":
            choice["position"], choice["distractor_index"] = combined["position"], combined["distractor_index"]
        else:
            choice["position"], choice["distractor_index"] = rng.choice(POSITIONS), rng.randrange(len(distractors))
        template = rng.choice(SPATIAL_TEMPLATES)
        variant = tidy(template.format(position=choice["position"], instruction=variant,
                                       instruction_lower=lower_first(variant)))
    choice["text_operation"] = "none"
    choice["instruction"] = variant
    return choice


def _draw_black(image: Any, rectangles: Sequence[Mapping[str, float]]) -> Any:
    from PIL import ImageDraw

    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size
    for rect in rectangles:
        x0, y0 = round(rect["x"] * width), round(rect["y"] * height)
        x1, y1 = round((rect["x"] + rect["width"]) * width), round((rect["y"] + rect["height"]) * height)
        draw.rectangle([x0, y0, max(x0, x1 - 1), max(y0, y1 - 1)], fill=(0, 0, 0))
    return image


def part_augmented_images(
    choice: Mapping[str, Any],
    source_path: str,
    image_a_path: str,
    image_b_path: str,
    distractor_path: Optional[str],
    work_dir: Path,
    tag: str,
) -> Tuple[str, str, str]:
    """Paths of (source, A, B) for the Student: only the source changes, in the order
    rotation -> black rectangles -> spatial composite -> degradation."""
    from PIL import Image

    if choice["rotation"] is None and not choice["mask"] and choice["layout"] == "none" and not choice["degradation"]:
        return source_path, image_a_path, image_b_path
    work_dir.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as opened:
        image = opened.convert("RGB")
    if choice["rotation"] is not None:
        image = image.rotate(float(choice["rotation"]), resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    if choice["mask"]:
        image = _draw_black(image, choice["mask"])
    if choice["layout"] == "spatial":
        if distractor_path is None:
            raise ValueError("The spatial layout needs a distractor image.")
        with Image.open(distractor_path) as distractor:
            image = compose_spatial(image, distractor, choice["position"])
    if choice["degradation"]:
        image = degrade_image(image, choice["degradation"], tag)
    path = work_dir / f"{tag}_source_{choice['scenario']}.png"
    image.save(path)
    return str(path), image_a_path, image_b_path


# =====================================================================================
# Single-operation augmentation (user, 2026-11-05): every experiment applies exactly ONE
# operation to the Student input, so the effect of each operation on the Student-Teacher
# agreement is measured alone; the 4-5 strongest are combined later. No A<->B swap.
# Reuses the combined scenarios file (text / layout / degradation eligibility, flip
# exclusion, distractor images). A sample on which the operation is not applicable keeps
# the original input and is logged with applied = False (and the reason).
#   drop_action             drop the request words before the first article / quote
#   mask_content            20-40% of the content words -> "___" (first word, quoted text kept)
#   typo                    20-40% of the words of 4+ letters misspelled (quoted text and
#                           position words kept)
#   distractor_instruction  the instruction of another sample added, marked as not to apply
#   injection               a neutral prefix and / or suffix (no new requirement)
#   composite               the SAME unrelated image put at the SAME side (left / right /
#                           top / bottom, uniform) of the source, A and B, so the three
#                           images keep the same layout; each composite is scaled back to
#                           the pixel count of its own original; the instruction names the
#                           main image and says to ignore the other one (COMPOSITE_TEMPLATES)
#   flip                    mirror source, A and B; left <-> right in the instruction (not
#                           for text / number / rotation / mirror instructions)
#   blur / noise            the same Gaussian blur / noise on source, A and B (normal levels,
#                           mild levels for text edits; not for quality / sharpness edits)
#   rotation                source only, 5-15 degrees, sign drawn, corners black
#   black_rectangles        source only, 1-3 black rectangles, total area 10-20%
# Strength values drawn per sample are logged (choice["strength"]).
# =====================================================================================

SINGLE_OPERATIONS = (
    "drop_action", "mask_content", "typo", "distractor_instruction", "injection",
    "composite", "flip", "blur", "noise", "rotation", "black_rectangles",
)
SINGLE_TEXT_FRACTION = (0.2, 0.4)        # mask_content / typo (user, 2026-11-05)
SINGLE_MASK_AREA = (0.10, 0.20)          # black_rectangles total area (user, 2026-11-05)
# The main (edited) image is at {position}; the unrelated image at {other}.
COMPOSITE_TEMPLATES = (
    "{instruction} (Only the {position} image is the one to edit; the {other} image is unrelated, ignore it.)",
    "Focus on the {position} image and ignore the unrelated {other} image: {instruction_lower}",
    "In the {position} image, {instruction_lower} The {other} image is unrelated; ignore it.",
    "The {other} image is unrelated and should be ignored. For the {position} image: {instruction_lower}",
)


def _fraction_of(candidates: Sequence[Tuple[int, int, str]], rng: random.Random,
                 keep_one: bool) -> Tuple[float, int]:
    """(drawn fraction, number of words changed) for mask_content / typo."""
    fraction = rng.uniform(*SINGLE_TEXT_FRACTION)
    count = max(1, round(len(candidates) * fraction))
    if keep_one:
        count = min(len(candidates) - 1, count)
    return fraction, count


def choose_single_augmentation(
    operation: str,
    scenario: Mapping[str, Any],
    sample_id: str,
    iteration: int,
    seed: int,
    distractors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """One single-operation augmentation of one sample in one iteration (seeded, online)."""
    if operation not in SINGLE_OPERATIONS:
        raise ValueError(f"Unknown single operation {operation}")
    original = scenario["instruction"].strip()
    rng = stable_random("single", operation, seed, iteration, sample_id)
    choice: Dict[str, Any] = {
        "scenario": "single", "operation": operation, "applied": False, "not_applied_reason": None,
        "swap": False, "layout": "none", "position": None, "degradation": [], "rotation": None,
        "mask": [], "text_operation": "none", "instruction": original, "strength": {},
    }

    def skip(reason: str) -> Dict[str, Any]:
        choice["not_applied_reason"] = reason
        return choice

    if operation in ("drop_action", "mask_content", "typo"):
        if operation not in scenario["text_eligible"]:
            return skip("not eligible")
        if operation == "drop_action":
            variant = drop_action(original)
        elif operation == "mask_content":
            candidates = mask_candidates(original)
            fraction, count = _fraction_of(candidates, rng, keep_one=True)
            chosen = rng.sample(candidates, count)
            variant = replace_spans(original, [(s, e, MASK_TOKEN) for s, e, _ in chosen])
            choice["strength"] = {"fraction_drawn": round(fraction, 4), "words_changed": count,
                                  "candidate_words": len(candidates),
                                  "fraction_changed": round(count / len(candidates), 4)}
        else:
            candidates = typo_candidates(original)
            fraction, count = _fraction_of(candidates, rng, keep_one=False)
            chosen = rng.sample(candidates, count)
            variant = replace_spans(original, [(s, e, misspell(w, rng)) for s, e, w in chosen])
            choice["strength"] = {"fraction_drawn": round(fraction, 4), "words_changed": count,
                                  "candidate_words": len(candidates),
                                  "fraction_changed": round(count / len(candidates), 4)}
        if variant is None:
            return skip("not applicable")
        problems = TEXT_CHECKS[operation](original, variant)
        if problems:
            choice["fallback"] = problems
            return skip("check failed: " + "; ".join(problems))
        choice.update(applied=True, text_operation=operation, instruction=tidy(variant))
        return choice
    if operation == "distractor_instruction":
        other_index = rng.randrange(len(distractors))
        other = quoted_other(distractors[other_index]["instruction"])
        if other.lower() == quoted_other(original).lower():
            return skip("distractor instruction equals the instruction")
        template = rng.choice(DISTRACTOR_TEMPLATES)
        choice.update(applied=True, text_operation=operation, instruction_distractor_index=other_index,
                      instruction=tidy(template.format(other=other, instruction=original)))
        return choice
    if operation == "injection":
        choice.update(applied=True, text_operation=operation, instruction=online_injection(original, rng))
        return choice
    if operation == "composite":
        position = rng.choice(POSITIONS)
        other = OPPOSITE_SIDE[position]
        template = rng.choice(COMPOSITE_TEMPLATES)
        choice.update(
            applied=True, layout="composite", position=position, distractor_index=rng.randrange(len(distractors)),
            instruction=tidy(template.format(position=position, other=other, instruction=with_end_punctuation(original),
                                             instruction_lower=lower_first(with_end_punctuation(original)))),
        )
        choice["strength"] = {"main_image_position": position, "unrelated_image_position": other,
                              "template_index": COMPOSITE_TEMPLATES.index(template)}
        return choice
    if operation == "flip":
        if "flip" not in scenario["layout_eligible"]:
            return skip("not eligible (text / number / rotation / mirror instruction)")
        choice.update(applied=True, layout="flip", instruction=flip_instruction(original))
        return choice
    if operation in ("blur", "noise"):
        if operation not in scenario["degradation_eligible"]:
            return skip("not eligible (quality / sharpness instruction)")
        params = degradation_parameters([operation], bool(scenario["degradation_mild"]), rng)
        choice.update(applied=True, degradation=params)
        choice["strength"] = {key: value for key, value in params[0].items() if key != "name"}
        choice["strength"]["mild"] = bool(scenario["degradation_mild"])
        return choice
    if operation == "rotation":
        angle = round(rng.uniform(*ROTATION_DEGREES) * rng.choice((-1, 1)), 2)
        choice.update(applied=True, rotation=angle)
        choice["strength"] = {"degrees": angle}
        return choice
    # black_rectangles
    count = rng.randint(*MASK_RECTANGLES)
    area = rng.uniform(*SINGLE_MASK_AREA)
    choice.update(applied=True, mask=[_black_rectangle(rng, area / count) for _ in range(count)],
                  mask_area=round(area, 4))
    choice["strength"] = {"area": round(area, 4), "rectangles": count}
    return choice


def single_augmented_images(
    choice: Mapping[str, Any],
    source_path: str,
    image_a_path: str,
    image_b_path: str,
    distractor_path: Optional[str],
    work_dir: Path,
    tag: str,
) -> Tuple[str, str, str]:
    """Paths of (source, A, B) for the Student under one single operation.

    composite / flip / blur / noise change source, A and B the same way; rotation and
    black_rectangles change the source only; text operations change no image."""
    from PIL import Image, ImageOps

    if not choice["applied"]:
        return source_path, image_a_path, image_b_path
    operation = choice["operation"]
    if operation in ("composite", "flip", "blur", "noise"):
        targets = (("source", source_path), ("a", image_a_path), ("b", image_b_path))
    elif operation in ("rotation", "black_rectangles"):
        targets = (("source", source_path),)
    else:
        return source_path, image_a_path, image_b_path
    work_dir.mkdir(parents=True, exist_ok=True)
    distractor = None
    if operation == "composite":
        if distractor_path is None:
            raise ValueError("The composite operation needs a distractor image.")
        with Image.open(distractor_path) as opened:
            distractor = opened.convert("RGB")
    out = {"source": source_path, "a": image_a_path, "b": image_b_path}
    for name, original in targets:
        with Image.open(original) as opened:
            image = opened.convert("RGB")
        if operation == "composite":
            image = compose_spatial(image, distractor, choice["position"])
        elif operation == "flip":
            image = ImageOps.mirror(image)
        elif operation in ("blur", "noise"):
            image = degrade_image(image, choice["degradation"], tag)  # same seed for the three images
        elif operation == "rotation":
            image = image.rotate(float(choice["rotation"]), resample=Image.BICUBIC, expand=False, fillcolor=(0, 0, 0))
        else:
            image = _draw_black(image, choice["mask"])
        path = work_dir / f"{tag}_{name}_{operation}.png"
        image.save(path)
        out[name] = str(path)
    return out["source"], out["a"], out["b"]


# =====================================================================================
# Combined operations (user, 2026-11-06): the 4 strongest single operations (composite, blur,
# noise, mask_content) applied TOGETHER on every sample, for every combination of 2, 3 or 4.
# Every operation reuses the draw of its single-operation run (choose_single_augmentation with
# the same seed, iteration and sample), so a combination differs from its single runs only by
# being combined. An operation a sample is not eligible for is skipped alone; the others apply.
# Order: instruction = mask_content on the original instruction, then the composite template
# around it; images = composite (source, A, B) -> blur -> noise (same parameters on the three).
# No A<->B swap.
# =====================================================================================

COMBINABLE_OPERATIONS = ("composite", "blur", "noise", "mask_content")  # fixed order


def validate_combined_operations(operations: Sequence[str]) -> None:
    if len(operations) < 2 or len(set(operations)) != len(operations):
        raise ValueError(f"A combination needs at least 2 different operations, got {list(operations)}.")
    unknown = [name for name in operations if name not in COMBINABLE_OPERATIONS]
    if unknown:
        raise ValueError(f"Operations {unknown} cannot be combined; choose from {COMBINABLE_OPERATIONS}.")
    order = [name for name in COMBINABLE_OPERATIONS if name in operations]
    if list(operations) != order:
        raise ValueError(f"Write the combination in the fixed order {'+'.join(order)}.")


def choose_combined_operations(
    operations: Sequence[str],
    scenario: Mapping[str, Any],
    sample_id: str,
    iteration: int,
    seed: int,
    distractors: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    validate_combined_operations(operations)
    original = scenario["instruction"].strip()
    singles = {name: choose_single_augmentation(name, scenario, sample_id, iteration, seed, distractors)
               for name in operations}
    applied = [name for name in operations if singles[name]["applied"]]
    choice: Dict[str, Any] = {
        "scenario": "combined_operations", "operation": "+".join(operations), "applied": bool(applied),
        "applied_operations": applied,
        "not_applied_reason": ({name: singles[name]["not_applied_reason"] for name in operations
                                if not singles[name]["applied"]} or None),
        "swap": False, "layout": "none", "position": None, "degradation": [], "rotation": None, "mask": [],
        "text_operation": "none", "instruction": original,
        "strength": {name: singles[name]["strength"] for name in applied},
    }
    text = original
    if "mask_content" in applied:
        text = singles["mask_content"]["instruction"]
        choice["text_operation"] = "mask_content"
    if "composite" in applied:
        single = singles["composite"]
        template = COMPOSITE_TEMPLATES[single["strength"]["template_index"]]
        other = OPPOSITE_SIDE[single["position"]]
        text = tidy(template.format(position=single["position"], other=other, instruction=with_end_punctuation(text),
                                    instruction_lower=lower_first(with_end_punctuation(text))))
        choice.update(layout="composite", position=single["position"], distractor_index=single["distractor_index"])
    choice["instruction"] = text
    for name in ("blur", "noise"):
        if name in applied:
            choice["degradation"] = list(choice["degradation"]) + list(singles[name]["degradation"])
    return choice


def combined_operations_images(
    choice: Mapping[str, Any],
    source_path: str,
    image_a_path: str,
    image_b_path: str,
    distractor_path: Optional[str],
    work_dir: Path,
    tag: str,
) -> Tuple[str, str, str]:
    """Paths of (source, A, B): composite -> blur -> noise on the three images."""
    from PIL import Image

    if choice["layout"] != "composite" and not choice["degradation"]:
        return source_path, image_a_path, image_b_path  # text only (or nothing applied)
    work_dir.mkdir(parents=True, exist_ok=True)
    distractor = None
    if choice["layout"] == "composite":
        if distractor_path is None:
            raise ValueError("The composite operation needs a distractor image.")
        with Image.open(distractor_path) as opened:
            distractor = opened.convert("RGB")
    out = []
    for name, original in (("source", source_path), ("a", image_a_path), ("b", image_b_path)):
        with Image.open(original) as opened:
            image = opened.convert("RGB")
        if distractor is not None:
            image = compose_spatial(image, distractor, choice["position"])
        if choice["degradation"]:
            image = degrade_image(image, choice["degradation"], tag)  # blur then noise, same seed for the three
        path = work_dir / f"{tag}_{name}_combined_operations.png"
        image.save(path)
        out.append(str(path))
    return out[0], out[1], out[2]
