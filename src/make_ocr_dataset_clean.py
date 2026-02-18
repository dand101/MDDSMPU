import json
import re
import unicodedata
from pathlib import Path

import easyocr
from tqdm import tqdm
from PIL import Image, ImageSequence
import numpy as np

HERE = Path(__file__).resolve().parent

IN_DIR = HERE / "data" / "final_dataset_2"

OUT_DIR = HERE / "data" / "final_dataset_2_ocr_clean"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = ["train", "val", "test"]

CACHE_PATH = OUT_DIR / "ocr_cache.json"

USE_GPU = True
MIN_OCR_CHARS = 8
MAX_OCR_CHARS = 400

TOKEN_URL = "<URL>"
TOKEN_EMAIL = "<EMAIL>"
TOKEN_USER = "<USER>"
TOKEN_HASHTAG = "<HASHTAG>"
TOKEN_NUM = "<NUM>"

TOKEN_EMOJI = "<EMOJI>"
TOKEN_EMOJI_POS = "<EMOJI_POS>"
TOKEN_EMOJI_NEG = "<EMOJI_NEG>"
TOKEN_EMOJI_NEU = "<EMOJI_NEU>"

TOKEN_ELONG = "<ELONG>"
KEEP_ELONG_TOKEN = True

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
USER_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"(?<!\w)#(\w+)")
LONG_NUM_RE = re.compile(r"\b\d{4,}\b")
MULTISPACE_RE = re.compile(r"\s+")
PUNCT_RUN_RE = re.compile(r"([!?.,])\1{2,}")
ELONG_CHAR_RE = re.compile(r"([a-zA-Z])\1{2,}")

EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F700, 0x1F77F),
    (0x1F780, 0x1F7FF),
    (0x1F800, 0x1F8FF),
    (0x1F900, 0x1F9FF),
    (0x1FA00, 0x1FA6F),
    (0x1FA70, 0x1FAFF),
    (0x2600, 0x26FF),
    (0x2700, 0x27BF),
]

NEG_EMOJI = set("😢😭😞😔😟😣😖😩😫😠😡💔")
POS_EMOJI = set("😀😄😆😂😊😍😘😇🙂🙌💖❤️")
NEU_EMOJI = set("😐😶🤔😕")


def normalize_unicode(s: str) -> str:
    return unicodedata.normalize("NFKC", s or "")


def is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    for a, b in EMOJI_RANGES:
        if a <= cp <= b:
            return True
    if cp in (0xFE0F, 0xFE0E):
        return True
    if 0x1F3FB <= cp <= 0x1F3FF:
        return True
    return False


def emoji_token(ch: str) -> str:
    if ch in NEG_EMOJI:
        return TOKEN_EMOJI_NEG
    if ch in POS_EMOJI:
        return TOKEN_EMOJI_POS
    if ch in NEU_EMOJI:
        return TOKEN_EMOJI_NEU
    return TOKEN_EMOJI


def replace_emojis(s: str) -> str:
    out = []
    for ch in s:
        if is_emoji_char(ch):
            out.append(" " + emoji_token(ch) + " ")
        else:
            out.append(ch)
    return "".join(out)


def squash_elongation(s: str) -> str:
    def repl(m):
        ch = m.group(1)
        if KEEP_ELONG_TOKEN:
            return ch * 2 + f" {TOKEN_ELONG}"
        return ch * 2

    return ELONG_CHAR_RE.sub(repl, s)


def clean_all_text(s: str) -> str:
    s = normalize_unicode(s)

    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")

    s = replace_emojis(s)

    s = URL_RE.sub(f" {TOKEN_URL} ", s)
    s = EMAIL_RE.sub(f" {TOKEN_EMAIL} ", s)
    s = USER_RE.sub(f" {TOKEN_USER} ", s)
    s = HASHTAG_RE.sub(rf" {TOKEN_HASHTAG} \1 ", s)

    s = LONG_NUM_RE.sub(f" {TOKEN_NUM} ", s)

    s = PUNCT_RUN_RE.sub(lambda m: m.group(1) * 2, s)

    s = squash_elongation(s)

    s = MULTISPACE_RE.sub(" ", s).strip()
    return s


def load_image_for_ocr(path: str):
    try:
        img = Image.open(path)
        if getattr(img, "is_animated", False):
            img = next(ImageSequence.Iterator(img))
        img = img.convert("RGB")
        return np.array(img)
    except Exception:
        return None


def ocr_is_meaningful(ocr_clean: str) -> bool:
    if not ocr_clean:
        return False
    if len(ocr_clean) < MIN_OCR_CHARS:
        return False
    return any(ch.isalpha() for ch in ocr_clean)


def count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


reader = easyocr.Reader(["en"], gpu=USE_GPU)

if CACHE_PATH.exists():
    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
else:
    cache = {}


def ocr_image(image_path: str) -> str:
    if not image_path:
        return ""

    if image_path in cache:
        return cache[image_path]

    img = load_image_for_ocr(image_path)
    if img is None:
        cache[image_path] = ""
        return ""

    try:
        lines = reader.readtext(img, detail=0, paragraph=True)
        txt = " ".join(lines).strip()
    except Exception:
        txt = ""

    if len(txt) > 2000:
        txt = txt[:2000]

    cache[image_path] = txt
    return txt


def process_split(split: str):
    in_path = IN_DIR / f"{split}.jsonl"
    out_path = OUT_DIR / f"{split}.jsonl"

    total_lines = count_lines(in_path)
    with_ocr = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        pbar = tqdm(total=total_lines, desc=f"CLEAN+OCR {split}", unit="rows")

        for line in fin:
            line = line.strip()
            pbar.update(1)
            if not line:
                continue

            r = json.loads(line)

            post_raw = r.get("text", "") or ""
            post_clean = clean_all_text(post_raw)

            img_path = r.get("image_path", "")
            ocr_raw = ocr_image(img_path)
            ocr_clean = clean_all_text(ocr_raw)

            if len(ocr_clean) > MAX_OCR_CHARS:
                ocr_clean = ocr_clean[:MAX_OCR_CHARS].rstrip()

            ok = ocr_is_meaningful(ocr_clean)
            if ok:
                with_ocr += 1

            r["text_clean"] = post_clean
            r["ocr_text"] = ocr_raw
            r["ocr_text_clean"] = ocr_clean if ok else ""
            r["has_ocr"] = 1 if ok else 0

            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

        pbar.close()

    print(f"[DONE] {split}: total={total_lines} | has_ocr={with_ocr} ({with_ocr / max(1, total_lines) * 100:.1f}%)")


def main():
    for split in SPLITS:
        process_split(split)

    CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote OCR cache:", CACHE_PATH)
    print("Clean OCR dataset dir:", OUT_DIR)


if __name__ == "__main__":
    main()
