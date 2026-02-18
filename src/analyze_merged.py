import json
import os
import re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "final_dataset_2_ocr_clean"

ALL_PATH = DATA_DIR / "all.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
TEST_PATH = DATA_DIR / "test.jsonl"


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON error {path.name}:{line_no}: {e}")
                continue


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def summarize(path: Path, n_examples: int = 3):
    rows = list(read_jsonl(path))
    n = len(rows)

    labels = Counter()
    src_tags = Counter()
    post_types = Counter()
    text_lens = []
    missing_img = 0
    ext_counts = Counter()
    img_sizes = []

    id_counts = Counter()
    text_counts = Counter()

    for r in rows:
        rid = r.get("id")
        if isinstance(rid, str) and rid:
            id_counts[rid] += 1

        t = r.get("text", "")
        if isinstance(t, str):
            text_lens.append(len(t))
            text_counts[norm_text(t)] += 1

        labels[int(r.get("label", -1))] += 1

        st = r.get("source_tag")
        if isinstance(st, str) and st:
            src_tags[st] += 1

        pt = r.get("post_type")
        if isinstance(pt, str) and pt:
            post_types[pt] += 1

        img_path = r.get("image_path")
        if not isinstance(img_path, str) or not img_path:
            missing_img += 1
            continue

        p = Path(img_path)
        if not p.exists():
            missing_img += 1
            continue

        ext_counts[p.suffix.lower()] += 1
        try:
            img_sizes.append(p.stat().st_size)
        except OSError:
            missing_img += 1

    dup_ids = sum(1 for _, v in id_counts.items() if v > 1)
    dup_text = sum(1 for _, v in text_counts.items() if v > 1)

    print(f"\n=== {path.name} ===")
    print(f"Records: {n}")
    print(f"Label counts: {dict(labels)}")
    print(f"Missing images: {missing_img} ({(100*missing_img/n):.1f}%)" if n else "Missing images: N/A")
    print(f"Duplicate IDs: {dup_ids} | Duplicate normalized texts: {dup_text}")

    if text_lens:
        tl = sorted(text_lens)
        def q(p): return tl[int(p*(len(tl)-1))]
        print(f"Text length min/med/max: {tl[0]} / {q(0.5)} / {tl[-1]}")
        print(f"Text length p10/p90: {q(0.1)} / {q(0.9)}")

    if img_sizes:
        sz = sorted(img_sizes)
        def q(p): return sz[int(p*(len(sz)-1))]
        print(f"Image bytes min/med/max: {sz[0]} / {q(0.5)} / {sz[-1]}")
        print(f"Image bytes p10/p90: {q(0.1)} / {q(0.9)}")

    print("\nTop source tags (top 10):")
    for k, v in src_tags.most_common(10):
        print(f"  {k:18s} {v:6d}")

    print("\nPost types:")
    for k, v in post_types.most_common():
        print(f"  {k:10s} {v:6d}")

    print("\nImage extensions:")
    for k, v in ext_counts.most_common():
        print(f"  {k:6s} {v:6d}")

    print("\nExample inputs:")
    for ex in rows[:n_examples]:
        preview = (ex.get("text", "") or "")[:160].replace("\n", " ")
        print({
            "id": ex.get("id"),
            "label": ex.get("label"),
            "image_path": ex.get("image_path"),
            "text[:160]": preview
        })


def main():
    assert DATA_DIR.exists(), f"Missing: {DATA_DIR}"
    for p in [ALL_PATH, TRAIN_PATH, VAL_PATH, TEST_PATH]:
        assert p.exists(), f"Missing: {p}"

    summarize(ALL_PATH, n_examples=3)
    summarize(TRAIN_PATH, n_examples=2)
    summarize(VAL_PATH, n_examples=2)
    summarize(TEST_PATH, n_examples=2)


if __name__ == "__main__":
    main()
