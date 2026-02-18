import os
import json
import re
from collections import Counter, defaultdict

DATA_DIR = "data/control_full"
META_PATH = os.path.join(DATA_DIR, "meta.jsonl")
IMG_DIR = os.path.join(DATA_DIR, "images")


def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error at line {line_no}: {e}")
                continue


def safe_len(x) -> int:
    try:
        return len(x)
    except Exception:
        return 0


def pct(a: int, b: int) -> float:
    return (100.0 * a / b) if b else 0.0


def main():
    assert os.path.exists(META_PATH), f"Missing {META_PATH}"
    assert os.path.isdir(IMG_DIR), f"Missing images dir {IMG_DIR}"

    records = list(read_jsonl(META_PATH))
    n = len(records)
    print(f"\n=== DATASET SUMMARY ===")
    print(f"Meta file: {META_PATH}")
    print(f"Images dir: {IMG_DIR}")
    print(f"Total records: {n}")

    missing_text = 0
    missing_images_field = 0
    missing_source_tag = 0

    ids = []
    text_norms = []
    id_counts = Counter()
    text_counts = Counter()

    text_lengths = []

    source_tag_counts = Counter()

    post_type_counts = Counter()

    all_tag_counts = Counter()

    missing_image_files = 0
    total_image_files = 0
    ext_counts = Counter()
    image_sizes = []
    image_missing_paths = []

    tiny_images = 0
    tiny_threshold = 10_000

    for r in records:
        rid = r.get("id")
        source_tag = r.get("source_tag")
        post_type = r.get("post_type")
        text = r.get("text", "")
        image_files = r.get("image_files", [])

        if not isinstance(text, str) or not text.strip():
            missing_text += 1

        if "image_files" not in r:
            missing_images_field += 1

        if not source_tag:
            missing_source_tag += 1
        else:
            source_tag_counts[source_tag] += 1

        if post_type:
            post_type_counts[str(post_type)] += 1

        tags = r.get("tags", [])
        if isinstance(tags, list):
            for t in tags:
                if isinstance(t, str) and t.strip():
                    all_tag_counts[t.strip().lower()] += 1

        if isinstance(rid, str) and rid:
            ids.append(rid)
            id_counts[rid] += 1

        if isinstance(text, str):
            tnorm = re.sub(r"\s+", " ", text.strip().lower())
            text_norms.append(tnorm)
            text_counts[tnorm] += 1
            text_lengths.append(len(text))

        if not isinstance(image_files, list):
            image_files = []
        total_image_files += len(image_files)

        for fname in image_files:
            if not isinstance(fname, str):
                continue
            total_path = os.path.join(IMG_DIR, fname)
            if not os.path.exists(total_path):
                missing_image_files += 1
                image_missing_paths.append(total_path)
                continue

            _, ext = os.path.splitext(fname.lower())
            if ext:
                ext_counts[ext] += 1

            try:
                size = os.path.getsize(total_path)
                image_sizes.append(size)
                if size < tiny_threshold:
                    tiny_images += 1
            except OSError:
                missing_image_files += 1
                image_missing_paths.append(total_path)

    dup_id = sum(1 for k, v in id_counts.items() if v > 1)
    dup_text = sum(1 for k, v in text_counts.items() if v > 1)
    total_dup_id_rows = sum(v - 1 for v in id_counts.values() if v > 1)
    total_dup_text_rows = sum(v - 1 for v in text_counts.values() if v > 1)

    print(f"\n=== VALIDATION ===")
    print(f"Records with missing/empty text: {missing_text} ({pct(missing_text, n):.1f}%)")
    print(f"Records missing 'image_files' field: {missing_images_field} ({pct(missing_images_field, n):.1f}%)")
    print(f"Records missing 'source_tag': {missing_source_tag} ({pct(missing_source_tag, n):.1f}%)")

    print(f"\n=== DEDUP CHECKS ===")
    print(f"Unique IDs: {len(id_counts)}")
    print(f"Duplicate ID keys: {dup_id} | Extra duplicate rows by ID: {total_dup_id_rows}")
    print(f"Unique normalized texts: {len(text_counts)}")
    print(f"Duplicate text keys: {dup_text} | Extra duplicate rows by text: {total_dup_text_rows}")

    print(f"\n=== TEXT STATS ===")
    if text_lengths:
        text_lengths_sorted = sorted(text_lengths)

        def q(p):
            return text_lengths_sorted[int(p * (len(text_lengths_sorted) - 1))]

        print(f"Min/Median/Max length: {text_lengths_sorted[0]} / {q(0.5)} / {text_lengths_sorted[-1]}")
        print(f"P10/P90 length: {q(0.1)} / {q(0.9)}")
    else:
        print("No text lengths available.")

    print(f"\n=== SOURCE TAG DISTRIBUTION (top 20) ===")
    for tag, c in source_tag_counts.most_common(20):
        print(f"{tag:20s} {c:6d}  ({pct(c, n):5.1f}%)")

    print(f"\n=== POST TYPE DISTRIBUTION ===")
    for t, c in post_type_counts.most_common():
        print(f"{t:12s} {c:6d}  ({pct(c, n):5.1f}%)")

    print(f"\n=== IMAGE STATS ===")
    print(f"Total image_files references: {total_image_files}")
    print(f"Missing image files on disk: {missing_image_files} ({pct(missing_image_files, total_image_files):.1f}%)")
    print(f"Tiny images (<10KB): {tiny_images} ({pct(tiny_images, len(image_sizes)):.1f}% of existing files)")

    if image_sizes:
        sizes_sorted = sorted(image_sizes)

        def qs(p):
            return sizes_sorted[int(p * (len(sizes_sorted) - 1))]

        print(f"Image size bytes Min/Median/Max: {sizes_sorted[0]} / {qs(0.5)} / {sizes_sorted[-1]}")
        print(f"Image size bytes P10/P90: {qs(0.1)} / {qs(0.9)}")

    print(f"\n=== IMAGE EXTENSIONS ===")
    for ext, c in ext_counts.most_common():
        print(f"{ext:6s} {c:6d}  ({pct(c, max(1, len(image_sizes))):5.1f}%)")

    print(f"\n=== TOP TAGS INSIDE POSTS (top 30) ===")
    for t, c in all_tag_counts.most_common(30):
        print(f"{t:25s} {c:6d}")

    report_path = os.path.join(DATA_DIR, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Depressed dataset analysis report\n")
        f.write(f"Total records: {n}\n")
        f.write(f"Missing/empty text: {missing_text}\n")
        f.write(f"Total image refs: {total_image_files}\n")
        f.write(f"Missing image files: {missing_image_files}\n")
        f.write("Top source tags:\n")
        for tag, c in source_tag_counts.most_common(20):
            f.write(f"  {tag}: {c}\n")
    print(f"\nWrote report: {report_path}")

    if image_missing_paths:
        print("\n[WARN] Example missing image paths (first 10):")
        for p in image_missing_paths[:10]:
            print("  ", p)


if __name__ == "__main__":
    main()
