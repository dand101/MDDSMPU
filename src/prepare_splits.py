import json
import random
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any


SEED = 42
SPLIT = (0.80, 0.10, 0.10)

HERE = Path(__file__).resolve().parent

SOURCES = [
    ("control_full",     HERE / "data" / "control_full"     / "meta.jsonl", HERE / "data" / "control_full"     / "images", 0),
    ("control_full_2",   HERE / "data" / "control_full_2"   / "meta.jsonl", HERE / "data" / "control_full_2"   / "images", 0),
    ("depressed_full",   HERE / "data" / "depressed_full"   / "meta.jsonl", HERE / "data" / "depressed_full"   / "images", 1),
    ("depressed_full_2", HERE / "data" / "depressed_full_2" / "meta.jsonl", HERE / "data" / "depressed_full_2" / "images", 1),
]

OUT_DIR = HERE / "data" / "final_dataset_2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ALL_PATH = OUT_DIR / "all.jsonl"
TRAIN_PATH = OUT_DIR / "train.jsonl"
VAL_PATH = OUT_DIR / "val.jsonl"
TEST_PATH = OUT_DIR / "test.jsonl"
DUP_REPORT_PATH = OUT_DIR / "duplicates_report.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

DUP_IMAGE_HASH_REPORT_PATH = OUT_DIR / "dup_image_hash_report.jsonl"

MAX_IMAGES_PER_POST = 1

HASH_COMPARE_K = 2


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] JSON decode error {path}:{line_no}: {e}")
                continue


def write_jsonl(path: Path, rows: List[Dict[str, Any]]):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def norm_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.strip().split())



def md5_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def analyze_duplicate_image_bytes(
    dup_report_path: Path,
    out_report_path: Path,
    k: int = 2,
) -> Dict[str, Any]:
    same_bytes = 0
    diff_bytes = 0
    skipped = 0
    read_errors = 0
    total = 0

    out_rows = []

    for row in read_jsonl(dup_report_path):
        total += 1
        imgs = row.get("images", [])[:k]
        if len(imgs) < 2:
            skipped += 1
            continue

        hashes = [md5_file(p) for p in imgs]
        if any(h == "" for h in hashes):
            read_errors += 1

        h0, h1 = hashes[0], hashes[1]
        same = (h0 != "" and h1 != "" and h0 == h1)

        if same:
            same_bytes += 1
        else:
            diff_bytes += 1

        out_rows.append({
            "id": row.get("id"),
            "n": row.get("n"),
            "same_text": row.get("same_text"),
            "same_image_path": row.get("same_image"),
            "images_compared": imgs,
            "md5": hashes,
            "same_image_bytes_first2": same,
            "kept_source_dataset": row.get("kept_source_dataset"),
        })

    write_jsonl(out_report_path, out_rows)

    return {
        "dup_ids_total_in_report": total,
        "dup_ids_skipped_lt2_images": skipped,
        "dup_ids_image_hash_read_errors": read_errors,
        "dup_ids_same_image_bytes_first2": same_bytes,
        "dup_ids_diff_image_bytes_first2": diff_bytes,
        "dup_image_hash_report_path": str(out_report_path),
    }


def build_records(meta_path: Path, img_dir: Path, expected_label: int, source_name: str) -> List[Dict[str, Any]]:
    records = []
    missing_images = 0
    bad_rows = 0
    label_mismatch = 0

    for r in read_jsonl(meta_path):
        rid = r.get("id")
        if rid is None:
            bad_rows += 1
            continue

        text = r.get("text", "")
        if not isinstance(text, str) or not text.strip():
            bad_rows += 1
            continue

        label = r.get("label")
        if label is None:
            label = expected_label

        if int(label) != int(expected_label):
            label_mismatch += 1
            continue

        img_files = r.get("image_files", [])
        if not isinstance(img_files, list) or len(img_files) == 0:
            missing_images += 1
            continue

        img_name = img_files[0] if MAX_IMAGES_PER_POST == 1 else img_files[:MAX_IMAGES_PER_POST]
        if isinstance(img_name, list):
            img_name = img_name[0]

        img_path = img_dir / img_name
        if not img_path.exists():
            missing_images += 1
            continue

        rec = {
            "id": str(rid),
            "label": int(expected_label),
            "text": text,
            "image_path": str(img_path.resolve()),
            "source_dataset": source_name,
            "source_tag": r.get("source_tag"),
            "timestamp": r.get("timestamp"),
            "post_type": r.get("post_type"),
        }
        records.append(rec)

    print(f"[INFO] {source_name}: kept={len(records)} missing_images={missing_images} bad_rows={bad_rows} label_mismatch={label_mismatch}")
    return records



def score_record(rec: Dict[str, Any]) -> Tuple[int, int, int, int]:
    t_norm = norm_text(rec.get("text", ""))
    t_raw = rec.get("text", "")
    has_ts = 1 if rec.get("timestamp") is not None else 0
    src = rec.get("source_dataset", "")
    prefer_non2 = 1 if not src.endswith("_2") else 0
    return (len(t_norm), len(t_raw), has_ts, prefer_non2)


def analyze_and_dedup_by_id(all_records: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    by_id: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in all_records:
        by_id[r["id"]].append(r)

    dup_ids = [rid for rid, items in by_id.items() if len(items) > 1]
    total_dups = len(dup_ids)

    same_text = 0
    diff_text = 0
    same_image = 0
    diff_image = 0
    diff_label = 0

    report_rows = []

    kept = []
    removed = 0

    for rid, items in by_id.items():
        if len(items) == 1:
            kept.append(items[0])
            continue

        norm_texts = [norm_text(it.get("text", "")) for it in items]
        image_paths = [it.get("image_path") for it in items]
        labels = [it.get("label") for it in items]

        text_set = set(norm_texts)
        img_set = set(image_paths)
        label_set = set(labels)

        is_same_text = len(text_set) == 1
        is_same_image = len(img_set) == 1
        is_same_label = len(label_set) == 1

        if is_same_text:
            same_text += 1
        else:
            diff_text += 1

        if is_same_image:
            same_image += 1
        else:
            diff_image += 1

        if not is_same_label:
            diff_label += 1

        best = max(items, key=score_record)

        report_rows.append({
            "id": rid,
            "n": len(items),
            "same_text": is_same_text,
            "same_image": is_same_image,
            "same_label": is_same_label,
            "labels": sorted(list(label_set)),
            "images": sorted(list(img_set))[:50],
            "texts_preview": [
                {
                    "source_dataset": it.get("source_dataset"),
                    "label": it.get("label"),
                    "image_path": it.get("image_path"),
                    "text_preview": norm_text(it.get("text", ""))[:200],
                    "score": score_record(it),
                }
                for it in items
            ],
            "kept_source_dataset": best.get("source_dataset"),
            "kept_score": score_record(best),
        })

        kept.append(best)
        removed += (len(items) - 1)

    write_jsonl(DUP_REPORT_PATH, report_rows)

    summary = {
        "total_records_before": len(all_records),
        "unique_ids": len(by_id),
        "duplicate_ids": total_dups,
        "duplicates_removed": removed,
        "dup_id_same_text": same_text,
        "dup_id_diff_text": diff_text,
        "dup_id_same_image": same_image,
        "dup_id_diff_image": diff_image,
        "dup_id_diff_label": diff_label,
        "duplicate_report_path": str(DUP_REPORT_PATH),
    }
    return kept, summary


def stratified_split(records: List[Dict[str, Any]], split: Tuple[float, float, float], seed: int):
    rnd = random.Random(seed)

    by_label: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_label[int(r["label"])].append(r)

    for lab in by_label:
        rnd.shuffle(by_label[lab])

    train, val, test = [], [], []

    for lab, items in by_label.items():
        n = len(items)
        n_train = int(n * split[0])
        n_val = int(n * split[1])
        n_test = n - n_train - n_val

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

        print(f"[INFO] label={lab} n={n} -> train={n_train} val={n_val} test={n_test}")

    rnd.shuffle(train)
    rnd.shuffle(val)
    rnd.shuffle(test)
    return train, val, test



def main():
    all_records: List[Dict[str, Any]] = []
    for source_name, meta_path, img_dir, expected_label in SOURCES:
        if not meta_path.exists():
            raise FileNotFoundError(f"Missing meta: {meta_path}")
        if not img_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {img_dir}")
        all_records.extend(build_records(meta_path, img_dir, expected_label, source_name))

    print(f"[INFO] Loaded total records: {len(all_records)}")

    deduped, dup_summary = analyze_and_dedup_by_id(all_records)
    print(f"[INFO] Unique IDs after dedup: {len(deduped)}")
    print(f"[INFO] Duplicate report written: {DUP_REPORT_PATH}")

    hash_summary = analyze_duplicate_image_bytes(
        dup_report_path=DUP_REPORT_PATH,
        out_report_path=DUP_IMAGE_HASH_REPORT_PATH,
        k=HASH_COMPARE_K,
    )
    print(f"[INFO] Dup image hash report written: {DUP_IMAGE_HASH_REPORT_PATH}")
    print(f"[INFO] Image-bytes check: same={hash_summary['dup_ids_same_image_bytes_first2']} "
          f"diff={hash_summary['dup_ids_diff_image_bytes_first2']} "
          f"(read_errors={hash_summary['dup_ids_image_hash_read_errors']}, skipped={hash_summary['dup_ids_skipped_lt2_images']})")

    n_dep = sum(1 for r in deduped if r["label"] == 1)
    n_ctl = sum(1 for r in deduped if r["label"] == 0)
    print(f"[INFO] Label counts after dedup: depressed={n_dep} control={n_ctl}")

    train, val, test = stratified_split(deduped, SPLIT, SEED)

    write_jsonl(ALL_PATH, deduped)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)
    write_jsonl(TEST_PATH, test)

    summary = {
        **dup_summary,
        **hash_summary,
        "after_dedup_label_counts": {"control": n_ctl, "depressed": n_dep},
        "split_seed": SEED,
        "split_ratio": {"train": SPLIT[0], "val": SPLIT[1], "test": SPLIT[2]},
        "paths": {
            "all": str(ALL_PATH),
            "train": str(TRAIN_PATH),
            "val": str(VAL_PATH),
            "test": str(TEST_PATH),
            "dup_report": str(DUP_REPORT_PATH),
            "dup_image_hash_report": str(DUP_IMAGE_HASH_REPORT_PATH),
            "summary": str(SUMMARY_PATH),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n--- WROTE ---")
    print("All  :", ALL_PATH)
    print("Train:", TRAIN_PATH)
    print("Val  :", VAL_PATH)
    print("Test :", TEST_PATH)
    print("Dup report:", DUP_REPORT_PATH)
    print("Dup image hash report:", DUP_IMAGE_HASH_REPORT_PATH)
    print("Summary:", SUMMARY_PATH)

    print("\n--- DUP SUMMARY (path vs bytes) ---")
    print(f"path-based: same_image={dup_summary['dup_id_same_image']} diff_image={dup_summary['dup_id_diff_image']}")
    print(f"bytes-based (first2): same={hash_summary['dup_ids_same_image_bytes_first2']} diff={hash_summary['dup_ids_diff_image_bytes_first2']}")

    print("\n--- EXAMPLE (first 2 train) ---")
    for ex in train[:2]:
        short = norm_text(ex["text"])[:120]
        print({"id": ex["id"], "label": ex["label"], "source_dataset": ex["source_dataset"], "text[:120]": short})


if __name__ == "__main__":
    main()
