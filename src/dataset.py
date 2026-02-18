import os, json, time, hashlib, re
from urllib.parse import urlparse
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config import TUMBLR_API_KEY

API_KEY = TUMBLR_API_KEY
assert API_KEY

BASE_URL = "https://api.tumblr.com/v2/tagged"

DEPRESSED_TAGS = [
    "depression", "depressed", "depressive",

    "mental health", "mentalhealth",

    "sad", "sadness", "numb", "empty",

    "lonely", "alone", "isolation",

    "tired", "exhausted", "burnout",

    "hopeless", "worthless"
]

TARGET_PER_TAG = 3
MAX_PAGES_PER_TAG = 2
LIMIT_PER_REQUEST = 20
SLEEP_SECONDS = 0.2
MIN_TEXT_CHARS = 20
MAX_IMAGES_PER_POST = 1

OUT_DIR = "data/depressed_fast_test3"
IMG_DIR = os.path.join(OUT_DIR, "images")
META_PATH = os.path.join(OUT_DIR, "meta.jsonl")
STATE_PATH = os.path.join(OUT_DIR, "state.json")

MAX_DOWNLOAD = 50
DOWNLOAD_WORKERS = 6
IMG_TIMEOUT = 10
API_TIMEOUT = 10

os.makedirs(IMG_DIR, exist_ok=True)

session = requests.Session()
session.headers.update({"User-Agent": "tumblr-multimodal-research/fast-test2"})

if os.path.exists(STATE_PATH):
    state = json.load(open(STATE_PATH, "r", encoding="utf-8"))
else:
    state = {"seen_post_ids": [], "seen_text_hashes": []}

seen_post_ids = set(state.get("seen_post_ids", []))
seen_text_hashes = set(state.get("seen_text_hashes", []))

IMG_RE = re.compile(r'<img[^>]+src="([^"]+)"', re.IGNORECASE)


def save_state():
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {"seen_post_ids": list(seen_post_ids), "seen_text_hashes": list(seen_text_hashes)},
            f,
            ensure_ascii=False,
            indent=2,
        )


def fetch_tagged(tag: str, before: int | None) -> list[dict]:
    params = {"tag": tag, "api_key": API_KEY, "limit": max(1, min(20, LIMIT_PER_REQUEST))}
    if before is not None:
        params["before"] = before
    try:
        r = session.get(BASE_URL, params=params, timeout=API_TIMEOUT)
        if r.status_code == 429:
            time.sleep(3.0)
            return []
        r.raise_for_status()
        return r.json().get("response", [])
    except requests.RequestException:
        return []


def strip_html(s: str) -> str:
    out, in_tag = [], False
    for ch in s:
        if ch == "<": in_tag = True; continue
        if ch == ">": in_tag = False; continue
        if not in_tag: out.append(ch)
    text = "".join(out)
    text = (text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
            .replace("&quot;", '"').replace("&#39;", "'"))
    return " ".join(text.split()).strip()


def extract_text(post: dict) -> str:
    parts = []
    for k in ("summary", "caption", "body", "title"):
        v = post.get(k)
        if isinstance(v, str) and v.strip():
            parts.append(v.strip())
    return strip_html("\n".join(parts))


def extract_image_urls(post: dict) -> list[str]:
    urls = []

    for ph in (post.get("photos") or []):
        orig = (ph.get("original_size") or {})
        u = orig.get("url")
        if u:
            urls.append(u)

    for k in ("caption", "body"):
        v = post.get(k)
        if isinstance(v, str) and v:
            urls.extend(IMG_RE.findall(v))

    urls = [u for u in urls if isinstance(u, str) and u.startswith("http")]

    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            out.append(u)
    return out


def safe_ext(url: str) -> str:
    path = urlparse(url).path.lower()
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
        if path.endswith(ext):
            return ext
    return ".jpg"


def post_key(post: dict) -> str:
    raw = f"{post.get('id', '')}-{post.get('blog_name', '')}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:16]


def append_jsonl(rec: dict):
    with open(META_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def collect_metadata_for_tag(tag: str) -> list[dict]:
    kept = 0
    before = None
    page = 0
    records = []
    api_calls = 0
    start = time.time()

    while kept < TARGET_PER_TAG and page < MAX_PAGES_PER_TAG:
        posts = fetch_tagged(tag, before)
        api_calls += 1
        page += 1
        if not posts:
            continue

        ts = [int(p.get("timestamp", 0)) for p in posts if p.get("timestamp") is not None]
        if ts:
            before = min(ts) - 1

        for post in posts:
            if kept >= TARGET_PER_TAG:
                break

            image_urls = extract_image_urls(post)
            if not image_urls:
                continue

            text = extract_text(post)
            if len(text) < MIN_TEXT_CHARS:
                continue

            pid = post_key(post)
            if pid in seen_post_ids:
                continue

            th = text_hash(text)
            if th in seen_text_hashes:
                continue

            rec = {
                "id": pid,
                "label": 1,
                "source_tag": tag,
                "timestamp": post.get("timestamp"),
                "post_type": post.get("type"),
                "text": text,
                "photo_urls": image_urls[:MAX_IMAGES_PER_POST],
                "tags": post.get("tags", []),
                "image_files": [],
            }

            records.append(rec)
            seen_post_ids.add(pid)
            seen_text_hashes.add(th)
            kept += 1

        time.sleep(SLEEP_SECONDS)

    elapsed = time.time() - start
    print(f"{tag}: meta kept {len(records)} | api_calls={api_calls} | pages={page} | {elapsed:.1f}s")
    save_state()
    return records


def download_one(pid: str, idx: int, url: str) -> tuple[bool, str]:
    ext = safe_ext(url)
    fname = f"{pid}_{idx}{ext}"
    out_path = os.path.join(IMG_DIR, fname)
    if os.path.exists(out_path):
        return True, fname
    try:
        r = session.get(url, stream=True, timeout=IMG_TIMEOUT)
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "image" not in ctype:
            return False, ""
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
        return True, fname
    except Exception:
        return False, ""


def download_images(records: list[dict]) -> list[dict]:
    jobs = []
    for rec in records:
        for idx, url in enumerate(rec["photo_urls"]):
            jobs.append((rec["id"], idx, url))
    jobs = jobs[:MAX_DOWNLOAD]

    if not jobs:
        return records

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as ex:
        futures = [ex.submit(download_one, pid, idx, url) for pid, idx, url in jobs]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="downloading", unit="img"):
            pass

    for rec in records:
        rec["image_files"] = []
        for idx, url in enumerate(rec["photo_urls"]):
            fname = f"{rec['id']}_{idx}{safe_ext(url)}"
            if os.path.exists(os.path.join(IMG_DIR, fname)):
                rec["image_files"].append(fname)
    return records


if __name__ == "__main__":
    all_records = []
    for tag in DEPRESSED_TAGS:
        all_records.extend(collect_metadata_for_tag(tag))

    all_records = download_images(all_records)

    kept_final = 0
    for rec in all_records:
        if rec["image_files"]:
            append_jsonl(rec)
            kept_final += 1

    print("FINAL kept with downloaded images:", kept_final)
    print("Meta:", META_PATH)
    print("Images:", IMG_DIR)
