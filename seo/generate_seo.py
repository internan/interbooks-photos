#!/usr/bin/env python3
"""SEO metadata generator for the Inter Books Squarespace import.

Reads inventory.xlsx, looks up each item's product photos (IB-0001-front.jpg,
IB-0001-back.jpg, ...), and asks Claude for:

  * an SEO title           (<= 60 chars)
  * an SEO / meta description (<= 155 chars)
  * alt text for every photo of the item (<= 125 chars each)

Outputs two CSVs:

  seo_metadata.csv   one row per product  (sku, seo_title, seo_description, url_slug, ...)
  seo_alt_text.csv   one row per photo    (sku, image_file, alt_text)

Usage:
  pip install -r seo/requirements.txt
  export ANTHROPIC_API_KEY=...            # or `ant auth login`
  python seo/generate_seo.py inventory.xlsx --dry-run --limit 3   # inspect prompts, no API calls
  python seo/generate_seo.py inventory.xlsx --limit 10            # try a small batch first
  python seo/generate_seo.py inventory.xlsx                       # full run (resumable)

Re-running skips SKUs already present in the output CSV, so an interrupted run
picks up where it stopped. Use --overwrite to start fresh.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sys
import threading
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import pandas as pd
from PIL import Image

MODEL = "claude-opus-5-5"

TITLE_MAX = 60
DESC_MIN, DESC_MAX = 120, 155
ALT_MAX = 125
MAX_IMAGES = 8          # per product; extras are skipped (and reported)
IMAGE_LONG_SIDE = 1024  # px; enough to read cover/title-page text

# Shop context the model writes for. Edit this to match how you describe the shop.
SHOP_CONTEXT = """\
Inter Books is an independent UK seller of secondhand, vintage and out-of-print books, \
with a strong focus on spirituality, healing, mysticism, esoteric and Eastern \
philosophy, and related non-fiction. Buyers are collectors and readers searching for \
specific titles, authors, or older editions that are hard to find new."""

# ---------------------------------------------------------------------------
# The prompt. This is where most of the quality comes from.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You write product SEO metadata for an online secondhand bookshop.

{SHOP_CONTEXT}

For each book you receive the shop's inventory record and photos of the actual copy \
for sale (front cover, back cover, title page, etc.). Write metadata that helps the \
listing rank for the searches real buyers make, and that is accurate about this \
specific copy.

## How buyers search
People look for books by title + author ("spiritual healing harry edwards"), sometimes \
adding "first edition", "hardback", "vintage", "1960s", or the publisher. Exact title \
and author name are the most valuable keywords; format, year and edition are the \
useful qualifiers. Subject keywords ("dowsing", "Tibetan Buddhism") help for less \
famous titles.

## SEO title (max {TITLE_MAX} characters, hard limit)
- Lead with the book title, then "by" + author. Use the title as printed (title case); \
drop a long subtitle if it doesn't fit, keep it if it carries the subject keyword.
- If room remains, add one or two distinguishing qualifiers in parentheses, e.g. \
"(1963 Hardback)", "(First Edition)", "(Vintage Paperback)". Only use qualifiers \
supported by the facts.
- No shop name, no "Buy", no ALL CAPS, no keyword lists, no emojis.

## Meta description ({DESC_MIN}-{DESC_MAX} characters, hard limit {DESC_MAX})
- One or two plain sentences a searcher would click. Include title and author \
naturally, say briefly what the book is about, then the most useful fact about this \
copy (format, publisher/year, edition, condition highlight).
- Write like a knowledgeable bookseller, not an advert. No hype ("amazing", "must-have"), \
no exclamation marks, no "Shop now".
- British English.

## Alt text (one per photo, max {ALT_MAX} characters each)
- Say which view it is and of which book, then the most identifying visual detail: \
"Front cover of Spiritual Healing by Harry Edwards, white and red dust jacket with a \
bronze bust of the author".
- Describe what is actually visible in that photo. Mention notable wear, inscriptions \
or signatures only if you can see them.
- Do not start with "Image of" or "Photo of". No keyword stuffing.

## Accuracy rules (most important)
- Facts about this copy (edition, printing, year, publisher, binding, signed, dust \
jacket, condition) may come ONLY from the inventory record or what is clearly visible \
in the photos. Never claim "first edition", "signed", or "rare" unless the record says \
so or the photo plainly shows it.
- What the book is about may come from the record, readable cover/flap/title-page \
text, or your own knowledge if the work is well known and you are confident. If you \
are unsure, describe the subject generally (e.g. "a guide to spiritual healing") \
rather than inventing specifics.
- If the record and the photos disagree (different title, author, year), trust the \
photos for what the item is, and say so in review_notes.
- Use review_notes for anything a human should check: conflicts, unreadable photos, \
guesses about the subject, missing key data. Leave it empty when nothing needs checking.
"""

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "seo_title": {"type": "string"},
        "seo_description": {"type": "string"},
        "alt_text": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "image_file": {"type": "string"},
                    "alt_text": {"type": "string"},
                },
                "required": ["image_file", "alt_text"],
                "additionalProperties": False,
            },
        },
        "review_notes": {"type": "string"},
    },
    "required": ["seo_title", "seo_description", "alt_text", "review_notes"],
    "additionalProperties": False,
}

# ---------------------------------------------------------------------------
# Inventory reading
# ---------------------------------------------------------------------------

# Canonical field -> header spellings we accept (compared after normalising).
COLUMN_ALIASES = {
    "sku": ["sku", "id", "stockid", "stockno", "stocknumber", "itemid", "code", "ref", "reference", "productid"],
    "title": ["title", "booktitle", "name", "productname", "product"],
    "author": ["author", "authors", "writer", "by"],
    "publisher": ["publisher", "pub", "publishedby"],
    "year": ["year", "date", "pubyear", "publicationyear", "published", "pubdate"],
    "edition": ["edition", "printing", "impression"],
    "format": ["format", "binding", "type", "cover", "hardbackpaperback"],
    "condition": ["condition", "grade", "bookcondition"],
    "isbn": ["isbn", "isbn13", "isbn10"],
    "category": ["category", "categories", "subject", "genre", "tags"],
    "notes": ["notes", "description", "comments", "details", "conditionnotes"],
    "price": ["price", "cost", "saleprice"],
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_columns(columns: list[str]) -> dict[str, str]:
    """Return {canonical_field: actual_column} for the columns we recognise."""
    mapping: dict[str, str] = {}
    normed = {_norm(c): c for c in columns}
    for field, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in normed and normed[alias] not in mapping.values():
                mapping[field] = normed[alias]
                break
    return mapping


def clean(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)  # 1963.0 -> 1963
    return str(value).strip()


def load_inventory(path: Path, sheet: str | None, sku_column: str | None) -> tuple[list[dict], dict[str, str]]:
    df = pd.read_excel(path, sheet_name=sheet or 0, dtype=object)
    df = df.dropna(how="all")
    mapping = map_columns(list(df.columns))
    if sku_column:
        if sku_column not in df.columns:
            sys.exit(f"--sku-column {sku_column!r} not found. Columns: {list(df.columns)}")
        mapping["sku"] = sku_column
    if "sku" not in mapping:
        sys.exit(f"Couldn't find a SKU column. Columns: {list(df.columns)}\nPass --sku-column NAME.")
    if "title" not in mapping:
        print(f"warning: no title column recognised in {list(df.columns)}; relying on photos", file=sys.stderr)

    rows = []
    for _, r in df.iterrows():
        sku = clean(r[mapping["sku"]])
        if not sku:
            continue
        record = {col: clean(r[col]) for col in df.columns if clean(r[col])}
        rows.append({"sku": sku, "record": record, "fields": {f: clean(r[c]) for f, c in mapping.items()}})
    return rows, mapping


# ---------------------------------------------------------------------------
# Photos
# ---------------------------------------------------------------------------

VIEW_ORDER = ["front", "spine", "back", "title-page", "jacket-flap", "inside-jacket-flap", "contents-page", "preface"]


def _view_rank(path: Path, sku: str) -> tuple[int, str]:
    view = path.stem[len(sku) + 1:]
    if view in VIEW_ORDER:
        return (VIEW_ORDER.index(view), view)
    m = re.match(r"(?:photo|inside)-(\d+)", view)
    return (len(VIEW_ORDER) + (int(m.group(1)) if m else 99), view)


def find_photos(photos_dir: Path, sku: str) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp"}
    photos = [p for p in photos_dir.glob(f"{sku}-*") if p.suffix.lower() in exts]
    return sorted(photos, key=lambda p: _view_rank(p, sku))


def encode_image(path: Path) -> dict:
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((IMAGE_LONG_SIDE, IMAGE_LONG_SIDE))
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.standard_b64encode(buf.getvalue()).decode()},
    }


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_user_content(item: dict, photos: list[Path], include_images: bool = True) -> list[dict]:
    content: list[dict] = []
    for p in photos:
        content.append({"type": "text", "text": f"Photo: {p.name}"})
        if include_images:
            content.append(encode_image(p))
    record = "\n".join(f"- {k}: {v}" for k, v in item["record"].items())
    files = ", ".join(p.name for p in photos) or "(no photos found)"
    content.append({
        "type": "text",
        "text": (
            f"<inventory_record>\n{record}\n</inventory_record>\n\n"
            f"Photos of this copy: {files}\n"
            "Write the SEO title, meta description, and one alt text per photo listed "
            "(use the exact file names)."
        ),
    })
    return content


def check(result: dict, photo_names: list[str]) -> list[str]:
    """Return a list of problems with the model output (empty = fine)."""
    problems = []
    t, d = result["seo_title"].strip(), result["seo_description"].strip()
    if not t:
        problems.append("seo_title is empty")
    elif len(t) > TITLE_MAX:
        problems.append(f"seo_title is {len(t)} characters; the limit is {TITLE_MAX}")
    if len(d) > DESC_MAX:
        problems.append(f"seo_description is {len(d)} characters; the limit is {DESC_MAX}")
    elif len(d) < DESC_MIN - 20:
        problems.append(f"seo_description is only {len(d)} characters; aim for {DESC_MIN}-{DESC_MAX}")
    got = {a["image_file"] for a in result["alt_text"]}
    missing = [n for n in photo_names if n not in got]
    if missing:
        problems.append(f"alt text missing for: {', '.join(missing)}")
    for a in result["alt_text"]:
        if len(a["alt_text"]) > ALT_MAX:
            problems.append(f"alt text for {a['image_file']} is {len(a['alt_text'])} characters; the limit is {ALT_MAX}")
    return problems


class RefusedError(Exception):
    pass


def call_claude(client: anthropic.Anthropic, messages: list[dict], effort: str) -> tuple[dict, list[dict]]:
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if response.stop_reason == "refusal":
        raise RefusedError(getattr(response.stop_details, "explanation", None) or "model declined")
    if response.stop_reason == "max_tokens":
        raise RuntimeError("response truncated (max_tokens)")
    text = next(b.text for b in response.content if b.type == "text")
    # Echo the assistant turn back unchanged (incl. thinking blocks) if we need a retry.
    return json.loads(text), [{"role": "assistant", "content": response.content}]


def generate(client: anthropic.Anthropic, item: dict, photos: list[Path], effort: str) -> dict:
    names = [p.name for p in photos]
    messages = [{"role": "user", "content": build_user_content(item, photos)}]
    result, assistant_turn = call_claude(client, messages, effort)
    problems = check(result, names)
    if problems:  # one corrective retry, keeping the conversation append-only
        messages += assistant_turn
        messages.append({"role": "user", "content": "Please fix these and return the full result again:\n- " + "\n- ".join(problems)})
        result, _ = call_claude(client, messages, effort)
        problems = check(result, names)
    result["_problems"] = problems
    return result


def slugify(*parts: str) -> str:
    text = " ".join(p for p in parts if p)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.lower()).strip("-")
    return text[:80].rstrip("-")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PRODUCT_FIELDS = ["sku", "seo_title", "title_chars", "seo_description", "description_chars",
                  "url_slug", "image_count", "needs_review", "review_notes"]
ALT_FIELDS = ["sku", "image_file", "alt_text", "alt_chars"]


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8-sig") as f:
        return {row["sku"] for row in csv.DictReader(f)}


def open_writer(path: Path, fields: list[str], overwrite: bool):
    new = overwrite or not path.exists()
    f = path.open("w" if new else "a", newline="", encoding="utf-8-sig" if new else "utf-8")
    w = csv.DictWriter(f, fieldnames=fields)
    if new:
        w.writeheader()
    return f, w


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inventory", type=Path, help="path to inventory.xlsx")
    ap.add_argument("--sheet", help="sheet name (default: first sheet)")
    ap.add_argument("--sku-column", help="column holding SKUs like IB-0001 (auto-detected if omitted)")
    ap.add_argument("--photos-dir", type=Path, default=repo_root, help=f"folder of product photos (default: {repo_root})")
    ap.add_argument("--out", type=Path, default=Path("seo_metadata.csv"), help="product CSV (alt text goes to <out>_alt_text.csv)")
    ap.add_argument("--limit", type=int, help="only process the first N products (after skipping finished ones)")
    ap.add_argument("--only", nargs="+", metavar="SKU", help="only process these SKUs")
    ap.add_argument("--workers", type=int, default=4, help="parallel requests (default 4)")
    ap.add_argument("--effort", default="medium", choices=["low", "medium", "high"], help="model effort (default medium)")
    ap.add_argument("--overwrite", action="store_true", help="start fresh instead of resuming")
    ap.add_argument("--dry-run", action="store_true", help="print the prompts and exit without calling the API")
    args = ap.parse_args()

    items, mapping = load_inventory(args.inventory, args.sheet, args.sku_column)
    print(f"Loaded {len(items)} products. Recognised columns: {mapping}", file=sys.stderr)

    alt_out = args.out.with_name(args.out.stem + "_alt_text.csv")
    done = set() if args.overwrite else read_done(args.out)
    todo = [i for i in items if i["sku"] not in done and (not args.only or i["sku"] in args.only)]
    if args.limit:
        todo = todo[: args.limit]
    if done:
        print(f"Skipping {len(done)} products already in {args.out}", file=sys.stderr)

    jobs = []
    for item in todo:
        photos = find_photos(args.photos_dir, item["sku"])
        if len(photos) > MAX_IMAGES:
            print(f"{item['sku']}: {len(photos)} photos, using the first {MAX_IMAGES}", file=sys.stderr)
            photos = photos[:MAX_IMAGES]
        jobs.append((item, photos))

    if args.dry_run:
        print("=== SYSTEM PROMPT ===\n" + SYSTEM_PROMPT)
        for item, photos in jobs:
            print(f"=== {item['sku']} ({len(photos)} photos attached) ===")
            for block in build_user_content(item, photos, include_images=False):
                print(block["text"])
            print()
        return

    if not jobs:
        print("Nothing to do.", file=sys.stderr)
        return

    client = anthropic.Anthropic(max_retries=5)
    pf, pw = open_writer(args.out, PRODUCT_FIELDS, args.overwrite)
    af, aw = open_writer(alt_out, ALT_FIELDS, args.overwrite)
    lock = threading.Lock()
    failed: list[str] = []

    def work(item, photos):
        return item, photos, generate(client, item, photos, args.effort)

    with pf, af, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, item, photos) for item, photos in jobs]
        for n, fut in enumerate(as_completed(futures), 1):
            try:
                item, photos, r = fut.result()
            except (RefusedError, anthropic.APIError, RuntimeError, json.JSONDecodeError, StopIteration) as e:
                sku = jobs[futures.index(fut)][0]["sku"]
                failed.append(sku)
                print(f"[{n}/{len(jobs)}] {sku}: FAILED - {type(e).__name__}: {e}", file=sys.stderr)
                continue
            f = item["fields"]
            notes = "; ".join(x for x in [r["review_notes"].strip(), *r["_problems"]] if x)
            if not photos:
                notes = "; ".join(x for x in ["no photos found", notes] if x)
            with lock:
                pw.writerow({
                    "sku": item["sku"],
                    "seo_title": r["seo_title"].strip(),
                    "title_chars": len(r["seo_title"].strip()),
                    "seo_description": r["seo_description"].strip(),
                    "description_chars": len(r["seo_description"].strip()),
                    "url_slug": slugify(f.get("title", "") or r["seo_title"], f.get("author", "")),
                    "image_count": len(photos),
                    "needs_review": "yes" if notes else "",
                    "review_notes": notes,
                })
                valid = {p.name for p in photos}
                for a in r["alt_text"]:
                    if a["image_file"] in valid:
                        aw.writerow({"sku": item["sku"], "image_file": a["image_file"],
                                     "alt_text": a["alt_text"].strip(), "alt_chars": len(a["alt_text"].strip())})
                pf.flush(); af.flush()
            flag = "  (needs review)" if notes else ""
            print(f"[{n}/{len(jobs)}] {item['sku']}: {r['seo_title']}{flag}", file=sys.stderr)

    print(f"\nWrote {args.out} and {alt_out}.", file=sys.stderr)
    if failed:
        print(f"{len(failed)} failed (re-run to retry them): {' '.join(failed)}", file=sys.stderr)


if __name__ == "__main__":
    main()
