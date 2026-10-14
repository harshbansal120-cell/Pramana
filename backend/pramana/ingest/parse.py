"""Turn raw sources into citable units. No LLM here: every unit's text is verbatim source text.

PDF slide decks, including n-up handouts (several slides printed on one page), become one
unit per slide. Textbook pages become one or more units per page. Videos become transcript
windows with start and end seconds.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from .. import config


@dataclass
class UnitDraft:
    kind: str                 # slide | text | transcript | figure
    loc_type: str             # slide | page | time
    loc_start: float
    loc_end: float
    text: str
    image_path: str | None = None
    meta: dict = field(default_factory=dict)


_LICENSE = re.compile(r"For information about citing these materials|ocw\.mit\.edu/terms", re.I)
_FIGURE = re.compile(r"\bFig(?:ure|\.)\s*\d+(?:\.\d+)?\s*[:.]")
MAX_CHARS = 1500


def _clean(text: str) -> str:
    text = text.replace("­", "")
    for lig, plain in (("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"), ("ﬃ", "ffi"), ("ﬄ", "ffl")):
        text = text.replace(lig, plain)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # re-join hyphenated line breaks
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"^([•–▪◦-])\n", r"\1 ", text, flags=re.M)   # bullet glyph extracted on its own line
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _handout_grid(page: pymupdf.Page) -> tuple[int, int]:
    """Detect slides tiled on a page by looking for empty gutters through the page centre."""
    words = page.get_text("words")
    if len(words) < 8:
        return (1, 1)
    w, h = page.rect.width, page.rect.height
    body = [wd for wd in words if wd[3] < h * 0.96]      # ignore the page-number footer
    cols = 2 if not any(wd[0] < w / 2 - 4 and wd[2] > w / 2 + 4 for wd in body) and \
        any(wd[2] < w / 2 for wd in body) and any(wd[0] > w / 2 for wd in body) else 1
    rows = 2 if cols == 2 and not any(wd[1] < h / 2 + 10 and wd[3] > h / 2 - 10 for wd in body) else 1
    return (rows, cols) if cols == 2 else (1, 1)


def parse_slides(path: Path, source_id: str) -> list[UnitDraft]:
    out: list[UnitDraft] = []
    doc = pymupdf.open(path)
    render_dir = config.RENDERS / source_id
    render_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    # A handout uses one layout throughout; a sparse last page must not fall back to full-page.
    grids = [_handout_grid(page) for page in doc]
    tiled = [g for g in grids if g != (1, 1)]
    doc_grid = max(set(tiled), key=tiled.count) if len(tiled) >= 0.4 * len(grids) else None
    for page in doc:
        rows, cols = doc_grid or grids[page.number]
        w, h = page.rect.width, page.rect.height
        for r in range(rows):
            for c in range(cols):
                clip = pymupdf.Rect(c * w / cols, r * h / rows, (c + 1) * w / cols, (r + 1) * h / rows)
                text = _clean(page.get_text("text", clip=clip))
                has_ink = any(pymupdf.Rect(d["rect"]).intersects(clip) for d in page.get_drawings()) or \
                    any(pymupdf.Rect(i["bbox"]).intersects(clip) for i in page.get_image_info())
                # Keep a slide with no text layer only if it holds a real image (a scanned or pictorial slide,
                # which the vision stage will read); stray rules and logos on a mostly empty tile are dropped.
                pictorial = any(pymupdf.Rect(i["bbox"]).intersect(clip).get_area() > 0.2 * clip.get_area()
                                for i in page.get_image_info())
                if _LICENSE.search(text) or (len(text) < 15 and not pictorial):
                    continue
                n += 1
                img = render_dir / f"slide_{n:03d}.png"
                page.get_pixmap(dpi=130, clip=clip).save(img)
                out.append(UnitDraft("slide", "slide", n, n, text, str(img.relative_to(config.DATA)),
                                     {"pdf_page": page.number + 1, "has_visual": has_ink}))
    return out


def _split(text: str) -> list[str]:
    if len(text) <= MAX_CHARS:
        return [text]
    parts, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) > MAX_CHARS and len(cur) > MAX_CHARS // 2:
            parts.append(cur.strip())
            cur = ""
        cur += line + "\n"
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parse_textbook(path: Path, source_id: str, pages: tuple[int, int] | None = None) -> list[UnitDraft]:
    out: list[UnitDraft] = []
    doc = pymupdf.open(path)
    first, last = pages or (1, len(doc))
    render_dir = config.RENDERS / source_id
    render_dir.mkdir(parents=True, exist_ok=True)
    for pno in range(first - 1, min(last, len(doc))):
        page = doc[pno]
        raw = page.get_text("text")
        lines = raw.split("\n")
        # Drop the running header: a short all-caps title line and/or a bare page number.
        while lines and (re.fullmatch(r"\s*\d{1,4}\s*", lines[0]) or
                         (len(lines[0]) < 70 and lines[0].upper() == lines[0] and re.search(r"[A-Z]{3}", lines[0]))):
            lines.pop(0)
        text = _clean("\n".join(lines))
        if len(text) < 40:
            continue
        has_figure = bool(_FIGURE.search(text)) or bool(page.get_image_info())
        img = None
        if has_figure:
            img_path = render_dir / f"page_{pno + 1:04d}.png"
            page.get_pixmap(dpi=100).save(img_path)
            img = str(img_path.relative_to(config.DATA))
        for i, part in enumerate(_split(text)):
            out.append(UnitDraft("text", "page", pno + 1, pno + 1, part, img,
                                 {"has_visual": has_figure and i == 0, "part": i}))
    return out


def chunk_transcript(snippets: list[tuple[float, float, str]], window: float = 55.0) -> list[UnitDraft]:
    """Group caption snippets (start, duration, text) into windows of roughly `window` seconds."""
    out: list[UnitDraft] = []
    buf: list[str] = []
    start = end = None
    for s, d, t in snippets:
        t = re.sub(r"\s+", " ", t).strip()
        if not t or re.fullmatch(r"\[.*\]", t):
            continue
        if start is None:
            start = s
        buf.append(t)
        end = s + d
        if end - start >= window and re.search(r"[.?!]$", t) or (end - start >= window * 1.6):
            out.append(UnitDraft("transcript", "time", round(start, 1), round(end, 1), " ".join(buf)))
            buf, start = [], None
    if buf and start is not None:
        out.append(UnitDraft("transcript", "time", round(start, 1), round(end, 1), " ".join(buf)))
    return out


def _captions_api(video_id: str) -> tuple[list[tuple[float, float, str]], dict]:
    from youtube_transcript_api import YouTubeTranscriptApi

    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception as e:
        if "block" in type(e).__name__.lower():
            raise
        fetched = next(iter(api.list(video_id))).fetch()      # no English track: take what exists
    return ([(s.start, s.duration, s.text) for s in fetched.snippets],
            {"language": fetched.language_code, "generated": fetched.is_generated})


def _captions_ytdlp(video_id: str) -> tuple[list[tuple[float, float, str]], dict]:
    """Second route to the same captions, used when the transcript endpoint rate-limits us."""
    import httpx
    import yt_dlp

    opts = {"quiet": True, "skip_download": True, "no_warnings": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
    for generated, tracks in ((False, info.get("subtitles") or {}), (True, info.get("automatic_captions") or {})):
        for lang in [k for k in tracks if k.startswith("en")] or list(tracks)[:1]:
            fmt = next((f for f in tracks[lang] if f.get("ext") == "json3"), None)
            if not fmt:
                continue
            events = httpx.get(fmt["url"], timeout=60).json().get("events", [])
            snippets = [(e["tStartMs"] / 1000, e.get("dDurationMs", 0) / 1000,
                         "".join(seg.get("utf8", "") for seg in e.get("segs", []))) for e in events if e.get("segs")]
            if snippets:
                return snippets, {"language": lang, "generated": generated}
    raise RuntimeError("no captions available for this video")


def parse_youtube(video_id: str) -> tuple[list[UnitDraft], dict]:
    """Fetch captions with timestamps. Prefers English; falls back to any language available.

    Captions are cached on disk, and fetches are retried with backoff and a second route,
    because YouTube rate-limits bursts of transcript requests.
    """
    import json
    import time

    cache = config.RAW / f"{video_id}.captions.json"
    if cache.exists():
        saved = json.loads(cache.read_text(encoding="utf-8"))
        return chunk_transcript([tuple(s) for s in saved["snippets"]]), saved["meta"]
    last: Exception | None = None
    for attempt, route in enumerate((_captions_api, _captions_ytdlp, _captions_api, _captions_ytdlp)):
        try:
            snippets, meta = route(video_id)
            cache.write_text(json.dumps({"snippets": snippets, "meta": meta}), encoding="utf-8")
            time.sleep(1.5)                       # be polite between lectures
            return chunk_transcript(snippets), meta
        except Exception as e:
            last = e
            time.sleep(4 * (attempt + 1))
    raise RuntimeError(f"could not fetch captions for {video_id}: {type(last).__name__}: {str(last).strip().splitlines()[0] if str(last).strip() else ''}")


def parse_media_file(path: Path) -> tuple[list[UnitDraft], dict]:
    """Transcribe a local audio/video file. Needs the optional faster-whisper package."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as e:
        raise RuntimeError("Local video files need speech-to-text: pip install faster-whisper") from e
    model = WhisperModel("small", compute_type="int8")
    segments, info = model.transcribe(str(path), vad_filter=True)
    snippets = [(seg.start, seg.end - seg.start, seg.text) for seg in segments]
    return chunk_transcript(snippets), {"language": info.language, "generated": True}


def youtube_id(url: str) -> str | None:
    m = re.search(r"(?:v=|youtu\.be/|embed/|shorts/)([\w-]{11})", url)
    return m.group(1) if m else (url if re.fullmatch(r"[\w-]{11}", url) else None)
