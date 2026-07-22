"""Import an existing transcript from Excel (.xlsx/.xls) or CSV.

Expected columns (case/space-insensitive, a few synonyms accepted):
  Speaker | Time From | Time To | Questions | Matter

Rules (per the satsang Q&A layout):
  * Questioner rows  → text from the "Questions" column
  * Pujyashree rows  → text from the "Matter" column
  * If both are filled on one row, they're joined.
  * Speaker labels are kept ONLY for voice assignment (never translated).
"""

from __future__ import annotations

import datetime as dt
import io
from pathlib import Path

import pandas as pd


def _norm(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


_COL_ALIASES = {
    "speaker": {"speaker", "speakers"},
    "from": {"timefrom", "from", "start", "starttime", "timestart"},
    "to": {"timeto", "to", "end", "endtime", "timeend", "till"},
    "questions": {"questions", "question", "ques"},
    "matter": {"matter", "answer", "answers", "response", "content"},
}


def _find_col(cols: list[str], kind: str) -> str | None:
    normed = {_norm(c): c for c in cols}
    for alias in _COL_ALIASES[kind]:
        if alias in normed:
            return normed[alias]
    return None


def parse_time(v) -> float | None:
    """Seconds as float from 'MM:SS', 'HH:MM:SS', number, or time/datetime cell."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dt.time):
        return v.hour * 3600 + v.minute * 60 + v.second + v.microsecond / 1e6
    if isinstance(v, (dt.datetime, dt.timedelta)):
        if isinstance(v, dt.timedelta):
            return v.total_seconds()
        return v.hour * 3600 + v.minute * 60 + v.second
    s = str(v).strip().replace(",", ".").rstrip("sS")
    if not s:
        return None
    if ":" in s:
        parts = s.split(":")
        try:
            parts = [float(p) for p in parts]
        except ValueError:
            return None
        sec = 0.0
        for p in parts:
            sec = sec * 60 + p
        return sec
    try:
        return float(s)
    except ValueError:
        return None


def import_transcript(data: bytes, filename: str) -> list[dict]:
    name = filename.lower()
    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(data))
    elif name.endswith((".csv", ".tsv")):
        df = pd.read_csv(io.BytesIO(data), sep="\t" if name.endswith(".tsv") else ",")
    else:
        raise ValueError("Upload an Excel (.xlsx/.xls) or CSV file.")

    df = df.dropna(how="all")
    cols = list(df.columns)
    c_speaker = _find_col(cols, "speaker")
    c_from = _find_col(cols, "from")
    c_to = _find_col(cols, "to")
    c_q = _find_col(cols, "questions")
    c_m = _find_col(cols, "matter")

    missing = [k for k, c in (("Time From", c_from), ("Time To", c_to)) if not c]
    if not c_q and not c_m:
        missing.append("Questions/Matter")
    if missing:
        raise ValueError(
            f"Couldn't find column(s): {', '.join(missing)}. "
            f"Your file has: {', '.join(map(str, cols))}"
        )

    segments = []
    for _, row in df.iterrows():
        qtext = str(row.get(c_q, "") or "").strip() if c_q else ""
        mtext = str(row.get(c_m, "") or "").strip() if c_m else ""
        if qtext.lower() in ("nan", "none"):
            qtext = ""
        if mtext.lower() in ("nan", "none"):
            mtext = ""
        if not qtext and not mtext:
            continue

        start = parse_time(row.get(c_from)) if c_from else None
        end = parse_time(row.get(c_to)) if c_to else None
        if start is None:
            continue
        if end is None or end <= start:
            end = start + max(2.0, min(15.0, (len(qtext) + len(mtext)) / 12.0))

        # Speaker comes from WHICH column the text is in (ground truth):
        #   Questions -> Questioner, Matter -> Pujyashree.
        # Rows containing BOTH (Q&A on one line) are SPLIT into two segments,
        # dividing the time range proportionally to the text lengths.
        parts: list[tuple[str, str, float, float]] = []
        if qtext and mtext:
            span = end - start
            ratio = len(qtext) / max(len(qtext) + len(mtext), 1)
            mid = start + min(max(span * ratio, 1.0), span - 1.0) if span > 2 else start + span / 2
            parts.append(("Questioner", qtext, start, mid))
            parts.append(("Pujyashree", mtext, mid, end))
        elif qtext:
            parts.append(("Questioner", qtext, start, end))
        else:
            parts.append(("Pujyashree", mtext, start, end))

        for role, text, ps, pe in parts:
            if pe <= ps:
                pe = ps + 1.0
            segments.append({
                "start": round(ps, 3), "end": round(pe, 3),
                "text": text, "speaker": role,
            })

    if not segments:
        raise ValueError("No usable rows found in the file.")
    segments.sort(key=lambda s: s["start"])
    return segments
