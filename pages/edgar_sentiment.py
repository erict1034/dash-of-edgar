# edgar_10k_dashboard_llm.py
import os
import re
import json
import html as html_stdlib
import sqlite3
import warnings
import threading
import logging
from pathlib import Path
from datetime import datetime

import requests
import pandas as pd
import dash
from dash import get_app, dcc, html, ctx
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objects as go
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers.utils import logging as transformers_logging
from huggingface_hub.utils import logging as hf_hub_logging

dash.register_page(__name__, path="/sentiment", name="EDGAR Sentiment", order=6)


# Reduce known noisy Hugging Face warnings on Windows and unauthenticated pulls.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
warnings.filterwarnings(
    "ignore",
    message=r".*cache-system uses symlinks by default.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*unauthenticated requests to the HF Hub.*",
    category=UserWarning,
)
transformers_logging.set_verbosity_error()
hf_hub_logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# -----------------------
# CONFIG
# -----------------------
DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "sentiment.db"
PARQUET_DIR = DATA_DIR / "parquet"
PARQUET_PATH = PARQUET_DIR / "filings_sentiment.parquet"
DATA_DIR.mkdir(parents=True, exist_ok=True)
PARQUET_DIR.mkdir(parents=True, exist_ok=True)

FORM = "10-K"
DEFAULT_TICKER = "AAPL"
DEFAULT_YEARS = 5
DEBUG_MODE = os.getenv("DASH_DEBUG", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HEADERS = {
    "User-Agent": os.getenv(
        "SEC_USER_AGENT", "EarlyWarningDashboard your_email@example.com"
    )
}
# FinBERT model name to use for inference. Override via environment variable.
FINBERT_MODEL = os.getenv("FINBERT_MODEL", "ProsusAI/finbert")
FINBERT_DEVICE = os.getenv("FINBERT_DEVICE", "auto").lower()
HF_TOKEN = os.getenv("HF_TOKEN")
PRELOAD_FINBERT = os.getenv("PRELOAD_FINBERT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
# Sentiment analysis mode: "chunked" (splits MD&A into multiple chunks, scores each, aggregates)
# or "single" (scores first 6000 chars only). Chunked is slower but more thorough.
ANALYSIS_MODE = os.getenv("ANALYSIS_MODE", "chunked").lower()
# Max characters per chunk sent to model (used in both single and chunked modes)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "12000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "500"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
SENTIMENT_NEUTRAL_BAND = float(os.getenv("SENTIMENT_NEUTRAL_BAND", "0.08"))
SENTIMENT_DIRECTIONAL_BAND = float(os.getenv("SENTIMENT_DIRECTIONAL_BAND", "0.10"))
# Fallback for single-mode: max chars of text to send if not chunking
EXTRACT_CHARS = CHUNK_SIZE
EMOTIONS = {"fear", "greed", "optimism", "skepticism", "hype", "neutral"}
_CIK_CACHE: dict[str, str] = {}
_SUBMISSIONS_CACHE: dict[str, dict] = {}
_FINBERT_TOKENIZER = None
_FINBERT_MODEL_OBJ = None
_FINBERT_PRELOAD_STARTED = False
_FINBERT_PRELOAD_STATE = "idle"
_FINBERT_PRELOAD_ERROR = ""


def _resolve_finbert_device() -> str:
    """Resolve runtime device, honoring override while supporting auto mode."""
    if FINBERT_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if FINBERT_DEVICE == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return "cuda" if FINBERT_DEVICE == "cuda" else "cpu"


def _get_active_finbert_device() -> str:
    """Return active model device if loaded, else resolved target device."""
    if _FINBERT_MODEL_OBJ is not None:
        try:
            return str(next(_FINBERT_MODEL_OBJ.parameters()).device)
        except Exception:
            pass
    return _resolve_finbert_device()


# -----------------------
# UTILITIES
# -----------------------
_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"[ \t]{2,}")
_BLANK_RE = re.compile(r"\n{3,}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[\"'\(\[]?[A-Z0-9])")
_FOCUS_SECTION_TERMS = (
    "results of operations",
    "liquidity and capital resources",
    "financial condition and results of operations",
    "net sales",
    "revenue",
    "gross margin",
    "operating income",
    "operating margin",
    "cash flow",
    "capital resources",
)


def _find_10k_text_block(raw: str) -> str:
    """Fast string-scan to extract the <TEXT> body of the 10-K <DOCUMENT> block.

    The full submission file contains multiple <DOCUMENT> sections (exhibits,
    cover pages, etc.).  We walk each one looking for <TYPE>10-K and then
    slice out its <TEXT>...</TEXT> content — no DOTALL regex backtracking over
    the full 9-10 MB file.
    """
    raw_lower = raw.lower()
    search_from = 0

    while True:
        doc_start = raw_lower.find("<document>", search_from)
        if doc_start == -1:
            break

        doc_end = raw_lower.find("</document>", doc_start)
        if doc_end == -1:
            doc_end = len(raw)

        block_lower = raw_lower[doc_start:doc_end]

        # Check this block is the primary 10-K (not an exhibit)
        type_pos = block_lower.find("<type>")
        if type_pos != -1:
            type_end = block_lower.find("\n", type_pos)
            type_val = block_lower[type_pos + 6 : type_end].strip()
            if type_val == "10-k":
                # Extract <TEXT>...</TEXT> within this block
                text_start = block_lower.find("<text>")
                text_end = block_lower.find("</text>")
                if text_start != -1:
                    content_start = doc_start + text_start + 6
                    content_end = doc_start + text_end if text_end != -1 else doc_end
                    return raw[content_start:content_end]
                # <TEXT> tag absent — return the whole block
                return raw[doc_start:doc_end]

        search_from = doc_end + 1

    return raw  # fallback: use full file


def _clean_10k_main_text(raw: str) -> str:
    """Return cleaned plain text for the primary 10-K document block."""
    main_content = _find_10k_text_block(raw)
    main_content = html_stdlib.unescape(main_content)
    text = _TAG_RE.sub(" ", main_content)
    text = _SPACE_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def _extract_item_section(
    text: str,
    start_pattern: str,
    end_patterns: list[str],
    min_substantive_chars: int,
) -> str:
    """Extract the best substantive Item span while skipping TOC-style snippets."""
    start_re = re.compile(start_pattern, re.IGNORECASE)
    end_res = [re.compile(pattern, re.IGNORECASE) for pattern in end_patterns]

    starts = [m.start() for m in start_re.finditer(text)]
    if not starts:
        return ""

    best_slice = None
    best_len = 0

    for start in starts:
        candidate_end = None
        for end_re in end_res:
            end_match = end_re.search(text, start + 1)
            if end_match and end_match.start() > start:
                if candidate_end is None or end_match.start() < candidate_end:
                    candidate_end = end_match.start()

        end = candidate_end if candidate_end is not None else len(text)
        span_len = end - start

        if span_len >= min_substantive_chars and span_len > best_len:
            best_len = span_len
            best_slice = (start, end)

    if best_slice:
        return text[best_slice[0] : best_slice[1]].strip()

    # Fallback to the last occurrence if no candidate met substantive-length threshold.
    fallback_start = starts[-1]
    fallback_end = len(text)
    for end_re in end_res:
        end_match = end_re.search(text, fallback_start + 1)
        if end_match and end_match.start() > fallback_start:
            fallback_end = min(fallback_end, end_match.start())

    return text[fallback_start:fallback_end].strip()


def _extract_risk_factors_text(raw: str) -> str:
    """Extract Item 1A (Risk Factors) section."""
    text = _clean_10k_main_text(raw)
    section = _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+1a\b[^\n]{0,40}\brisk\s+factors\b",
        end_patterns=[r"\bitem\s+1b\b", r"\bitem\s+2\b"],
        min_substantive_chars=1800,
    )
    if section:
        return section

    return _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+1a\b",
        end_patterns=[r"\bitem\s+1b\b", r"\bitem\s+2\b"],
        min_substantive_chars=1800,
    )


def _extract_mda_text(raw: str) -> str:
    """Extract Item 7 (MD&A) section."""
    text = _clean_10k_main_text(raw)
    section = _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+7\b[^\n]{0,80}\bmanagement\b[^\n]{0,120}\bdiscussion\s+and\s+analysis\b",
        end_patterns=[r"\bitem\s+7a\b", r"\bitem\s+8\b"],
        min_substantive_chars=2500,
    )
    if section:
        return section

    return _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+7\b",
        end_patterns=[r"\bitem\s+7a\b", r"\bitem\s+8\b"],
        min_substantive_chars=2500,
    )


def _extract_text(raw: str, max_chars: int | None = None) -> str:
    """Return MD&A text, optionally capped to a max length."""
    if max_chars is None:
        max_chars = EXTRACT_CHARS

    text = _extract_mda_text(raw)
    return text[:max_chars]


def _build_focus_excerpt(text: str, max_chars: int = CHUNK_SIZE) -> str:
    """Build a more informative excerpt than the opening MD&A boilerplate.

    Fast mode and the UI preview should favor sections discussing actual
    operating results, liquidity, margins, and cash flows instead of the
    repeated introductory company summary that often appears at the top.
    """
    if len(text) <= max_chars:
        return text

    lower_text = text.lower()
    slices: list[tuple[int, int]] = []
    window_size = max(1200, min(2200, max_chars // 2))
    min_keyword_pos = min(1500, max(0, len(text) // 12))

    for term in _FOCUS_SECTION_TERMS:
        start_search = 0
        while True:
            position = lower_text.find(term, start_search)
            if position == -1:
                break
            if position >= min_keyword_pos:
                start = max(0, position - 250)
                end = min(len(text), start + window_size)
                slices.append((start, end))
                break
            start_search = position + len(term)

        if sum(end - start for start, end in slices) >= max_chars:
            break

    if not slices:
        midpoint = len(text) // 2
        start = max(0, midpoint - (max_chars // 2))
        end = min(len(text), start + max_chars)
        return text[start:end]

    merged_slices: list[tuple[int, int]] = []
    for start, end in sorted(slices):
        if not merged_slices or start > merged_slices[-1][1] + 200:
            merged_slices.append((start, end))
        else:
            prev_start, prev_end = merged_slices[-1]
            merged_slices[-1] = (prev_start, max(prev_end, end))

    parts: list[str] = []
    current_len = 0
    separator = "\n\n...\n\n"
    for start, end in merged_slices:
        piece = text[start:end].strip()
        if not piece:
            continue

        projected_len = current_len + len(piece)
        if parts:
            projected_len += len(separator)

        if projected_len > max_chars:
            remaining = max_chars - current_len - (len(separator) if parts else 0)
            if remaining > 300:
                parts.append(piece[:remaining].rstrip())
            break

        if parts:
            parts.append(separator)
            current_len += len(separator)

        parts.append(piece)
        current_len += len(piece)

    excerpt = "".join(parts).strip()
    return excerpt if excerpt else text[:max_chars]


def _normalize_result(data: dict) -> dict:
    """Guard against malformed or inconsistent model output."""
    try:
        sentiment = float(data.get("sentiment", 0))
    except Exception:
        sentiment = 0.0
    sentiment = max(-1.0, min(1.0, sentiment))

    try:
        confidence = float(data.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    emotion = str(data.get("emotion", "neutral")).strip().lower()
    if emotion not in EMOTIONS:
        emotion = "neutral"

    # Keep the categorical emotion aligned with the numeric score.
    if abs(sentiment) < SENTIMENT_NEUTRAL_BAND:
        emotion = "neutral"
    elif sentiment >= 0.15 and emotion in {"skepticism", "fear"}:
        emotion = "optimism"
    elif sentiment <= -0.15 and emotion in {"optimism", "greed", "hype"}:
        emotion = "skepticism"

    # When confidence is low, avoid overconfident skeptical labels on near-neutral text.
    if (
        confidence < 0.35
        and abs(sentiment) < SENTIMENT_DIRECTIONAL_BAND
        and emotion == "skepticism"
    ):
        emotion = "neutral"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "emotion": emotion,
    }


def _get_finbert_components():
    """Lazily load and cache FinBERT tokenizer/model."""
    global \
        _FINBERT_TOKENIZER, \
        _FINBERT_MODEL_OBJ, \
        _FINBERT_PRELOAD_STATE, \
        _FINBERT_PRELOAD_ERROR
    if _FINBERT_TOKENIZER is not None and _FINBERT_MODEL_OBJ is not None:
        _FINBERT_PRELOAD_STATE = "ready"
        return _FINBERT_TOKENIZER, _FINBERT_MODEL_OBJ

    _FINBERT_PRELOAD_STATE = "warming"
    _FINBERT_PRELOAD_ERROR = ""

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            FINBERT_MODEL,
            token=HF_TOKEN,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            FINBERT_MODEL,
            token=HF_TOKEN,
        )
        model.eval()

        device = _resolve_finbert_device()
        model = model.to(device)

        _FINBERT_TOKENIZER = tokenizer
        _FINBERT_MODEL_OBJ = model
        _FINBERT_PRELOAD_STATE = "ready"
        return _FINBERT_TOKENIZER, _FINBERT_MODEL_OBJ
    except Exception as exc:
        _FINBERT_PRELOAD_STATE = "failed"
        _FINBERT_PRELOAD_ERROR = str(exc)
        raise


def _start_finbert_preload() -> None:
    """Warm FinBERT on startup so first live pull is faster."""
    global _FINBERT_PRELOAD_STARTED
    if not PRELOAD_FINBERT or _FINBERT_PRELOAD_STARTED:
        return

    _FINBERT_PRELOAD_STARTED = True

    def _preload_worker() -> None:
        try:
            _get_finbert_components()
        except Exception:
            # Keep startup resilient; model load errors are handled at inference time too.
            pass

    threading.Thread(target=_preload_worker, daemon=True).start()


def _finbert_predict(text: str) -> dict:
    """Run FinBERT on text and map labels to dashboard sentiment schema."""
    tokenizer, model = _get_finbert_components()

    inputs = tokenizer(
        text,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze(0)

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    label_probs = {id2label[i]: float(probs[i]) for i in range(len(probs))}

    p_pos = label_probs.get("positive", 0.0)
    p_neg = label_probs.get("negative", 0.0)
    p_neu = label_probs.get("neutral", 0.0)

    # Scale to [-1, 1] by net polarity while preserving FinBERT confidence.
    sentiment = max(-1.0, min(1.0, p_pos - p_neg))
    confidence = max(p_pos, p_neg, p_neu)

    top_label = (
        max(label_probs.items(), key=lambda kv: kv[1])[0] if label_probs else "neutral"
    )
    if top_label == "positive":
        emotion = "optimism"
    elif top_label == "negative":
        emotion = "skepticism"
    else:
        emotion = "neutral"

    # If the top label is neutral but polarity is directional, preserve direction.
    if emotion == "neutral" and abs(sentiment) >= SENTIMENT_DIRECTIONAL_BAND:
        emotion = "optimism" if sentiment > 0 else "skepticism"

    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "emotion": emotion,
    }


def _finbert_predict_batch(texts: list[str]) -> list[dict]:
    """Run FinBERT for a batch of chunks in one forward pass."""
    tokenizer, model = _get_finbert_components()

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

    model_device = next(model.parameters()).device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1)

    id2label = {int(k): str(v).lower() for k, v in model.config.id2label.items()}
    results = []

    for row in probs:
        label_probs = {id2label[i]: float(row[i]) for i in range(len(row))}

        p_pos = label_probs.get("positive", 0.0)
        p_neg = label_probs.get("negative", 0.0)
        p_neu = label_probs.get("neutral", 0.0)

        sentiment = max(-1.0, min(1.0, p_pos - p_neg))
        confidence = max(p_pos, p_neg, p_neu)

        top_label = max(label_probs, key=label_probs.get)
        emotion = (
            "optimism"
            if top_label == "positive"
            else "skepticism"
            if top_label == "negative"
            else "neutral"
        )

        if emotion == "neutral" and abs(sentiment) >= SENTIMENT_DIRECTIONAL_BAND:
            emotion = "optimism" if sentiment > 0 else "skepticism"

        results.append(
            {
                "sentiment": sentiment,
                "confidence": confidence,
                "emotion": emotion,
            }
        )

    return results


def _predict_chunks_batched(chunks: list[str]) -> list[dict]:
    """Predict chunk sentiments in micro-batches to reduce total runtime."""
    all_results = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        all_results.extend(_finbert_predict_batch(batch))
    return all_results


def _analyze_single_chunk(text: str) -> dict:
    """Single FinBERT inference on a text chunk — fast."""
    try:
        data = _finbert_predict(text)
        return _normalize_result(data)
    except Exception:
        return {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}


def _chunk_text(
    text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks for analysis."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text) - overlap:
            break
    return chunks


def _split_sentences(text: str, min_chars: int = 40, max_chars: int = 500) -> list[str]:
    """Split filing text into sentence-like units suitable for FinBERT scoring."""
    if not text:
        return []

    normalized = _SPACE_RE.sub(" ", text.replace("\n", " ")).strip()
    if not normalized:
        return []

    parts = _SENTENCE_SPLIT_RE.split(normalized)
    cleaned = []
    for part in parts:
        sentence = part.strip()
        if not sentence:
            continue
        if len(sentence) < min_chars:
            continue
        if len(sentence) > max_chars:
            sentence = sentence[:max_chars].rstrip()
        cleaned.append(sentence)
    return cleaned


def categorize_non_neutral_sentences(
    text: str,
    top_n_per_bucket: int = 12,
) -> dict[str, list[dict]]:
    """Score sentences and return strongest non-neutral statements by category."""
    sentences = _split_sentences(text)
    if not sentences:
        return {"optimism": [], "skepticism": []}

    raw_results = _predict_chunks_batched(sentences)
    records: list[dict] = []
    for sentence, raw_result in zip(sentences, raw_results):
        result = _normalize_result(raw_result)
        if result["emotion"] == "neutral":
            continue
        if abs(float(result["sentiment"])) < SENTIMENT_NEUTRAL_BAND:
            continue
        bucket = "optimism" if result["sentiment"] > 0 else "skepticism"
        records.append(
            {
                "bucket": bucket,
                "sentence": sentence,
                "sentiment": round(float(result["sentiment"]), 4),
                "confidence": round(float(result["confidence"]), 4),
            }
        )

    records.sort(key=lambda item: abs(item["sentiment"]), reverse=True)

    grouped = {"optimism": [], "skepticism": []}
    for record in records:
        bucket = record["bucket"]
        if len(grouped[bucket]) >= top_n_per_bucket:
            continue
        grouped[bucket].append(
            {
                "sentence": record["sentence"],
                "sentiment": record["sentiment"],
                "confidence": record["confidence"],
            }
        )

    return grouped


def analyze_filing_chunked(text: str) -> dict:
    """Analyze full MD&A by chunking, scoring each chunk, and aggregating results."""
    chunks = _chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
    if not chunks:
        return {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}

    raw_results = _predict_chunks_batched(chunks)
    results = [_normalize_result(r) for r in raw_results]

    # Confidence-weighted aggregation reduces cancellation from weak/noisy chunks.
    weights = [max(float(r["confidence"]), 0.15) for r in results]
    weight_sum = sum(weights) if weights else 1.0
    avg_sentiment = (
        sum(r["sentiment"] * w for r, w in zip(results, weights)) / weight_sum
    )
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    if avg_sentiment >= SENTIMENT_NEUTRAL_BAND:
        best_emotion = "optimism"
    elif avg_sentiment <= -SENTIMENT_NEUTRAL_BAND:
        best_emotion = "skepticism"
    else:
        best_emotion = "neutral"

    return {
        "sentiment": max(-1.0, min(1.0, avg_sentiment)),
        "confidence": max(0.0, min(1.0, avg_confidence)),
        "emotion": best_emotion,
    }


def analyze_filing(text: str) -> dict:
    """Analyze filing text using configured mode (chunked or single)."""
    if ANALYSIS_MODE == "chunked":
        return analyze_filing_chunked(text)
    else:
        return _analyze_single_chunk(text)


# -----------------------
# EDGAR DATA
# -----------------------
def _filing_year_from_accession(accession: str) -> int:
    """Extract year from accession number format XXXXXXXXXX-YY-NNNNNN."""
    parts = (accession or "").split("-")
    if len(parts) >= 2 and parts[1].isdigit():
        yy = int(parts[1])
        return 1900 + yy if yy >= 50 else 2000 + yy
    return datetime.utcnow().year


def _filing_year(
    report_date: str | None, filing_date: str | None, accession: str
) -> int:
    # Prefer filing_date so chart years match when the 10-K was filed (e.g., 2025).
    for value in (filing_date, report_date):
        if value:
            try:
                return datetime.strptime(value, "%Y-%m-%d").year
            except ValueError:
                pass
    return _filing_year_from_accession(accession)


def _get_cik(ticker: str) -> str | None:
    ticker = ticker.strip().upper()
    if not ticker:
        return None

    cached = _CIK_CACHE.get(ticker)
    if cached:
        return cached

    try:
        resp = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return None

    for item in payload.values():
        if str(item.get("ticker", "")).upper() == ticker:
            cik = str(item.get("cik_str", "")).zfill(10)
            if cik:
                _CIK_CACHE[ticker] = cik
                return cik
    return None


def _get_company_submissions(cik: str) -> dict | None:
    cached = _SUBMISSIONS_CACHE.get(cik)
    if cached:
        return cached

    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/CIK{cik}.json",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        _SUBMISSIONS_CACHE[cik] = payload
        return payload
    except Exception:
        return None


def _recent_filing_rows(submissions: dict, form: str) -> list[dict]:
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])

    rows: list[dict] = []
    # SEC arrays can have uneven lengths (often reportDate is shorter); index safely.
    for i, frm in enumerate(forms):
        if frm != form:
            continue
        accession = accessions[i] if i < len(accessions) else ""
        filing_date = filing_dates[i] if i < len(filing_dates) else None
        report_date = report_dates[i] if i < len(report_dates) else None
        rows.append(
            {
                "accession": accession,
                "filing_date": filing_date,
                "report_date": report_date,
            }
        )
    return rows


def _fetch_submissions_history_file(cik: str, filename: str) -> dict | None:
    """Fetch a paginated historical submissions JSON file for a company."""
    try:
        resp = requests.get(
            f"https://data.sec.gov/submissions/{filename}",
            headers=HEADERS,
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _all_filing_rows(submissions: dict, cik: str, form: str) -> list[dict]:
    """Collect filing rows from recent + paginated historical submission files."""
    rows = _recent_filing_rows(submissions, form)

    history_files = submissions.get("filings", {}).get("files", [])
    for file_info in history_files:
        filename = file_info.get("name")
        if not filename:
            continue
        payload = _fetch_submissions_history_file(cik, filename)
        if not payload:
            continue
        rows.extend(_recent_filing_rows(payload, form))

    # Deduplicate accessions and sort newest-first by filing date.
    by_accession: dict[str, dict] = {}
    for row in rows:
        accession = row.get("accession", "")
        if accession and accession not in by_accession:
            by_accession[accession] = row

    deduped_rows = list(by_accession.values())
    deduped_rows.sort(key=lambda r: r.get("filing_date") or "", reverse=True)
    return deduped_rows


def _fetch_filing_index(cik: str, accession: str) -> dict | None:
    acc_no = accession.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/index.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def _select_submission_text_filename(
    index_json: dict | None, accession: str
) -> str | None:
    if not index_json:
        return None

    items = index_json.get("directory", {}).get("item", [])
    if not items:
        return None

    names = [item.get("name", "") for item in items if item.get("name")]
    accession_txt = f"{accession}.txt"

    # Prefer the canonical accession text file when available.
    if accession_txt in names:
        return accession_txt

    txt_candidates = [
        name
        for name in names
        if name.lower().endswith(".txt")
        and "index" not in name.lower()
        and "readme" not in name.lower()
    ]
    if txt_candidates:
        return txt_candidates[0]

    # Fallback: use the primary filing HTML document if no .txt exists.
    htm_candidates = [
        name
        for name in names
        if name.lower().endswith(".htm") and "index" not in name.lower()
    ]
    if htm_candidates:
        return htm_candidates[0]

    return None


def _fetch_full_submission(cik: str, accession: str) -> str | None:
    acc_no = accession.replace("-", "")
    index_json = _fetch_filing_index(cik, accession)
    filename = _select_submission_text_filename(index_json, accession)
    if not filename:
        return None

    url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no}/{filename}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        return resp.text
    except Exception:
        return None


def collect_edgar(ticker: str, num_years: int = DEFAULT_YEARS, form: str = FORM):
    ticker = ticker.strip().upper()
    cik = _get_cik(ticker)
    if not cik:
        return pd.DataFrame()

    submissions = _get_company_submissions(cik)
    if not submissions:
        return pd.DataFrame()

    filings = _all_filing_rows(submissions, cik, form)
    rows = []
    for filing in filings:
        if len(rows) >= num_years:
            break

        accession = filing.get("accession", "")
        if not accession:
            continue
        raw = _fetch_full_submission(cik, accession)
        if not raw:
            continue

        risk_text = _extract_risk_factors_text(raw)
        full_mda = _extract_mda_text(raw)
        if not full_mda and not risk_text:
            continue

        mda_preview_text = _build_focus_excerpt(full_mda, max_chars=CHUNK_SIZE)
        risk_preview_text = (risk_text or "")[:CHUNK_SIZE]

        if ANALYSIS_MODE == "chunked":
            mda_result = (
                analyze_filing(full_mda)
                if full_mda
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
            risk_result = (
                analyze_filing_chunked(risk_text)
                if risk_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
        else:
            mda_result = (
                analyze_filing(mda_preview_text)
                if mda_preview_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
            risk_result = (
                analyze_filing(risk_preview_text)
                if risk_preview_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )

        # Keep top-level sentiment backward-compatible by using MD&A when available.
        primary_result = mda_result if full_mda else risk_result
        text = mda_preview_text or risk_preview_text
        non_neutral = categorize_non_neutral_sentences(
            (mda_preview_text or "") + "\n\n" + (risk_preview_text or "")
        )

        filing_year = _filing_year(
            filing.get("report_date"), filing.get("filing_date"), accession
        )
        rows.append(
            {
                "ticker": ticker,
                "filing_year": filing_year,
                "sentiment": primary_result["sentiment"],
                "confidence": primary_result["confidence"],
                "emotion": primary_result["emotion"],
                "mda_sentiment": mda_result["sentiment"],
                "mda_confidence": mda_result["confidence"],
                "mda_emotion": mda_result["emotion"],
                "risk_sentiment": risk_result["sentiment"],
                "risk_confidence": risk_result["confidence"],
                "risk_emotion": risk_result["emotion"],
                "extracted_text": text,
                "mda_extracted_text": mda_preview_text,
                "risk_extracted_text": risk_preview_text,
                "non_neutral_sentences_json": json.dumps(non_neutral),
            }
        )
    return pd.DataFrame(rows)


# -----------------------
# 10-Q EXTRACTION & COLLECTION
# -----------------------


def _find_10q_text_block(raw: str) -> str:
    """Fast string-scan to extract the <TEXT> body of the 10-Q <DOCUMENT> block."""
    raw_lower = raw.lower()
    search_from = 0

    while True:
        doc_start = raw_lower.find("<document>", search_from)
        if doc_start == -1:
            break

        doc_end = raw_lower.find("</document>", doc_start)
        if doc_end == -1:
            doc_end = len(raw)

        block_lower = raw_lower[doc_start:doc_end]

        type_pos = block_lower.find("<type>")
        if type_pos != -1:
            type_end = block_lower.find("\n", type_pos)
            type_val = block_lower[type_pos + 6 : type_end].strip()
            if type_val == "10-q":
                text_start = block_lower.find("<text>")
                text_end = block_lower.find("</text>")
                if text_start != -1:
                    content_start = doc_start + text_start + 6
                    content_end = doc_start + text_end if text_end != -1 else doc_end
                    return raw[content_start:content_end]
                return raw[doc_start:doc_end]

        search_from = doc_end + 1

    return raw


def _clean_10q_main_text(raw: str) -> str:
    """Return cleaned plain text for the primary 10-Q document block."""
    main_content = _find_10q_text_block(raw)
    main_content = html_stdlib.unescape(main_content)
    text = _TAG_RE.sub(" ", main_content)
    text = _SPACE_RE.sub(" ", text)
    text = _BLANK_RE.sub("\n\n", text)
    return text.strip()


def _extract_10q_mda_text(raw: str) -> str:
    """Extract Item 2 (MD&A) from a 10-Q (Part I)."""
    text = _clean_10q_main_text(raw)
    section = _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+2\b[^\n]{0,80}\bmanagement\b[^\n]{0,120}\bdiscussion\s+and\s+analysis\b",
        end_patterns=[r"\bitem\s+3\b", r"\bitem\s+4\b", r"\bpart\s+ii\b"],
        min_substantive_chars=1500,
    )
    if section:
        return section
    return _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+2\b",
        end_patterns=[r"\bitem\s+3\b", r"\bitem\s+4\b", r"\bpart\s+ii\b"],
        min_substantive_chars=1500,
    )


def _extract_10q_risk_factors_text(raw: str) -> str:
    """Extract Item 1A (Risk Factors) from a 10-Q (Part II — often brief updates)."""
    text = _clean_10q_main_text(raw)
    section = _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+1a\b[^\n]{0,40}\brisk\s+factors\b",
        end_patterns=[r"\bitem\s+1b\b", r"\bitem\s+2\b", r"\bitem\s+3\b"],
        min_substantive_chars=300,
    )
    if section:
        return section
    return _extract_item_section(
        text=text,
        start_pattern=r"\bitem\s+1a\b",
        end_patterns=[r"\bitem\s+1b\b", r"\bitem\s+2\b", r"\bitem\s+3\b"],
        min_substantive_chars=300,
    )


def _period_label(
    report_date: str | None, filing_date: str | None, accession: str
) -> str:
    """Derive a human-readable quarter label (e.g. '2024 Q3') from date info."""
    date_str = report_date or filing_date
    if date_str:
        try:
            dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
            q = (dt.month - 1) // 3 + 1
            return f"{dt.year} Q{q}"
        except Exception:
            pass
    if accession:
        parts = accession.split("-")
        if len(parts) >= 2:
            try:
                return f"20{int(parts[1]):02d}"
            except Exception:
                pass
    return "Unknown"


def collect_edgar_10q(ticker: str, num_quarters: int = 4):
    """Collect and analyze recent 10-Q filings for a ticker."""
    ticker = ticker.strip().upper()
    cik = _get_cik(ticker)
    if not cik:
        return pd.DataFrame()

    submissions = _get_company_submissions(cik)
    if not submissions:
        return pd.DataFrame()

    filings = _all_filing_rows(submissions, cik, "10-Q")
    rows = []
    for filing in filings:
        if len(rows) >= num_quarters:
            break

        accession = filing.get("accession", "")
        if not accession:
            continue
        raw = _fetch_full_submission(cik, accession)
        if not raw:
            continue

        risk_text = _extract_10q_risk_factors_text(raw)
        full_mda = _extract_10q_mda_text(raw)
        if not full_mda and not risk_text:
            continue

        mda_preview_text = _build_focus_excerpt(full_mda, max_chars=CHUNK_SIZE)
        risk_preview_text = (risk_text or "")[:CHUNK_SIZE]

        if ANALYSIS_MODE == "chunked":
            mda_result = (
                analyze_filing(full_mda)
                if full_mda
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
            risk_result = (
                analyze_filing_chunked(risk_text)
                if risk_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
        else:
            mda_result = (
                analyze_filing(mda_preview_text)
                if mda_preview_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )
            risk_result = (
                analyze_filing(risk_preview_text)
                if risk_preview_text
                else {"sentiment": 0, "confidence": 0.5, "emotion": "neutral"}
            )

        primary_result = mda_result if full_mda else risk_result
        non_neutral = categorize_non_neutral_sentences(
            (mda_preview_text or "") + "\n\n" + (risk_preview_text or "")
        )

        period = _period_label(
            filing.get("report_date"), filing.get("filing_date"), accession
        )
        rows.append(
            {
                "ticker": ticker,
                "accession": accession,
                "period_label": period,
                "filing_date": filing.get("filing_date", ""),
                "sentiment": primary_result["sentiment"],
                "confidence": primary_result["confidence"],
                "emotion": primary_result["emotion"],
                "mda_sentiment": mda_result["sentiment"],
                "mda_confidence": mda_result["confidence"],
                "mda_emotion": mda_result["emotion"],
                "risk_sentiment": risk_result["sentiment"],
                "risk_confidence": risk_result["confidence"],
                "risk_emotion": risk_result["emotion"],
                "non_neutral_sentences_json": json.dumps(non_neutral),
            }
        )
    return pd.DataFrame(rows)


# -----------------------
# SQLITE STORAGE
# -----------------------
def _ensure_table(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filings_sentiment (
            ticker        TEXT NOT NULL,
            filing_year   INTEGER NOT NULL,
            sentiment     REAL,
            confidence    REAL,
            emotion       TEXT,
            extracted_text TEXT,
            mda_sentiment REAL,
            mda_confidence REAL,
            mda_emotion   TEXT,
            risk_sentiment REAL,
            risk_confidence REAL,
            risk_emotion  TEXT,
            mda_extracted_text TEXT,
            risk_extracted_text TEXT,
            non_neutral_sentences_json TEXT,
            PRIMARY KEY (ticker, filing_year)
        )
        """
    )
    # Migrate old tables: add extracted_text column if it doesn't exist
    cursor = conn.execute("PRAGMA table_info(filings_sentiment)")
    columns = {row[1] for row in cursor.fetchall()}
    if "extracted_text" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN extracted_text TEXT")
    if "mda_sentiment" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN mda_sentiment REAL")
    if "mda_confidence" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN mda_confidence REAL")
    if "mda_emotion" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN mda_emotion TEXT")
    if "risk_sentiment" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN risk_sentiment REAL")
    if "risk_confidence" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN risk_confidence REAL")
    if "risk_emotion" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN risk_emotion TEXT")
    if "mda_extracted_text" not in columns:
        conn.execute("ALTER TABLE filings_sentiment ADD COLUMN mda_extracted_text TEXT")
    if "risk_extracted_text" not in columns:
        conn.execute(
            "ALTER TABLE filings_sentiment ADD COLUMN risk_extracted_text TEXT"
        )
    if "non_neutral_sentences_json" not in columns:
        conn.execute(
            "ALTER TABLE filings_sentiment ADD COLUMN non_neutral_sentences_json TEXT"
        )


def upsert_sqlite(df: pd.DataFrame):
    if df.empty:
        return
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table(conn)
        conn.executemany(
            """
            INSERT OR REPLACE INTO filings_sentiment
                (
                    ticker,
                    filing_year,
                    sentiment,
                    confidence,
                    emotion,
                    extracted_text,
                    mda_sentiment,
                    mda_confidence,
                    mda_emotion,
                    risk_sentiment,
                    risk_confidence,
                    risk_emotion,
                    mda_extracted_text,
                    risk_extracted_text,
                    non_neutral_sentences_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.ticker,
                    int(r.filing_year),
                    r.sentiment,
                    r.confidence,
                    r.emotion,
                    getattr(r, "extracted_text", ""),
                    getattr(r, "mda_sentiment", None),
                    getattr(r, "mda_confidence", None),
                    getattr(r, "mda_emotion", None),
                    getattr(r, "risk_sentiment", None),
                    getattr(r, "risk_confidence", None),
                    getattr(r, "risk_emotion", None),
                    getattr(r, "mda_extracted_text", ""),
                    getattr(r, "risk_extracted_text", ""),
                    getattr(
                        r,
                        "non_neutral_sentences_json",
                        json.dumps({"optimism": [], "skepticism": []}),
                    ),
                )
                for r in df.itertuples(index=False)
            ],
        )
    save_parquet(df)


def save_parquet(df: pd.DataFrame):
    """Merge new rows into the parquet snapshot, keyed on (ticker, filing_year)."""
    if df.empty:
        return
    if PARQUET_PATH.exists():
        existing = pd.read_parquet(PARQUET_PATH)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(
            subset=["ticker", "filing_year"], keep="last"
        )
    else:
        combined = df.copy()
    combined = combined.sort_values(["ticker", "filing_year"]).reset_index(drop=True)
    combined.to_parquet(PARQUET_PATH, index=False)


def load_sqlite(ticker: str = "") -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='filings_sentiment'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()
        if ticker:
            return pd.read_sql(
                "SELECT * FROM filings_sentiment WHERE ticker = ? ORDER BY filing_year",
                conn,
                params=(ticker.strip().upper(),),
            )
        return pd.read_sql(
            "SELECT * FROM filings_sentiment ORDER BY ticker, filing_year", conn
        )


# -----------------------
# SQLITE STORAGE (10-Q)
# -----------------------


def _ensure_table_10q(conn):
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS filings_sentiment_10q (
            ticker          TEXT NOT NULL,
            accession       TEXT NOT NULL,
            period_label    TEXT,
            filing_date     TEXT,
            sentiment       REAL,
            confidence      REAL,
            emotion         TEXT,
            mda_sentiment   REAL,
            mda_confidence  REAL,
            mda_emotion     TEXT,
            risk_sentiment  REAL,
            risk_confidence REAL,
            risk_emotion    TEXT,
            non_neutral_sentences_json TEXT,
            PRIMARY KEY (ticker, accession)
        )
        """
    )


def upsert_sqlite_10q(df: pd.DataFrame):
    if df.empty:
        return
    with sqlite3.connect(DB_PATH) as conn:
        _ensure_table_10q(conn)
        conn.executemany(
            """
            INSERT OR REPLACE INTO filings_sentiment_10q
                (
                    ticker, accession, period_label, filing_date,
                    sentiment, confidence, emotion,
                    mda_sentiment, mda_confidence, mda_emotion,
                    risk_sentiment, risk_confidence, risk_emotion,
                    non_neutral_sentences_json
                )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    r.ticker,
                    r.accession,
                    getattr(r, "period_label", ""),
                    getattr(r, "filing_date", ""),
                    r.sentiment,
                    r.confidence,
                    r.emotion,
                    getattr(r, "mda_sentiment", None),
                    getattr(r, "mda_confidence", None),
                    getattr(r, "mda_emotion", None),
                    getattr(r, "risk_sentiment", None),
                    getattr(r, "risk_confidence", None),
                    getattr(r, "risk_emotion", None),
                    getattr(
                        r,
                        "non_neutral_sentences_json",
                        json.dumps({"optimism": [], "skepticism": []}),
                    ),
                )
                for r in df.itertuples(index=False)
            ],
        )
    save_parquet_10q(df)


def load_sqlite_10q(ticker: str = "") -> pd.DataFrame:
    if not DB_PATH.exists():
        return pd.DataFrame()
    with sqlite3.connect(DB_PATH) as conn:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='filings_sentiment_10q'"
        ).fetchone()
        if not exists:
            return pd.DataFrame()
        if ticker:
            return pd.read_sql(
                "SELECT * FROM filings_sentiment_10q WHERE ticker = ? ORDER BY filing_date",
                conn,
                params=(ticker.strip().upper(),),
            )
        return pd.read_sql(
            "SELECT * FROM filings_sentiment_10q ORDER BY ticker, filing_date", conn
        )


def save_parquet_10q(df: pd.DataFrame):
    """Merge new 10-Q rows into the parquet snapshot, keyed on (ticker, accession)."""
    if df.empty:
        return
    parquet_path_10q = PARQUET_DIR / "filings_sentiment_10q.parquet"
    if parquet_path_10q.exists():
        existing = pd.read_parquet(parquet_path_10q)
        combined = pd.concat([existing, df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["ticker", "accession"], keep="last")
    else:
        combined = df.copy()
    combined = combined.sort_values(["ticker", "filing_date"]).reset_index(drop=True)
    combined.to_parquet(parquet_path_10q, index=False)


# -----------------------
# DASH APP
# -----------------------
def build_layout():
    return html.Div(
        style={
            "fontFamily": "Arial, sans-serif",
            "maxWidth": "960px",
            "margin": "0 auto",
            "padding": "20px",
        },
        children=[
            html.H1("EDGAR 10-K Sentiment (FinBERT)"),
            html.H2(
                "10-K Analysis",
                style={"fontSize": "18px", "marginBottom": "12px", "marginTop": "8px"},
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
                children=[
                    dcc.Input(
                        id="ticker-input",
                        type="text",
                        placeholder="Enter ticker here",
                        value="",
                        debounce=False,
                        style={"padding": "8px", "fontSize": "14px", "width": "140px"},
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "fontSize": "12px",
                        },
                        children=[
                            html.Label("Years of 10-Ks", style={"marginBottom": "2px"}),
                            dcc.Slider(
                                id="years-slider",
                                min=1,
                                max=10,
                                step=1,
                                value=DEFAULT_YEARS,
                                marks={i: str(i) for i in range(1, 11)},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                    ),
                    html.Button(
                        "Load from DB",
                        id="load-db-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Button(
                        "Pull Live",
                        id="load-live-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Span(
                        id="status-label",
                        style={"fontStyle": "italic", "fontSize": "13px"},
                    ),
                    html.Span(
                        id="model-status-label",
                        style={"fontSize": "12px", "marginLeft": "4px"},
                    ),
                ],
            ),
            dcc.Interval(id="model-status-interval", interval=1500, n_intervals=0),
            html.H3(
                "Non-Neutral Sentence Categories",
                style={"fontSize": "15px", "marginBottom": "10px", "marginTop": "8px"},
            ),
            html.Div(id="non-neutral-container"),
            html.Hr(style={"marginTop": "24px", "marginBottom": "16px"}),
            html.H3(
                "Sentiment Trend",
                style={"fontSize": "15px", "marginBottom": "10px"},
            ),
            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Button(
                        "Load Sentiment Trend",
                        id="trend-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Span(
                        " — reads from DB only",
                        style={
                            "fontSize": "12px",
                            "color": "#64748b",
                            "marginLeft": "8px",
                        },
                    ),
                ],
            ),
            dcc.Graph(id="sentiment-trend-chart"),
            html.Hr(style={"marginTop": "32px", "marginBottom": "16px"}),
            html.H2(
                "10-Q Analysis",
                style={"fontSize": "18px", "marginBottom": "12px"},
            ),
            html.Div(
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "marginBottom": "16px",
                },
                children=[
                    dcc.Input(
                        id="ticker-10q-input",
                        type="text",
                        placeholder="Enter ticker here",
                        value="",
                        debounce=False,
                        style={"padding": "8px", "fontSize": "14px", "width": "140px"},
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "flexDirection": "column",
                            "fontSize": "12px",
                        },
                        children=[
                            html.Label(
                                "Quarters of 10-Qs", style={"marginBottom": "2px"}
                            ),
                            dcc.Slider(
                                id="quarters-slider",
                                min=1,
                                max=8,
                                step=1,
                                value=4,
                                marks={i: str(i) for i in range(1, 9)},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                    ),
                    html.Button(
                        "Load 10-Q from DB",
                        id="load-10q-db-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Button(
                        "Pull Live 10-Q",
                        id="load-10q-live-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Span(
                        id="10q-status-label",
                        style={"fontStyle": "italic", "fontSize": "13px"},
                    ),
                ],
            ),
            html.H3(
                "Non-Neutral Sentence Categories (10-Q)",
                style={"fontSize": "15px", "marginBottom": "10px", "marginTop": "8px"},
            ),
            html.Div(id="non-neutral-10q-container"),
            html.Hr(style={"marginTop": "24px", "marginBottom": "16px"}),
            html.H3(
                "10-Q Sentiment Trend",
                style={"fontSize": "15px", "marginBottom": "10px"},
            ),
            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Button(
                        "Load 10-Q Sentiment Trend",
                        id="trend-10q-btn",
                        n_clicks=0,
                        style={"padding": "8px 14px"},
                    ),
                    html.Span(
                        " — reads from DB only",
                        style={
                            "fontSize": "12px",
                            "color": "#64748b",
                            "marginLeft": "8px",
                        },
                    ),
                ],
            ),
            dcc.Graph(id="sentiment-trend-10q-chart"),
        ],
    )


layout = build_layout()


def register_callbacks(app):
    @app.callback(
        [
            Output("non-neutral-container", "children"),
            Output("status-label", "children"),
        ],
        [
            Input("load-db-btn", "n_clicks"),
            Input("load-live-btn", "n_clicks"),
        ],
        [
            State("ticker-input", "value"),
            State("years-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_dashboard(db_clicks, live_clicks, ticker_val, num_years):
        ticker = (ticker_val or DEFAULT_TICKER).strip().upper()
        num_years = int(num_years or DEFAULT_YEARS)
        triggered = ctx.triggered_id

        if triggered == "load-live-btn":
            df = collect_edgar(ticker, num_years)
            if df.empty:
                return (
                    html.Div("No data returned from SEC.", style={"color": "#999"}),
                    "Live pull returned no data.",
                )
            upsert_sqlite(df)
            status = f"Live pull complete — {len(df)} filing(s) stored."
        else:
            status = "Loaded from DB."

        df_sql = load_sqlite(ticker)
        if df_sql.empty:
            return (
                html.Div(
                    f"No cached data for {ticker}. Click 'Pull Live'.",
                    style={"color": "#999"},
                ),
                "No cached data.",
            )

        def _render_non_neutral_list(items: list[dict], tone: str):
            if not items:
                return html.Div(
                    "No non-neutral sentences identified.",
                    style={"fontSize": "12px", "color": "#666", "padding": "4px 0"},
                )
            tone_color = "#0f766e" if tone == "optimism" else "#9f1239"
            blocks = []
            for item in items:
                sentence = str(item.get("sentence", "")).strip()
                if not sentence:
                    continue
                sentiment_val = float(item.get("sentiment", 0.0))
                confidence_val = float(item.get("confidence", 0.0))
                blocks.append(
                    html.Li(
                        [
                            html.Div(sentence, style={"marginBottom": "4px"}),
                            html.Span(
                                f"sentiment {sentiment_val:+.2f} | confidence {confidence_val:.2f}",
                                style={
                                    "fontSize": "11px",
                                    "color": tone_color,
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    )
                )
            if not blocks:
                return html.Div(
                    "No non-neutral sentences identified.",
                    style={"fontSize": "12px", "color": "#666", "padding": "4px 0"},
                )
            return html.Ul(
                blocks,
                style={
                    "margin": "0",
                    "paddingLeft": "18px",
                    "lineHeight": "1.3",
                    "fontSize": "12px",
                },
            )

        year_elements = []
        for _, row in df_sql.sort_values("filing_year", ascending=False).iterrows():
            raw_non_neutral = (
                row.get("non_neutral_sentences_json")
                if "non_neutral_sentences_json" in row.index
                else None
            )
            non_neutral: dict = {"optimism": [], "skepticism": []}
            if isinstance(raw_non_neutral, str) and raw_non_neutral.strip():
                try:
                    parsed = json.loads(raw_non_neutral)
                    if isinstance(parsed, dict):
                        non_neutral["optimism"] = parsed.get("optimism", [])
                        non_neutral["skepticism"] = parsed.get("skepticism", [])
                except Exception:
                    pass

            year_elements.append(
                html.Details(
                    [
                        html.Summary(
                            f"{int(row['filing_year'])}",
                            style={
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "padding": "8px",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Optimism / Positive",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#0f766e",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        _render_non_neutral_list(
                                            non_neutral.get("optimism", []), "optimism"
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "minWidth": "280px",
                                        "padding": "8px",
                                        "backgroundColor": "#f0fdfa",
                                        "borderRadius": "4px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Skepticism / Negative",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#9f1239",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        _render_non_neutral_list(
                                            non_neutral.get("skepticism", []),
                                            "skepticism",
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "minWidth": "280px",
                                        "padding": "8px",
                                        "backgroundColor": "#fff1f2",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "flexWrap": "wrap",
                            },
                        ),
                    ],
                    style={"marginBottom": "12px"},
                )
            )

        container = (
            html.Div(year_elements)
            if year_elements
            else html.Div("No data available.", style={"color": "#999"})
        )
        return container, status

    @app.callback(
        Output("sentiment-trend-chart", "figure"),
        Input("trend-btn", "n_clicks"),
        State("ticker-input", "value"),
        prevent_initial_call=True,
    )
    def update_trend(n_clicks, ticker_val):
        ticker = (ticker_val or DEFAULT_TICKER).strip().upper()
        df = load_sqlite(ticker)
        empty_fig = go.Figure().update_layout(
            title=f"{ticker} — No sentiment data in DB",
            template="plotly_white",
            height=380,
        )
        if df.empty:
            return empty_fig

        df = df.copy()
        df["filing_year_int"] = df["filing_year"].astype(int)
        df = df.sort_values("filing_year_int")
        x_labels = df["filing_year_int"].astype(str).tolist()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df["sentiment"].tolist(),
                mode="lines+markers",
                name="Overall Sentiment",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.2f}<extra></extra>",
            )
        )
        if "mda_sentiment" in df.columns and df["mda_sentiment"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["mda_sentiment"].tolist(),
                    mode="lines+markers",
                    name="MD&A Sentiment",
                    line=dict(color="#10b981", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>MD&A: %{y:.2f}<extra></extra>",
                )
            )
        if "risk_sentiment" in df.columns and df["risk_sentiment"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["risk_sentiment"].tolist(),
                    mode="lines+markers",
                    name="Risk Sentiment",
                    line=dict(color="#ef4444", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>Risk: %{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"{ticker} — Sentiment Trend (from DB)",
            xaxis_title="Filing Year",
            yaxis_title="Sentiment (-1 to +1)",
            yaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinecolor="#cbd5e1"),
            template="plotly_white",
            height=380,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    @app.callback(
        [
            Output("non-neutral-10q-container", "children"),
            Output("10q-status-label", "children"),
        ],
        [
            Input("load-10q-db-btn", "n_clicks"),
            Input("load-10q-live-btn", "n_clicks"),
        ],
        [
            State("ticker-10q-input", "value"),
            State("quarters-slider", "value"),
        ],
        prevent_initial_call=True,
    )
    def update_10q_dashboard(db_clicks, live_clicks, ticker_val, num_quarters):
        ticker = (ticker_val or DEFAULT_TICKER).strip().upper()
        num_quarters = int(num_quarters or 4)
        triggered = ctx.triggered_id

        if triggered == "load-10q-live-btn":
            df = collect_edgar_10q(ticker, num_quarters)
            if df.empty:
                return (
                    html.Div(
                        "No 10-Q data returned from SEC.", style={"color": "#999"}
                    ),
                    "Live pull returned no data.",
                )
            upsert_sqlite_10q(df)
            status = f"Live pull complete — {len(df)} 10-Q filing(s) stored."
        else:
            status = "Loaded from DB."

        df_sql = load_sqlite_10q(ticker)
        if df_sql.empty:
            return (
                html.Div(
                    f"No cached 10-Q data for {ticker}. Click 'Pull Live 10-Q'.",
                    style={"color": "#999"},
                ),
                "No cached data.",
            )

        def _render_non_neutral_list(items: list[dict], tone: str):
            if not items:
                return html.Div(
                    "No non-neutral sentences identified.",
                    style={"fontSize": "12px", "color": "#666", "padding": "4px 0"},
                )
            tone_color = "#0f766e" if tone == "optimism" else "#9f1239"
            blocks = []
            for item in items:
                sentence = str(item.get("sentence", "")).strip()
                if not sentence:
                    continue
                sentiment_val = float(item.get("sentiment", 0.0))
                confidence_val = float(item.get("confidence", 0.0))
                blocks.append(
                    html.Li(
                        [
                            html.Div(sentence, style={"marginBottom": "4px"}),
                            html.Span(
                                f"sentiment {sentiment_val:+.2f} | confidence {confidence_val:.2f}",
                                style={
                                    "fontSize": "11px",
                                    "color": tone_color,
                                    "fontWeight": "bold",
                                },
                            ),
                        ],
                        style={"marginBottom": "8px"},
                    )
                )
            if not blocks:
                return html.Div(
                    "No non-neutral sentences identified.",
                    style={"fontSize": "12px", "color": "#666", "padding": "4px 0"},
                )
            return html.Ul(
                blocks,
                style={
                    "margin": "0",
                    "paddingLeft": "18px",
                    "lineHeight": "1.3",
                    "fontSize": "12px",
                },
            )

        period_elements = []
        for _, row in df_sql.sort_values("filing_date", ascending=False).iterrows():
            raw_non_neutral = (
                row.get("non_neutral_sentences_json")
                if "non_neutral_sentences_json" in row.index
                else None
            )
            non_neutral: dict = {"optimism": [], "skepticism": []}
            if isinstance(raw_non_neutral, str) and raw_non_neutral.strip():
                try:
                    parsed = json.loads(raw_non_neutral)
                    if isinstance(parsed, dict):
                        non_neutral["optimism"] = parsed.get("optimism", [])
                        non_neutral["skepticism"] = parsed.get("skepticism", [])
                except Exception:
                    pass

            label = str(row.get("period_label") or row.get("filing_date") or "Unknown")
            period_elements.append(
                html.Details(
                    [
                        html.Summary(
                            label,
                            style={
                                "cursor": "pointer",
                                "fontWeight": "bold",
                                "padding": "8px",
                                "marginBottom": "4px",
                            },
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div(
                                            "Optimism / Positive",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#0f766e",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        _render_non_neutral_list(
                                            non_neutral.get("optimism", []), "optimism"
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "minWidth": "280px",
                                        "padding": "8px",
                                        "backgroundColor": "#f0fdfa",
                                        "borderRadius": "4px",
                                    },
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            "Skepticism / Negative",
                                            style={
                                                "fontWeight": "bold",
                                                "color": "#9f1239",
                                                "marginBottom": "4px",
                                            },
                                        ),
                                        _render_non_neutral_list(
                                            non_neutral.get("skepticism", []),
                                            "skepticism",
                                        ),
                                    ],
                                    style={
                                        "flex": "1",
                                        "minWidth": "280px",
                                        "padding": "8px",
                                        "backgroundColor": "#fff1f2",
                                        "borderRadius": "4px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "gap": "10px",
                                "flexWrap": "wrap",
                            },
                        ),
                    ],
                    style={"marginBottom": "12px"},
                )
            )

        container = (
            html.Div(period_elements)
            if period_elements
            else html.Div("No data available.", style={"color": "#999"})
        )
        return container, status

    @app.callback(
        Output("sentiment-trend-10q-chart", "figure"),
        Input("trend-10q-btn", "n_clicks"),
        State("ticker-10q-input", "value"),
        prevent_initial_call=True,
    )
    def update_10q_trend(n_clicks, ticker_val):
        ticker = (ticker_val or DEFAULT_TICKER).strip().upper()
        df = load_sqlite_10q(ticker)
        empty_fig = go.Figure().update_layout(
            title=f"{ticker} — No 10-Q sentiment data in DB",
            template="plotly_white",
            height=380,
        )
        if df.empty:
            return empty_fig

        df = df.copy().sort_values("filing_date")
        x_labels = df["period_label"].tolist()

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=df["sentiment"].tolist(),
                mode="lines+markers",
                name="Overall Sentiment",
                line=dict(color="#3b82f6", width=3),
                marker=dict(size=8),
                hovertemplate="<b>%{x}</b><br>Sentiment: %{y:.2f}<extra></extra>",
            )
        )
        if "mda_sentiment" in df.columns and df["mda_sentiment"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["mda_sentiment"].tolist(),
                    mode="lines+markers",
                    name="MD&A Sentiment",
                    line=dict(color="#10b981", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>MD&A: %{y:.2f}<extra></extra>",
                )
            )
        if "risk_sentiment" in df.columns and df["risk_sentiment"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=x_labels,
                    y=df["risk_sentiment"].tolist(),
                    mode="lines+markers",
                    name="Risk Sentiment",
                    line=dict(color="#ef4444", width=2, dash="dot"),
                    marker=dict(size=6),
                    hovertemplate="<b>%{x}</b><br>Risk: %{y:.2f}<extra></extra>",
                )
            )
        fig.update_layout(
            title=f"{ticker} — 10-Q Sentiment Trend (from DB)",
            xaxis_title="Quarter",
            yaxis_title="Sentiment (-1 to +1)",
            yaxis=dict(range=[-1.05, 1.05], zeroline=True, zerolinecolor="#cbd5e1"),
            template="plotly_white",
            height=380,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        return fig

    @app.callback(
        [
            Output("model-status-label", "children"),
            Output("model-status-label", "style"),
        ],
        [Input("model-status-interval", "n_intervals")],
    )
    def update_model_status(_):
        state = _FINBERT_PRELOAD_STATE
        device = _get_active_finbert_device().upper()
        base_style = {"fontSize": "12px", "marginLeft": "4px", "fontWeight": "bold"}

        if state == "warming":
            return f"Model: warming... ({device})", {**base_style, "color": "#f59e0b"}
        if state == "ready":
            return f"Model: ready ({device})", {**base_style, "color": "#10b981"}
        if state == "failed":
            if _FINBERT_PRELOAD_ERROR:
                short_error = _FINBERT_PRELOAD_ERROR[:100]
                return (
                    f"Model: failed [{device}] ({short_error})",
                    {**base_style, "color": "#ef4444"},
                )
            return f"Model: failed ({device})", {**base_style, "color": "#ef4444"}

        return f"Model: idle ({device})", {**base_style, "color": "#64748b"}


register_callbacks(get_app())
