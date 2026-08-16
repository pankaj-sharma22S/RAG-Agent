"""Safe YouTube transcript extractor with multi-language fallback and untrusted boundary wrapping."""

from __future__ import annotations

import logging
from urllib.parse import parse_qs, urlparse
from langchain_core.documents import Document
from youtube_transcript_api import YouTubeTranscriptApi

try:
    from app.core.document import create_document
    from app.guardrails.prompt_injection import global_injection_guard
    from app.tracing import safe_traceable
except ModuleNotFoundError:
    from core.document import create_document
    from guardrails.prompt_injection import global_injection_guard
    from tracing import safe_traceable

logger = logging.getLogger("YouTubeLoader")


def extract_video_id(url: str) -> str:
    """Extract 11-character video ID from any valid YouTube URL format."""
    if not url or not url.strip():
        raise ValueError("Please provide a valid YouTube URL.")

    url_str = url.strip()
    if "youtube.com" not in url_str and "youtu.be" not in url_str:
        # If user passed raw video id
        if len(url_str) == 11 and not url_str.startswith("http"):
            return url_str
        return url_str

    parsed = urlparse(url_str)
    hostname = (parsed.hostname or "").lower()

    if hostname in ["youtube.com", "www.youtube.com", "m.youtube.com"]:
        # Standard watch URL
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id:
            return query_video_id

        # Shorts / embed / live URLs
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live", "v"}:
            return path_parts[1]
        elif len(path_parts) == 1:
            return path_parts[0]

        raise ValueError("YouTube URL does not contain a valid video ID.")

    elif hostname in ["youtu.be", "www.youtu.be"]:
        path_clean = parsed.path.lstrip("/")
        if path_clean:
            return path_clean.split("?")[0]
        raise ValueError("Short YouTube URL (youtu.be) does not contain a video ID.")

    else:
        raise ValueError(f"Invalid YouTube hostname: '{hostname}'.")


def extract_timestamp(url: str) -> int | None:
    """Extract start timestamp in seconds from YouTube URL if present (e.g. &t=4036s or ?t=120)."""
    try:
        parsed = urlparse(url)
        t_param = parse_qs(parsed.query).get("t", [None])[0]
        if not t_param and parsed.fragment:
            t_param = parse_qs(parsed.fragment).get("t", [None])[0]
        if t_param:
            t_str = t_param.rstrip("sS")
            if t_str.isdigit():
                return int(t_str)
    except Exception:
        pass
    return None


@safe_traceable(name="Load YouTube Transcript", run_type="chain", tags=["rag-agent", "youtube", "loader"])
def load_youtube(url: str) -> Document:
    """Download video transcripts safely with multi-language fallback, timestamp parsing, and wrap as untrusted data."""
    video_id = extract_video_id(url)
    timestamp_sec = extract_timestamp(url)
    api = YouTubeTranscriptApi()

    snippets_text: str = ""
    error_reasons: list[str] = []

    # Attempt 1: Standard English transcript
    try:
        fetched = api.fetch(video_id, languages=["en", "en-US", "en-GB"])
        if hasattr(fetched, "snippets"):
            snippets_text = " ".join(item.text for item in fetched.snippets if hasattr(item, "text"))
    except Exception as e1:
        error_reasons.append(str(e1))

    # Attempt 2: Fallback across any available transcript (including auto-generated)
    if not snippets_text.strip():
        try:
            transcript_list = api.list(video_id)
            target_transcript = None

            # Look for English first
            try:
                target_transcript = transcript_list.find_transcript(["en", "en-US", "en-GB"])
            except Exception:
                pass

            # If not found, take the first available transcript
            if not target_transcript:
                for t in transcript_list:
                    target_transcript = t
                    break

            if target_transcript:
                # If non-English, try translating to English if available
                if target_transcript.language_code not in {"en", "en-US", "en-GB"} and target_transcript.is_translatable:
                    try:
                        target_transcript = target_transcript.translate("en")
                    except Exception:
                        pass

                fetched_data = target_transcript.fetch()
                if hasattr(fetched_data, "snippets"):
                    snippets_text = " ".join(item.text for item in fetched_data.snippets if hasattr(item, "text"))
                elif isinstance(fetched_data, list):
                    snippets_text = " ".join(
                        (item.text if hasattr(item, "text") else item.get("text", "")) for item in fetched_data
                    )
        except Exception as e2:
            error_reasons.append(str(e2))

    if not snippets_text.strip():
        raise ValueError(
            f"YouTube transcript unavailable for video '{video_id}'. "
            f"The video may have captions disabled, be private, or be region-restricted. "
            f"(Details: {'; '.join(error_reasons[:1])})"
        )

    # 1. Content Security Scan: Redact passwords, API keys, private keys, connection strings, PII
    try:
        from guardrails.pii import global_pii_guard
    except ModuleNotFoundError:
        from app.guardrails.pii import global_pii_guard
    pii_res = global_pii_guard.scan_and_mask(snippets_text)
    sanitized_snippets = pii_res.sanitized_text

    # 2. Wrap transcript in untrusted boundary envelope
    wrapped_text = global_injection_guard.wrap_untrusted_content(
        sanitized_snippets, source_name=f"YouTube Video ({video_id})", content_type="youtube_transcript"
    )

    meta: dict[str, Any] = {
        "source": url,
        "source_url": url,
        "source_id": video_id,
        "source_type": "youtube",
        "type": "youtube",
        "video_id": video_id,
        "title": f"YouTube Video {video_id}",
    }
    if timestamp_sec is not None:
        meta["timestamp"] = timestamp_sec

    return create_document(wrapped_text, meta)
