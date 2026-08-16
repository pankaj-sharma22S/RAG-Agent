from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from urllib.parse import urlparse, parse_qs
from langsmith import traceable
@traceable(
    metadata={
        "module": "loader",
        "tool":"youtube_loader"
    }
)
def extract_video_id(url: str):
    if "youtube.com" not in url and "youtu.be" not in url:
        return url

    parsed = urlparse(url)

    hostname = (parsed.hostname or "").lower()

    if hostname in [
        "youtube.com",
        "www.youtube.com",
        "m.youtube.com",
    ]:
        query_video_id = parse_qs(parsed.query).get("v", [None])[0]
        if query_video_id:
            return query_video_id

        # Also support share URLs such as /shorts/<id> and /embed/<id>.
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]

        raise ValueError("YouTube URL does not contain a video ID")


    elif hostname == "youtu.be":

        return parsed.path[1:]

    else:
        raise ValueError(
            "Invalid YouTube URL"
        )

def load_youtube(url: str):

    try:

        video_id = extract_video_id(url)
        transcript = YouTubeTranscriptApi().fetch(video_id)


        text = " ".join(
            item.text
            for item in transcript.snippets
        )


        if not text.strip():

            raise ValueError(
                "Transcript empty"
            )


        document = Document(

            page_content=text,

            metadata={
                "source": url,
                "type": "youtube",
                "video_id": video_id
            }
        )


        return document


    except Exception as e:

        raise ValueError(
            f"YouTube transcript extraction failed: {e}"
        )
