import subprocess
import os
import whisper
from core.document import create_document

_model = None


def get_model():
    """Load Whisper only if a video is actually being indexed."""
    global _model
    if _model is None:
        _model = whisper.load_model("base")
    return _model

from langsmith import traceable

@traceable(
    metadata={
        "module": "audio",
        "tool":"whisper"
    }
)
def extract_audio(video_path: str):

    audio_path = video_path.rsplit(".", 1)[0] + ".mp3"


    command = [
        "ffmpeg",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "mp3",
        audio_path,
        "-y"
    ]


    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {completed.stderr.strip()}")


    return audio_path



def load_video(file_path: str):

    try:

        # 1. Extract audio
        audio_file = extract_audio(
            file_path
        )


        # 2. Transcribe
        result = get_model().transcribe(
            audio_file
        )


        text = result["text"]


        if not text.strip():

            raise ValueError(
                "No speech detected in video"
            )


        return create_document(
            
                    text,
            
                    {
                        "source": file_path,
                        "type": "video"
                    }
            
                )


    except Exception as e:

        raise ValueError(
            f"Video processing failed: {e}"
        )
    finally:
        # Do not leave extracted audio files beside the user's uploads.
        if "audio_file" in locals() and os.path.exists(audio_file):
            os.remove(audio_file)
