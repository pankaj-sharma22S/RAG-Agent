import whisper
from core.document import create_document

model = whisper.load_model(
    "base"
)
def load_audio(file_path: str):

    try:

        result = model.transcribe(
            file_path
        )


        text = result["text"]


        if not text.strip():

            raise ValueError(
                "No speech detected"
            )


        return create_document(
            
                    text,
            
                    {
                        "source": file_path,
                        "type": "audio"
                    }
            
                )


    except Exception as e:

        raise ValueError(
            f"Audio processing failed: {e}"
        )
