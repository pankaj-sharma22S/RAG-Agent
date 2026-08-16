import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

print("Gemini key loaded:", api_key is not None)

llm = ChatGoogleGenerativeAI(

    model="gemini-2.5-flash",

    temperature=0.2,

    api_key=os.getenv(
        "GEMINI_API_KEY"
    )

)