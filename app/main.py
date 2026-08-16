from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.upload import router as upload_router


app = FastAPI(
    title="Knowledge AI Backend"
)

# The upload page is commonly served by a separate local frontend during
# development.  Without CORS, browsers block its multipart upload before the
# API receives the selected file.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(
    upload_router,
    prefix="/api"
)


@app.get("/")
def home():

    return {
        "message":"Backend Running"
    }
