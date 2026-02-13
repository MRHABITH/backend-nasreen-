from dotenv import load_dotenv
import os

# Load env vars before any other imports to ensure they are available
load_dotenv()

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.routes import api

app = FastAPI(
    title="AI-Based Plagiarism Detection System",
    description="A real-time plagiarism detection system using NLP and Sentence-BERT.",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi import Request
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url} from {request.client.host}")
    response = await call_next(request)
    return response

# Include API Routes
app.include_router(api.router)

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("path/to/favicon.ico") if os.path.exists("path/to/favicon.ico") else {"message": "No favicon"}

@app.get("/")
async def root():
    return {"message": "Welcome to the AI-Based Plagiarism Detection System API"}
