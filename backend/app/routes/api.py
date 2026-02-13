from fastapi import APIRouter
from app.routes import check

router = APIRouter()

router.include_router(check.router, prefix="/api", tags=["Plagiarism Check"])
