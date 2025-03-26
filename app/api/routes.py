from fastapi import APIRouter
from app.api.endpoints import auth, users, chat, documents, meta_prompt

api_router = APIRouter()

api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(users.router, prefix="/users", tags=["users"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(meta_prompt.router, prefix="/meta-prompt", tags=["meta-prompt"]) 