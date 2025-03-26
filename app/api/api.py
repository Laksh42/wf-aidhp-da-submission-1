from fastapi import APIRouter

from app.api import auth, chat, recommendations, images

api_router = APIRouter()

# Include all API routers
api_router.include_router(auth.router)
api_router.include_router(chat.router)
api_router.include_router(recommendations.router)
api_router.include_router(images.router) 