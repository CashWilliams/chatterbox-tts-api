"""
Model listing endpoints (OpenAI compatibility)
"""

from fastapi import APIRouter

from app.models import ModelsResponse, ModelInfo
from app.core import add_route_aliases
from app.core.tts_model import get_model_id

# Create router with aliasing support
base_router = APIRouter()
router = add_route_aliases(base_router)


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="List models",
    description="List available models (OpenAI API compatibility)"
)
async def list_models():
    """List available models (OpenAI API compatibility)"""
    configured_model_id = get_model_id()
    model_id = _normalize_model_id(configured_model_id)
    return ModelsResponse(
        object="list",
        data=[
            ModelInfo(
                id=model_id,
                object="model", 
                created=1677649963,
                owned_by="resemble-ai"
            )
        ]
    )


def _normalize_model_id(model_id: str) -> str:
    if not model_id or model_id == "default":
        return "chatterbox-tts-1"

    normalized = model_id.lower()
    if "chatterbox-turbo" in normalized:
        return "chatterbox-turbo"

    return model_id

# Export the base router for the main app to use
__all__ = ["base_router"] 
