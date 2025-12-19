"""
TTS model initialization and management
"""

import os
import asyncio
import inspect
import importlib
from enum import Enum
from typing import Optional, Dict, Any
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from app.core.mtl import SUPPORTED_LANGUAGES
from app.config import Config, detect_device

# Global model instance
_model = None
_device = None
_initialization_state = "not_started"
_initialization_error = None
_initialization_progress = ""
_is_multilingual = None
_supported_languages = {}
_model_id = None
_model_variant = None


class InitializationState(Enum):
    NOT_STARTED = "not_started"
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"


async def initialize_model():
    """Initialize the Chatterbox TTS model"""
    global _model, _device, _initialization_state, _initialization_error, _initialization_progress, _is_multilingual, _supported_languages, _model_id, _model_variant
    
    try:
        _initialization_state = InitializationState.INITIALIZING.value
        _initialization_progress = "Validating configuration..."
        
        Config.validate()
        _device = detect_device()
        
        print(f"Initializing Chatterbox TTS model...")
        print(f"Device: {_device}")
        print(f"Voice sample: {Config.VOICE_SAMPLE_PATH}")
        print(f"Model cache: {Config.MODEL_CACHE_DIR}")

        model_id = Config.MODEL_ID.strip() if Config.MODEL_ID else ""
        model_id = model_id or None
        model_variant = _resolve_model_variant(model_id)
        _model_id = model_id or "default"
        _model_variant = model_variant
        if model_id:
            print(f"Model ID override: {model_id}")
        if model_variant != "default":
            print(f"Model variant: {model_variant}")
        
        _initialization_progress = "Creating model cache directory..."
        # Ensure model cache directory exists
        os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
        
        _initialization_progress = "Checking voice sample..."
        # Check voice sample exists
        if not os.path.exists(Config.VOICE_SAMPLE_PATH):
            raise FileNotFoundError(f"Voice sample not found: {Config.VOICE_SAMPLE_PATH}")
        
        _initialization_progress = "Configuring device compatibility..."
        # Patch torch.load for CPU compatibility if needed
        if _device == 'cpu':
            import torch
            original_load = torch.load
            original_load_file = None
            
            # Try to patch safetensors if available
            try:
                import safetensors.torch
                original_load_file = safetensors.torch.load_file
            except ImportError:
                pass
            
            def force_cpu_torch_load(f, map_location=None, **kwargs):
                # Always force CPU mapping if we're on a CPU device
                return original_load(f, map_location='cpu', **kwargs)
            
            def force_cpu_load_file(filename, device=None):
                # Force CPU for safetensors loading too
                return original_load_file(filename, device='cpu')
            
            torch.load = force_cpu_torch_load
            if original_load_file:
                safetensors.torch.load_file = force_cpu_load_file
        
        # Determine if we should use multilingual model
        use_multilingual = Config.USE_MULTILINGUAL_MODEL
        
        _initialization_progress = "Loading TTS model (this may take a while)..."
        _patch_perth_watermarker()
        # Initialize model with run_in_executor for non-blocking
        loop = asyncio.get_event_loop()
        
        if model_variant == "turbo":
            if use_multilingual:
                print("⚠ Turbo model does not support multilingual mode; forcing standard model settings.")
                use_multilingual = False
            print("Loading Chatterbox Turbo TTS model...")
            turbo_cls = _get_turbo_class()
            model_kwargs = _build_pretrained_kwargs(turbo_cls, model_id)
            model_kwargs["device"] = _device
            _model = await loop.run_in_executor(
                None,
                lambda: turbo_cls.from_pretrained(**model_kwargs)
            )
            _is_multilingual = False
            _supported_languages = {"en": "English"}
            print("✓ Turbo model initialized (English only)")
        elif use_multilingual:
            print(f"Loading Chatterbox Multilingual TTS model...")
            model_kwargs = _build_pretrained_kwargs(ChatterboxMultilingualTTS, model_id)
            model_kwargs["device"] = _device
            _model = await loop.run_in_executor(
                None, 
                lambda: ChatterboxMultilingualTTS.from_pretrained(**model_kwargs)
            )
            _is_multilingual = True
            _supported_languages = SUPPORTED_LANGUAGES.copy()
            print(f"✓ Multilingual model initialized with {len(_supported_languages)} languages")
        else:
            print(f"Loading standard Chatterbox TTS model...")
            model_kwargs = _build_pretrained_kwargs(ChatterboxTTS, model_id)
            model_kwargs["device"] = _device
            _model = await loop.run_in_executor(
                None, 
                lambda: ChatterboxTTS.from_pretrained(**model_kwargs)
            )
            _is_multilingual = False
            _supported_languages = {"en": "English"}  # Standard model only supports English
            print(f"✓ Standard model initialized (English only)")
        
        _initialization_state = InitializationState.READY.value
        _initialization_progress = "Model ready"
        _initialization_error = None
        print(f"✓ Model initialized successfully on {_device}")
        return _model
        
    except Exception as e:
        _initialization_state = InitializationState.ERROR.value
        _initialization_error = str(e)
        _initialization_progress = f"Failed: {str(e)}"
        print(f"✗ Failed to initialize model: {e}")
        raise e


def get_model():
    """Get the current model instance"""
    return _model


def get_model_id():
    """Get the current model ID"""
    if _model_id:
        return _model_id
    config_model_id = Config.MODEL_ID.strip() if Config.MODEL_ID else ""
    return config_model_id or "default"


def get_model_variant():
    """Get the current model variant"""
    if _model_variant:
        return _model_variant
    return _resolve_model_variant(get_model_id())


def get_device():
    """Get the current device"""
    return _device


def get_initialization_state():
    """Get the current initialization state"""
    return _initialization_state


def get_initialization_progress():
    """Get the current initialization progress message"""
    return _initialization_progress


def get_initialization_error():
    """Get the initialization error if any"""
    return _initialization_error


def is_ready():
    """Check if the model is ready for use"""
    return _initialization_state == InitializationState.READY.value and _model is not None


def is_initializing():
    """Check if the model is currently initializing"""
    return _initialization_state == InitializationState.INITIALIZING.value 


def is_multilingual():
    """Check if the loaded model supports multilingual generation"""
    return _is_multilingual


def get_supported_languages():
    """Get the dictionary of supported languages"""
    return _supported_languages.copy()


def supports_language(language_id: str):
    """Check if the model supports a specific language"""
    return language_id in _supported_languages


def get_model_info() -> Dict[str, Any]:
    """Get comprehensive model information"""
    return {
        "model_type": "multilingual" if _is_multilingual else "standard",
        "is_multilingual": _is_multilingual,
        "supported_languages": _supported_languages,
        "language_count": len(_supported_languages),
        "device": _device,
        "model_id": _model_id,
        "model_variant": _model_variant,
        "is_ready": is_ready(),
        "initialization_state": _initialization_state
    }


def _build_pretrained_kwargs(model_cls, model_id: Optional[str]) -> Dict[str, Any]:
    if not model_id:
        return {}

    signature = inspect.signature(model_cls.from_pretrained)
    for param_name in ("model_id", "model_name", "repo_id", "hf_model_id"):
        if param_name in signature.parameters:
            return {param_name: model_id}

    if any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    ):
        return {"model_id": model_id}

    if _apply_model_id_override(model_cls, model_id):
        print(f"Model repo override applied via module REPO_ID: {model_id}")
        return {}

    print(
        "⚠ Model ID override is set but this chatterbox-tts version "
        "does not accept a model ID or REPO_ID override; using default weights."
    )
    return {}


def _apply_model_id_override(model_cls, model_id: str) -> bool:
    try:
        module = importlib.import_module(model_cls.__module__)
    except Exception:
        return False

    if hasattr(module, "REPO_ID"):
        setattr(module, "REPO_ID", model_id)
        return True

    return False


def _patch_perth_watermarker() -> None:
    try:
        import perth
    except Exception:
        return

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        perth.PerthImplicitWatermarker = getattr(perth, "DummyWatermarker", None)
        print("⚠ Perth watermarker not available; falling back to DummyWatermarker.")


def _resolve_model_variant(model_id: Optional[str]) -> str:
    config_variant = (Config.MODEL_VARIANT or "").strip().lower()
    if config_variant and config_variant != "auto":
        return config_variant

    if model_id and "turbo" in model_id.lower():
        return "turbo"

    return "default"


def _get_turbo_class():
    try:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
    except ImportError as exc:
        raise ImportError(
            "ChatterboxTurboTTS is not available in this chatterbox-tts install. "
            "Install the Turbo-capable version of chatterbox-tts to use MODEL_ID with turbo."
        ) from exc

    return ChatterboxTurboTTS
