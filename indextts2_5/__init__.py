"""
IndexTTS-2.5 integration package for ComfyUI
提供 IndexTTS-2.5 的模型加载与推理封装（多语言、情感控制、语速控制）。
"""

from .model_loader import IndexTTS25Loader, DEFAULT_GPT_CHOICE, GPT_CHECKPOINT_DIRNAME
from .infer import IndexTTS25Engine
from .comfy_integration import install_free_hook

__all__ = [
    "IndexTTS25Loader",
    "IndexTTS25Engine",
    "DEFAULT_GPT_CHOICE",
    "GPT_CHECKPOINT_DIRNAME",
    "install_free_hook",
]
