"""
@title: IndexTTS-2.5 for ComfyUI
@author: ComfyUI-Index-TTS
@description: IndexTTS-2.5 专用 ComfyUI 节点：零样本音色克隆、多语言、语速控制、发音标注、情感控制、音色持久化、后训练 GPT checkpoint 切换
"""

# IndexTTS-2.5 节点（多语言 / 语速控制 / 发音标注 / 情感控制）
from .index_tts2_5_nodes import (
    IndexTTS25BaseNode,
    IndexTTS25EmotionAudioNode,
    IndexTTS25EmotionVectorNode,
    IndexTTS25EmotionTextNode,
    IndexTTS25CacheControlNode,
    IndexTTS25GPTCheckpointSelectorNode,
)
# IndexTTS-2.5 持久化音色 conditioning 节点
from .index_tts2_5_voice_nodes import (
    IndexTTS25ExtractVoiceNode,
    IndexTTS25SaveVoiceNode,
    IndexTTS25LoadVoiceNode,
    IndexTTS25VoiceBaseNode,
)

# 注册ComfyUI节点
NODE_CLASS_MAPPINGS = {
    "IndexTTS25BaseNode": IndexTTS25BaseNode,
    "IndexTTS25EmotionAudioNode": IndexTTS25EmotionAudioNode,
    "IndexTTS25EmotionVectorNode": IndexTTS25EmotionVectorNode,
    "IndexTTS25EmotionTextNode": IndexTTS25EmotionTextNode,
    "IndexTTS25CacheControlNode": IndexTTS25CacheControlNode,
    "IndexTTS25GPTCheckpointSelectorNode": IndexTTS25GPTCheckpointSelectorNode,
    "IndexTTS25ExtractVoiceNode": IndexTTS25ExtractVoiceNode,
    "IndexTTS25SaveVoiceNode": IndexTTS25SaveVoiceNode,
    "IndexTTS25LoadVoiceNode": IndexTTS25LoadVoiceNode,
    "IndexTTS25VoiceBaseNode": IndexTTS25VoiceBaseNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IndexTTS25BaseNode": "Index TTS 2.5 - Base (多语言/语速)",
    "IndexTTS25EmotionAudioNode": "Index TTS 2.5 - Emotion Audio",
    "IndexTTS25EmotionVectorNode": "Index TTS 2.5 - Emotion Vector",
    "IndexTTS25EmotionTextNode": "Index TTS 2.5 - Emotion Text",
    "IndexTTS25CacheControlNode": "Index TTS 2.5 - Cache Control",
    "IndexTTS25GPTCheckpointSelectorNode": "Index TTS 2.5 - GPT Checkpoint Selector",
    "IndexTTS25ExtractVoiceNode": "Index TTS 2.5 - Extract Voice",
    "IndexTTS25SaveVoiceNode": "Index TTS 2.5 - Save Voice",
    "IndexTTS25LoadVoiceNode": "Index TTS 2.5 - Load Voice",
    "IndexTTS25VoiceBaseNode": "Index TTS 2.5 - Voice Base",
}

# 前端扩展：为 Load Voice 提供 .pt 文件选择/上传和动态刷新。
WEB_DIRECTORY = "./web"

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS', 'WEB_DIRECTORY']
