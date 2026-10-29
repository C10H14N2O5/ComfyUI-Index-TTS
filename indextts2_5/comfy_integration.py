"""
Hook IndexTTS-2.5 into ComfyUI's model unloading.

ComfyUI's ``POST /free`` (and the "Unload Models" button) only sets queue flags; the prompt
worker then calls ``comfy.model_management.unload_all_models()``, which frees models registered
in ``current_loaded_models``. Our IndexTTS2 instance is held by the plugin's own loader and was
never freed. We wrap ``unload_all_models`` so it also releases the loader's instance.

``main.py`` looks the function up on the module at call time, so replacing the module attribute
is enough. Unloading runs on the prompt worker thread, i.e. never concurrently with a node.
``free_memory(memory_required, ...)`` is intentionally NOT hooked: it runs whenever other models
need VRAM, and evicting TTS there would change behaviour of unrelated workflows.
"""
import functools

_HOOK_MARKER = "_indextts25_free_hook"


def install_free_hook(loader) -> bool:
    """Install once; returns True when the hook is active. No-op outside ComfyUI."""
    try:
        import comfy.model_management as mm
    except Exception:
        return False

    original = getattr(mm, "unload_all_models", None)
    if original is None:
        print("[IndexTTS2.5] comfy.model_management.unload_all_models not found; /free hook not installed")
        return False
    if getattr(original, _HOOK_MARKER, False):
        return True

    @functools.wraps(original)
    def unload_all_models(*args, **kwargs):
        result = original(*args, **kwargs)
        try:
            loader.unload_tts(reason="ComfyUI /free")
        except Exception as e:  # never break ComfyUI's own unloading
            print(f"[IndexTTS2.5] /free hook failed to unload: {e!r}")
        return result

    setattr(unload_all_models, _HOOK_MARKER, True)
    mm.unload_all_models = unload_all_models
    print("[IndexTTS2.5] /free hook installed (ComfyUI unload_all_models also releases IndexTTS-2.5)")
    return True
