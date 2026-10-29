import os
import sys
import gc
import threading
import torch
import yaml
from typing import Optional, List, Tuple


GPT_CHECKPOINT_DIRNAME = "gpt_checkpoints"
DEFAULT_GPT_CHOICE = "(config default)"


class IndexTTS25Loader:
    """
    Lightweight model manager for IndexTTS-2.5.
    - Resolves model root: <ComfyUI>/models/IndexTTS-2.5
    - Validates required files
    - Keeps at most ONE IndexTTS2 (v2.5) instance alive, keyed by the GPT checkpoint
      (path + mtime + size). Same checkpoint -> reuse; different -> unload old, then load new.
    - QwenEmotion (Emotion Text) is attached lazily to the current instance and released with it.
    - Post-trained GPT checkpoints live in <model_dir>/gpt_checkpoints/**.pth
    """

    DEFAULT_DIRNAME = "IndexTTS-2.5"

    def __init__(self, models_root: Optional[str] = None, device: Optional[str] = None, dtype: Optional[str] = None):
        self._models_root = models_root or self._default_models_root()
        self._model_dir = os.path.join(self._models_root, self.DEFAULT_DIRNAME)
        self._device = torch.device(device) if device else (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        # IndexTTS-2.5 官方推荐 CUDA 下用 bf16（更快更省显存），CPU 下强制 fp32
        if dtype is None and self._device.type == "cuda":
            dtype = "bf16"
        self._dtype = self._resolve_dtype(dtype)
        self._lock = threading.RLock()
        self._tts = None
        self._tts_key: Optional[Tuple[str, int, int]] = None
        # vendored IndexTTS-2.5 source: indextts2_5/vendor/indextts
        self._vendor_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
        self._vendor_pkg_root = os.path.join(self._vendor_root, "indextts")
        if os.path.isdir(self._vendor_root):
            try:
                sys.path.remove(self._vendor_root)
            except ValueError:
                pass
            sys.path.insert(0, self._vendor_root)

    @staticmethod
    def _default_models_root() -> str:
        # <repo>/ComfyUI/models
        # This file is at: .../ComfyUI/custom_nodes/ComfyUI-Index-TTS/indextts2_5/model_loader.py
        # Go up 4 levels to reach .../ComfyUI/
        here = os.path.abspath(__file__)
        for _ in range(4):
            here = os.path.dirname(here)
        return os.path.join(here, "models")

    @staticmethod
    def _resolve_dtype(dtype: Optional[str]):
        if isinstance(dtype, torch.dtype):
            return dtype
        if dtype == "fp16":
            return torch.float16
        if dtype == "bf16":
            return torch.bfloat16
        return torch.float32

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def model_dir(self):
        return self._model_dir

    @property
    def gpt_checkpoint_dir(self) -> str:
        return os.path.join(self._model_dir, GPT_CHECKPOINT_DIRNAME)

    @property
    def current_gpt_checkpoint(self) -> Optional[str]:
        """Absolute path of the GPT checkpoint of the loaded instance (None when nothing is loaded)."""
        return self._tts_key[0] if self._tts_key else None

    # ------------------------------------------------------------------ checkpoints
    def list_gpt_checkpoints(self) -> List[str]:
        """Relative posix paths of every *.pth below gpt_checkpoints/ (recursive, sorted)."""
        root = self.gpt_checkpoint_dir
        if not os.path.isdir(root):
            return []
        found = []
        for dirpath, _dirnames, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".pth"):
                    rel = os.path.relpath(os.path.join(dirpath, fn), root)
                    found.append(rel.replace(os.sep, "/"))
        return sorted(found, key=str.lower)

    def resolve_gpt_checkpoint(self, name: Optional[str]) -> Optional[str]:
        """
        Map a Selector choice to an absolute checkpoint path.
        None / "" / DEFAULT_GPT_CHOICE -> None (use config.yaml's gpt_checkpoint).
        """
        if name is None:
            return None
        name = str(name).strip()
        if not name or name == DEFAULT_GPT_CHOICE:
            return None
        root = os.path.realpath(self.gpt_checkpoint_dir)
        path = os.path.realpath(os.path.join(root, name))
        try:
            inside = os.path.commonpath([root, path]) == root and path != root
        except ValueError:  # different drives on Windows
            inside = False
        if not inside:
            raise ValueError(f"[IndexTTS2.5] GPT checkpoint 必须位于 {root} 目录内: {name}")
        if not path.lower().endswith(".pth"):
            raise ValueError(f"[IndexTTS2.5] GPT checkpoint 必须是 .pth 文件: {name}")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"[IndexTTS2.5] 找不到 GPT checkpoint: {path}")
        return path

    def _default_gpt_path(self) -> str:
        cfg_path = os.path.join(self._model_dir, "config.yaml")
        name = "gpt.pth"
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            name = cfg.get("gpt_checkpoint") or name
        except Exception:
            pass
        return os.path.realpath(os.path.join(self._model_dir, name))

    def _checkpoint_key(self, gpt_checkpoint: Optional[str]) -> Tuple[str, int, int]:
        path = os.path.realpath(gpt_checkpoint) if gpt_checkpoint else self._default_gpt_path()
        try:
            st = os.stat(path)
            return (path, st.st_mtime_ns, st.st_size)
        except OSError:
            return (path, -1, -1)

    def validate(self, gpt_checkpoint: Optional[str] = None) -> None:
        required = [
            "config.yaml",
            "feat1.pt",
            "feat2.pt",
            "codec.pth",
            "s2mel.pth",
            "wav2vec2bert_stats.pt",
            "multilingual_zh_ja_yue_char_del.tiktoken",
        ]
        missing = [f for f in required if not os.path.exists(os.path.join(self._model_dir, f))]
        gpt_path = gpt_checkpoint or self._default_gpt_path()
        if not os.path.exists(gpt_path):
            missing.append(os.path.relpath(gpt_path, self._model_dir) if not gpt_checkpoint else gpt_path)
        if missing:
            raise FileNotFoundError(f"IndexTTS-2.5 missing files in {self._model_dir}: {', '.join(missing)}")

    # ------------------------------------------------------------------ instances
    def _import_indextts2(self):
        try:
            # Purge any previously imported 'indextts' top-level modules to avoid
            # shadowing by other packages/custom nodes that also ship an 'indextts' package
            for k in list(sys.modules.keys()):
                if k == "indextts" or k.startswith("indextts."):
                    sys.modules.pop(k, None)
            if os.path.isdir(self._vendor_root):
                try:
                    sys.path.remove(self._vendor_root)
                except ValueError:
                    pass
                sys.path.insert(0, self._vendor_root)
            from indextts.infer_v2_5 import IndexTTS2  # imported from vendored 2.5 package
            return IndexTTS2
        except Exception as e:
            # Fallback: import by file path to avoid package name collisions
            try:
                import importlib.util
                infer_path = os.path.join(self._vendor_pkg_root, "infer_v2_5.py")
                spec = importlib.util.spec_from_file_location("indextts_infer_v2_5", infer_path)
                if spec is None or spec.loader is None:
                    raise ImportError(f"spec load failed for {infer_path}")
                mod = importlib.util.module_from_spec(spec)
                sys.modules["indextts_infer_v2_5"] = mod
                spec.loader.exec_module(mod)
                return getattr(mod, "IndexTTS2")
            except Exception as e2:
                raise ImportError(
                    f"Failed to import IndexTTS2 (v2.5) from vendored source at {self._vendor_pkg_root}. Error: {e}. Fallback failed: {e2}. "
                    "Ensure project dependencies (transformers, modelscope, librosa, tiktoken, omegaconf, torchaudio, safetensors, etc.) are installed."
                )

    @staticmethod
    def _attach_qwen(tts) -> None:
        """Lazily load QwenEmotion onto an existing instance (Emotion Text)."""
        if getattr(tts, "qwen_emo", None) is not None:
            return
        mod = sys.modules.get(type(tts).__module__)
        qwen_cls = getattr(mod, "QwenEmotion", None) if mod else None
        if qwen_cls is None:
            raise ImportError("[IndexTTS2.5] QwenEmotion class not found in vendored infer_v2_5")
        qwen_dir = os.path.join(tts.model_dir, tts.cfg.qwen_emo_path)
        print(f"[IndexTTS2.5] loading QwenEmotion from {qwen_dir}")
        tts.qwen_emo = qwen_cls(qwen_dir)

    @staticmethod
    def _label(key: Optional[Tuple[str, int, int]]) -> str:
        return os.path.basename(key[0]) if key else "<none>"

    def get_tts(self, use_qwen_emo: bool = False, gpt_checkpoint: Optional[str] = None):
        """
        Return the IndexTTS2 (v2.5) instance for the requested GPT checkpoint.
        gpt_checkpoint: absolute path (from resolve_gpt_checkpoint) or None for config default.
        use_qwen_emo=True ensures QwenEmotion is attached (Emotion Text).
        """
        with self._lock:
            key = self._checkpoint_key(gpt_checkpoint)
            if self._tts is not None and key == self._tts_key:
                print(f"[IndexTTS2.5] reuse cached instance (gpt: {self._label(key)})")
                tts = self._tts
            else:
                if self._tts is not None:
                    print(f"[IndexTTS2.5] switching gpt checkpoint {self._label(self._tts_key)} -> {self._label(key)}")
                    self.unload_tts(reason="checkpoint switch")
                self.validate(gpt_checkpoint)
                IndexTTS2 = self._import_indextts2()
                cfg_path = os.path.join(self._model_dir, "config.yaml")
                print(f"[IndexTTS2.5] loading instance (gpt: {key[0]})")
                tts = IndexTTS2(
                    cfg_path=cfg_path,
                    model_dir=self._model_dir,
                    use_bf16=(self._dtype == torch.bfloat16),
                    device=str(self._device),
                    # Windows 下自定义 CUDA kernel 需要 JIT 编译，默认关闭，自动回退到 torch 实现
                    use_cuda_kernel=False,
                    use_deepspeed=False,
                    use_qwen_emo=use_qwen_emo,
                    gpt_checkpoint=gpt_checkpoint,
                )
                self._tts = tts
                self._tts_key = key
            if use_qwen_emo:
                self._attach_qwen(tts)
            return tts

    def get_current_or_default_tts(self):
        """Current instance if any (no checkpoint switch), otherwise load the config default."""
        with self._lock:
            if self._tts is not None:
                return self._tts
            return self.get_tts(use_qwen_emo=False, gpt_checkpoint=None)

    def unload_tts(self, reason: str = "") -> None:
        """
        Release the cached instance (incl. QwenEmotion) and free GPU cache.
        Safe to call even if nothing is loaded.
        """
        with self._lock:
            tts, label = self._tts, self._label(self._tts_key)
            self._tts = None
            self._tts_key = None
            if tts is not None:
                try:
                    tts.qwen_emo = None
                except Exception:
                    pass
                del tts
                print(f"[IndexTTS2.5] unloaded (gpt: {label}{', ' + reason if reason else ''})")
            try:
                gc.collect()
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            except Exception:
                pass
