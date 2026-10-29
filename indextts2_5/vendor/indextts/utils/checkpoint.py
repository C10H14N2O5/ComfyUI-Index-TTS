# Copyright (c) 2020 Mobvoi Inc. (authors: Binbin Zhang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path

import torch
import yaml


def load_checkpoint(model: torch.nn.Module, model_pth: str) -> dict:
    checkpoint = torch.load(model_pth, map_location='cpu')
    checkpoint = checkpoint['model'] if 'model' in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(checkpoint, strict=False)
    if missing:
        print(f">> load_checkpoint: missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f">> load_checkpoint: skipping unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    info_path = str(Path(model_pth).with_suffix('.yaml'))
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


# [ComfyUI-Index-TTS patch] 后训练 GPT checkpoint 的严格加载
_WRAPPER_KEYS = ("model", "state_dict")


def _unwrap_state_dict(obj):
    """Peel {"model": ...} / {"state_dict": ...} wrappers until a flat tensor dict remains."""
    while isinstance(obj, dict):
        inner = None
        for key in _WRAPPER_KEYS:
            if key in obj and isinstance(obj[key], dict):
                inner = obj[key]
                break
        if inner is None:
            break
        obj = inner
    if not isinstance(obj, dict) or not obj:
        raise RuntimeError("checkpoint does not contain a state dict")
    if all(isinstance(k, str) and k.startswith("module.") for k in obj):
        obj = {k[len("module."):]: v for k, v in obj.items()}
    return obj


def load_gpt_checkpoint_strict(model: torch.nn.Module, model_pth: str) -> None:
    """
    Load a (fine-tuned) GPT checkpoint and fail loudly when weights are missing.

    - Accepts plain state dicts, {"model": sd}, {"state_dict": sd} (nested) and 'module.' prefixes,
      e.g. the merged output of index-tts25-lora-finetune (torch.save({"model": state_dict})).
    - Any missing *parameter* raises (LoRA-only deltas or a mismatched architecture would
      otherwise silently produce noise). Unexpected keys (e.g. 'inference_model.*') only warn.
    """
    name = os.path.basename(model_pth)
    try:
        checkpoint = torch.load(model_pth, map_location='cpu', weights_only=True)
    except Exception as e:
        raise RuntimeError(
            f"[IndexTTS2.5] 无法读取 GPT checkpoint '{name}'（需要是只含张量的权重文件）: {e}"
        ) from e
    try:
        state_dict = _unwrap_state_dict(checkpoint)
    except RuntimeError as e:
        raise RuntimeError(f"[IndexTTS2.5] GPT checkpoint '{name}' 格式不正确: {e}") from e

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    param_names = {n for n, _ in model.named_parameters()}
    missing_params = [k for k in missing if k in param_names]
    if missing_params:
        raise RuntimeError(
            f"[IndexTTS2.5] GPT checkpoint '{name}' 缺少 {len(missing_params)} 个权重，"
            f"例如: {missing_params[:10]}。可能是 LoRA 增量权重（需先合并）或模型结构不匹配。"
        )
    if unexpected:
        print(f">> load_gpt_checkpoint_strict: ignoring {len(unexpected)} unexpected keys in {name}: "
              f"{unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    del state_dict, checkpoint
