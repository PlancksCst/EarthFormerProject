"""Official EarthFormer backbone with minimal SEVIRI compatibility changes."""

from __future__ import annotations

import os
import sys
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


_THIS_DIR = Path(__file__).resolve().parent
_PREP_MODELS_DIR = _THIS_DIR.parent
_OFFICIAL_SRC_DIR = _PREP_MODELS_DIR / "earthformer_src" / "src"
if str(_OFFICIAL_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_SRC_DIR))

from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel  # noqa: E402


OFFICIAL_SEVIR_CHECKPOINT_URL = (
    "https://earthformer.s3.amazonaws.com/pretrained_checkpoints/earthformer_sevir.pt"
)
DEFAULT_CHECKPOINT_PATH = (
    _PREP_MODELS_DIR
    / "earthformer_src"
    / "pretrained_checkpoints"
    / "earthformer_sevir.pt"
)


@dataclass
class WeightLoadReport:
    """Summary of how the official SEVIR checkpoint was loaded."""

    checkpoint_path: str
    loaded_keys: int = 0
    adapted_keys: list[str] = field(default_factory=list)
    skipped_shape_keys: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=dict
    )
    unexpected_keys: list[str] = field(default_factory=list)
    missing_keys_after_load: list[str] = field(default_factory=list)
    unexpected_keys_after_load: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "loaded_keys": self.loaded_keys,
            "adapted_keys": self.adapted_keys,
            "skipped_shape_keys": {
                key: {"checkpoint": src, "model": dst}
                for key, (src, dst) in self.skipped_shape_keys.items()
            },
            "unexpected_keys": self.unexpected_keys,
            "missing_keys_after_load": self.missing_keys_after_load,
            "unexpected_keys_after_load": self.unexpected_keys_after_load,
        }


def _interp_1d_table(weight: torch.Tensor, target_len: int) -> torch.Tensor:
    """Interpolate a learned table shaped `(length, channels)`."""
    if weight.shape[0] == target_len:
        return weight
    table = weight.float().transpose(0, 1).unsqueeze(0)
    if weight.shape[0] == 1 or target_len == 1:
        resized = F.interpolate(table, size=target_len, mode="nearest")
    else:
        resized = F.interpolate(table, size=target_len, mode="linear", align_corners=True)
    return resized.squeeze(0).transpose(0, 1).to(dtype=weight.dtype)


def _adapt_single_channel_conv(weight: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Expand a 1-channel pretrained conv to 7 channels while preserving scale."""
    if weight.ndim != 4:
        raise ValueError("Expected Conv2d weight")
    target_out, target_in, target_h, target_w = target_shape
    src_out, src_in, src_h, src_w = weight.shape
    if (src_out, src_in, src_h, src_w) != (target_out, 1, target_h, target_w):
        raise ValueError(f"Cannot adapt {tuple(weight.shape)} to {tuple(target_shape)}")
    return weight.repeat(1, target_in, 1, 1) / float(target_in)


def _default_model_kwargs(
    image_size: int = 200,
    input_length: int = 13,
    output_length: int = 12,
    input_channels: int = 7,
    output_channels: int = 1,
) -> dict[str, Any]:
    """SEVIR v1 config with only input shape changed for SEVIRI geometry/channels."""
    num_blocks = 2
    return {
        "input_shape": (input_length, image_size, image_size, input_channels),
        "target_shape": (output_length, image_size, image_size, output_channels),
        "base_units": 128,
        "block_units": None,
        "scale_alpha": 1.0,
        "enc_depth": [1, 1],
        "dec_depth": [1, 1],
        "enc_use_inter_ffn": True,
        "dec_use_inter_ffn": True,
        "dec_hierarchical_pos_embed": False,
        "downsample": 2,
        "downsample_type": "patch_merge",
        "upsample_type": "upsample",
        "num_global_vectors": 8,
        "use_dec_self_global": False,
        "dec_self_update_global": True,
        "use_dec_cross_global": False,
        "use_global_vector_ffn": False,
        "use_global_self_attn": True,
        "separate_global_qkv": True,
        "global_dim_ratio": 1,
        "enc_attn_patterns": ["axial"] * num_blocks,
        "dec_self_attn_patterns": ["axial"] * num_blocks,
        "dec_cross_attn_patterns": ["cross_1x1"] * num_blocks,
        "dec_cross_last_n_frames": None,
        "attn_drop": 0.1,
        "proj_drop": 0.1,
        "ffn_drop": 0.1,
        "num_heads": 4,
        "ffn_activation": "gelu",
        "gated_ffn": False,
        "norm_layer": "layer_norm",
        "padding_type": "zeros",
        "pos_embed_type": "t+h+w",
        "use_relative_pos": True,
        "self_attn_use_final_proj": True,
        "dec_use_first_self_attn": False,
        "z_init_method": "zeros",
        "checkpoint_level": 0,
        "initial_downsample_type": "stack_conv",
        "initial_downsample_activation": "leaky",
        "initial_downsample_stack_conv_num_layers": 3,
        "initial_downsample_stack_conv_dim_list": [16, 64, 128],
        "initial_downsample_stack_conv_downscale_list": [3, 2, 2],
        "initial_downsample_stack_conv_num_conv_list": [2, 2, 2],
        "attn_linear_init_mode": "0",
        "ffn_linear_init_mode": "0",
        "conv_init_mode": "0",
        "down_up_linear_init_mode": "0",
        "norm_init_mode": "0",
    }


def build_seviri_earthformer(
    image_size: int = 200,
    input_length: int = 13,
    output_length: int = 12,
    input_channels: int = 7,
    output_channels: int = 1,
) -> CuboidTransformerModel:
    """Build the official CuboidTransformerModel with SEVIRI-compatible shapes."""
    kwargs = _default_model_kwargs(
        image_size=image_size,
        input_length=input_length,
        output_length=output_length,
        input_channels=input_channels,
        output_channels=output_channels,
    )
    return CuboidTransformerModel(**kwargs)


def ensure_sevir_pretrained_checkpoint(
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT_PATH,
    url: str = OFFICIAL_SEVIR_CHECKPOINT_URL,
) -> str:
    """Download the official SEVIR checkpoint if it is not already present."""
    checkpoint_path = os.fspath(checkpoint_path)
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    urllib.request.urlretrieve(url, checkpoint_path)
    return checkpoint_path


def load_sevir_pretrained_weights(
    model: nn.Module,
    checkpoint_path: str | os.PathLike[str] = DEFAULT_CHECKPOINT_PATH,
    map_location: str | torch.device = "cpu",
) -> WeightLoadReport:
    """Load official SEVIR weights, adapting only compatibility-forced tensors."""
    checkpoint_path = os.fspath(checkpoint_path)
    try:
        raw_state = torch.load(
            checkpoint_path,
            map_location=map_location,
            weights_only=True,
        )
    except TypeError:
        raw_state = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(raw_state, dict) and "state_dict" in raw_state:
        raw_state = raw_state["state_dict"]

    model_state = model.state_dict()
    adapted_state: OrderedDict[str, torch.Tensor] = OrderedDict()
    report = WeightLoadReport(checkpoint_path=checkpoint_path)

    for key, value in raw_state.items():
        if key not in model_state:
            report.unexpected_keys.append(key)
            continue

        target = model_state[key]
        if tuple(value.shape) == tuple(target.shape):
            adapted_state[key] = value
            continue

        adapted_value: torch.Tensor | None = None
        if key == "initial_encoder.conv_block_list.0.0.weight":
            adapted_value = _adapt_single_channel_conv(value, target.shape)
        elif key.endswith(("H_embed.weight", "W_embed.weight")) and value.ndim == 2:
            adapted_value = _interp_1d_table(value, target.shape[0])
        elif key.endswith("relative_position_bias_table") and value.ndim == 2:
            if value.shape[1] == target.shape[1]:
                adapted_value = _interp_1d_table(value, target.shape[0])

        if adapted_value is None or tuple(adapted_value.shape) != tuple(target.shape):
            report.skipped_shape_keys[key] = (tuple(value.shape), tuple(target.shape))
            continue

        adapted_state[key] = adapted_value.to(dtype=target.dtype)
        report.adapted_keys.append(key)

    load_result = model.load_state_dict(adapted_state, strict=False)
    report.loaded_keys = len(adapted_state)
    report.missing_keys_after_load = list(load_result.missing_keys)
    report.unexpected_keys_after_load = list(load_result.unexpected_keys)
    return report


class EarthFormerSEVIRIMigration(nn.Module):
    """Thin wrapper exposing the pre-head latent of the official EarthFormer model."""

    def __init__(self, model: CuboidTransformerModel) -> None:
        super().__init__()
        self.model = model

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return tuple(self.model.input_shape)

    @property
    def target_shape(self) -> tuple[int, int, int, int]:
        return tuple(self.model.target_shape)

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Accept `(B,T,C,H,W)` or `(B,T,H,W,C)` and return official `(B,T,H,W,C)`."""
        if x.ndim != 5:
            raise ValueError(f"Expected 5D tensor, got shape={tuple(x.shape)}")

        expected_t, expected_h, expected_w, expected_c = self.input_shape
        if x.shape[-1] == expected_c:
            out = x
        elif x.shape[2] == expected_c:
            out = x.permute(0, 1, 3, 4, 2).contiguous()
        else:
            raise ValueError(
                f"Could not find channel dimension of size {expected_c} in shape={tuple(x.shape)}"
            )

        if out.shape[1] > expected_t:
            raise ValueError(f"Input T={out.shape[1]} exceeds configured T={expected_t}")
        if out.shape[2] != expected_h or out.shape[3] != expected_w:
            raise ValueError(
                f"Expected spatial size {(expected_h, expected_w)}, got "
                f"{tuple(out.shape[2:4])}"
            )
        if out.shape[1] < expected_t:
            pad_t = expected_t - out.shape[1]
            pad = out.new_zeros(out.shape[0], pad_t, expected_h, expected_w, expected_c)
            out = torch.cat([out, pad], dim=1)
        return out

    def forward(self, x: torch.Tensor, return_latent: bool = False) -> Any:
        """Run the unchanged official forward pass and optionally return pre-head latent."""
        x = self.prepare_input(x)
        holder: dict[str, torch.Tensor] = {}

        def capture_pre_head(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            holder["pre_head_latent"] = inputs[0]

        handle = self.model.dec_final_proj.register_forward_pre_hook(capture_pre_head)
        try:
            prediction = self.model(x)
        finally:
            handle.remove()

        if return_latent:
            return {
                "prediction": prediction,
                "pre_head_latent": holder["pre_head_latent"],
            }
        return prediction

    def forward_latent(self, x: torch.Tensor, return_trace: bool = False) -> Any:
        """Return the tensor immediately before the official projection head."""
        x = self.prepare_input(x)
        trace: OrderedDict[str, Any] = OrderedDict()
        trace["input"] = tuple(x.shape)

        bsz = x.shape[0]
        t_out = self.model.target_shape[0]
        encoded = self.model.initial_encoder(x)
        trace["after_initial_encoder"] = tuple(encoded.shape)
        encoded = self.model.enc_pos_embed(encoded)
        trace["after_encoder_pos_embed"] = tuple(encoded.shape)

        if self.model.num_global_vectors > 0:
            init_global_vectors = self.model.init_global_vectors.expand(
                bsz,
                self.model.num_global_vectors,
                self.model.global_dim_ratio * self.model.base_units,
            )
            mem_l, mem_global_vector_l = self.model.encoder(encoded, init_global_vectors)
            trace["global_vectors"] = [tuple(g.shape) for g in mem_global_vector_l]
        else:
            mem_l = self.model.encoder(encoded)
            mem_global_vector_l = None
        trace["encoder_memory"] = [tuple(mem.shape) for mem in mem_l]

        initial_z = self.model.get_initial_z(final_mem=mem_l[-1], T_out=t_out)
        trace["initial_decoder_query"] = tuple(initial_z.shape)

        if self.model.num_global_vectors > 0:
            decoded = self.model.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            decoded = self.model.decoder(initial_z, mem_l)
        trace["after_cuboid_decoder"] = tuple(decoded.shape)

        latent = self.model.final_decoder(decoded)
        trace["pre_head_latent"] = tuple(latent.shape)
        if return_trace:
            return {"pre_head_latent": latent, "trace": trace}
        return latent

    def forward_trace(self, x: torch.Tensor) -> dict[str, Any]:
        """Trace major EarthFormer stages without changing any submodule."""
        x = self.prepare_input(x)
        trace: OrderedDict[str, Any] = OrderedDict()
        trace["input"] = tuple(x.shape)

        bsz = x.shape[0]
        t_out = self.model.target_shape[0]
        encoded = self.model.initial_encoder(x)
        trace["after_initial_encoder"] = tuple(encoded.shape)
        encoded = self.model.enc_pos_embed(encoded)
        trace["after_encoder_pos_embed"] = tuple(encoded.shape)

        if self.model.num_global_vectors > 0:
            init_global_vectors = self.model.init_global_vectors.expand(
                bsz,
                self.model.num_global_vectors,
                self.model.global_dim_ratio * self.model.base_units,
            )
            mem_l, mem_global_vector_l = self.model.encoder(encoded, init_global_vectors)
            trace["global_vectors"] = [tuple(g.shape) for g in mem_global_vector_l]
        else:
            mem_l = self.model.encoder(encoded)
            mem_global_vector_l = None
        trace["encoder_memory"] = [tuple(mem.shape) for mem in mem_l]

        initial_z = self.model.get_initial_z(final_mem=mem_l[-1], T_out=t_out)
        trace["initial_decoder_query"] = tuple(initial_z.shape)

        if self.model.num_global_vectors > 0:
            dec_out = self.model.decoder(initial_z, mem_l, mem_global_vector_l)
        else:
            dec_out = self.model.decoder(initial_z, mem_l)
        trace["after_cuboid_decoder"] = tuple(dec_out.shape)

        latent = self.model.final_decoder(dec_out)
        trace["pre_head_latent"] = tuple(latent.shape)
        prediction = self.model.dec_final_proj(latent)
        trace["prediction"] = tuple(prediction.shape)

        return {
            "prediction": prediction,
            "pre_head_latent": latent,
            "trace": trace,
        }
