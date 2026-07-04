"""EarthFormer backbone with a Perceiver IO output-query readout."""

from __future__ import annotations

from typing import Any, Iterator

import torch
from torch import nn

from earthformer_migration.model import EarthFormerSEVIRIMigration
from readout import PerceiverReadout


class EarthFormerPerceiverReadoutModel(nn.Module):
    """Attach a Perceiver IO readout after the unchanged EarthFormer decoder."""

    def __init__(
        self,
        earthformer: EarthFormerSEVIRIMigration,
        readout: PerceiverReadout,
    ) -> None:
        super().__init__()
        self.earthformer = earthformer
        self.readout = readout

    def forward(self, x: torch.Tensor, return_debug: bool = False) -> Any:
        """Return CSI sequence predictions, optionally with intermediate tensors."""
        if return_debug:
            latent_result = self.earthformer.forward_latent(x, return_trace=True)
            pre_head_latent = latent_result["pre_head_latent"]
            readout_result = self.readout(pre_head_latent, return_debug=True)
            return {
                "prediction": readout_result["prediction"],
                "pre_head_latent": pre_head_latent,
                "earthformer_trace": latent_result["trace"],
                "readout": readout_result,
            }

        pre_head_latent = self.earthformer.forward_latent(x, return_trace=False)
        return self.readout(pre_head_latent, return_debug=False)

    def earthformer_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over pretrained EarthFormer parameters."""
        return self.earthformer.parameters()

    def readout_parameters(self) -> Iterator[nn.Parameter]:
        """Iterate over newly initialized readout parameters."""
        return self.readout.parameters()

    def freeze_earthformer(self) -> None:
        """Freeze the pretrained EarthFormer backbone."""
        for parameter in self.earthformer_parameters():
            parameter.requires_grad = False

    def unfreeze_earthformer(self) -> None:
        """Unfreeze the pretrained EarthFormer backbone."""
        for parameter in self.earthformer_parameters():
            parameter.requires_grad = True
