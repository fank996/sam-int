import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import List, Tuple, Type, Union
from .unet import Unet_decoder, Conv, TwoConv
from monai.networks.nets import UNet


class LayerNorm3d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None, None] * x + self.bias[:, None, None, None]
        return x


class MaskDecoder3D(nn.Module):
    def __init__(
        self,
        args,
        *,
        transformer_dim: int = 384,
        multiple_outputs: bool = False,
        num_multiple_outputs: int = 3,
    ) -> None:
        super().__init__()
        self.args = args
        self.multiple_outputs = multiple_outputs
        self.num_multiple_outputs = num_multiple_outputs
        # if self.args.use_sam3d_turbo:
        #     self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, 48, 3) for i in range(num_multiple_outputs + 1)])
        # else:
        self.output_hypernetworks_mlps = nn.ModuleList([MLP(transformer_dim, transformer_dim, 32, 3) for i in range(num_multiple_outputs + 1)])
        self.iou_prediction_head = MLP(transformer_dim, 256, num_multiple_outputs + 1, 3, sigmoid_output=True)

        self.decoder = Unet_decoder(spatial_dims=3, features=(32, 32, 64, 128, transformer_dim, 32))

        if self.args.refine:
            self.refine = Refine(self.args)

    def forward(
        self,
        prompt_embeddings: torch.Tensor, # prompt_embedding --> [b, self.num_mask_tokens, c]
        image_embeddings, # image_embedding --> [b, c, low_res / 4, low_res / 4, low_res / 4]
        feature_list: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        upscaled_embedding = self.decoder(image_embeddings, feature_list)
        masks, iou_pred = self._predict_mask(upscaled_embedding, prompt_embeddings)
        return masks, iou_pred


    def _predict_mask(self, upscaled_embedding, prompt_embeddings):

        b, c, x, y, z = upscaled_embedding.shape
        iou_token_out = prompt_embeddings[:, 0, :]
        mask_tokens_out = prompt_embeddings[:, 1: (self.num_multiple_outputs + 1 + 1), :]  # multiple masks + iou

        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_multiple_outputs + 1):
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        hyper_in = torch.stack(hyper_in_list, dim=1)

        masks = (hyper_in @ upscaled_embedding.view(b, c, x * y * z)).view(b, -1, x, y, z)
        iou_pred = self.iou_prediction_head(iou_token_out)

        if self.multiple_outputs:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]
        return masks, iou_pred
l
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x

