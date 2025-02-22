# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Any, Dict, List, Tuple
from torchvision.transforms import functional as Func
from torchvision.transforms import InterpolationMode
from PIL import Image
from .image_encoder import ImageEncoderViT
from .mask_decoder import MaskDecoder
from .prompt_encoder import PromptEncoder
from .clip_ImageEncoder import ImageEncoder
from .clip_TextEncoder import TextEncoder
from models.UNet import U_Net

class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder: ImageEncoderViT,
            prompt_encoder: PromptEncoder,
            mask_decoder: MaskDecoder,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
    ) -> None:
        """
        SAM predicts object masks from an image and input prompts.

        Arguments:
          image_encoder (ImageEncoderViT): The backbone used to encode the
            image into image embeddings that allow for efficient mask prediction.
          prompt_encoder (PromptEncoder): Encodes various types of input prompts.
          mask_decoder (MaskDecoder): Predicts masks from the image embeddings
            and encoded prompts.
          pixel_mean (list(float)): Mean values for normalizing pixels in the input image.
          pixel_std (list(float)): Std values for normalizing pixels in the input image.
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder
        self.mask_decoder = mask_decoder
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        self.text_align = nn.Linear(768, 256)
        self.text_process = nn.Linear(256, 8)
        self.text_process2 = nn.Linear(8, 256)
        self.layerNorm_text = nn.LayerNorm(256)
        self.activation1 = nn.PReLU()
        self.activation2 = nn.PReLU()
        self.dropout_text = nn.Dropout(0.5)
        self.weight = nn.Parameter(torch.empty(1, 1, 8).cuda())
        self.shift = nn.Parameter(torch.empty(1, 1, 256).cuda())
        self.learnable_weight = nn.Parameter(torch.empty(1, 1, 256).cuda())

        nn.init.normal_(self.shift, mean=0, std=0.01)
        nn.init.normal_(self.weight, mean=0, std=0.01)
        nn.init.normal_(self.learnable_weight, mean=0, std=0.01)
    @property
    def device(self) -> Any:
        return self.pixel_mean.device


    def forward(
            self,
            imgs: torch.Tensor,
            pt: Tuple[torch.Tensor, torch.Tensor],
            bbox: torch.Tensor,  # b 4
            text: torch.Tensor,
            image_filename: List
    ) -> torch.Tensor:
        # imge = self.image_encoder(imgs, text, image_filename)
        imge = self.image_encoder(imgs, image_filename)

        text_embedding = self.text_align(text)
        text_embedding = self.activation1(text_embedding)
        text_embedding_processed = self.text_process(text_embedding)
        weight = torch.sigmoid(self.weight)
        text_embedding_processed = text_embedding_processed * weight
        text_embedding_processed = self.text_process2(text_embedding_processed)
        learnable_weight = torch.sigmoid(self.learnable_weight)
        text_embedding_processed = learnable_weight * text_embedding_processed + (1-learnable_weight) * self.shift
        text_embedding_processed = self.dropout_text(text_embedding_processed)
        text_embedding = text_embedding + text_embedding_processed
        text_embedding = self.layerNorm_text(text_embedding)
        text_embedding = self.activation2(text_embedding)

        se, de = self.prompt_encoder(
            points=pt,
            boxes=None,
            masks=None,
            prompt=text_embedding
        )
        low_res_masks, _ = self.mask_decoder(
            image_embeddings=imge,
            image_pe=self.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=se,
            dense_prompt_embeddings=de,
            multimask_output=False,
            text_embedding=text_embedding,
        )

        outputs = {"low_res_logits": low_res_masks}  # stage3
        return outputs

    def postprocess_masks(
            self,
            masks: torch.Tensor,
            input_size: Tuple[int, ...],
            original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.image_encoder.img_size - h
        padw = self.image_encoder.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
