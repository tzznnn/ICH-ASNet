import numpy as np
import torch.nn.functional as F
import cv2
import torch
import torch.nn as nn
from torchvision import transforms

from transformers import XLMRobertaTokenizer
from models.UNet import network
from models.sam2.beit3.modeling_utils import BEiT3Wrapper, _get_base_config

def random_sample_points(mask, class_id=1, num_points=10, max_positive_points=1):
    batch_points = []
    batch_labels = []
    for m in mask:
        background_indices = np.argwhere(m == 0)
        background_indices[:, [0, 1]] = background_indices[:, [1, 0]]

        num_labels, labels = cv2.connectedComponents((m == class_id).astype(np.uint8))

        positive_points = []

        if num_labels > 1:
            region_sizes = [(label, np.sum(labels == label)) for label in range(1, num_labels)]
            region_sizes = sorted(region_sizes, key=lambda x: x[1], reverse=True)[:max_positive_points]

            for label, _ in region_sizes:
                region_points = np.argwhere(labels == label)
                region_points[:, [0, 1]] = region_points[:, [1, 0]]

                selected_point = region_points[np.random.choice(len(region_points), 1, replace=False)]
                positive_points.append(selected_point[0])

        positive_points = np.array(positive_points)
        num_positive = len(positive_points)

        if num_positive > 0:
            num_negative = num_points - num_positive
            if num_negative > 0 and len(background_indices) > 0:
                selected_negative_points = background_indices[
                    np.random.choice(len(background_indices), num_negative, replace=False)]
            else:
                selected_negative_points = np.zeros((0, 2), dtype=int)

            image_points = np.vstack([positive_points, selected_negative_points])
            image_labels = np.array([1] * num_positive + [-1] * num_negative)

        else:
            if len(background_indices) > 0:
                selected_background_point = background_indices[np.random.choice(len(background_indices), 1)]
                image_points = np.vstack([selected_background_point, np.zeros((3, 2), dtype=int)])
                image_labels = np.array([0] + [0] * 3)

        batch_points.append(image_points)
        batch_labels.append(image_labels)

    batch_points = np.array(batch_points)
    batch_labels = np.array(batch_labels)

    return batch_points, batch_labels

class PromptGenerator(nn.Module):
    def __init__(self, point_generator, **kwargs):
        super().__init__()
        self.point_generator = network.U_Net(1, 1, point_generator)

        beit_config = _get_base_config()
        self.text_generator = BEiT3Wrapper(beit_config)
        beit_state_dict = torch.load("./pretrained/beit3_base_indomain_patch16_224.pth")
        self.text_generator.load_state_dict(
            beit_state_dict,
            strict=False
        )
        for param in self.text_generator.parameters():
            param.requires_grad = False

        self.image_preprocessor = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3, antialias=None),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])

        self.tokenizer = XLMRobertaTokenizer("./pretrained/beit3.spm")
        self.text = "The hemorrhagic region appears as a hyperdense area, representing an abnormal accumulation of blood within the brain tissue."
        self.text_tokens = self.tokenizer(self.text, return_tensors="pt").input_ids
        self.attention_masks = self.text_tokens.ne(self.tokenizer.pad_token_id)

    def generator_points(self, imgs):
        downsampled_imgs = F.interpolate(imgs, size=(256, 256), mode='bilinear', align_corners=False).to(
            dtype=torch.float32, device=imgs.device)

        min_vals = downsampled_imgs.amin(dim=(2, 3), keepdim=True)
        max_vals = downsampled_imgs.amax(dim=(2, 3), keepdim=True)

        downsampled_imgs = (downsampled_imgs - min_vals) / (max_vals - min_vals)
        coarse_mask = self.point_generator(downsampled_imgs)
        coarse_mask = torch.sigmoid(coarse_mask)
        coarse_predict = coarse_mask.detach().cpu().numpy()
        coarse_seg = coarse_predict[:, 0, :, :] > 0.5
        seg_ = np.zeros((coarse_seg.shape[0], coarse_seg.shape[1], coarse_seg.shape[2]))
        seg_[coarse_seg[:, :, :] == 1] = 1
        seg_ = torch.tensor(seg_, dtype=torch.float32).unsqueeze(1).to(dtype=torch.float32, device=imgs.device)
        seg_ = F.interpolate(seg_, size=(1024, 1024), mode='bilinear', align_corners=False)
        pt, point_labels = random_sample_points(np.array(seg_.squeeze().unsqueeze(0).cpu().numpy()), 1)
        point_coords = pt
        coords_torch = torch.as_tensor(point_coords, dtype=torch.float32, device=imgs.device)
        labels_torch = torch.as_tensor(point_labels, dtype=torch.int, device=imgs.device)
        pt = (coords_torch, labels_torch)
        return pt
    def generator_text(self, imgs):
        if imgs.size()[1] == 1:
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = self.image_preprocessor(imgs)

        output = self.text_generator.beit3(
            visual_tokens=imgs,
            textual_tokens=self.text_tokens.expand(imgs.shape[0], -1).to("cuda"),
            text_padding_position=~self.attention_masks.expand(imgs.shape[0], -1).to("cuda")
        )
        feat = output["encoder_out"][:, :1, ...]
        return feat
