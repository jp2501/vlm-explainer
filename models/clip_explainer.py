# models/clip_explainer.py

from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class ClipExplainer:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str | None = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def image_text_similarity(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            image_embeds = outputs.image_embeds  # [1, D]
            text_embeds = outputs.text_embeds    # [1, D]
            sim = F.cosine_similarity(image_embeds, text_embeds).item()

        return float(sim)

    def gradcam_for_image_text(self, image: Image.Image, text: str) -> torch.Tensor:
        """
        Grad-CAM over CLIP's vision encoder, using cosine similarity between
        image and text embeddings as the target.
        Returns [H_patches, W_patches] heatmap normalized to [0,1].
        """
        self.model.eval()
        self.model.zero_grad()

        inputs = self.processor(
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Ask CLIPModel for full vision_model_output so we can get gradients
        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

        vision_output = outputs.vision_model_output  # CLIPVisionModelOutput
        fmap = vision_output.last_hidden_state       # [B, N, C]
        fmap.retain_grad()

        image_embeds = outputs.image_embeds          # [B, D]
        text_embeds = outputs.text_embeds            # [B, D]

        sim = F.cosine_similarity(image_embeds, text_embeds).mean()
        sim.backward()

        grad = fmap.grad                             # [B, N, C]

        fmap = fmap[0]                               # [N, C]
        grad = grad[0]                               # [N, C]

        if fmap.numel() == 0 or grad.numel() == 0:
            return torch.ones(1, 1, device=self.device)

        weights = grad.mean(dim=0)                   # [C]
        cam_tokens = torch.relu((fmap * weights).sum(dim=-1))  # [N]

        num_tokens = cam_tokens.shape[0]
        if num_tokens <= 1:
            return torch.ones(1, 1, device=self.device)

        num_patches = num_tokens - 1
        if num_patches <= 0:
            return torch.ones(1, 1, device=self.device)

        side = int(num_patches ** 0.5)
        if side < 1:
            side = 1
        max_patches = min(side * side, num_patches)
        side = int(max_patches ** 0.5) or 1
        max_patches = side * side

        cam_vec = cam_tokens[1:1 + max_patches]
        if cam_vec.numel() == 0:
            return torch.ones(1, 1, device=self.device)

        cam_patches = cam_vec.reshape(side, side)

        cam_patches = cam_patches - cam_patches.min()
        if cam_patches.max() > 0:
            cam_patches = cam_patches / cam_patches.max()

        return cam_patches
