# models/blip_explainer.py

from typing import List, Tuple, Optional
import torch
from torch import nn
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import warnings

class BlipExplainer:
    def __init__(self, model_name: str = "Salesforce/blip-image-captioning-base", device: Optional[str] = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Use fast image processor to avoid the "slow image processor" warning
        self.processor = None
        try:
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=True)
        except Exception:
            warnings.filterwarnings(
                "ignore",
                message=r"Using a slow image processor as `use_fast` is unset.*",
                category=UserWarning,
            )
            self.processor = BlipProcessor.from_pretrained(model_name, use_fast=False)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # how many layers in the vision encoder (BLIP-base: 12)
        self.num_vision_layers = len(self.model.vision_model.encoder.layers)

    # ---------- basic captioning ----------

    def generate_caption(self, image: Image.Image, max_new_tokens: int = 20) -> Tuple[str, torch.Tensor]:
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
            )

        caption_ids = output_ids[0]
        caption = self.processor.decode(caption_ids, skip_special_tokens=True)
        return caption, caption_ids

    def tokens_from_ids(self, caption_ids: torch.Tensor) -> List[str]:
        if caption_ids.dim() > 1:
            caption_ids = caption_ids[0]
        tokens = self.processor.tokenizer.convert_ids_to_tokens(caption_ids.tolist())
        return tokens

    # ---------- optional: global self-attention over vision encoder ----------

    def vision_attention_heatmap(self, image: Image.Image) -> torch.Tensor:
        """
        Returns [H_patches, W_patches] normalized heatmap from vision encoder self-attention.
        """
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=inputs["pixel_values"],
                output_attentions=True,
                return_dict=True,
            )

        attentions = vision_outputs.attentions  # tuple of [B, H, N, N]
        attn_tensor = torch.stack(attentions, dim=0)  # [L, B, H, N, N]
        attn_mean = attn_tensor.mean(dim=(0, 2))      # [B, N, N]
        attn_mean = attn_mean[0]                      # [N, N]

        cls_attn = attn_mean[0]                       # [N]
        num_tokens = cls_attn.shape[0]
        num_patches = num_tokens - 1

        side = int(num_patches ** 0.5)
        num_patches = side * side

        patch_attn = cls_attn[1:1 + num_patches].reshape(side, side)
        patch_attn = patch_attn - patch_attn.min()
        if patch_attn.max() > 0:
            patch_attn = patch_attn / patch_attn.max()

        return patch_attn  # [H_patches, W_patches]

    # ---------- Grad-CAM per token, for a chosen vision layer ----------

    def gradcam_for_token(
        self,
        image: Image.Image,
        caption_ids: torch.Tensor,
        token_index: int,
        layer_idx: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Compute Grad-CAM over a chosen BLIP vision encoder layer for a selected token.
        layer_idx: which vision encoder layer to use (0 .. num_vision_layers-1).
                   If None, uses the last layer (deepest).
        Returns [H_patches, W_patches] heatmap normalized to [0, 1].
        """
        self.model.eval()
        self.model.zero_grad()

        if caption_ids.dim() == 1:
            caption_ids = caption_ids.unsqueeze(0)  # [1, T]

        if layer_idx is None:
            layer_idx = self.num_vision_layers - 1

        # safety clamp
        layer_idx = max(0, min(layer_idx, self.num_vision_layers - 1))

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        input_ids = caption_ids.to(self.device)

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def fwd_hook(module: nn.Module, inp, out):
            activations["value"] = out  # [B, N, C]

        def bwd_hook(module: nn.Module, grad_input, grad_output):
            gradients["value"] = grad_output[0]  # [B, N, C]

        target_layer = self.model.vision_model.encoder.layers[layer_idx]
        handle_fwd = target_layer.register_forward_hook(fwd_hook)
        handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

        outputs = self.model(
            pixel_values=inputs["pixel_values"],
            input_ids=input_ids,
            output_attentions=False,
            return_dict=True,
            use_cache=False,
        )

        logits = outputs.logits  # [B, T, V]
        target_id = caption_ids[0, token_index]
        target_logit = logits[0, token_index, target_id]

        target_logit.backward()

        handle_fwd.remove()
        handle_bwd.remove()

        fmap = activations["value"][0]  # [N, C]
        grad = gradients["value"][0]    # [N, C]

        # Grad-CAM weights
        weights = grad.mean(dim=0)      # [C]
        cam_tokens = torch.relu((fmap * weights).sum(dim=-1))  # [N]

        num_tokens = cam_tokens.shape[0]
        num_patches = num_tokens - 1
        side = int(num_patches ** 0.5)
        num_patches = side * side

        cam_patches = cam_tokens[1:1 + num_patches].reshape(side, side)
        cam_patches = cam_patches - cam_patches.min()
        if cam_patches.max() > 0:
            cam_patches = cam_patches / cam_patches.max()

        return cam_patches  # [H_patches, W_patches]
