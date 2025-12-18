# app.py

import tempfile
import streamlit as st
from PIL import Image, ImageDraw
import torch

from streamlit_image_coordinates import streamlit_image_coordinates

from models.blip_explainer import BlipExplainer
from models.clip_explainer import ClipExplainer
from utils.patch_utils import mask_patches_from_indices, rect_patch_indices
from utils.image_utils import overlay_heatmap_on_image
from utils.video_utils import build_blip_layer_evolution_frames, save_mp4

st.set_page_config(page_title="From Patches to Phrases", layout="wide")


@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    blip = BlipExplainer(device=device)
    clip = ClipExplainer(device=device)
    return blip, clip


blip, clip = load_models()

st.title("From Patches to Phrases – BLIP & CLIP Explainer")

st.sidebar.header("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("Please upload an image to begin.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")

col_left, col_right = st.columns(2)


# ===================== BLIP COLUMN =====================
with col_left:
    st.subheader("BLIP – Captioning & Explanations")

    caption, caption_ids = blip.generate_caption(image)
    tokens = blip.tokens_from_ids(caption_ids)
    st.markdown(f"**BLIP Caption:** {caption}")

    token_options = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
    default_idx = 1 if len(token_options) > 1 else 0
    selected_token_str = st.selectbox("Select token:", token_options, index=default_idx)
    token_index = int(selected_token_str.split(":")[0])
    selected_token = tokens[token_index]

    layer_idx = st.selectbox(
        "BLIP Vision Layer (0 = shallow, last = deep)",
        list(range(blip.num_vision_layers)),
        index=blip.num_vision_layers - 1,
    )

    cam_blip = blip.gradcam_for_token(image, caption_ids, token_index, layer_idx)
    Hp_blip, Wp_blip = cam_blip.shape

    st.pyplot(
        overlay_heatmap_on_image(
            image,
            cam_blip,
            title=f"BLIP Grad-CAM – token '{selected_token}' (layer {layer_idx})",
        )
    )

    st.markdown("---")
    st.markdown("### BLIP Layer Evolution Video")

    if st.button("Generate Layer Evolution Video"):
        with st.spinner("Computing video..."):
            frames = build_blip_layer_evolution_frames(
                blip=blip,
                image=image,
                caption_ids=caption_ids,
                token_index=token_index,
            )

            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            video_path = tmp.name
            tmp.close()

            video_path = save_mp4(
                frames,
                video_path,
                seconds_per_layer=2.0,
                transition_seconds=0.6,
                fps=10,
            )

        st.video(video_path)
        with open(video_path, "rb") as f:
            st.download_button(
                "Download Video",
                f.read(),
                file_name="blip_layer_evolution.mp4",
                mime="video/mp4",
            )

    st.markdown("---")
    st.markdown("## Manual Masking (BLIP) – 2-click selection + Confirm")

    st.markdown(
        f"BLIP patch grid: **{Hp_blip} rows × {Wp_blip} cols**. "
        "Click once for **top-left**, once for **bottom-right**, then press **Confirm**."
    )

    if "blip_points" not in st.session_state:
        st.session_state.blip_points = []
    if "blip_confirmed" not in st.session_state:
        st.session_state.blip_confirmed = False
    if "blip_last_coord" not in st.session_state:
        st.session_state.blip_last_coord = None

    if st.button("🔄 Reset BLIP Selection"):
        st.session_state.blip_points = []
        st.session_state.blip_confirmed = False
        st.session_state.blip_last_coord = None

    display_img = image.copy()
    draw = ImageDraw.Draw(display_img)

    if len(st.session_state.blip_points) >= 1:
        x1, y1 = st.session_state.blip_points[0]
        draw.ellipse([x1 - 5, y1 - 5, x1 + 5, y1 + 5], fill="yellow", outline="black")

    if len(st.session_state.blip_points) == 2:
        x1, y1 = st.session_state.blip_points[0]
        x2, y2 = st.session_state.blip_points[1]

        draw.ellipse([x2 - 5, y2 - 5, x2 + 5, y2 + 5], fill="cyan", outline="black")

        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

    coords = streamlit_image_coordinates(display_img, key="blip_clicks")

    if coords is not None and not st.session_state.blip_confirmed:
        new_point = (coords["x"], coords["y"])
        if st.session_state.blip_last_coord != new_point:
            st.session_state.blip_points.append(new_point)
            st.session_state.blip_last_coord = new_point
            if len(st.session_state.blip_points) > 2:
                st.session_state.blip_points = st.session_state.blip_points[-2:]

    if len(st.session_state.blip_points) == 2 and not st.session_state.blip_confirmed:
        x1, y1 = st.session_state.blip_points[0]
        x2, y2 = st.session_state.blip_points[1]
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        st.info(f"Selected region (pixels): TL=({xmin:.0f},{ymin:.0f}) → BR=({xmax:.0f},{ymax:.0f})")

        if st.button("✅ Confirm BLIP Mask Region"):
            st.session_state.blip_confirmed = True

    if st.session_state.blip_confirmed and len(st.session_state.blip_points) == 2:
        (x1, y1), (x2, y2) = st.session_state.blip_points
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        img_w, img_h = image.size
        patch_h = img_h / Hp_blip
        patch_w = img_w / Wp_blip

        r1 = max(0, min(Hp_blip - 1, int(ymin // patch_h)))
        r2 = max(0, min(Hp_blip - 1, int(ymax // patch_h)))
        c1 = max(0, min(Wp_blip - 1, int(xmin // patch_w)))
        c2 = max(0, min(Wp_blip - 1, int(xmax // patch_w)))

        st.markdown(f"Mapped patch indices (BLIP): rows **{r1}-{r2}**, cols **{c1}-{c2}**")

        patch_ids = rect_patch_indices(r1, r2, c1, c2)
        masked_img = mask_patches_from_indices(image, patch_ids, Hp_blip, Wp_blip)

        st.image(masked_img, caption="BLIP Masked Image", width="stretch")

        new_caption, _ = blip.generate_caption(masked_img)
        st.markdown(f"**Caption after masking (BLIP):** {new_caption}")


# ===================== CLIP COLUMN =====================
with col_right:
    st.subheader("CLIP – Image–Text Alignment (Verification)")

    st.markdown("#### Reference: BLIP Caption")
    st.code(caption)

    st.markdown("#### Text used for CLIP (editable)")
    text_for_clip = st.text_area(
        "Caption for CLIP",
        value=caption,
        height=80,
        help="Edit this to test how CLIP similarity + heatmap changes.",
    )

    sim_blip_caption = clip.image_text_similarity(image, caption)
    sim_current = clip.image_text_similarity(image, text_for_clip)

    st.markdown(
        f"**CLIP similarity with BLIP caption:** `{sim_blip_caption:.4f}`  \n"
        f"**CLIP similarity with current text:** `{sim_current:.4f}`  \n"
        f"**Δ similarity (current − BLIP caption):** `{sim_current - sim_blip_caption:.4f}`"
    )

    st.markdown("---")
    st.markdown("### CLIP Grad-CAM (Verification Heatmap)")

    cam_clip = clip.gradcam_for_image_text(image, text_for_clip)

    st.pyplot(
        overlay_heatmap_on_image(
            image,
            cam_clip,
            title="CLIP Grad-CAM – evidence for this sentence",
        )
    )
