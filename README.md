# VLM Explainer (BLIP + CLIP)

**From Patches to Phrases** — An interactive Streamlit application for interpreting  
how **Vision–Language Models (VLMs) look and talk**.

This project focuses on **explainability**, not training.  
We analyze *why* pretrained VLMs generate specific words or align images with text.

---

##  What This App Does

### BLIP — Captioning & Explainability
- Token-level **Grad-CAM**
- Vision-layer selection (shallow → deep)
- Manual region masking  
  - 2-click rectangle + confirm
- Vision-layer evolution visualization

### CLIP — Verification & Alignment
- Image–text similarity scoring
- Grad-CAM for alignment verification
- Used as a **sanity-check model**, not for captioning

---

## Key Idea

- **BLIP explains** *what the model says* and *why*
- **CLIP verifies** whether the image truly matches the text

Together, they enable **causal analysis** of vision–language grounding.

---

##  Setup

### Option A: Using `pip` (recommended)


git clone https://github.com/jp2501/vlm-explainer
cd vlm-explainer

python -m venv .venv

# Windows
.venv\Scripts\activate


# source .venv/bin/activate

pip install -r requirements.txt

### Option B: Using Comnda
git clone https://github.com/jp2501/vlm-explainer

cd vlm-explainer

conda create -n vlm python=3.10 -y

conda activate vlm

pip install -r requirements.txt

### Run the App
streamlit run app.py

