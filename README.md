# VLM Explainer (BLIP + CLIP)

A Streamlit app for interpreting Vision–Language Models:
- **BLIP**: token-level Grad-CAM + layer selection + manual region masking (2-click rectangle + confirm) + layer-evolution video
- **CLIP**: verification tool (text–image similarity + Grad-CAM)

## 1) Setup

### Option A: Using pip (recommended)
```bash
git clone <YOUR_GITHUB_REPO_URL>
cd vlm-explainer

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt


### Option B: Using conda
git clone <YOUR_GITHUB_REPO_URL>
cd vlm-explainer

conda create -n vlm python=3.10 -y
conda activate vlm
pip install -r requirements.txt
