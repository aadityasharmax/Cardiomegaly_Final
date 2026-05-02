import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import timm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from PIL import Image
import cv2
import time
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CardioVision — Cardiomegaly Detection",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    .main .block-container { padding: 1.5rem 2rem; max-width: 1200px; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #1a1d2e;
        border-right: 1px solid #2d3142;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #e0e6ff;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1e2235;
        border: 1px solid #2d3142;
        border-radius: 10px;
        padding: 12px 16px;
    }
    [data-testid="metric-container"] label { color: #7b8ab8 !important; font-size: 12px !important; }
    [data-testid="metric-container"] [data-testid="stMetricValue"] {
        color: #4f8ef7 !important; font-size: 22px !important; font-weight: 600 !important;
    }

    /* Prediction result boxes */
    .pred-cardio {
        background: rgba(248,113,113,0.1);
        border: 1px solid rgba(248,113,113,0.3);
        border-radius: 12px; padding: 20px 24px; text-align: center;
    }
    .pred-normal {
        background: rgba(52,211,153,0.1);
        border: 1px solid rgba(52,211,153,0.3);
        border-radius: 12px; padding: 20px 24px; text-align: center;
    }
    .pred-title-cardio { color: #f87171; font-size: 28px; font-weight: 700; margin: 0; }
    .pred-title-normal { color: #34d399; font-size: 28px; font-weight: 700; margin: 0; }
    .pred-conf { color: #9ca3af; font-size: 14px; margin-top: 6px; }

    /* Info box */
    .info-box {
        background: rgba(251,191,36,0.08);
        border: 1px solid rgba(251,191,36,0.25);
        border-radius: 8px; padding: 12px 16px;
        color: #fbbf24; font-size: 13px;
    }

    /* History table */
    .hist-row {
        background: #1e2235; border: 1px solid #2d3142;
        border-radius: 8px; padding: 10px 14px;
        margin-bottom: 8px; display: flex;
        align-items: center; justify-content: space-between;
    }

    /* Divider */
    hr { border-color: #2d3142 !important; }

    /* Button styling */
    .stButton > button {
        background: #4f8ef7; color: white; border: none;
        border-radius: 8px; font-weight: 600;
        padding: 0.5rem 2rem; width: 100%;
        transition: background 0.2s;
    }
    .stButton > button:hover { background: #3a7bd5; border: none; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: #1e2235;
        border: 2px dashed #2d3142;
        border-radius: 12px;
    }

    /* Hide streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] { background: #1e2235; border-radius: 8px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { color: #7b8ab8; border-radius: 6px; }
    .stTabs [aria-selected="true"] { background: #4f8ef7 !important; color: white !important; }

    /* Progress bar */
    .stProgress > div > div { background: #4f8ef7; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
IMG_SIZE   = 224
CLASS_NAMES = ['Normal', 'Cardiomegaly']
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Transforms ────────────────────────────────────────────────────────────────
eval_tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.Grayscale(num_output_channels=3),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD)
])

# ── Model builders ────────────────────────────────────────────────────────────
def make_head(in_features):
    return nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 2)
    )

def build_efficientnet():
    m = timm.create_model('efficientnet_b3', pretrained=False, num_classes=0)
    m.classifier = make_head(m.num_features)
    return m

def build_densenet():
    m = models.densenet121(weights=None)
    m.classifier = make_head(m.classifier.in_features)
    return m

# ── Load model (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model(model_type, weights_path):
    """Load model from .pth file. Cached so it only loads once."""
    try:
        if model_type == 'EfficientNet-B3':
            model = build_efficientnet()
        else:
            model = build_densenet()

        state_dict = torch.load(weights_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        return model, None
    except Exception as e:
        return None, str(e)

# ── Prediction ────────────────────────────────────────────────────────────────
def predict(model, image: Image.Image):
    """Run inference on a PIL image. Returns probs array."""
    tensor = eval_tf(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return probs  # [normal_prob, cardio_prob]

# ── Grad-CAM ──────────────────────────────────────────────────────────────────
def get_gradcam(model, model_type, image: Image.Image):
    """Generate Grad-CAM heatmap. Returns overlay as numpy RGB array."""
    try:
        # Get target layer
        if model_type == 'EfficientNet-B3':
            target_layer = list(model.blocks.children())[-1][-1].conv_pwl
        else:
            target_layer = model.features.denseblock4.denselayer16.conv2

        grads, acts = [], []

        def fwd_hook(m, inp, out):
            acts.append(out.detach())
            out.register_hook(lambda g: grads.append(g.detach()))

        handle = target_layer.register_forward_hook(fwd_hook)

        tensor = eval_tf(image).unsqueeze(0).to(DEVICE)
        tensor.requires_grad_(True)

        model.eval()
        output = model(tensor)
        pred_class = output.argmax(1).item()

        model.zero_grad()
        output[0, pred_class].backward()
        handle.remove()

        if not grads or not acts:
            return None

        w   = grads[0].mean(dim=[2, 3], keepdim=True)
        cam = torch.relu((w * acts[0]).sum(dim=1)).squeeze().cpu().numpy()

        # Normalize
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        cam = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))

        # Create overlay
        img_resized = image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB')
        img_np      = np.array(img_resized)
        heatmap     = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        overlay     = np.clip(0.55 * img_np + 0.45 * heatmap, 0, 255).astype(np.uint8)

        return overlay
    except Exception:
        return None

# ── Probability bar chart ─────────────────────────────────────────────────────
def make_prob_chart(probs):
    fig, ax = plt.subplots(figsize=(5, 2.2))
    fig.patch.set_facecolor('#1e2235')
    ax.set_facecolor('#1e2235')

    colors = ['#34d399', '#f87171']
    bars   = ax.barh(CLASS_NAMES, probs * 100, color=colors,
                     height=0.45, edgecolor='none')

    for bar, p in zip(bars, probs * 100):
        ax.text(min(p + 1.5, 95), bar.get_y() + bar.get_height() / 2,
                f'{p:.1f}%', va='center', fontsize=12, fontweight='600',
                color='white')

    ax.set_xlim(0, 105)
    ax.set_xlabel('Probability (%)', color='#7b8ab8', fontsize=10)
    ax.tick_params(colors='#9ca3af', labelsize=10)
    ax.spines[:].set_visible(False)
    for spine in ax.spines.values():
        spine.set_edgecolor('#2d3142')

    plt.tight_layout(pad=0.5)
    return fig

# ── Confidence gauge ──────────────────────────────────────────────────────────
def make_gauge(confidence, label, is_cardio):
    fig, ax = plt.subplots(figsize=(3.5, 2.2), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor('#1e2235')
    ax.set_facecolor('#1e2235')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis('off')

    # Background arc
    theta = np.linspace(np.pi, 0, 100)
    ax.plot(0.5 + 0.38 * np.cos(theta), 0.3 + 0.38 * np.sin(theta),
            color='#2d3142', linewidth=12, solid_capstyle='round')

    # Foreground arc
    fill_theta = np.linspace(np.pi, np.pi - confidence * np.pi, 100)
    color = '#f87171' if is_cardio else '#34d399'
    ax.plot(0.5 + 0.38 * np.cos(fill_theta), 0.3 + 0.38 * np.sin(fill_theta),
            color=color, linewidth=12, solid_capstyle='round')

    ax.text(0.5, 0.28, f'{confidence*100:.0f}%', ha='center', va='center',
            fontsize=22, fontweight='700', color=color)
    ax.text(0.5, 0.05, label, ha='center', va='center',
            fontsize=11, color='#9ca3af')

    plt.tight_layout(pad=0)
    return fig

# ── Session state ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🫀 CardioVision")
    st.markdown("---")

    st.markdown("### Model Selection")
    model_type = st.radio(
        "Choose architecture",
        ['EfficientNet-B3', 'DenseNet-121'],
        help="EfficientNet-B3 achieves ~84% accuracy. DenseNet-121 achieves ~80%."
    )

    st.markdown("---")
    st.markdown("### Load Model Weights")
    weights_path = st.text_input(
        "Path to .pth file",
        placeholder="e.g. C:/Users/Aaditya/efficientnet_b3_best.pth",
        help="Paste the full path to your saved model weights file"
    )

    load_btn = st.button("🔄 Load Model", use_container_width=True)

    if load_btn:
        if not weights_path:
            st.error("Please enter the path to your .pth file.")
        elif not os.path.exists(weights_path):
            st.error(f"File not found:\n`{weights_path}`")
        else:
            with st.spinner(f"Loading {model_type}..."):
                model, err = load_model(model_type, weights_path)
            if err:
                st.error(f"Failed to load model:\n{err}")
                st.session_state.model_loaded = False
            else:
                st.session_state.model_loaded = True
                st.session_state.loaded_model = model
                st.session_state.loaded_model_type = model_type
                st.success(f"✅ {model_type} loaded!")

    # Model status
    st.markdown("---")
    st.markdown("### Model Status")
    if st.session_state.model_loaded:
        st.markdown(f"""
        <div style='background:rgba(52,211,153,0.1);border:1px solid rgba(52,211,153,0.3);
        border-radius:8px;padding:10px 14px;'>
        <span style='color:#34d399;font-weight:600;'>● Loaded</span><br>
        <span style='color:#9ca3af;font-size:12px;'>{st.session_state.loaded_model_type}</span>
        </div>
        """, unsafe_allow_html=True)

        # Model stats
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.loaded_model_type == 'EfficientNet-B3':
            st.metric("Test Accuracy", "~84%")
            st.metric("AUC", "~0.91")
            st.metric("Parameters", "~12M")
        else:
            st.metric("Test Accuracy", "~80%")
            st.metric("AUC", "~0.88")
            st.metric("Parameters", "~8M")
    else:
        st.markdown("""
        <div style='background:rgba(248,113,113,0.1);border:1px solid rgba(248,113,113,0.3);
        border-radius:8px;padding:10px 14px;'>
        <span style='color:#f87171;font-weight:600;'>● No model loaded</span><br>
        <span style='color:#9ca3af;font-size:12px;'>Enter path and click Load</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Device")
    device_color = '#34d399' if str(DEVICE) == 'cuda' else '#fbbf24'
    st.markdown(f"""
    <div style='color:{device_color};font-size:13px;font-weight:500;'>
    {'🟢 GPU (CUDA)' if str(DEVICE) == 'cuda' else '🟡 CPU (no GPU detected)'}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Aaditya Sharma · MIET · Hestabit Trainee")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🫀 Cardiomegaly Prediction Dashboard")
st.markdown("Upload a chest X-ray image to classify it as **Normal** or **Cardiomegaly** using your trained deep learning model.")

st.markdown("---")

# ── Warning strip if model not loaded ────────────────────────────────────────
if not st.session_state.model_loaded:
    st.markdown("""
    <div class='info-box'>
    ⚠️ <strong>No model loaded.</strong> Use the sidebar to enter the path to your
    <code>efficientnet_b3_best.pth</code> or <code>densenet121_best.pth</code> file and click Load Model.
    </div>
    <br>
    """, unsafe_allow_html=True)

# ── Upload ────────────────────────────────────────────────────────────────────
col_upload, col_spacer = st.columns([2, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload Chest X-ray",
        type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
        help="Upload a chest X-ray image. DICOM files should be converted to PNG/JPG first."
    )

if uploaded is not None:
    image = Image.open(uploaded).convert('RGB')

    # ── Two-column layout: image | results ───────────────────────────────────
    col_img, col_res = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown("### X-ray Preview")
        st.image(image, use_container_width=True, caption=uploaded.name)

        # Image metadata
        w, h = image.size
        size_kb = len(uploaded.getvalue()) / 1024
        c1, c2, c3 = st.columns(3)
        c1.metric("Width", f"{w}px")
        c2.metric("Height", f"{h}px")
        c3.metric("Size", f"{size_kb:.0f} KB")

    with col_res:
        st.markdown("### Prediction Result")

        if not st.session_state.model_loaded:
            st.warning("Load a model from the sidebar first to run prediction.")
        else:
            # Run prediction
            with st.spinner("Running inference..."):
                progress = st.progress(0)
                for i in range(0, 80, 20):
                    time.sleep(0.08)
                    progress.progress(i)

                probs = predict(st.session_state.loaded_model, image)
                progress.progress(100)
                time.sleep(0.1)
                progress.empty()

            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class])
            cardio_prob = float(probs[1])
            normal_prob = float(probs[0])
            is_cardio = pred_class == 1

            # ── Main prediction box ───────────────────────────────────────
            if is_cardio:
                st.markdown(f"""
                <div class='pred-cardio'>
                    <p class='pred-title-cardio'>⚠ Cardiomegaly Detected</p>
                    <p class='pred-conf'>Confidence: {confidence*100:.1f}% · {st.session_state.loaded_model_type}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='pred-normal'>
                    <p class='pred-title-normal'>✓ Normal</p>
                    <p class='pred-conf'>Confidence: {confidence*100:.1f}% · {st.session_state.loaded_model_type}</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Probability bars ──────────────────────────────────────────
            st.markdown("**Class Probabilities**")
            st.progress(int(normal_prob * 100),
                        text=f"Normal: {normal_prob*100:.1f}%")
            st.progress(int(cardio_prob * 100),
                        text=f"Cardiomegaly: {cardio_prob*100:.1f}%")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Gauge + bar chart ─────────────────────────────────────────
            gc1, gc2 = st.columns(2)
            with gc1:
                fig_gauge = make_gauge(confidence, CLASS_NAMES[pred_class], is_cardio)
                st.pyplot(fig_gauge, use_container_width=True)
                plt.close()
            with gc2:
                fig_bar = make_prob_chart(probs)
                st.pyplot(fig_bar, use_container_width=True)
                plt.close()

            # ── Add to history ────────────────────────────────────────────
            st.session_state.history.insert(0, {
                'filename':    uploaded.name,
                'prediction':  CLASS_NAMES[pred_class],
                'confidence':  f'{confidence*100:.1f}%',
                'cardio_prob': f'{cardio_prob*100:.1f}%',
                'normal_prob': f'{normal_prob*100:.1f}%',
                'model':       st.session_state.loaded_model_type,
                'time':        datetime.now().strftime('%H:%M:%S'),
                'is_cardio':   is_cardio
            })

    # ── Grad-CAM section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🔍 Grad-CAM Visualization")
    st.markdown("Heatmap showing which regions of the X-ray influenced the prediction most.")

    if st.session_state.model_loaded:
        with st.spinner("Generating Grad-CAM heatmap..."):
            gradcam_img = get_gradcam(
                st.session_state.loaded_model,
                st.session_state.loaded_model_type,
                image
            )

        if gradcam_img is not None:
            gc_col1, gc_col2, gc_col3 = st.columns(3)
            with gc_col1:
                st.image(image.resize((IMG_SIZE, IMG_SIZE)),
                         caption="Original X-ray", use_container_width=True)
            with gc_col2:
                st.image(gradcam_img, caption="Grad-CAM Heatmap", use_container_width=True)
            with gc_col3:
                # Side-by-side blend
                orig_np = np.array(image.resize((IMG_SIZE, IMG_SIZE)).convert('RGB'))
                blend   = np.clip(0.5 * orig_np + 0.5 * gradcam_img, 0, 255).astype(np.uint8)
                st.image(blend, caption="50/50 Overlay", use_container_width=True)

            st.markdown("""
            <div class='info-box'>
            🔴 <strong>Red/warm regions</strong> = high activation (model focused here) &nbsp;|&nbsp;
            🔵 <strong>Blue/cool regions</strong> = low activation
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Grad-CAM could not be generated for this model/image combination.")

# ══════════════════════════════════════════════════════════════════════════════
# TABS — History & About
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
tab_hist, tab_about = st.tabs(["📋 Prediction History", "ℹ️ About"])

with tab_hist:
    if not st.session_state.history:
        st.markdown("""
        <div style='text-align:center;padding:40px;color:#7b8ab8;'>
        No predictions yet. Upload a chest X-ray above.
        </div>
        """, unsafe_allow_html=True)
    else:
        # Clear history button
        if st.button("🗑 Clear History"):
            st.session_state.history = []
            st.rerun()

        # Table header
        hcols = st.columns([3, 2, 2, 2, 2, 2])
        for col, header in zip(hcols, ['Filename', 'Prediction', 'Confidence', 'Cardiomegaly %', 'Model', 'Time']):
            col.markdown(f"**{header}**")
        st.markdown("---")

        # Table rows
        for entry in st.session_state.history[:20]:
            row = st.columns([3, 2, 2, 2, 2, 2])
            row[0].markdown(f"📄 `{entry['filename'][:22]}`")

            color = '#f87171' if entry['is_cardio'] else '#34d399'
            row[1].markdown(f"<span style='color:{color};font-weight:600;'>{entry['prediction']}</span>",
                            unsafe_allow_html=True)
            row[2].markdown(entry['confidence'])
            row[3].markdown(entry['cardio_prob'])
            row[4].markdown(f"`{entry['model'][:14]}`")
            row[5].markdown(entry['time'])

with tab_about:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        ### About This Project
        This dashboard is the deployment interface for the research paper:

        **"Cardiomegaly Prediction Using Deep Learning"**
        *Aaditya Sharma, Akash Sharma, Abhishek Tyagi, Abhinav Gupta*
        *Meerut Institute of Engineering and Technology*

        The system compares two architectures for automated detection of
        cardiomegaly from chest X-ray images.

        ---
        ### Models
        | Model | Accuracy | AUC | Params |
        |-------|----------|-----|--------|
        | EfficientNet-B3 | ~84% | ~0.91 | 12M |
        | DenseNet-121 | ~80% | ~0.88 | 8M |

        ---
        ### Dataset
        [rahimanshu/cardiomegaly-disease-prediction-using-cnn](https://www.kaggle.com/datasets/rahimanshu/cardiomegaly-disease-prediction-using-cnn)
        - 5,552 chest X-rays (balanced)
        - `true` folder = Cardiomegaly (label 1)
        - `false` folder = Normal (label 0)
        """)
    with c2:
        st.markdown("""
        ### How to Use
        1. **Load your model** — enter the full path to your `.pth` file in the sidebar
           (e.g. `C:/Users/Aaditya/efficientnet_b3_best.pth`)
        2. **Upload a chest X-ray** — JPG, PNG, or JPEG
        3. **View prediction** — probability bars, confidence gauge, and result box
        4. **Inspect Grad-CAM** — see which heart regions activated the prediction
        5. **Check history** — all predictions in this session are logged

        ---
        ### Interpreting Results
        - **Normal** — no cardiomegaly detected
        - **Cardiomegaly** — enlarged heart detected → recommend clinical review
        - **High confidence (>80%)** — model is certain
        - **Low confidence (50–65%)** — borderline case, consider second opinion

        ---
        ### ⚠️ Disclaimer
        This tool is for **research and educational purposes only**.
        It is not a certified medical device and should not replace
        professional radiological diagnosis.

        ---
        ### Tech Stack
        `PyTorch` · `timm` · `Streamlit` · `OpenCV` · `Grad-CAM`
        """)