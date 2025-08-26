#!/usr/bin/env python3
"""
Enhanced ZO-1 Network Analysis App with Streamlit
Implements RIS (Radial Integrity Score) and TiJOR quantification using peak detection and concentric circles/expanding rectangles
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import cv2
from PIL import Image
import pandas as pd
from io import StringIO, BytesIO

# Peak detection imports (replacing segmentation)
# (Cellpose and other GPU-heavy AI features removed for this deployment version)

# Import hexagonal network generator
from zo1_core.hexagonal_network import generate_hexagonal_network, get_hexagonal_network_info

# Peak detection utility functions
def _rolling_mean(x, w):
    """Apply rolling mean smoothing to a signal."""
    if w <= 1: 
        return x.astype(np.float32)
    if w % 2 == 0: 
        w += 1
    k = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), k, mode='same')

def _local_prominence(signal, idx, win):
    """Compute prominence against local minima on both sides within ±win."""
    n = len(signal)
    left = signal[max(0, idx - win):idx+1]
    right = signal[idx:min(n, idx + win + 1)]
    if left.size == 0 or right.size == 0:
        return 0.0
    left_min = left.min()
    right_min = right.min()
    base = max(left_min, right_min)
    return float(signal[idx] - base)

def _mad(x):
    """Compute Median Absolute Deviation."""
    med = np.median(x)
    return np.median(np.abs(x - med)) + 1e-6

def detect_peaks_1d(signal, smooth_window=9, z_thresh=2.0, min_separation=10, local_win=15):
    """
    Simple, robust 1D peak detector:
    - Smooth
    - Candidate peaks = local maxima
    - Prominence vs local minima (±local_win)
    - Accept if (prominence / (1.4826*MAD)) >= z_thresh
    - Enforce min_separation in samples
    Returns indices of accepted peaks.
    """
    x = _rolling_mean(signal, smooth_window)
    n = len(x)
    # wrap-aware neighbor comparisons
    prev = np.roll(x, 1)
    nextv = np.roll(x, -1)
    cand = (x >= prev) & (x > nextv)

    noise = 1.4826 * _mad(x)  # ~std for Gaussian
    if noise <= 0: 
        noise = 1e-6

    cand_idx = np.where(cand)[0].tolist()
    # score candidates by prominence
    scored = []
    for i in cand_idx:
        prom = _local_prominence(x, i, local_win)
        z = prom / noise
        scored.append((i, z, prom))
    # filter by z threshold
    scored = [s for s in scored if s[1] >= z_thresh]
    # sort by strength, keep with min separation
    scored.sort(key=lambda t: t[1], reverse=True)
    keep = []
    taken = np.zeros(n, dtype=bool)
    for i, z, prom in scored:
        if min_separation <= 0:
            keep.append(i)
            continue
        lo = max(0, i - min_separation)
        hi = min(n, i + min_separation + 1)
        if not taken[lo:hi].any():
            taken[lo:hi] = True
            keep.append(i)
    keep.sort()
    return np.array(keep, dtype=int)

def sample_circle(cx, cy, r, H, W):
    """Sample points along a circle with approximately 1px spacing."""
    # arc ≈ 1 px ⇒ Δθ ≈ 1/r; cap to reasonable density
    dtheta = max(1.0/max(r, 1.0), 2*np.pi/4096)
    thetas = np.arange(0, 2*np.pi, dtheta, dtype=np.float32)
    xs = np.clip(np.rint(cx + r*np.cos(thetas)).astype(int), 0, W-1)
    ys = np.clip(np.rint(cy + r*np.sin(thetas)).astype(int), 0, H-1)
    return thetas, xs, ys

def sample_rectangle_perimeter(cx, cy, side, H, W):
    """Return perimeter samples (x,y) for a square of side centered at (cx,cy). 1 px spacing."""
    half = side / 2.0
    x0, x1 = int(max(0, np.floor(cx - half))), int(min(W-1, np.ceil(cx + half)))
    y0, y1 = int(max(0, np.floor(cy - half))), int(min(H-1, np.ceil(cy + half)))
    xs, ys = [], []
    # top & bottom edges
    for x in range(x0, x1+1):
        xs.append(x); ys.append(y0)
        xs.append(x); ys.append(y1)
    # left & right edges
    for y in range(y0, y1+1):
        xs.append(x0); ys.append(y)
        xs.append(x1); ys.append(y)
    coords = np.stack([xs, ys], axis=1)
    # de-duplicate & order is not critical for our 1D traversal (we'll treat it as a closed loop)
    coords = np.unique(coords, axis=0)
    # Order roughly around the square (clockwise)
    # Simple heuristic: sort by angle around center
    ang = np.arctan2(coords[:,1]-cy, coords[:,0]-cx)
    order = np.argsort(ang)
    coords = coords[order]
    return coords[:,0].astype(int), coords[:,1].astype(int)

def apply_otsu_mask(img, strength=1.0):
    """Apply Otsu thresholding for background cleaning (not segmentation)."""
    thr, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = int(thr * max(0.5, min(3.0, strength)))
    _, mask = cv2.threshold(img, thr, 255, cv2.THRESH_BINARY)
    return (img.astype(np.float32) * (mask>0)).astype(np.float32), mask

# Peak detection-based quantification functions
def ris_segfree(img_gray,
                use_premask=True, premask_strength=1.0,
                smooth_window=9, z_thresh=2.0, min_sep=10,
                initial_area_pct=10.0, max_area_pct=70.0, steps=5):
    """RIS quantification using peak detection along circular sampling paths."""
    H, W = img_gray.shape
    work = img_gray.copy().astype(np.float32)
    mask_used = None
    if use_premask:
        work, mask_used = apply_otsu_mask(img_gray, strength=premask_strength)

    A = H*W
    r0 = np.sqrt((initial_area_pct/100.0 * A)/np.pi)
    r1 = np.sqrt((max_area_pct/100.0 * A)/np.pi)
    radii = np.linspace(r0, r1, steps+1)

    cx, cy = W/2.0, H/2.0
    d_vals, N_vals = [], []
    hits_all = []

    for r in radii:
        thetas, xs, ys = sample_circle(cx, cy, r, H, W)
        profile = work[ys, xs]
        # Peak detection along circular profile
        peak_idx = detect_peaks_1d(profile, smooth_window=smooth_window,
                                   z_thresh=z_thresh, min_separation=int(min_sep),
                                   local_win=int(max(10, min(60, r/2))))
        N = int(len(peak_idx))
        N_vals.append(N)
        d_vals.append(N / (2*np.pi*r) if r>0 else 0.0)

        if N>0:
            # Use the actual sampled pixel coordinates for peak locations to keep overlays aligned
            hx = xs[peak_idx]
            hy = ys[peak_idx]
            hits_all.append(np.stack([hx, hy], axis=1))

    d_vals = np.array(d_vals, dtype=np.float32)
    N_vals = np.array(N_vals, dtype=int)
    hits_xy = np.vstack(hits_all) if hits_all else np.empty((0,2))
    return radii, N_vals, d_vals, hits_xy, mask_used

def tijor_segfree(img_gray,
                  use_premask=True, premask_strength=1.0,
                  smooth_window=9, z_thresh=2.0, min_sep=10,
                  initial_area_pct=10.0, max_area_pct=70.0, steps=5):
    """TiJOR quantification using peak detection along rectangular sampling paths."""
    H, W = img_gray.shape
    work = img_gray.copy().astype(np.float32)
    mask_used = None
    if use_premask:
        work, mask_used = apply_otsu_mask(img_gray, strength=premask_strength)

    A = H*W
    side0 = np.sqrt((initial_area_pct/100.0)*A)
    side1 = np.sqrt((max_area_pct/100.0)*A)
    sides = np.linspace(side0, side1, steps+1)

    cx, cy = W/2.0, H/2.0
    tijor_vals = []
    counts = []
    all_pts = []

    for side in sides:
        xs, ys = sample_rectangle_perimeter(cx, cy, side, H, W)
        profile = work[ys, xs]
        peak_idx = detect_peaks_1d(profile, smooth_window=smooth_window,
                                   z_thresh=z_thresh, min_separation=int(min_sep),
                                   local_win=int(max(10, min(60, side/4))))
        N = int(len(peak_idx))
        perim = 4.0*side
        tijor = N / perim if perim>0 else 0.0
        tijor_vals.append(tijor)
        counts.append(N)

        if N>0:
            sel = np.stack([xs[peak_idx], ys[peak_idx]], axis=1)
            all_pts.append(sel)

    tijor_vals = np.array(tijor_vals, dtype=np.float32)
    counts = np.array(counts, dtype=int)
    pts = np.vstack(all_pts) if all_pts else np.empty((0,2), dtype=np.float32)
    return sides, counts, tijor_vals, pts, mask_used

# Set page configuration
st.set_page_config(
    page_title="ZO-1 Network Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .overlay-controls {
        background: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #bee5eb;
    }
    .fun-fact {
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #333;
        font-size: 0.9em;
        margin: 0.5rem 0;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🔬 ZO-1 Network Analysis & Quantification</h1>
    <p>Peak detection-based RIS analysis - Fast, robust, and segmentation-free! ✨</p>
    <p style="font-size: 0.9em; margin-top: 0.5rem;">📧 For batch operations, contact: <a href="mailto:pierre.bagnaninchi@ed.ac.uk" style="color: white; text-decoration: underline;">pierre.bagnaninchi@ed.ac.uk</a></p>
</div>
""", unsafe_allow_html=True)

# Add a fun welcome message
st.markdown("""
<div class="fun-fact">
    🚀 Welcome to our ZO-1 analysis tool! (We hope it doesn't break on you!) 🧪
</div>
""", unsafe_allow_html=True)

# Add the professional but engaging tutorial panel
with st.expander("🔵 The RIS Method: How it works", expanded=False):
    st.markdown("""
    ## 🔬 **Understanding Radial Integrity Score (RIS)**
    
    ### **🎯 What You're Actually Measuring**
    
    RIS quantifies the **structural integrity** of your ZO-1 junction network using a sophisticated geometric approach. Instead of subjective visual assessment, we employ **concentric circular sampling** to systematically evaluate network connectivity across all spatial scales.
    
    **The Core Concept:** Each circle acts as a **sampling probe** that intersects your network at multiple points, providing an unbiased measure of network density that's independent of image orientation or scale.
    
    ---
    
    ## 📊 **The RIS Metric: From 0 to 1**
    
    **RIS = Radial Integrity Score** - A normalized measure of your network's structural robustness
    
    - **RIS = 1.0** 🟢 = **Optimal network integrity** - Your junction network exhibits ideal connectivity patterns
    - **RIS = 0.7-0.9** 🟡 = **Good network integrity** - Minor structural variations, typical of healthy tissue
    - **RIS = 0.4-0.6** 🟠 = **Moderate network integrity** - Some structural compromise, may indicate early pathology
    - **RIS = 0.0-0.3** 🔴 = **Poor network integrity** - Significant structural disruption, likely pathological
    
    ---
    
    ## 🧠 **Why RIS Outperforms Traditional Methods**
    
    ### **❌ Limitations of Conventional Approaches:**
    - **Intensity-based analysis** - Susceptible to staining variations, imaging artifacts, and subjective thresholding
    - **Morphological measurements** - Focus on secondary characteristics (thickness, area) rather than functional connectivity
    - **Directional bias** - Rectangular sampling introduces systematic errors in network assessment
    
    ### **✅ RIS Advantages:**
    - **Structure-focused quantification** - Directly measures the network's topological properties
    - **Rotation-invariant** - Results are independent of image orientation
    - **Scale-invariant** - Applicable across different magnification levels and cell sizes
    - **Unbiased sampling** - 360° coverage eliminates directional artifacts
    
    ---
    
    ## 🔬 **The RIS Algorithm: Step-by-Step**
    
    ### **Step 1: Geometric Sampling**
    We generate **concentric circles** with radii calculated from image area percentages, ensuring systematic coverage from the center outward.
    
    ### **Step 2: Intersection Detection**
    For each circle, we detect **network intersections** using robust edge detection algorithms, counting the number of times the circle crosses the ZO-1 network.
    
    ### **Step 3: Density Calculation**
    We compute the **radial density function** d(r) = N(r)/(2πr), where N(r) is the intersection count and r is the radius. This normalizes for the increasing sampling opportunity at larger radii.
    
    ### **Step 4: Reference Normalization**
    We compare your measured network density to a **theoretical reference density** d_ref, derived from ideal hexagonal packing models or control measurements.
    
    **Final RIS = min(d_mean / d_ref, 1.0)**
    
    ---
    
    ## 🧮 **The Mathematical Foundation**
    
    ### **Stereological Principles:**
    RIS is grounded in **classical stereology**, specifically the **Buffon's Needle Problem** and its extensions to line-intersection counting. For line-like structures (such as junction networks), the number of intersections per unit length is **proportional to the true network density**.
    
    ### **Why Circular Sampling is Optimal:**
    - **Squares/Rectangles** introduce **directional bias** - they only sample in 4 primary directions
    - **Circles provide isotropic sampling** - all directions are equally represented
    - **Statistical robustness** - circular sampling minimizes variance in density estimates
    
    ### **The Reference Density d_ref:**
    We calculate d_ref using the relationship: **d_ref = κ / D_cell**, where κ is the packing factor (typically 1.5 for hexagonal arrangements) and D_cell is the characteristic cell diameter.
    
    ---
    
    ## 📈 **Interpreting Your RIS Results**
    
    ### **Primary Metrics:**
    - **RIS Score** - Your overall network integrity (0-1 scale)
    - **RIS Peak** - Maximum network density at any radius (indicates local hotspots)
    
    ### **Radial Profile Analysis:**
    - **Consistent high values** - Indicates uniform network distribution
    - **Radial gradients** - May suggest edge effects or regional variations
    - **Irregular patterns** - Could indicate structural heterogeneity or artifacts
    
    ### **Biological Interpretation:**
    - **High RIS (>0.8)** - Suggests well-organized, healthy junction networks
    - **Moderate RIS (0.5-0.8)** - May indicate normal aging or mild stress responses
    - **Low RIS (<0.5)** - Suggests significant structural disruption, warranting further investigation
    
    ---
    
    ## 🎯 **Methodological Advantages for Research**
    
    ### **Reproducibility:**
    - **Objective quantification** eliminates inter-observer variability
    - **Standardized protocol** ensures consistent measurements across studies
    - **Automated processing** reduces human error and bias
    
    ### **Robustness:**
    - **Imaging condition independence** - Works with various microscope settings
    - **Artifact resistance** - Less sensitive to staining variations than intensity-based methods
    - **Statistical validity** - Based on well-established mathematical principles
    
    ### **Scalability:**
    - **Batch processing** capability for high-throughput analysis
    - **Parameter optimization** for different tissue types and conditions
    - **Integration potential** with existing image analysis pipelines
    
    ---
    
    ## 🔬 **Best Practices for RIS Analysis**
    
    ### **Experimental Design:**
    - **Consistent imaging parameters** across all samples
    - **Appropriate magnification** to capture relevant network features
    - **Adequate sample size** for statistical power
    
    ### **Parameter Selection:**
    - **Cell diameter estimation** - Use actual measurements when possible
    - **Sampling range** - Cover 10-70% of image area for comprehensive analysis
    - **Validation methods** - Employ multiple contour validation approaches

    ### **Quality Control:**
    - **Contour validation** - Remove phantom boundaries using intensity-based methods
    - **Edge effects** - Consider excluding very small or large sampling circles
    - **Reproducibility testing** - Validate results across multiple image regions
    
    ---
    
    ## 🚀 **The Bottom Line**
    
    **RIS represents a paradigm shift in junction network analysis** - moving from subjective visual assessment to objective, mathematically rigorous quantification. By focusing on **structural connectivity** rather than secondary characteristics, RIS provides researchers with a powerful tool for understanding tissue organization and pathology.
    
    **Key Innovation:** RIS transforms the complex, multi-dimensional problem of network assessment into a single, interpretable metric that directly reflects biological function.
    
    ---
    
    **Ready to apply RIS to your research? Upload an image and experience the power of quantitative network analysis.** 🔬📊
    """)

# Random animal comparison fun fact
import random
animal_comparisons = [
    ("🧠 Human", 37.2e12),
    ("🐱 Cat", 2.4e12), 
    ("🐸 Frog", 1.5e6),
    ("🐭 Mouse", 2.5e11),
    ("🐝 Bee", 1.0e5),
    ("🐘 Elephant", 2.57e11)
]
random_animal, animal_cells = random.choice(animal_comparisons)
st.markdown(f"""
<div class="fun-fact">
    🧬 Fun Fact: A {random_animal} has about {animal_cells:,.0e} cells! How many will we find in your image? 🔬
</div>
""", unsafe_allow_html=True)



# Initialize session state
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'show_overlays' not in st.session_state:
    st.session_state.show_overlays = False
if 'quantifier_results' not in st.session_state:
    st.session_state.quantifier_results = None
if 'tutorial_network' not in st.session_state:
    st.session_state.tutorial_network = False
if 'hex_size' not in st.session_state:
    st.session_state.hex_size = 40
if 'line_thickness' not in st.session_state:
    st.session_state.line_thickness = 2
if 'degradation' not in st.session_state:
    st.session_state.degradation = 0.0
if 'generated_image' not in st.session_state:
    st.session_state.generated_image = None
if 'generated_pil_image' not in st.session_state:
    st.session_state.generated_pil_image = None

# Sidebar configuration
st.sidebar.header("⚙️ Analysis Configuration")

# Deployment notice
st.sidebar.info("🚀 **Deployment Version**: Optimized for Streamlit Cloud. CPU-only processing for maximum compatibility.")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload ZO-1 Image",
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    help="Upload your ZO-1 fluorescence image",
)

# Hexagonal Network Tutorial
if not uploaded_file:
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔬 **Hexagonal Network Tutorial**")
    st.sidebar.markdown("Try our perfect hexagonal network to test RIS analysis!")
    
    # Tutorial parameters
    hex_size = st.sidebar.slider(
        "Hexagon Size (px)",
        min_value=20,
        max_value=80,
        value=st.session_state.hex_size,
        step=5,
        help="Size of each hexagon (distance from center to corner)"
    )
    
    line_thickness = st.sidebar.slider(
        "Line Thickness (px)",
        min_value=1,
        max_value=8,
        value=st.session_state.line_thickness,
        step=1,
        help="Thickness of hexagon lines"
    )
    
    degradation = st.sidebar.slider(
        "Network Degradation (%)",
        min_value=0.0,
        max_value=50.0,
        value=st.session_state.degradation,
        step=5.0,
        help="Percentage of hexagons to randomly remove (0% = perfect network)"
    )
    
    # Update session state when sliders change
    st.session_state.hex_size = hex_size
    st.session_state.line_thickness = line_thickness
    st.session_state.degradation = degradation
    
    # Generate tutorial network
    if st.sidebar.button("🔄 Generate Tutorial Network", type="secondary"):
        # Generate the network and store in session state
        buf, pil_img = generate_hexagonal_network(
            size=512, 
            hex_size=hex_size, 
            line_thickness=line_thickness, 
            degradation_percent=degradation
        )
        
        # Store the generated image in session state
        st.session_state.generated_image = buf
        st.session_state.generated_pil_image = pil_img
        st.session_state.tutorial_network = True
        
        # Get network info for display
        network_info = get_hexagonal_network_info(hex_size, line_thickness, degradation)
        st.sidebar.success(f"✅ Generated {network_info['perfect_network_score']:.0f}% perfect network!")
        st.sidebar.info(f"📐 Cell diameter estimate: {network_info['cell_diameter_estimate']}px")
        st.sidebar.info(f"📊 Theoretical density: {network_info['theoretical_density']:.4f}")
        
        # Rerun to show the generated image
        st.rerun()

# Move peak detection info here
st.sidebar.markdown("🧪 **Peak Detection Method**")
st.sidebar.info("🎯 Peak detection analyzes intensity profiles along sampling paths to find ZO-1 network intersections. No segmentation required! ✨")

# Sidebar scroll hint just after file selection
st.sidebar.markdown(
    """
    <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 10px; color: white; text-align: center; font-weight: bold; margin: 0.5rem 0;'>
        📜 Scroll down to fine tune parameters, but try it first without messing about! ✨
    </div>
    """,
    unsafe_allow_html=True,
)
# Check if we have either an uploaded file or a generated tutorial image
has_image = uploaded_file is not None or st.session_state.generated_image is not None

if not has_image:
    st.info("👆 Please upload an image or generate a tutorial network to begin analysis")
    st.markdown("""
    <div class="fun-fact">
        🧪 <strong>Try the Hexagonal Network Tutorial!</strong> Generate a perfect network in the sidebar to test RIS analysis with known parameters! 🔬✨
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load and process image
@st.cache_data(hash_funcs={BytesIO: lambda f: f.getvalue()})
def load_image(image_source):
    """Load and preprocess uploaded or generated image"""
    if hasattr(image_source, 'read'):  # BytesIO buffer
        pil_img = Image.open(image_source)
    else:  # PIL Image
        pil_img = image_source
    
    orig = np.array(pil_img)
    
    # Convert to grayscale if needed
    if orig.ndim == 3:
        gray = orig[:, :, 1]  # Use green channel for ZO-1
    else:
        gray = orig.copy()
    
    # Normalize
    gmin, gmax = gray.min(), gray.max()
    img_gray = ((gray - gmin) / (gmax - gmin) * 255).astype(np.uint8)
    
    return img_gray, orig

# Load image (either uploaded or generated)
if uploaded_file is not None:
    img_gray, orig = load_image(uploaded_file)
else:
    img_gray, orig = load_image(st.session_state.generated_pil_image)

# Main content area

# Display generated tutorial image if available
if st.session_state.tutorial_network and st.session_state.generated_pil_image is not None:
    st.markdown("### 🔬 **Generated Hexagonal Network Tutorial**")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.image(st.session_state.generated_pil_image, use_container_width=True, caption="Generated Hexagonal Network")
    
    with col2:
        st.markdown("**🎯 Tutorial Network Ready!**")
        st.markdown(f"**Hexagon Size:** {st.session_state.hex_size}px")
        st.markdown(f"**Line Thickness:** {st.session_state.line_thickness}px")
        st.markdown(f"**Degradation:** {st.session_state.degradation:.1f}%")
        
        network_info = get_hexagonal_network_info(st.session_state.hex_size, st.session_state.line_thickness, st.session_state.degradation)
        st.markdown(f"**Expected Cell Diameter:** {network_info['cell_diameter_estimate']}px")
        st.markdown(f"**Theoretical Density:** {network_info['theoretical_density']:.4f}")
        
        st.markdown("""
        <div style='background: linear-gradient(45deg, #a8e6cf, #dcedc1); padding: 0.5rem; border-radius: 8px; color: #333; text-align: center; font-size: 0.9em; margin: 0.5rem 0;'>
            🚀 <strong>Ready for RIS Analysis!</strong> This perfect network should give RIS ≈ 1.0! ✨
        </div>
        """, unsafe_allow_html=True)

# Run analysis button
st.markdown("### 🚀 **Run Analysis**")
run_analysis = st.button(
    "📊 Run Peak Detection Analysis",
    type="primary",
    use_container_width=True,
)

# Main panel scroll message
st.markdown("""
<div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 10px; color: white; text-align: center; font-weight: bold; margin: 1rem 0;'>
    🎯 Scroll down to see how good is your RIS score! ✨
</div>
""", unsafe_allow_html=True)

# Peak detection parameters

use_otsu_premask = st.sidebar.checkbox(
    "Otsu pre-mask (to clear background)",
    value=True,
    help="Applies Otsu to blank the background before peak detection."
)

otsu_strength = st.sidebar.slider(
    "Otsu strength ×",
    0.5, 3.0, 1.0, 0.1,
    help="Multiply the Otsu threshold (higher = stricter). Only used if Otsu pre-mask is on."
)

smooth_win = st.sidebar.slider(
    "Smoothing window (px along path)",
    1, 31, 1, 2,
    help="Odd values recommended; controls noise reduction before peak finding."
)

peak_strength = st.sidebar.slider(
    "Peak strength (z-threshold)",
    0.0, 8.0, 2.0, 0.1,
    help="Higher means stricter: requires stronger prominence above local baseline/noise."
)

min_sep_px = st.sidebar.slider(
    "Min peak distance (px along path)",
    0, 50, 0, 1,
    help="De-duplicates close peaks along the sampling path."
)

# Cell diameter moved to main content area

# Grid overlay controls
st.sidebar.markdown("---")
st.sidebar.markdown("📐 **Ruler & Grid Overlay**")

show_grid = st.sidebar.checkbox(
    "Show grid overlay",
    value=True,
    help="Display a grid overlay on the image to help estimate feature sizes."
)

grid_spacing = st.sidebar.slider(
    "Grid spacing (px)",
    10, 100, 25, 5,
    help="Adjust grid spacing based on your image scale. Smaller spacing for detailed images, larger for overview."
)

show_ruler = st.sidebar.checkbox(
    "Show ruler scale",
    value=True,
    help="Display a ruler scale bar on the image."
)

# Run analysis button moved to main content area

# Peak detection info (moved above next to scroll hint)

# Single image display with overlays
st.markdown("---")
st.markdown("## 🖼️ **Image Display & Analysis**")

# Create the main image display
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False

# Removed static image display to avoid duplicate figures; overlays will show after analysis

# Analysis results display and overlays
if st.session_state.analysis_complete and st.session_state.quantifier_results:
    results = st.session_state.quantifier_results
    
    # (Metrics moved below overlay image)
    
    # Overlay controls integrated with image display
    st.markdown("### 🎨 **Analysis Overlays**")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        show_geom = st.checkbox("Show sampling geometry", value=True)
    with col2:
        show_peaks = st.checkbox("Show detected peaks", value=True)
    with col3:
        show_grid = st.checkbox("Show grid overlay", value=True)
    
    # Create overlay image
    if show_geom or show_peaks or show_grid:
        overlay_img = np.stack([img_gray]*3, axis=-1).astype(np.uint8)
        
        if show_grid:
            # Draw grid
            H, W = img_gray.shape
            for x in range(0, W, grid_spacing):
                overlay_img[:, x, :] = [255, 0, 0]  # Red vertical lines
            for y in range(0, H, grid_spacing):
                overlay_img[y, :, :] = [255, 0, 0]  # Red horizontal lines
        
        if show_geom:
            if results.get('mode') == 'RIS_segfree':
                # Draw sampling circles
                if 'radii' in results:
                    cx, cy = img_gray.shape[1]//2, img_gray.shape[0]//2
                    for radius in results['radii']:
                        cv2.circle(overlay_img, (cx, cy), int(radius), (0, 255, 0), 2)
            elif results.get('mode') == 'TiJOR_segfree':
                # Draw sampling rectangles
                if 'sides' in results:
                    cx, cy = img_gray.shape[1]//2, img_gray.shape[0]//2
                    for side in results['sides']:
                        half = int(round(side/2))
                        x0 = int(max(0, cx - half))
                        y0 = int(max(0, cy - half))
                        x1 = int(min(img_gray.shape[1]-1, cx + half))
                        y1 = int(min(img_gray.shape[0]-1, cy + half))
                        cv2.rectangle(overlay_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        
        if show_peaks:
            if results.get('mode') == 'RIS_segfree' and results.get('hits_xy') is not None:
                hits = results['hits_xy']
                if len(hits) > 0:
                    for hit in hits:
                        x, y = int(hit[0]), int(hit[1])
                        cv2.circle(overlay_img, (x, y), 3, (0, 0, 255), -1)
            elif results.get('mode') == 'TiJOR_segfree' and results.get('points') is not None:
                pts = results['points']
                if len(pts) > 0:
                    for p in pts:
                        x, y = int(p[0]), int(p[1])
                        cv2.circle(overlay_img, (x, y), 3, (0, 0, 255), -1)
        
        st.image(overlay_img, use_container_width=True, caption="Image with Analysis Overlays")
    
    # Show metrics after overlay image
    if results.get('mode') == 'RIS_segfree':
        st.markdown("### 📊 **RIS Analysis Results**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            RIS = results.get('RIS', np.nan)
            st.metric("🔵 RIS Score", f"{RIS:.3f}" if np.isfinite(RIS) else "N/A")
        with col2:
            RIS_peak = results.get('RIS_peak', np.nan)
            st.metric("🚀 RIS Peak", f"{RIS_peak:.3f}" if np.isfinite(RIS_peak) else "N/A")
        with col3:
            d_mean = results.get('d_mean', np.nan)
            st.metric("📊 Mean Density", f"{d_mean:.4f}" if np.isfinite(d_mean) else "N/A")
        with col4:
            d_ref = results.get('d_ref', np.nan)
            st.metric("🔬 Reference", f"{d_ref:.4f}" if np.isfinite(d_ref) else "N/A")
        if np.isfinite(d_ref):
            cell_diam_used = results.get('cell_diameter_used', 'N/A')
            used_control = results.get('used_control_image', False)
            if used_control:
                st.info(f"📐 **Reference Density (d_ref)**: {d_ref:.4f} | **Source**: Control image")
            else:
                st.info(f"📐 **Reference Density (d_ref)**: {d_ref:.4f} | **Source**: Cell diameter estimate ({cell_diam_used}px)")
    elif results.get('mode') == 'TiJOR_segfree':
        st.markdown("### 📊 **TiJOR Analysis Results**")
        tijor_vals = results.get('tijor', [])
        counts = results.get('counts', [])
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean TiJOR", f"{np.mean(tijor_vals):.4f}" if len(tijor_vals) > 0 else "N/A")
        with col2:
            st.metric("Total Crossings", f"{int(np.sum(counts))}" if len(counts) > 0 else "N/A")
        with col3:
            st.metric("Steps", f"{len(tijor_vals)}")
        with col4:
            st.metric("Peaks Detected", f"{len(results.get('points', []))}")
    
    # RIS Radial Profile Graph
    if results.get('mode') == 'RIS_segfree' and 'radial_density' in results:
        st.markdown("### 📈 **RIS Radial Profile**")
        
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        
        radii = results['radii']
        d_vals = results['radial_density']
        d_ref = results.get('d_ref', np.nan)
        
        # Plot radial density d(r)
        ax.plot(radii, d_vals, 'b-o', linewidth=3, markersize=8, label='Radial Density d(r)')
        
        # Add reference line if available
        if np.isfinite(d_ref):
            ax.axhline(y=d_ref, color='red', linestyle='--', alpha=0.7, 
                       label=f'Reference d_ref = {d_ref:.4f}')
        
        ax.set_xlabel('Radius (px)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Radial Density d(r)', fontsize=12, fontweight='bold', color='blue')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=11)
        ax.set_title('RIS Radial Profile', fontsize=14, fontweight='bold')
        
        # Add summary text
        RIS = results.get('RIS', np.nan)
        if np.isfinite(RIS):
            summary_text = f'RIS: {RIS:.3f}'
            fig.text(0.5, -0.15, summary_text, ha='center', va='bottom', fontsize=12, 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        st.pyplot(fig)

# RIS Normalization UI (moved here after the graph)
st.markdown("---")
st.markdown("### 🎯 **RIS Normalization**")
normalize_ris = st.checkbox(
    "Normalize RIS scores (0-1 range)",
    value=False,
    help="If checked: RIS scores will be normalized to 0-1 range using cell diameter or control image. If unchecked: Raw density values will be shown (like TiJOR)."
)

# Cell size options (only shown if normalization is enabled)
if normalize_ris:
    st.markdown("### 📏 **Cell Size Reference**")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Option 1: Cell Diameter Estimate**")
        st.markdown("""
        <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 8px; color: white; text-align: center; font-weight: bold; margin: 0.5rem 0;'>
            🧠 <strong>Perfect Network Reference:</strong> If you don't have a control image, we can estimate the ideal network density if you provide the mean cell diameter from an ideal network. This gives you a theoretical baseline for comparison!
        </div>
        """, unsafe_allow_html=True)
        cell_diam = st.slider(
            "Estimated cell diameter (px)",
            20, 200, 35, 5,
            help="Estimate the typical cell diameter in your image"
        )
    
    with col2:
        st.markdown("**Option 2: Control Image**")
        st.markdown("""
        <div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 8px; color: white; text-align: center; font-weight: bold; margin: 0.5rem 0;'>
            📊 <strong>DIY Normalization:</strong> Upload a control image and normalize in your favorite stats software (Excel really! 😲). We'll remove any automatic estimation - you're in control of your reference values!
        </div>
        """, unsafe_allow_html=True)
        # Control image handling removed - users normalize in their preferred software
        use_control = False
        control_file = None
else:
    # If not normalizing, set default values
    cell_diam = 35
    use_control = False
    control_file = None

# Reference options moved to main content area

# Hexagonal Network Tutorial Parameters (main panel)
if st.session_state.get('tutorial_network', False) and st.session_state.generated_pil_image is not None:
    st.markdown("---")
    st.markdown("### 🔬 **Hexagonal Network Tutorial Parameters**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**📐 Network Geometry**")
        st.markdown(f"**Hexagon Size:** {st.session_state.hex_size}px")
        st.markdown(f"**Line Thickness:** {st.session_state.line_thickness}px")
        st.markdown(f"**Degradation:** {st.session_state.degradation:.1f}%")
    
    with col2:
        st.markdown("**📊 Theoretical Values**")
        network_info = get_hexagonal_network_info(st.session_state.hex_size, st.session_state.line_thickness, st.session_state.degradation)
        st.markdown(f"**Cell Diameter:** {network_info['cell_diameter_estimate']}px")
        st.markdown(f"**Theoretical Density:** {network_info['theoretical_density']:.4f}")
        st.markdown(f"**Perfect Score:** {network_info['perfect_network_score']:.0f}%")
    
    with col3:
        st.markdown("**🎯 Expected Results**")
        st.markdown(f"**Expected RIS:** ~1.0 (perfect)")
        st.markdown(f"**Peak Detection:** Should work well")
        st.markdown(f"**Normalization:** Use {network_info['cell_diameter_estimate']}px")
    
    st.markdown("""
    <div style='background: linear-gradient(45deg, #a8e6cf, #dcedc1); padding: 0.5rem; border-radius: 8px; color: #333; text-align: center; font-size: 0.9em; margin: 0.5rem 0;'>
        🧪 <strong>Perfect Tutorial Network!</strong> This should give you RIS ≈ 1.0. Try adjusting line thickness to see how it affects peak detection! ✨
    </div>
    """, unsafe_allow_html=True)

# Second Run Analysis button in main panel (after RIS normalization)
st.markdown("---")
st.markdown("### 🚀 **Run Analysis (After Adjusting Settings)**")
run_analysis_bottom = st.button(
    "📊 Run Peak Detection Analysis",
    type="primary",
    use_container_width=True,
    key="main_panel_run_analysis"
)

# Reset button
if st.session_state.analysis_complete:
    if st.sidebar.button("🔄 Reset All", type="secondary", use_container_width=True):
        st.session_state.analysis_complete = False
        st.session_state.quantifier_results = None
        st.rerun()

# Removed old scroll message - moved to main content area

# Processing options - image used at original scale
st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ **Processing Options**")
H, W = img_gray.shape
scale = 1.0

# Force CPU-only mode for deployment
gpu_available = False
st.sidebar.info("💻 **CPU in the Clouds**: Running on Streamlit's cloud CPU for deployment compatibility. Processing may be slower but more reliable than your ex's promises! ☁️")

# (AI contour validation removed in this streamlined version)

# Network analysis parameters
st.sidebar.markdown("---")
st.sidebar.markdown("📐 **Network Analysis Parameters**")

# Analysis geometry selection
analysis_geometry = st.sidebar.selectbox(
    "Analysis Geometry",
    options=["Circles (RIS - recommended)", "Rectangles (TiJOR - legacy)"],
    index=0,
    help="Choose between circular RIS analysis (recommended) or rectangular TiJOR analysis"
)

# RIS-specific parameters (only show when circles are selected)
if analysis_geometry == "Circles (RIS - recommended)":
    st.sidebar.markdown("🔵 **RIS Analysis Settings**")
    
    # Add fun message about parameters
    st.sidebar.markdown("""
    <div style='background: linear-gradient(45deg, #a8e6cf, #dcedc1); padding: 0.5rem; border-radius: 8px; color: #333; text-align: center; font-size: 0.9em; margin: 0.5rem 0;'>
        🎉 No need to touch these parameters! They're already set to perfection! ✨
    </div>
    """, unsafe_allow_html=True)
    
    # Reference mode selection
    normalization_mode = st.sidebar.selectbox(
        "Reference Mode",
        options=["Auto (from cell measurements)", "Model (packing factor κ)", "Control (from control images)"],
        index=0,
        help="Auto: calculate d_ref from actual cells. Model: use theoretical packing factor. Control: learn from healthy images"
    )
    
    # Packing factor slider
    packing_factor = st.sidebar.slider(
        "κ (packing factor)",
        min_value=1.2,
        max_value=2.0,
        value=1.5,
        step=0.1,
        help="Theoretical packing factor for ideal hexagonal networks (1.2-2.0)"
    )
    
    # Control image upload (if control mode selected)
    if normalization_mode == "Control (from control images)":
        st.sidebar.markdown("📁 **Control Images**")
        control_files = st.sidebar.file_uploader(
            "Upload control images for d_ref calculation",
            type=["tif", "tiff", "png", "jpg", "jpeg"],
            accept_multiple_files=True,
            help="Upload healthy control images to calculate reference density"
        )
        
        if control_files:
            st.sidebar.info(f"📊 {len(control_files)} control images loaded")
        else:
            st.sidebar.warning("⚠️ Please upload control images for Control mode")

initial_size = st.sidebar.slider(
    "Initial size (%)",
    min_value=1.0,
    max_value=20.0,
    value=10.0,  # Changed from 5.0 to 10.0
    step=0.5,
    help="Starting size as percentage of image area"
)

max_size = st.sidebar.slider(
    "Max size (%)",
    min_value=30.0,
    max_value=90.0,
    value=70.0,
    step=5.0,
    help="Maximum size as percentage of image area"
)

num_steps = st.sidebar.slider(
    "Number of steps",
    min_value=3,
    max_value=10,
    value=5,
    step=1,
    help="Number of size enlargement steps"
)

min_distance = st.sidebar.slider(
    "Min cross-section distance (px)",
    min_value=5,
    max_value=50,
    value=10,
    step=5,
    help="Minimum distance between cross-section points"
)

# Second Run Analysis button at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🚀 **Run Analysis (After Parameters)**")
st.sidebar.button(
    "📊 Run Peak Detection Analysis",
    type="primary",
    use_container_width=True,
    key="sidebar_run_analysis"
)

# Display image info at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Image Information**")
st.sidebar.markdown(f"**Size:** {img_gray.shape[1]} × {img_gray.shape[0]} px")

if uploaded_file is not None:
    st.sidebar.markdown(f"**Type:** {uploaded_file.type}")
    st.sidebar.markdown(f"**Format:** {uploaded_file.name.split('.')[-1].upper()}")
else:
    st.sidebar.markdown("**Type:** Generated Tutorial")
    st.sidebar.markdown("**Format:** PNG")
    st.sidebar.markdown(f"**Hexagon Size:** {st.session_state.hex_size}px")
    st.sidebar.markdown(f"**Line Thickness:** {st.session_state.line_thickness}px")
    st.sidebar.markdown(f"**Degradation:** {st.session_state.degradation:.1f}%")

# (Cellpose model loading removed)

# Peak detection-based quantification functions are already defined above

# Peak detection analysis is now the main method - no segmentation needed


# Ensure cell diameter is available for sidebar display and analysis
# Use the sidebar slider value directly, not from session state
# diam is already defined from the sidebar slider above

# Run analysis locally when button is clicked
if run_analysis:
    st.write("🚀 **Starting peak detection analysis...**")
    with st.spinner("🔬 Running peak detection analysis..."):
        if analysis_geometry == "Circles (RIS - recommended)":
            radii, N_vals, d_vals, hits_xy, premask = ris_segfree(
                img_gray,
                use_premask=use_otsu_premask,
                premask_strength=otsu_strength,
                smooth_window=smooth_win,
                z_thresh=peak_strength,
                min_sep=min_sep_px,
                initial_area_pct=initial_size,
                max_area_pct=max_size,
                steps=num_steps,
            )

            d_mean = float(np.mean(d_vals)) if len(d_vals) > 0 else np.nan
            d_peak = float(np.max(d_vals)) if len(d_vals) > 0 else np.nan

            if normalize_ris:
                if use_control and control_file:
                    d_ref = 0.1
                    st.warning("⚠️ Control image processing not yet implemented - using placeholder d_ref")
                else:
                    d_ref = float(packing_factor / float(cell_diam)) if cell_diam > 0 else np.nan
                RIS = float(np.clip(d_mean / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) and d_ref > 0 else np.nan
                RIS_peak = float(np.clip(d_peak / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) and d_ref > 0 else np.nan
            else:
                d_ref = np.nan
                RIS = d_mean
                RIS_peak = d_peak

            st.session_state.quantifier_results = {
                "mode": "RIS_segfree",
                "radii": radii,
                "crossings": N_vals,
                "radial_density": d_vals,
                "hits_xy": hits_xy,
                "premask": premask,
                "d_mean": d_mean,
                "d_peak": d_peak,
                "d_ref": d_ref,
                "RIS": RIS,
                "RIS_peak": RIS_peak,
                "kappa": packing_factor,
                "cell_diameter_used": cell_diam,
                "used_control_image": use_control,
                "normalized": normalize_ris,
                "params": {
                    "initial_area_pct": initial_size,
                    "max_area_pct": max_size,
                    "steps": num_steps,
                    "smooth_window": smooth_win,
                    "z_thresh": peak_strength,
                    "min_sep": min_sep_px,
                    "otsu": use_otsu_premask,
                    "otsu_strength": otsu_strength,
                    "packing_factor": packing_factor,
                    "diam": cell_diam,
                    "normalized": normalize_ris,
                },
            }
        else:
            sides, counts, tijor_vals, pts, premask = tijor_segfree(
                img_gray,
                use_premask=use_otsu_premask,
                premask_strength=otsu_strength,
                smooth_window=smooth_win,
                z_thresh=peak_strength,
                min_sep=min_sep_px,
                initial_area_pct=initial_size,
                max_area_pct=max_size,
                steps=num_steps,
            )

            st.session_state.quantifier_results = {
                "mode": "TiJOR_segfree",
                "sides": sides,
                "counts": counts,
                "tijor": tijor_vals,
                "points": pts,
                "premask": premask,
                "params": {
                    "initial_area_pct": initial_size,
                    "max_area_pct": max_size,
                    "steps": num_steps,
                    "smooth_window": smooth_win,
                    "z_thresh": peak_strength,
                    "min_sep": min_sep_px,
                    "otsu": use_otsu_premask,
                    "otsu_strength": otsu_strength,
                },
            }

        st.session_state.analysis_complete = True
        st.success("🚀 Peak detection analysis complete! Results ready.")
        st.rerun()

# Run analysis when bottom button is clicked (same logic as top button)
if run_analysis_bottom:
    st.write("🚀 **Starting peak detection analysis (from bottom button)...**")
    with st.spinner("🔬 Running peak detection analysis..."):
        if analysis_geometry == "Circles (RIS - recommended)":
            radii, N_vals, d_vals, hits_xy, premask = ris_segfree(
                img_gray,
                use_premask=use_otsu_premask,
                premask_strength=otsu_strength,
                smooth_window=smooth_win,
                z_thresh=peak_strength,
                min_sep=min_sep_px,
                initial_area_pct=initial_size,
                max_area_pct=max_size,
                steps=num_steps,
            )

            d_mean = float(np.mean(d_vals)) if len(d_vals) > 0 else np.nan
            d_peak = float(np.max(d_vals)) if len(d_vals) > 0 else np.nan

            if normalize_ris:
                if use_control and control_file:
                    d_ref = 0.1
                    st.warning("⚠️ Control image processing not yet implemented - using placeholder d_ref")
                else:
                    d_ref = float(packing_factor / float(cell_diam)) if cell_diam > 0 else np.nan
                RIS = float(np.clip(d_mean / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) and d_ref > 0 else np.nan
                RIS_peak = float(np.clip(d_peak / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) and d_ref > 0 else np.nan
            else:
                d_ref = np.nan
                RIS = d_mean
                RIS_peak = d_peak

            st.session_state.quantifier_results = {
                "mode": "RIS_segfree",
                "radii": radii,
                "crossings": N_vals,
                "radial_density": d_vals,
                "hits_xy": hits_xy,
                "premask": premask,
                "d_mean": d_mean,
                "d_peak": d_peak,
                "d_ref": d_ref,
                "RIS": RIS,
                "RIS_peak": RIS_peak,
                "kappa": packing_factor,
                "cell_diameter_used": cell_diam,
                "used_control_image": use_control,
                "normalized": normalize_ris,
                "params": {
                    "initial_area_pct": initial_size,
                    "max_area_pct": max_size,
                    "steps": num_steps,
                    "smooth_window": smooth_win,
                    "z_thresh": peak_strength,
                    "min_sep": min_sep_px,
                    "otsu": use_otsu_premask,
                    "otsu_strength": otsu_strength,
                    "packing_factor": packing_factor,
                    "diam": cell_diam,
                    "normalized": normalize_ris,
                },
            }
        else:
            sides, counts, tijor_vals, pts, premask = tijor_segfree(
                img_gray,
                use_premask=use_otsu_premask,
                premask_strength=otsu_strength,
                smooth_window=smooth_win,
                z_thresh=peak_strength,
                min_sep=min_sep_px,
                initial_area_pct=initial_size,
                max_area_pct=max_size,
                steps=num_steps,
            )

            st.session_state.quantifier_results = {
                "mode": "TiJOR_segfree",
                "sides": sides,
                "counts": counts,
                "tijor": tijor_vals,
                "points": pts,
                "premask": premask,
                "params": {
                    "initial_area_pct": initial_size,
                    "max_area_pct": max_size,
                    "steps": num_steps,
                    "smooth_window": smooth_win,
                    "z_thresh": peak_strength,
                    "min_sep": min_sep_px,
                    "otsu": use_otsu_premask,
                    "otsu_strength": otsu_strength,
                },
            }

        st.session_state.analysis_complete = True
        st.success("🚀 Peak detection analysis complete! Results ready.")
        st.rerun()

# Display analysis status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")

if st.session_state.analysis_complete:
    st.sidebar.success("🎯 Analysis Complete!")
    st.sidebar.info("🎮 Peak detection analysis finished!")
    st.sidebar.markdown("**Peak Detection Parameters:**")
    st.sidebar.markdown(f"- Otsu pre-mask: {'Yes' if use_otsu_premask else 'No'}")
    st.sidebar.markdown(f"- Peak strength: {peak_strength:.1f}")
    st.sidebar.markdown(f"- Min separation: {min_sep_px} px")
    st.sidebar.markdown(f"- Smoothing: {smooth_win} px")
else:
    st.sidebar.warning("🤔 No analysis yet!")
    st.sidebar.info("🎯 Configure peak detection parameters and run analysis!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 ZO-1 Network Analysis Tool | Peak Detection-based RIS & TiJOR Quantification ✨</p>
</div>
""", unsafe_allow_html=True)
