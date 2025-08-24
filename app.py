#!/usr/bin/env python3
"""
Enhanced ZO-1 Network Analysis App with Streamlit
Implements RIS (Radial Integrity Score) and TiJOR quantification using concentric circles and expanding rectangles
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import cv2
from PIL import Image
import pandas as pd
from io import StringIO

# Cellpose imports
from cellpose import models
from skimage.segmentation import find_boundaries

# AI validation imports
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# Classic CPU segmentation
from classic_seg import (
    segment_zo1_gmm,
    segment_zo1_kmeans,
    segment_zo1_otsu,
    compute_cell_metrics,
)

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
    <p>Advanced segmentation and RIS analysis using cutting-edge AI magic ✨</p>
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
    - **Validation methods** - Employ multiple AI-based contour validation approaches
    
    ### **Quality Control:**
    - **Contour validation** - Remove phantom boundaries using AI methods
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
if 'segmentation_complete' not in st.session_state:
    st.session_state.segmentation_complete = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'show_overlays' not in st.session_state:
    st.session_state.show_overlays = False

# Sidebar configuration
st.sidebar.header("⚙️ Analysis Configuration")

# Deployment notice
st.sidebar.info("🚀 **Deployment Version**: Optimized for Streamlit Cloud. CPU-only processing for maximum compatibility.")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "📁 Upload ZO-1 Image", 
    type=["tif", "tiff", "png", "jpg", "jpeg"],
    help="Upload your ZO-1 fluorescence image"
)

if not uploaded_file:
    st.info("👆 Please upload an image to begin analysis")
    st.markdown("""
    <div class="fun-fact">
        📸 Don't forget to upload your ZO-1 image! (We promise not to judge the quality 😉) 🔍
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# Load and process image
@st.cache_data
def load_image(uploaded_file):
    """Load and preprocess uploaded image"""
    pil_img = Image.open(uploaded_file)
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

# Load image
img_gray, orig = load_image(uploaded_file)

# Segmentation engine selection
seg_engine = st.sidebar.selectbox(
    "Segmentation Engine",
    ["Cellpose (AI)", "Classic (CPU)"],
    index=0,
)

if seg_engine == "Cellpose (AI)":
    st.sidebar.markdown("🔬 **AI Segmentation Parameters**")
    diam = st.sidebar.slider(
        "Estimated Cell diameter (px) for better qulity segmentation! 🎯",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="Estimate the size of cells in pixels from your image. This is just an estimate - try again if you're not happy with the segmentation! 🧪",
    )
else:
    st.sidebar.markdown("🧪 **Classic Segmentation Parameters**")
    classic_method = st.sidebar.selectbox(
        "Method",
        ["Otsu", "GMM", "K-means", "Adaptive"],
        index=0,
    )
    smooth_sigma = st.sidebar.slider("Smoothing σ", 0.0, 2.0, 1.0, 0.1)
    min_obj = st.sidebar.slider("Min object size (px)", 50, 1000, 200, 50)
    min_peak_dist = st.sidebar.slider("Seed min_distance (px)", 1, 20, 8)
    use_ridge = st.sidebar.checkbox("Ridge filter", True)
    skeleton_thickness = st.sidebar.slider("Skeleton thickness (px)", 1, 5, 1)
    adaptive = False
    block = 51
    offset = 0.0
    thresh_multiplier = 1.0
    if classic_method in ("Otsu", "Adaptive"):
        if classic_method == "Adaptive":
            adaptive = True
            block = st.sidebar.slider("Block size", 3, 255, 51, step=2)
            offset = st.sidebar.slider("Offset", -30, 30, 0)
        else:
            thresh_multiplier = st.sidebar.slider(
                "Otsu threshold multiplier", 0.5, 1.5, 1.0, 0.1
            )
    pixel_size = st.sidebar.number_input(
        "Pixel size (µm/px)",
        min_value=0.0,
        value=0.0,
    )

# Run buttons - moved to top for easy access
st.sidebar.markdown("---")
st.sidebar.markdown("🚀 **Run Analysis**")

col1, col2 = st.sidebar.columns(2)

with col1:
    seg_button_label = (
        "🔬 Run Segmentation" if seg_engine == "Cellpose (AI)" else "🔬 Run Classic Segmentation"
    )
    run_segmentation = st.button(
        seg_button_label,
        type="primary",
        use_container_width=True,
    )

with col2:
    run_analysis = st.button(
        "📊 Run Analysis",
        type="secondary",
        use_container_width=True,
        disabled=not st.session_state.segmentation_complete
    )

# Reset button
if st.session_state.segmentation_complete:
    if st.sidebar.button("🔄 Reset All", type="secondary", use_container_width=True):
        st.session_state.segmentation_complete = False
        st.session_state.analysis_complete = False
        st.session_state.masks = None
        st.session_state.membrane_mask = None
        st.session_state.quantifier = None
        st.rerun()

# Add updated scroll message
st.sidebar.markdown("""
<div style='background: linear-gradient(45deg, #ff6b6b, #4ecdc4); padding: 0.5rem; border-radius: 10px; color: white; text-align: center; font-weight: bold; margin: 1rem 0;'>
    📜 Scroll down to fine tune, but try it first without messing about! ✨
</div>
""", unsafe_allow_html=True)

# Processing options - moved after run buttons
st.sidebar.markdown("---")
st.sidebar.markdown("⚙️ **Processing Options**")

# Resampling options
rescale_option = st.sidebar.selectbox(
    "Image Processing",
    options=["Rescale to 512x512 (default)", "No rescale (be ready to wait!)"],
    index=0,  # 512x512 is default
    help="Rescale to 512x512 for faster processing, or keep original size (much slower!)"
)

# Convert to scale factor for internal use - REAL 512px resampling
H, W = img_gray.shape
if rescale_option == "Rescale to 512x512 (default)":
    TARGET = 512
    scale = min(1.0, TARGET / max(H, W))
    w_small, h_small = int(round(W * scale)), int(round(H * scale))
    st.sidebar.info(f"🔍 **Resampling Active**: {W}×{H} → {w_small}×{h_small} (scale={scale:.3f})")
else:
    scale = 1.0
    w_small, h_small = W, H
    st.sidebar.warning("🐌 **Full Resolution**: Processing at original image size. This will take much longer - grab a coffee! ☕")

# Force CPU-only mode for deployment
gpu_available = False
st.sidebar.info("💻 **CPU in the Clouds**: Running on Streamlit's cloud CPU for deployment compatibility. Processing may be slower but more reliable than your ex's promises! ☁️")

# Contour validation parameters
st.sidebar.markdown("---")
st.sidebar.markdown("🔍 **Contour Validation Parameters**")

enable_contour_validation = st.sidebar.checkbox(
    "Enable AI Contour Validation", 
    value=True,
    help="Remove phantom boundaries using AI-powered methods"
)

if enable_contour_validation:
    validation_method = st.sidebar.selectbox(
        "AI Validation Method",
        options=["K-means clustering", "Gaussian Mixture Model (GMM)", "Otsu thresholding"],
        index=1,  # GMM as default
        help="Choose AI method for contour validation"
    )
    
    if validation_method == "K-means clustering":
        st.sidebar.info("🎯 **K-means clustering**: Handles bimodal images better than Otsu when histogram isn't cleanly split")
    elif validation_method == "Gaussian Mixture Model (GMM)":
        st.sidebar.info("🔮 **GMM**: Smoother than K-means (probabilistic) and can adapt to skewed histograms")
    else:  # Otsu
        st.sidebar.info("📊 **Otsu**: Classic automatic thresholding (legacy method)")
    
    # Method-specific parameters
    if validation_method == "K-means clustering":
        kmeans_iterations = st.sidebar.slider(
            "K-means Iterations",
            min_value=100,
            max_value=1000,
            value=300,
            step=100,
            help="Number of iterations for K-means convergence"
        )
    elif validation_method == "Gaussian Mixture Model (GMM)":
        gmm_covariance_type = st.sidebar.selectbox(
            "GMM Covariance Type",
            options=["full", "tied", "diag", "spherical"],
            index=0,
            help="Covariance type for GMM fitting"
        )
    else:  # Otsu
        otsu_strength = st.sidebar.slider(
            "💪 Otsu Strength Multiplier",
            min_value=0.5,
            max_value=3.0,
            value=1.0,
            step=0.1,
            help="Make Otsu threshold more aggressive for noisy images (higher = stricter filtering)"
        )

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



# Display image info at bottom of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("📊 **Image Information**")
st.sidebar.markdown(f"**Size:** {img_gray.shape[1]} × {img_gray.shape[0]} px")
st.sidebar.markdown(f"**Type:** {uploaded_file.type}")
st.sidebar.markdown(f"**Format:** {uploaded_file.name.split('.')[-1].upper()}")

# Load AI model
@st.cache_resource
def load_ai_model():
    """Load our super-smart AI model with caching (CPU-only for deployment)"""
    try:
        model = models.CellposeModel(pretrained_model='cyto2', gpu=False)
        return model
    except Exception as e:
        st.error(f"🤖 Oops! Our AI model is having a moment: {e}")
        return None

cp_model = load_ai_model()

# Network quantification class
class ZO1NetworkQuantifier:
    """ZO-1 Network Quantification using TiJOR methodology (Legacy)"""
    
    def __init__(self):
        self.results = {}
    
    def quantify_network(self, membrane_mask, image_shape, 
                        initial_size=0.05, max_size=0.7, num_steps=5, 
                        min_cross_section_distance=10):
        """Quantify ZO-1 network organization using expanding rectangles"""
        
        image_height, image_width = image_shape
        total_image_area = image_height * image_width
        
        # Calculate rectangle sizes
        initial_side_length = np.sqrt(total_image_area * initial_size)
        max_side_length = np.sqrt(total_image_area * max_size)
        step_size = (max_side_length - initial_side_length) / num_steps
        
        # Center of image
        center_x = image_width / 2
        center_y = image_height / 2
        
        # Initialize storage arrays
        tijor_values = np.zeros(num_steps + 1)
        all_cross_section_counts = np.zeros(num_steps + 1)
        
        # Process each rectangle size
        for step in range(num_steps + 1):
            current_side_length = initial_side_length + step * step_size
            half_side = current_side_length / 2
            
            # Calculate cross-section points for this rectangle
            cross_sections = self._get_cross_section_points(
                center_x, center_y, half_side, membrane_mask
            )
            
            # Store cross-section count for this rectangle
            num_cross_sections = len(cross_sections)
            all_cross_section_counts[step] = num_cross_sections
            
            # Calculate TiJOR based on cross-sections
            polygon_perimeter = 4 * current_side_length
            tijor_values[step] = num_cross_sections / polygon_perimeter if polygon_perimeter > 0 else 0
        
        # Collect all cross-section points from all rectangles
        all_cross_sections = []
        for step in range(num_steps + 1):
            current_side_length = initial_side_length + step * step_size
            half_side = current_side_length / 2
            
            cross_sections = self._get_cross_section_points(
                center_x, center_y, half_side, membrane_mask
            )
            all_cross_sections.extend(cross_sections)
        
        # Filter cross-section points to avoid overlap
        filtered_cross_sections = self._filter_cross_sections_by_distance(
            all_cross_sections, min_cross_section_distance
        )
        
        # Calculate filtered counts per rectangle for consistent visualization
        filtered_counts_per_rectangle = np.zeros(num_steps + 1)
        for step in range(num_steps + 1):
            current_side_length = initial_side_length + step * step_size
            half_side = current_side_length / 2
            
            rect_cross_sections = self._get_cross_section_points(
                center_x, center_y, half_side, membrane_mask
            )
            
            filtered_count = 0
            for rect_point in rect_cross_sections:
                for filtered_point in filtered_cross_sections:
                    if rect_point[0] == filtered_point[0] and rect_point[1] == filtered_point[1]:
                        filtered_count += 1
                        break
            
            filtered_counts_per_rectangle[step] = filtered_count
        
        # Store results
        self.results = {
            'tijor_values': tijor_values,
            'cross_section_counts': all_cross_section_counts,
            'filtered_cross_section_counts': filtered_counts_per_rectangle,
            'cross_section_points': filtered_cross_sections,
            'rectangle_sizes': [initial_side_length + step * step_size for step in range(num_steps + 1)],
            'parameters': {
                'initial_size': initial_size,
                'max_size': max_size,
                'num_steps': num_steps,
                'min_cross_section_distance': min_cross_section_distance
            }
        }
        
        return self.results
    
    def _get_cross_section_points(self, center_x, center_y, half_side, membrane_mask):
        """Get points where rectangle edges intersect with membrane network"""
        cross_sections = []
        
        x_min = max(0, int(center_x - half_side))
        x_max = min(membrane_mask.shape[1], int(center_x + half_side))
        y_min = max(0, int(center_y - half_side))
        y_max = min(membrane_mask.shape[0], int(center_y + half_side))
        
        # Check all four edges
        for x in range(x_min, x_max + 1):
            if y_min < membrane_mask.shape[0] and membrane_mask[y_min, x] > 0:
                cross_sections.append([y_min, x])
            if y_max < membrane_mask.shape[0] and membrane_mask[y_max, x] > 0:
                cross_sections.append([y_max, x])
        
        for y in range(y_min, y_max + 1):
            if x_min < membrane_mask.shape[1] and membrane_mask[y, x_min] > 0:
                cross_sections.append([y, x_min])
            if x_max < membrane_mask.shape[1] and membrane_mask[y, x_max] > 0:
                cross_sections.append([y, x_max])
        
        return cross_sections
    
    def _filter_cross_sections_by_distance(self, cross_sections, min_distance):
        """Filter cross-section points to maintain minimum distance"""
        if not cross_sections or len(cross_sections) <= 1:
            return cross_sections
        
        points = np.array(cross_sections)
        filtered_points = [points[0]]
        
        for i in range(1, len(points)):
            current_point = points[i]
            keep_point = True
            
            for kept_point in filtered_points:
                distance = np.sqrt((current_point[0] - kept_point[0])**2 + 
                                 (current_point[1] - kept_point[1])**2)
                if distance < min_distance:
                    keep_point = False
                    break
            
            if keep_point:
                filtered_points.append(current_point)
        
        return filtered_points
    
    def get_summary_stats(self):
        """Get summary statistics"""
        if not self.results:
            return {}
        
        tijor_values = self.results['tijor_values']
        filtered_counts = self.results['filtered_cross_section_counts']
        
        return {
            'mean_tijor': np.mean(tijor_values),
            'std_tijor': np.std(tijor_values),
            'max_tijor': np.max(tijor_values),
            'min_tijor': np.min(tijor_values),
            'total_cross_sections': np.sum(filtered_counts),
            'mean_cross_sections_per_rectangle': np.mean(filtered_counts)
        }

# New RIS quantifier class
class ZO1RISQuantifier:
    """
    Circular (Sholl-style) crossings-per-circumference for ZO-1 networks.
    Produces d(r), RIS in [0,1], and a few companion metrics.
    """
    def __init__(self, packing_factor=1.5):
        self.kappa = float(packing_factor)  # "packing factor" slider (1.2–2.0)
        self.results = {}

    def calculate_dref_from_cells(self, masks, membrane_mask, d_eff_pixels, scale_factor=1.0):
        """
        Calculate reference density d_ref from actual cell measurements.
        This is more accurate than using theoretical packing factors.
        
        Args:
            masks: Cellpose segmentation masks
            membrane_mask: Validated membrane network mask
            d_eff_pixels: User-provided cell diameter estimate
            scale_factor: Image scaling factor (1.0 = original, 0.5 = half size, etc.)
            
        Returns:
            d_ref: Calculated reference density
            cell_stats: Dictionary with cell measurements
        """
        if masks is None or masks.max() == 0:
            # Fallback to theoretical calculation
            return self.kappa / float(d_eff_pixels), {}
        
        # Calculate actual cell properties
        cell_areas = []
        cell_diameters = []
        
        for i in range(1, int(masks.max()) + 1):
            cell_mask = (masks == i)
            if np.sum(cell_mask) > 0:
                # Calculate cell area
                area = np.sum(cell_mask)
                cell_areas.append(area)
                
                # Calculate equivalent diameter
                diameter = np.sqrt(4 * area / np.pi)
                cell_diameters.append(diameter)
        
        if not cell_areas:
            # Fallback to theoretical calculation
            return self.kappa / float(d_eff_pixels), {}
        
        # Calculate statistics
        avg_cell_area = np.mean(cell_areas)
        avg_cell_diameter = np.mean(cell_diameters)
        std_cell_diameter = np.std(cell_diameters)
        
        # 🚨 CRITICAL FIX: Scale the measured cell diameter back to original image coordinates
        # If we resampled the image, the measured diameters are in resampled coordinates
        # We need to scale them back to match the user's diameter estimate
        if scale_factor != 1.0:
            avg_cell_diameter_original = avg_cell_diameter / scale_factor
            std_cell_diameter_original = std_cell_diameter / scale_factor
            avg_cell_area_original = avg_cell_area / (scale_factor ** 2)
        else:
            avg_cell_diameter_original = avg_cell_diameter
            std_cell_diameter_original = std_cell_diameter
            avg_cell_area_original = avg_cell_area
        
        # Calculate actual network properties
        total_network_pixels = np.sum(membrane_mask > 0)
        image_area = membrane_mask.shape[0] * membrane_mask.shape[1]
        
        # Network density (pixels of network per total image area)
        network_density = total_network_pixels / image_area
        
        # Cell density (cells per total image area)
        cell_density = len(cell_areas) / image_area
        
        # 🚨 IMPORTANT: Use Cellpose's calculated diameter for d_ref (scaled to original coordinates)
        # This gives us the most accurate cell size from the actual image
        
        # Method 1: Cellpose-measured diameter (most accurate, properly scaled)
        d_ref_cellpose = self.kappa / avg_cell_diameter_original
        
        # Method 2: User-provided diameter (for reference only)
        d_ref_user = self.kappa / float(d_eff_pixels)
        
        # Use Cellpose d_ref for accuracy
        d_ref_final = d_ref_cellpose
        
        # Store cell statistics
        cell_stats = {
            'num_cells': len(cell_areas),
            'avg_cell_area': avg_cell_area_original,  # Scaled to original coordinates
            'avg_cell_diameter': avg_cell_diameter_original,  # Scaled to original coordinates
            'std_cell_diameter': std_cell_diameter_original,  # Scaled to original coordinates
            'avg_cell_area_resampled': avg_cell_area,  # In resampled coordinates (for reference)
            'avg_cell_diameter_resampled': avg_cell_diameter,  # In resampled coordinates (for reference)
            'network_density': network_density,
            'cell_density': cell_density,
            'd_ref_cellpose': d_ref_cellpose,
            'd_ref_user': d_ref_user,
            'd_ref_final': d_ref_final,
            'scale_factor': scale_factor,
            'note': f'Using Cellpose-measured d_ref for accuracy (scaled from {scale_factor}x resampled image)'
        }
        
        return d_ref_final, cell_stats

    def quantify_network(
        self,
        membrane_mask, image_shape, masks=None,
        initial_area_frac=0.05, max_area_frac=0.70, num_steps=5,
        min_cross_section_distance=10,
        center=None,
        d_eff_pixels=100,  # effective cell diameter from the UI
        normalization_mode="model",  # "model", "control", or "auto"
        control_dref=None,          # pass a float if normalization_mode=="control"
        use_auto_dref=True          # whether to use automatic d_ref calculation
    ):
        H, W = image_shape
        cx = W/2 if center is None else float(center[0])
        cy = H/2 if center is None else float(center[1])

        # radii from area %
        A = H * W
        r0 = np.sqrt((initial_area_frac * A) / np.pi)
        r1 = np.sqrt((max_area_frac * A) / np.pi)
        radii = np.linspace(r0, r1, num_steps + 1)

        d_vals, N_vals, hits_all = [], [], []

        for r in radii:
            thetas, pts = self._sample_circle(cx, cy, r, H, W)
            ring = membrane_mask[pts[:,1], pts[:,0]] > 0
            # 0->1 transitions (wrap)
            shifted = np.roll(ring.astype(np.uint8), 1)
            transitions = (ring.astype(np.uint8) - shifted) == 1
            N = int(transitions.sum())

            # optional de-dupe for plotted "hits"
            ang = thetas[transitions]
            ang = self._dedupe_angles(ang, max(1e-6, min_cross_section_distance/max(1.0, r)))
            if ang.size:
                hx = cx + r*np.cos(ang)
                hy = cy + r*np.sin(ang)
                hits_all.append(np.stack([hx, hy], axis=1))

            N_vals.append(N)
            d_vals.append(N / (2*np.pi*r) if r > 0 else 0.0)

        radii = np.asarray(radii)
        N_vals = np.asarray(N_vals)
        d_vals = np.asarray(d_vals)

        # Calculate reference density d_ref
        d_ref = np.nan
        cell_stats = {}
        
        if normalization_mode == "control" and control_dref is not None and control_dref > 0:
            # Use user-provided control reference
            d_ref = float(control_dref)
        elif normalization_mode == "auto" and use_auto_dref and masks is not None:
            # Calculate d_ref from actual cell measurements
            d_ref, cell_stats = self.calculate_dref_from_cells(masks, membrane_mask, d_eff_pixels)
        else:
            # Fallback to theoretical calculation
            d_ref = (self.kappa / float(d_eff_pixels)) if d_eff_pixels > 0 else np.nan

        d_mean = float(np.mean(d_vals))
        d_peak = float(np.max(d_vals))
        RIS = float(np.clip(d_mean / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) else np.nan
        RIS_peak = float(np.clip(d_peak / d_ref, 0.0, 1.0)) if np.isfinite(d_ref) else np.nan

        self.results = {
            "radii": radii,
            "crossings": N_vals,          # N(r)
            "radial_density": d_vals,     # d(r) = N(r)/(2πr)
            "d_mean": d_mean,
            "d_peak": d_peak,
            "d_ref": d_ref,
            "RIS": RIS,                   # 0..1
            "RIS_peak": RIS_peak,         # 0..1
            "hits_xy": np.vstack(hits_all) if len(hits_all) else np.empty((0,2)),
            "cell_stats": cell_stats,     # Store cell measurements
            "parameters": {
                "initial_area_frac": initial_area_frac,
                "max_area_frac": max_area_frac,
                "num_steps": num_steps,
                "min_cross_section_distance": min_cross_section_distance,
                "packing_factor": self.kappa,
                "d_eff_pixels": d_eff_pixels,
                "normalization_mode": normalization_mode,
                "use_auto_dref": use_auto_dref
            }
        }
        return self.results

    def _sample_circle(self, cx, cy, r, H, W):
        # arc ≈ 1 px => dθ ≈ 1/r ; cap to reasonable max points
        dtheta = max(1.0/max(r, 1.0), 2*np.pi/4096)
        thetas = np.arange(0, 2*np.pi, dtheta, dtype=np.float32)
        xs = np.clip(np.rint(cx + r*np.cos(thetas)).astype(int), 0, W-1)
        ys = np.clip(np.rint(cy + r*np.sin(thetas)).astype(int), 0, H-1)
        return thetas, np.stack([xs, ys], axis=1)

    def _dedupe_angles(self, ang, dtheta_min):
        if ang.size == 0: return ang
        a = np.sort((ang + 2*np.pi) % (2*np.pi))
        keep = [a[0]]
        for x in a[1:]:
            if (x - keep[-1]) >= dtheta_min:
                keep.append(x)
        if len(keep) > 1 and ((keep[0] + 2*np.pi) - keep[-1]) < dtheta_min:
            keep.pop()
        return np.array(keep, dtype=np.float32)
    
    def get_summary_stats(self):
        """Get summary statistics for RIS analysis"""
        if not self.results:
            return {}
        
        return {
            'RIS': self.results.get('RIS', np.nan),
            'RIS_peak': self.results.get('RIS_peak', np.nan),
            'd_mean': self.results.get('d_mean', np.nan),
            'd_peak': self.results.get('d_peak', np.nan),
            'd_ref': self.results.get('d_ref', np.nan),
            'total_crossings': np.sum(self.results.get('crossings', [])),
            'mean_crossings': np.mean(self.results.get('crossings', [])),
            'packing_factor': self.kappa
        }

# Segmentation function
def run_segmentation_only():
    """Run only the AI-powered segmentation step"""
    
    with st.spinner("🤖 Our AI is working its magic on your image... (First run takes longer to load models - grab a coffee! ☕)"):
        # Downsample if requested - using pre-computed dimensions
        h, w = img_gray.shape
        if scale < 1.0:
            small = cv2.resize(img_gray, (w_small, h_small), interpolation=cv2.INTER_AREA)
            diam_small = max(1, int(round(diam * scale)))
        else:
            small = img_gray
            diam_small = diam
        
        # Let our AI do its thing! 🚀
        masks_small, flows, styles = cp_model.eval(
            small,
            diameter=diam_small,
            channels=[0, 0],
            flow_threshold=0.4,
            batch_size=4,
            resample=True,
            augment=True
        )
        
        # Upsample labels back to ORIGINAL size without truncating IDs
        if scale < 1.0:
            masks = cv2.resize(
                masks_small.astype(np.float32), (w, h),
                interpolation=cv2.INTER_NEAREST
            ).astype(np.int32)
        else:
            masks = masks_small.astype(np.int32)
    
    with st.spinner("🔍 Drawing the cell boundaries (our AI found them!)..."):
        # Create membrane mask for network analysis
        contours = find_boundaries(masks, mode='inner')
        
        # Apply AI-powered validation to remove phantom boundaries
        if enable_contour_validation:
            membrane_mask = validate_contours_with_ai(contours, img_gray, validation_method, 4)  # Fixed 4-pixel dilation
        else:
            membrane_mask = contours.astype(np.uint8)
    
    return masks, membrane_mask

def validate_contours_with_ai(contours, image, method="K-means clustering", dilation_pixels=4):
    """
    Validate Cellpose contours using AI-powered methods to remove phantom boundaries.
    Only keeps contours where there's actual signal in the image.
    
    Args:
        contours: Boolean array from find_boundaries
        image: Original grayscale image
        method: AI validation method ("K-means clustering", "Gaussian Mixture Model (GMM)", "Otsu thresholding")
        dilation_pixels: Number of pixels to dilate mask for tolerance
    
    Returns:
        Validated membrane mask
    """
    if method == "K-means clustering":
        # K-means clustering for bimodal images
        # Reshape image to 1D array for clustering
        pixels = image.reshape(-1, 1).astype(np.float32)
        
        # Apply K-means with 2 clusters (foreground vs background)
        kmeans = KMeans(n_clusters=2, n_init=10, max_iter=300, random_state=42)
        labels = kmeans.fit_predict(pixels)
        
        # Use the brighter cluster as junction (foreground)
        cluster_centers = kmeans.cluster_centers_.flatten()
        foreground_cluster = np.argmax(cluster_centers)
        
        # Create mask from K-means results
        ai_mask = (labels == foreground_cluster).reshape(image.shape).astype(np.uint8) * 255
        
    elif method == "Gaussian Mixture Model (GMM)":
        # GMM for probabilistic classification
        pixels = image.reshape(-1, 1).astype(np.float32)
        
        # Fit 2-component GMM
        gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=42)
        gmm.fit(pixels)
        
        # Get posterior probabilities and classify
        probabilities = gmm.predict_proba(pixels)
        labels = np.argmax(probabilities, axis=1)
        
        # Use the brighter component as junction
        cluster_centers = gmm.means_.flatten()
        foreground_cluster = np.argmax(cluster_centers)
        
        # Create mask from GMM results
        ai_mask = (labels == foreground_cluster).reshape(image.shape).astype(np.uint8) * 255
        
    else:  # Otsu thresholding (legacy)
        # Apply Otsu thresholding to get signal mask
        otsu_threshold, ai_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create a dilated mask to be more permissive (allow some tolerance)
    # This helps with slight misalignments between Cellpose and AI validation
    if dilation_pixels > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        ai_mask_dilated = cv2.dilate(ai_mask, kernel, iterations=dilation_pixels)
    else:
        ai_mask_dilated = ai_mask
    
    # Combine: contour must exist AND there must be signal in AI mask
    validated_mask = np.logical_and(contours > 0, ai_mask_dilated > 0).astype(np.uint8)
    
    return validated_mask

# Analysis function
def run_network_analysis(masks, membrane_mask, diam):
    """Run only the network analysis step"""
    
    with st.spinner("🧮 Crunching numbers and analyzing your network like a pro..."):
        # Choose quantifier based on geometry selection
        if analysis_geometry == "Circles (RIS - recommended)":
            # Use RIS quantifier for circular analysis
            quantifier = ZO1RISQuantifier(packing_factor=packing_factor)
            
            # Calculate control d_ref if in control mode
            control_dref = None
            if normalization_mode == "Control (from control images)" and 'control_files' in locals() and control_files:
                # TODO: Implement control image processing to calculate d_ref
                # For now, use a placeholder
                control_dref = 0.1  # Placeholder - should be calculated from control images
                st.warning("⚠️ Control mode d_ref calculation not yet implemented - using placeholder value")
            
            # Determine normalization mode for the quantifier
            if normalization_mode == "Auto (from cell measurements)":
                quantifier_mode = "auto"
            elif normalization_mode == "Model (packing factor κ)":
                quantifier_mode = "model"
            else:  # Control mode
                quantifier_mode = "control"
            
            results = quantifier.quantify_network(
                membrane_mask,
                img_gray.shape,
                initial_area_frac=initial_size/100.0,
                max_area_frac=max_size/100.0,
                num_steps=num_steps,
                min_cross_section_distance=min_distance,
                d_eff_pixels=diam,
                normalization_mode=quantifier_mode,
                control_dref=control_dref
            )
            
            # If auto mode, calculate d_ref from actual cell measurements
            if normalization_mode == "Auto (from cell measurements)":
                d_ref, cell_stats = quantifier.calculate_dref_from_cells(
                    st.session_state.masks, membrane_mask, diam, scale_factor=1.0  # <- critical: no second scaling
                )
                # Update the results with the calculated d_ref
                quantifier.results['d_ref'] = d_ref
                quantifier.results['cell_stats'] = cell_stats
                # Recalculate RIS with the new d_ref
                d_mean = quantifier.results['d_mean']
                quantifier.results['RIS'] = d_mean / d_ref if d_ref > 0 else 0
                quantifier.results['RIS_peak'] = np.max(quantifier.results['radial_density']) / d_ref if d_ref > 0 else 0
        else:
            # Use legacy TiJOR quantifier for rectangular analysis
            quantifier = ZO1NetworkQuantifier()
            results = quantifier.quantify_network(
                membrane_mask,
                img_gray.shape,
                initial_size=initial_size/100.0,
                max_size=max_size/100.0,
                num_steps=num_steps,
                min_cross_section_distance=min_distance
            )
    
    return quantifier

# Run segmentation when button is clicked
if run_segmentation:
    if seg_engine == "Cellpose (AI)" and cp_model is not None:
        # Run segmentation only using Cellpose
        masks, membrane_mask = run_segmentation_only()
        st.session_state.segmentation_stats = None
        st.session_state.pixel_size = None
        st.session_state.cell_diameter = diam
    elif seg_engine == "Classic (CPU)":
        img_u8 = img_gray.astype(np.uint8)
        if classic_method == "GMM":
            res = segment_zo1_gmm(
                img_u8,
                min_obj=min_obj,
                smooth_sigma=smooth_sigma,
                use_ridge=use_ridge,
                min_peak_dist=min_peak_dist,
                skeleton_thickness=skeleton_thickness,
            )
        elif classic_method == "K-means":
            res = segment_zo1_kmeans(
                img_u8,
                min_obj=min_obj,
                smooth_sigma=smooth_sigma,
                use_ridge=use_ridge,
                min_peak_dist=min_peak_dist,
                skeleton_thickness=skeleton_thickness,
            )
        else:
            res = segment_zo1_otsu(
                img_u8,
                adaptive=adaptive if classic_method == "Adaptive" else False,
                block=block,
                offset=offset,
                min_obj=min_obj,
                smooth_sigma=smooth_sigma,
                use_ridge=use_ridge,
                min_peak_dist=min_peak_dist,
                thresh_multiplier=thresh_multiplier,
                skeleton_thickness=skeleton_thickness,
            )
        masks = res.labels
        membrane_mask = res.membrane
        st.session_state.segmentation_stats = res.stats
        st.session_state.pixel_size = pixel_size
        diam = res.stats.get('mean_equiv_diam', 0.0)
        st.session_state.cell_diameter = diam
    else:
        masks = None
        membrane_mask = None

    if masks is not None:
        st.session_state.segmentation_complete = True
        st.session_state.masks = masks
        st.session_state.membrane_mask = membrane_mask
        st.session_state.analysis_complete = False  # Reset analysis state

        st.success("🎉 Woohoo! Segmentation finished! Now you can play with the analysis parameters!")

        st.markdown("""
<div class=\"fun-fact\">
    🎮 Ready to analyze? Click that 'Run Analysis' button and cross your fingers! ✨
</div>
""", unsafe_allow_html=True)

        st.rerun()  # Force refresh to update button state


# Ensure cell diameter is available for sidebar display and analysis
diam = st.session_state.get('cell_diameter', 0.0)

# Run analysis when button is clicked
if run_analysis and st.session_state.segmentation_complete:
    # Run analysis only
    quantifier = run_network_analysis(
        st.session_state.masks, st.session_state.membrane_mask, diam
    )

    # Update session state
    st.session_state.analysis_complete = True
    st.session_state.quantifier = quantifier

    st.success("🚀 Bam! Analysis complete! Your ZO-1 network is now quantified!")

    # Add a fun completion message
    st.markdown("""
    <div class="fun-fact">
        🎯 Analysis complete! (We're as surprised as you are that it worked!) 📚
    </div>
    """, unsafe_allow_html=True)

# Display segmentation status
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Current Status")

if st.session_state.segmentation_complete:
    st.sidebar.success("🎯 Segmentation Complete!")
    st.sidebar.info("🎮 Time to play with the analysis parameters!")
    st.sidebar.markdown("**Current Segmentation:**")
    if seg_engine == "Cellpose (AI)":
        st.sidebar.markdown(f"- Cell diameter: {diam} px")
        st.sidebar.markdown(f"- Scale: {scale}x")
        st.sidebar.markdown("- Processing: CPU (deployment mode)")
    else:
        st.sidebar.markdown(f"- Method: {classic_method}")
        st.sidebar.markdown(f"- Min object size: {min_obj}px")
        st.sidebar.markdown(f"- Mean cell diameter: {diam:.1f} px")
        st.sidebar.markdown("- Processing: Classic CPU")
else:
    st.sidebar.warning("🤔 Hmm... no segmentation yet!")
    st.sidebar.info("🎯 Configure parameters and let our AI work its magic!")

# Show segmentation preview if available
if st.session_state.segmentation_complete:
    masks = st.session_state.masks
    membrane_mask = st.session_state.membrane_mask

    # Quick segmentation preview
    st.markdown("## 🔬 Segmentation Results")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(img_gray, use_container_width=True)
    
    with col2:
        st.subheader("Cell Boundaries (Validated)")
        # Create overlay with contours - ensure same dimensions
        if img_gray.shape != masks.shape:
            # Resize img_gray to match masks dimensions
            img_gray_resized = cv2.resize(img_gray, (masks.shape[1], masks.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            img_gray_resized = img_gray

        overlay = np.stack([img_gray_resized]*3, axis=-1)

        if seg_engine == "Classic (CPU)":
            validated_contours = find_boundaries(masks, mode="outer")
        else:
            validated_contours = membrane_mask

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thickened_contours = cv2.dilate(validated_contours.astype(np.uint8), kernel, iterations=1)

        overlay[thickened_contours > 0, 0] = 255  # Red contours
        overlay[thickened_contours > 0, 1] = 0    # No green
        overlay[thickened_contours > 0, 2] = 0    # No blue
        st.image(overlay, use_container_width=True)

        if seg_engine == "Classic (CPU)" and st.session_state.segmentation_stats:
            stats = st.session_state.segmentation_stats
            st.markdown("### Summary")
            st.write(f"Cells: {stats['n_cells']} | Mean area: {stats['mean_area']:.1f} px² | Mean equiv. diam: {stats['mean_equiv_diam']:.2f} px")
            if st.session_state.pixel_size:
                st.write(f"Mean area: {stats.get('mean_area_um2',0):.1f} µm² | Mean equiv. diam: {stats.get('mean_equiv_diam_um',0):.2f} µm")
            df, _ = compute_cell_metrics(masks, pixel_size=st.session_state.pixel_size)
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button('Export CSV', csv, 'cell_metrics.csv', 'text/csv')
            report_lines = [f"{k}: {v}" for k, v in stats.items()]
            report = '\n'.join(report_lines)
            st.download_button('Download report (.txt)', report, 'summary.txt', 'text/plain')

    
    # Add a fun fact with animal comparison
    cell_count = int(masks.max()) if masks is not None else 0
    random_animal, animal_cells = random.choice(animal_comparisons)
    st.markdown(f"""
    <div class="fun-fact">
        🧬 Fun Fact: Our AI found {cell_count} cells! That's about {cell_count/animal_cells*100:.2e}% of a {random_animal}'s total cells! 🎯
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

# Display analysis results if complete
if st.session_state.analysis_complete and st.session_state.segmentation_complete:
    quantifier = st.session_state.quantifier
    
    # Results header
    if analysis_geometry == "Circles (RIS - recommended)":
        st.markdown("## 📊 RIS Analysis Results")
        st.markdown("""
        <div class="fun-fact">
            🎊 Wow, it actually worked! Your ZO-1 network has been quantified with RIS! 🎊
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("## 📊 TiJOR Analysis Results")
        st.markdown("""
        <div class="fun-fact">
            🎊 Wow, it actually worked! Your ZO-1 network has been quantified with TiJOR! 🎊
        </div>
        """, unsafe_allow_html=True)
    
    # Summary metrics
    summary = quantifier.get_summary_stats()
    
    # Show validation status
    if enable_contour_validation:
        if validation_method == "K-means clustering":
            validation_status = "✅ AI Validated (K-means) 🎯"
            threshold_info = "K-means clustering with 2 components (foreground vs background)"
        elif validation_method == "Gaussian Mixture Model (GMM)":
            validation_status = "✅ AI Validated (GMM) 🔮"
            threshold_info = "Gaussian Mixture Model with probabilistic classification"
        else:  # Otsu
            # Calculate and display current Otsu threshold
            otsu_threshold, _ = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            adjusted_threshold = int(otsu_threshold * otsu_strength)
            
            if otsu_strength > 1.0:
                strength_emoji = "💪💪💪"  # Super strong
                strength_desc = "Super aggressive filtering!"
            elif otsu_strength > 1.5:
                strength_emoji = "💪💪"     # Strong
                strength_desc = "Strong noise filtering"
            elif otsu_strength > 1.0:
                strength_emoji = "💪"        # Normal+
                strength_desc = "Enhanced filtering"
            else:
                strength_emoji = "🤏"        # Gentle
                strength_desc = "Gentle filtering"
            
            validation_status = f"✅ Otsu Validated {strength_emoji}"
            threshold_info = f"Base: {otsu_threshold}, Adjusted: {adjusted_threshold} ({strength_desc})"
    else:
        validation_status = "⚠️ Raw Cellpose"
        threshold_info = "No validation applied"

    st.info(f"🔍 **Analysis Mode**: {validation_status}")
    st.info(f"📊 **Threshold**: {threshold_info}")
    
    # Display metrics based on analysis type
    if analysis_geometry == "Circles (RIS - recommended)":
        # RIS metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🔵 RIS (0-1)", f"{summary.get('RIS', 'N/A'):.3f}" if 'RIS' in summary and not np.isnan(summary['RIS']) else "N/A")
        with col2:
            st.metric("🚀 RIS Peak", f"{summary.get('RIS_peak', 'N/A'):.3f}" if 'RIS_peak' in summary and not np.isnan(summary['RIS_peak']) else "N/A")
        with col3:
            st.metric("🔬 Cells Detected", f"{int(masks.max()) if masks is not None else 0}")
        with col4:
            st.metric("📊 Mean Crossings", f"{summary.get('mean_crossings', 'N/A'):.1f}" if 'mean_crossings' in summary and not np.isnan(summary['mean_crossings']) else "N/A")
        
        # Show RIS reference info
        if 'd_ref' in summary and not np.isnan(summary['d_ref']):
            st.info(f"📐 **Reference Density (d_ref)**: {summary['d_ref']:.4f} | **Packing Factor (κ)**: {summary.get('packing_factor', 'N/A')}")
        
        # Show additional cell statistics if available (from auto mode)
        if hasattr(quantifier, 'results') and 'cell_stats' in quantifier.results and quantifier.results['cell_stats']:
            cell_stats = quantifier.results['cell_stats']
            if cell_stats:  # Only show if we have cell statistics
                st.markdown("---")
                st.markdown("### 🔬 **Cell Measurements (Auto Mode)**")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📊 Cells Counted", f"{cell_stats.get('num_cells', 'N/A')}")
                with col2:
                    st.metric("📏 Avg Cell Diameter", f"{cell_stats.get('avg_cell_diameter', 'N/A'):.1f} px")
                with col3:
                    st.metric("📐 Cell Diameter Std", f"{cell_stats.get('std_cell_diameter', 'N/A'):.1f} px")
                with col4:
                    st.metric("🔍 Network Density", f"{cell_stats.get('network_density', 'N/A'):.4f}")
                
                # Show d_ref calculation details
                st.info(f"""
                 **🔬 Auto d_ref Calculation (Updated):**
                 - **Measured cell diameter**: {cell_stats.get('avg_cell_diameter', 'N/A'):.1f} px (scaled from {cell_stats.get('scale_factor', 1.0)}x resampled image)
                 - **Cellpose d_ref (κ/measured_diameter)**: {cell_stats.get('d_ref_cellpose', 'N/A'):.4f}
                 - **User d_ref (κ/user_diameter)**: {cell_stats.get('d_ref_user', 'N/A'):.4f}
                 - **Final d_ref used**: {summary.get('d_ref', 'N/A'):.4f}
                 
                 **💡 Note**: Using Cellpose-measured d_ref for accuracy (properly scaled from resampled image)
                """)
    else:
        # Legacy TiJOR metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean TiJOR", f"{summary['mean_tijor']:.4f}")
        with col2:
            st.metric("Total Cross-sections", f"{summary['total_cross_sections']:.0f}")
        with col3:
            st.metric("Cells Detected", f"{int(masks.max()) if masks is not None else 0}")
        with col4:
            network_score_display = f"{summary['mean_tijor']/summary['std_tijor']:.2f}" if summary['std_tijor'] > 0 else "N/A"
            st.metric("Network Score", network_score_display)
    
    # Overlay controls
    st.markdown("---")
    st.markdown("### 🎨 Visualization Controls")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        show_contours = st.checkbox("Show Cell Contours", value=False)
    with col2:
        show_rectangles = st.checkbox("Show Analysis Rectangles", value=True)
    with col3:
        show_cross_sections = st.checkbox("Show Cross-section Points", value=True)
    
    # Create visualization - Single image with overlay (as requested)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Main image with overlays - ensure same dimensions
    if img_gray.shape != masks.shape:
        # Resize img_gray to match masks dimensions
        img_gray_resized = cv2.resize(img_gray, (masks.shape[1], masks.shape[0]), interpolation=cv2.INTER_LINEAR)
    else:
        img_gray_resized = img_gray
        
    ax.imshow(img_gray_resized, cmap='gray')
    
    if analysis_geometry == "Circles (RIS - recommended)":
        ax.set_title('ZO-1 Network with RIS Analysis Overlays', fontsize=14, fontweight='bold')
    else:
        ax.set_title('ZO-1 Network with TiJOR Analysis Overlays', fontsize=14, fontweight='bold')
    
    ax.axis('off')
    
    # Draw contours if requested - use validated membrane mask
    if show_contours:
        # Use the validated membrane mask from the quantifier analysis
        validated_membrane_mask = st.session_state.membrane_mask
        ax.contour(validated_membrane_mask, [0.5], colors='red', linewidths=1, alpha=0.9)
    
    # Draw analysis overlays based on geometry type
    if analysis_geometry == "Circles (RIS - recommended)":
        # Draw concentric circles and scatter hits for RIS analysis
        if show_rectangles or show_cross_sections:
            center_x = img_gray.shape[1] / 2
            center_y = img_gray.shape[0] / 2
            
            # Draw concentric circles
            if show_rectangles and hasattr(quantifier, 'results') and 'radii' in quantifier.results:
                radii = quantifier.results['radii']
                colors = plt.cm.Blues(np.linspace(0.3, 1, len(radii)))
                
                for i, r in enumerate(radii):
                    circle = plt.Circle((center_x, center_y), r, 
                                      linewidth=2, edgecolor=colors[i], 
                                      facecolor='none', linestyle='--', alpha=0.7)
                    ax.add_patch(circle)
            
            # Plot crossing points (hits) if requested
            if show_cross_sections and hasattr(quantifier, 'results') and 'hits_xy' in quantifier.results:
                hits = quantifier.results['hits_xy']
                if len(hits) > 0:
                    ax.scatter(hits[:, 0], hits[:, 1], 
                              c='cyan', s=15, alpha=0.8, edgecolors='white', linewidth=0.5,
                              label=f'Crossings ({len(hits)})')
                    ax.legend(loc='upper right', fontsize=10)
    else:
        # Draw rectangles and cross-sections for legacy TiJOR analysis
        if show_rectangles or show_cross_sections:
            center_x = img_gray.shape[1] / 2
            center_y = img_gray.shape[0] / 2
            
            colors = plt.cm.Reds(np.linspace(0.3, 1, len(quantifier.results['rectangle_sizes'])))
            
            for i, size in enumerate(quantifier.results['rectangle_sizes']):
                half_side = size / 2
                
                if show_rectangles:
                    rect = Rectangle(
                        (center_x - half_side, center_y - half_side),
                        size, size,
                        linewidth=2,
                        edgecolor=colors[i],
                        facecolor='none',
                        linestyle='--',
                        alpha=0.7
                    )
                    ax.add_patch(rect)
        
        # Plot cross-section points if requested
        if show_cross_sections and quantifier.results.get('cross_section_points'):
            cross_sections = np.array(quantifier.results['cross_section_points'])
            ax.scatter(cross_sections[:, 1], cross_sections[:, 0], 
                       c='blue', s=20, alpha=1.0, edgecolors='white', linewidth=1,
                       label=f'Cross-sections ({len(cross_sections)})')
            ax.legend(loc='upper right', fontsize=10)
    
    # Add summary text below the image
    if analysis_geometry == "Circles (RIS - recommended)":
        if hasattr(quantifier, 'results') and 'RIS' in quantifier.results:
            summary_text = f'RIS: {quantifier.results["RIS"]:.3f}'
        else:
            summary_text = 'RIS analysis in progress...'
    else:
        summary_text = f'Mean TiJOR: {summary["mean_tijor"]:.4f} | Total Cross-sections: {summary["total_cross_sections"]:.0f}'
    
    fig.text(0.5, -0.05, summary_text, ha='center', va='bottom', fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Add the analysis results plot below the main image
    st.markdown("---")
    
    if analysis_geometry == "Circles (RIS - recommended)":
        st.markdown("### 📈 RIS — Radial Profile")
        
        if hasattr(quantifier, 'results') and 'radial_density' in quantifier.results:
            fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
            
            radii = quantifier.results['radii']
            d_vals = quantifier.results['radial_density']
            d_ref = quantifier.results.get('d_ref', np.nan)
            
            # Plot radial density d(r)
            ax2.plot(radii, d_vals, 'b-o', linewidth=3, markersize=8, label='Radial Density d(r)')
            
            # Add reference line if available
            if not np.isnan(d_ref):
                ax2.axhline(y=d_ref, color='red', linestyle='--', alpha=0.7, 
                           label=f'Reference d_ref = {d_ref:.4f}')
            
            ax2.set_xlabel('Radius (px)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Radial Density d(r)', fontsize=12, fontweight='bold', color='blue')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper right', fontsize=11)
            ax2.set_title('RIS Radial Profile', fontsize=14, fontweight='bold')
            
            # Add summary text
            RIS = quantifier.results.get('RIS', np.nan)
            summary_text = f'RIS: {RIS:.3f}'
            fig2.text(0.5, -0.15, summary_text, ha='center', va='bottom', fontsize=12, 
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.warning("⚠️ RIS analysis results not available yet")
            
    else:
        st.markdown("### 📈 TiJOR Analysis Results")
        
        fig2, ax2 = plt.subplots(1, 1, figsize=(8, 4))
        ax2_twin = ax2.twinx()
        
        x_vals = np.arange(len(quantifier.results['tijor_values']))
        
        # Plot TiJOR values
        line1 = ax2.plot(x_vals, quantifier.results['tijor_values'], 'b-o', 
                         linewidth=3, markersize=10, label='TiJOR Values')
        ax2.set_xlabel('Rectangle Size Step', fontsize=12, fontweight='bold')
        ax2.set_ylabel('TiJOR (cross-sections/pixel)', fontsize=12, fontweight='bold', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        # Plot filtered cross-section counts
        line2 = ax2_twin.plot(x_vals, quantifier.results['filtered_cross_section_counts'], 'r-s', 
                               linewidth=3, markersize=10, label='Filtered Cross-sections')
        ax2_twin.set_ylabel('Number of Cross-sections', fontsize=12, fontweight='bold', color='red')
        ax2_twin.tick_params(axis='y', labelcolor='red')
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper left', fontsize=11)
        
        ax2.set_title('TiJOR Analysis Results', fontsize=14, fontweight='bold')
        
        # Add rectangle size labels
        size_labels = [f'{size:.0f}px' for size in quantifier.results['rectangle_sizes']]
        ax2.set_xticks(x_vals)
        ax2.set_xticklabels(size_labels, rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig2)
    
    # Detailed results table
    st.markdown("---")
    
    if analysis_geometry == "Circles (RIS - recommended)":
        st.markdown("### 📋 RIS Analysis Details")
        
        if hasattr(quantifier, 'results') and 'radii' in quantifier.results:
            # Create RIS results dataframe
            results_data = []
            for i, (radius, crossings, density) in enumerate(zip(
                quantifier.results['radii'],
                quantifier.results['crossings'],
                quantifier.results['radial_density']
            )):
                results_data.append({
                    'Step': i,
                    'Radius (px)': f'{radius:.1f}',
                    'Crossings N(r)': int(crossings),
                    'Radial Density d(r)': f'{density:.4f}'
                })
            
            df = pd.DataFrame(results_data)
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # Export functionality for RIS
            st.markdown("---")
            st.markdown("### 💾 Export RIS Results")
            
            st.markdown("""
            <div class="fun-fact">
                💡 Pro Tip: Save your RIS results! (You never know when the app might crash 😅)
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export as CSV
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📊 Download CSV",
                    data=csv,
                    file_name="zo1_ris_analysis.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Export detailed results as text
                RIS = quantifier.results.get('RIS', np.nan)
                RIS_peak = quantifier.results.get('RIS_peak', np.nan)
                d_ref = quantifier.results.get('d_ref', np.nan)
                kappa = quantifier.results.get('parameters', {}).get('packing_factor', 'N/A')
                d_eff = quantifier.results.get('parameters', {}).get('d_eff_pixels', 'N/A')
                normalization_mode = quantifier.results.get('parameters', {}).get('normalization_mode', 'N/A')
                
                detailed_results = f"""ZO-1 RIS Analysis Results
{'='*50}

Analysis Parameters:
  Initial area fraction: {initial_size}%
  Max area fraction: {max_size}%
  Number of steps: {num_steps}
  Min cross-section distance: {min_distance} px
  Packing factor (κ): {kappa}
  Effective cell diameter: {d_eff} px
  Normalization mode: {normalization_mode}

Summary Statistics:
  RIS: {RIS:.3f}
  RIS Peak: {RIS_peak:.3f}
  Reference density (d_ref): {d_ref:.4f}
  Cells detected: {int(masks.max()) if masks is not None else 0}

Radial analysis:
{df.to_string(index=False)}
"""
                
                # Add cell statistics if available
                if hasattr(quantifier, 'results') and 'cell_stats' in quantifier.results and quantifier.results['cell_stats']:
                    cell_stats = quantifier.results['cell_stats']
                    if cell_stats:
                        detailed_results += f"""

Cell Measurements (Auto Mode):
  Number of cells: {cell_stats.get('num_cells', 'N/A')}
  Average cell area: {cell_stats.get('avg_cell_area', 'N/A'):.1f} px²
  Average cell diameter: {cell_stats.get('avg_cell_diameter', 'N/A'):.1f} px
  Cell diameter std: {cell_stats.get('std_cell_diameter', 'N/A'):.1f} px
  Network density: {cell_stats.get('network_density', 'N/A'):.4f}
  Cell density: {cell_stats.get('cell_density', 'N/A'):.6f}

d_ref Calculation Details:
  Measured cell diameter: {cell_stats.get('avg_cell_diameter', 'N/A'):.1f} px (scaled from {cell_stats.get('scale_factor', 1.0)}x resampled image)
  Cellpose d_ref (κ/measured_diameter): {cell_stats.get('d_ref_cellpose', 'N/A'):.4f}
  User d_ref (κ/user_diameter): {cell_stats.get('d_ref_user', 'N/A'):.4f}
  Note: Using Cellpose-measured d_ref for accuracy (properly scaled)
"""
                
                st.download_button(
                    label="📝 Download Detailed Report",
                    data=detailed_results,
                    file_name="zo1_ris_analysis_report.txt",
                    mime="text/plain"
                )
        else:
            st.warning("⚠️ RIS analysis results not available yet")
            
    else:
        st.markdown("### 📋 TiJOR Analysis Details")
        
        # Create TiJOR results dataframe
        results_data = []
        for i, (size, tijor, original_count, filtered_count) in enumerate(zip(
            quantifier.results['rectangle_sizes'],
            quantifier.results['tijor_values'],
            quantifier.results['cross_section_counts'],
            quantifier.results['filtered_cross_section_counts']
        )):
            results_data.append({
                'Step': i,
                'Size (px)': f'{size:.1f}',
                'TiJOR': f'{tijor:.4f}',
                'Original Count': int(original_count),
                'Filtered Count': int(filtered_count)
            })
        
        df = pd.DataFrame(results_data)
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Export functionality for TiJOR
        st.markdown("---")
        st.markdown("### 💾 Export TiJOR Results")
        
        st.markdown("""
        <div class="fun-fact">
            💡 Pro Tip: Save your TiJOR results! (You never know when the app might crash 😅)
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export as CSV
            csv = df.to_csv(index=False)
            st.download_button(
                label="📊 Download CSV",
                data=csv,
                file_name="zo1_tijor_analysis.csv",
                mime="text/csv"
            )
        
        with col2:
            # Export detailed results as text
            network_score = f"{summary['mean_tijor']/summary['std_tijor']:.2f}" if summary['std_tijor'] > 0 else "N/A"
            
            detailed_results = f"""ZO-1 TiJOR Network Quantification Results
{'='*50}

Analysis Parameters:
  Initial rectangle size: {initial_size}%
  Max rectangle size: {max_size}%
  Number of steps: {num_steps}
  Min cross-section distance: {min_distance} px

Summary Statistics:
  Mean TiJOR: {summary['mean_tijor']:.4f} cross-sections/pixel
  Total cross-sections: {summary['total_cross_sections']:.0f}
  Cells detected: {int(masks.max()) if masks is not None else 0}
  Network organization score: {network_score}

Rectangle-by-rectangle analysis:
{df.to_string(index=False)}
"""
            
            st.download_button(
                label="📝 Download Detailed Report",
                data=detailed_results,
                file_name="zo1_tijor_analysis_report.txt",
                mime="text/plain"
            )

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🔬 ZO-1 Network Analysis Tool | Powered by Cutting-Edge AI | Enhanced with RIS & TiJOR Quantification ✨</p>
</div>
""", unsafe_allow_html=True)
