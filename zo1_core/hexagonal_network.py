import numpy as np
import cv2
from PIL import Image
import io
from math import sqrt, cos, sin, pi

rng = np.random.default_rng(42)  # reproducible

def hex_vertices(center, a, orientation="flat"):
    """
    Return the 6×2 array of vertices for a hexagon of circumradius a.
    orientation='flat' => flat-top hexagons (good for horizontal rows)
    """
    cx, cy = center
    if orientation == "flat":
        angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    else:  # 'pointy'
        angles = np.deg2rad([30, 90, 150, 210, 270, 330])
    return np.stack([cx + a*np.cos(angles), cy + a*np.sin(angles)], axis=1).astype(np.float32)

def make_hex_centers(W, H, a, orientation="flat", margin=3):
    """
    Generate centers for a hex grid that fills a WxH canvas.
    a = hex circumradius in pixels.
    """
    centers = []
    if orientation == "flat":
        dx = 1.5 * a
        dy = sqrt(3) * a
        # columns advance by dx; odd columns shifted vertically by dy/2
        q = 0
        while True:
            x = margin + a + q*dx
            if x > W - margin - a: break
            r = 0
            y_offset = 0.5*dy if (q % 2) else 0.0
            while True:
                y = margin + a + y_offset + r*dy
                if y > H - margin - a: break
                centers.append((x, y))
                r += 1
            q += 1
    else:  # 'pointy'
        dy = 1.5 * a
        dx = sqrt(3) * a
        r = 0
        while True:
            y = margin + a + r*dy
            if y > H - margin - a: break
            q = 0
            x_offset = 0.5*dx if (r % 2) else 0.0
            while True:
                x = margin + a + x_offset + q*dx
                if x > H - margin - a: break
                centers.append((x, y))
                q += 1
            r += 1
    return np.array(centers, dtype=np.float32)

def draw_hex_grid(
    W=1024, H=1024, a=18, orientation="flat",
    loss_pct=30.0,  # % of hexagons to remove entirely
    edge_dropout_pct=0.0,  # % of edges to randomly omit (extra fragmentation)
    thickness=1, antialias=True
):
    """
    Returns a uint8 binary image (H×W) where hex edges are 255 and background 0.
    """
    img = np.zeros((H, W), dtype=np.uint8)
    centers = make_hex_centers(W, H, a, orientation=orientation, margin=3)

    # choose which hexagons to keep
    keep_mask = rng.random(len(centers)) >= (loss_pct/100.0)
    kept_centers = centers[keep_mask]

    # optional per-edge dropout
    drop_edge = lambda: (rng.random() < (edge_dropout_pct/100.0))

    # line type
    linetype = cv2.LINE_AA if antialias else cv2.LINE_8

    for c in kept_centers:
        poly = hex_vertices(c, a, orientation=orientation)  # (6,2)
        # close polygon by repeating first vertex at end for edge-wise control
        cyc = np.vstack([poly, poly[0:1]])
        # draw each edge (optionally drop some)
        for i in range(6):
            if edge_dropout_pct > 0 and drop_edge():
                continue
            p1 = tuple(np.round(cyc[i]).astype(int))
            p2 = tuple(np.round(cyc[i+1]).astype(int))
            cv2.line(img, p1, p2, color=255, thickness=thickness, lineType=linetype)

    return img

def generate_hexagonal_network(size=512, hex_size=40, line_thickness=2, degradation_percent=0.0):
    """
    Generate a perfect hexagonal network image for tutorial purposes.
    
    Parameters:
    -----------
    size : int
        Size of the square image (size x size pixels)
    hex_size : int
        Size of each hexagon (circumradius in pixels)
    line_thickness : int
        Thickness of the hexagon lines in pixels
    degradation_percent : float
        Percentage of hexagons to randomly remove (0.0 = perfect, 100.0 = completely degraded)
    
    Returns:
    --------
    tuple
        (BytesIO buffer, PIL Image) for Streamlit compatibility
    """
    # Generate the hexagonal grid
    img = draw_hex_grid(
        W=size, 
        H=size, 
        a=hex_size, 
        orientation="flat",
        loss_pct=degradation_percent,
        edge_dropout_pct=0.0,  # No edge dropout for tutorial
        thickness=line_thickness,
        antialias=True
    )
    
    # Convert to PIL Image
    pil_img = Image.fromarray(img)
    
    # Create BytesIO buffer for Streamlit
    buf = io.BytesIO()
    pil_img.save(buf, format='PNG')
    buf.name = f"hexagonal_network_{int(100-degradation_percent)}.png"
    buf.type = "image/png"
    buf.seek(0)
    
    return buf, pil_img

def calculate_theoretical_density(hex_size, line_thickness):
    """
    Calculate theoretical network density for perfect hexagonal packing.
    
    Parameters:
    -----------
    hex_size : int
        Size of each hexagon (circumradius)
    line_thickness : int
        Thickness of hexagon lines
    
    Returns:
    --------
    float
        Theoretical density (line length per unit area)
    """
    # Area of one hexagon
    hex_area = 3 * np.sqrt(3) * hex_size**2 / 2
    
    # Perimeter of one hexagon
    hex_perimeter = 6 * hex_size
    
    # Theoretical density = perimeter / area
    density = hex_perimeter / hex_area
    
    return density

def get_hexagonal_network_info(hex_size, line_thickness, degradation_percent):
    """
    Get information about the generated hexagonal network for tutorial purposes.
    
    Returns:
    --------
    dict
        Dictionary with network information
    """
    theoretical_density = calculate_theoretical_density(hex_size, line_thickness)
    
    info = {
        "hex_size_px": hex_size,
        "line_thickness_px": line_thickness,
        "degradation_percent": degradation_percent,
        "theoretical_density": theoretical_density,
        "expected_intersections_per_circle": f"~{theoretical_density * 2 * np.pi * 50:.1f} at radius 50px",
        "perfect_network_score": 100 - degradation_percent,
        "cell_diameter_estimate": hex_size * 2,  # Approximate cell diameter
        "packing_factor": 1.5,  # For hexagonal packing
    }
    
    return info
