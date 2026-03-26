"""
Generate the Autonomica GitHub social preview image (1280 x 640 px).
Usage: python scripts/gen_social_preview.py
Output: assets/social_preview.png
"""

import math
import os
import random
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# ── canvas ────────────────────────────────────────────────────────────────────
W, H = 1280, 640
img = Image.new("RGB", (W, H), (0, 0, 0))
draw = ImageDraw.Draw(img)

# ── background gradient (dark navy → very dark teal) ─────────────────────────
for y in range(H):
    t = y / H
    r = int(6  + t * 4)
    g = int(8  + t * 14)
    b = int(22 + t * 18)
    draw.line([(0, y), (W, y)], fill=(r, g, b))

# ── neural-network nodes ──────────────────────────────────────────────────────
# Deterministic seed so the image is reproducible
random.seed(42)

# Node positions — two "hemispheres" to evoke a brain
def hemisphere_nodes(cx, cy, count, rx, ry, seed_offset=0):
    nodes = []
    for i in range(count):
        random.seed(42 + seed_offset + i)
        angle = random.uniform(0, 2 * math.pi)
        r_x   = random.uniform(0.15, 1.0)
        r_y   = random.uniform(0.15, 1.0)
        x = cx + rx * r_x * math.cos(angle)
        y = cy + ry * r_y * math.sin(angle)
        nodes.append((int(x), int(y)))
    return nodes

left_cx,  left_cy  = W * 0.22, H * 0.50
right_cx, right_cy = W * 0.78, H * 0.50
rx, ry = 190, 210

left_nodes  = hemisphere_nodes(left_cx,  left_cy,  28, rx, ry, seed_offset=0)
right_nodes = hemisphere_nodes(right_cx, right_cy, 28, rx, ry, seed_offset=100)

# Corpus callosum — a few cross-hemisphere connections through the centre
def corpus_edges(ln, rn, count=6):
    random.seed(99)
    edges = []
    for _ in range(count):
        a = random.choice(ln)
        b = random.choice(rn)
        edges.append((a, b))
    return edges

# ── draw edges ────────────────────────────────────────────────────────────────
def draw_edges(nodes, max_dist, color_base, alpha_max=55):
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            dist = math.hypot(a[0]-b[0], a[1]-b[1])
            if dist < max_dist:
                fade = int(alpha_max * (1 - dist / max_dist))
                c = (*color_base, fade)
                # Pillow doesn't support RGBA lines on RGB directly — use a
                # separate RGBA overlay layer
                edge_layer.line([a, b], fill=c, width=1)

edge_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
edge_draw  = ImageDraw.Draw(edge_layer)

# Temporarily rebind draw_edges to use edge_draw
def draw_edges_ex(nodes, max_dist, color_base, alpha_max=55):
    for i, a in enumerate(nodes):
        for j, b in enumerate(nodes):
            if j <= i:
                continue
            dist = math.hypot(a[0]-b[0], a[1]-b[1])
            if dist < max_dist:
                fade = int(alpha_max * (1 - dist / max_dist))
                edge_draw.line([a, b], fill=(*color_base, fade), width=1)

draw_edges_ex(left_nodes,  160, (100, 220, 255))   # cyan-ish left
draw_edges_ex(right_nodes, 160, (180, 130, 255))   # purple-ish right

for a, b in corpus_edges(left_nodes, right_nodes, count=7):
    edge_draw.line([a, b], fill=(140, 200, 255, 30), width=1)

img = img.convert("RGBA")
img = Image.alpha_composite(img, edge_layer)

# ── draw nodes ────────────────────────────────────────────────────────────────
node_layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
node_draw  = ImageDraw.Draw(node_layer)

def draw_nodes(nodes, core_color, glow_color):
    for n in nodes:
        x, y = n
        # outer glow
        for r in range(9, 1, -1):
            alpha = int(40 * (1 - r / 9))
            node_draw.ellipse(
                [(x-r, y-r), (x+r, y+r)],
                fill=(*glow_color, alpha)
            )
        # core dot
        node_draw.ellipse([(x-3, y-3), (x+3, y+3)], fill=(*core_color, 220))

draw_nodes(left_nodes,  (120, 230, 255), (80,  180, 255))
draw_nodes(right_nodes, (210, 160, 255), (160, 100, 255))

img = Image.alpha_composite(img, node_layer)

# ── subtle centre glow (corpus callosum area) ─────────────────────────────────
glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
gd = ImageDraw.Draw(glow)
for r in range(120, 0, -10):
    alpha = int(18 * (1 - r / 120))
    gd.ellipse(
        [(W//2 - r*2, H//2 - r), (W//2 + r*2, H//2 + r)],
        fill=(100, 200, 255, alpha)
    )
glow = glow.filter(ImageFilter.GaussianBlur(40))
img = Image.alpha_composite(img, glow)

# ── convert back to RGB for text layer ───────────────────────────────────────
img = img.convert("RGB")
draw = ImageDraw.Draw(img)

# ── fonts ─────────────────────────────────────────────────────────────────────
SFNS    = "/System/Library/Fonts/SFNS.ttf"
SFMONO  = "/System/Library/Fonts/SFNSMono.ttf"
NEWYORK = "/System/Library/Fonts/NewYork.ttf"

def load_font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

font_title    = load_font(NEWYORK, 96)
font_tagline  = load_font(SFNS,    32)
font_badge    = load_font(SFMONO,  20)

# ── helper: centred text with optional drop shadow ────────────────────────────
def centred_text(draw, y, text, font, fill, shadow=None):
    bbox = draw.textbbox((0, 0), text, font=font)
    tw = bbox[2] - bbox[0]
    x = (W - tw) // 2
    if shadow:
        draw.text((x+2, y+2), text, font=font, fill=shadow)
    draw.text((x, y), text, font=font, fill=fill)

# ── title: "Autonomica" ───────────────────────────────────────────────────────
centred_text(draw, 195, "Autonomica", font_title,
             fill=(255, 255, 255),
             shadow=(0, 0, 0))

# ── tagline ───────────────────────────────────────────────────────────────────
centred_text(draw, 320, "Runtime adaptive governance for AI agents",
             font_tagline,
             fill=(180, 220, 255),
             shadow=(0, 0, 0))

# ── thin divider line ─────────────────────────────────────────────────────────
line_y = 372
lw = 340
draw.line([(W//2 - lw, line_y), (W//2 + lw, line_y)], fill=(80, 140, 200, 120), width=1)

# ── badge row ─────────────────────────────────────────────────────────────────
badges = [
    ("< 10 ms",     (40, 180, 100)),
    ("287 tests",   (60, 140, 220)),
    ("5 modes",     (160, 100, 220)),
    ("Python 3.11", (200, 140, 40)),
]

badge_h = 34
badge_pad_x = 22
badge_pad_y = 8
badge_gap = 18
badge_y = 400

# measure total width first
total_w = 0
badge_sizes = []
for label, _ in badges:
    bb = draw.textbbox((0, 0), label, font=font_badge)
    tw = bb[2] - bb[0]
    bw = tw + badge_pad_x * 2
    badge_sizes.append((label, bw, tw))
    total_w += bw
total_w += badge_gap * (len(badges) - 1)

bx = (W - total_w) // 2
for (label, color), (_, bw, tw) in zip(badges, badge_sizes):
    r, g, b = color
    # filled pill
    draw.rounded_rectangle(
        [(bx, badge_y), (bx + bw, badge_y + badge_h)],
        radius=badge_h // 2,
        fill=(r, g, b, 200),
    )
    # label
    tx = bx + (bw - tw) // 2
    ty = badge_y + badge_pad_y - 2
    draw.text((tx, ty), label, font=font_badge, fill=(255, 255, 255))
    bx += bw + badge_gap

# ── bottom micro-caption ──────────────────────────────────────────────────────
font_micro = load_font(SFNS, 18)
centred_text(draw, 480, "github.com/hsbhatia1993-blip/autonomica",
             font_micro, fill=(100, 140, 180))

# ── save ──────────────────────────────────────────────────────────────────────
os.makedirs("assets", exist_ok=True)
out = "assets/social_preview.png"
img.save(out, "PNG", optimize=True)
print(f"Saved → {out}  ({img.size[0]}×{img.size[1]})")
