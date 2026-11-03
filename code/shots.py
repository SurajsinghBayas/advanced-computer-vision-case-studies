"""Render code excerpts and captured terminal output as editor/terminal style images."""
import ast, io, os, sys
from PIL import Image, ImageDraw, ImageFont
from pygments import highlight
from pygments.formatters import ImageFormatter
from pygments.lexers import PythonLexer

OUT = sys.argv[1]; FONT = "/System/Library/Fonts/Menlo.ttc"
BG = (39, 40, 34)

def chrome(img, title):
    pad, bar = 26, 58
    W, H = img.width + 2 * pad, img.height + bar + pad
    c = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(c)
    d.rectangle([0, 0, W, bar - 8], fill=(44, 44, 52))
    for i, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([20 + i * 34, 15, 40 + i * 34, 35], fill=col)
    f = ImageFont.truetype(FONT, 22)
    tw = d.textlength(title, font=f); d.text(((W - tw) / 2, 13), title, font=f, fill=(190, 190, 198))
    c.paste(img, (pad, bar)); return c

def code_shot(path, names, name, title):
    src = open(path).read(); tree = ast.parse(src); lines = src.splitlines()
    parts = []
    for n in names:
        if n == "__imports__":
            nodes = [x for x in tree.body if isinstance(x, (ast.Import, ast.ImportFrom, ast.Assign))]
            parts.append("\n".join("\n".join(lines[x.lineno - 1:x.end_lineno]) for x in nodes)); continue
        node = next(x for x in tree.body if getattr(x, "name", None) == n)
        parts.append("\n".join(lines[node.lineno - 1:node.end_lineno]))
    txt = "\n\n".join(parts)
    png = highlight(txt, PythonLexer(), ImageFormatter(font_name="Menlo", font_size=26, line_numbers=False, style="monokai", image_pad=0, line_pad=8))
    img = Image.open(io.BytesIO(png)).convert("RGB")
    chrome(img, title).save(os.path.join(OUT, name)); print(name, img.size)

def term_shot(text, name, title):
    f = ImageFont.truetype(FONT, 26); lines = text.rstrip().splitlines()
    W = int(max(f.getlength(l) for l in lines)) + 10; H = len(lines) * 36
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        col = (120, 220, 120) if l.startswith("$") else (225, 225, 230)
        d.text((0, i * 36), l, font=f, fill=col)
    chrome(img, title).save(os.path.join(OUT, name)); print(name, img.size)

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    code = os.path.dirname(os.path.abspath(__file__))
    bm, vd = os.path.join(code, "book_matcher.py"), os.path.join(code, "vehicle_detector.py")
    code_shot(bm, ["__imports__", "extract_features", "knn_match"], "code_cs1_a.png", "book_matcher.py: SIFT extraction and KNN matching")
    code_shot(bm, ["verify_geometry", "match_images"], "code_cs1_b.png", "book_matcher.py: RANSAC verification and pipeline")
    code_shot(bm, ["draw_matches"], "code_cs1_c.png", "book_matcher.py: match visualisation")
    code_shot(vd, ["__imports__", "detect"], "code_cs2_a.png", "vehicle_detector.py: model and vehicle classes")
    code_shot(vd, ["draw"], "code_cs2_b.png", "vehicle_detector.py: bounding boxes and labels")
    code_shot(vd, ["process_video"], "code_cs2_c.png", "vehicle_detector.py: video processing loop")
    for t, n, title in [(sys.argv[2], "term_cs1.png", "Terminal: book_matcher.py"), (sys.argv[3], "term_cs2.png", "Terminal: vehicle_detector.py")]:
        term_shot(open(t).read(), n, title)
