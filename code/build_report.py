"""Builds the Word report. Usage: python build_report.py <cv_dir> <out.docx> [toc_pages.json]"""
import json
import os
import sys

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING, WD_TAB_ALIGNMENT, WD_TAB_LEADER
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image
import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
TOC = json.load(open(sys.argv[3])) if len(sys.argv) > 3 else {}
FIG = os.path.join(ROOT, "figures")
R1 = json.load(open(os.path.join(ROOT, "outputs/cs1/cs1_results.json")))
R2 = json.load(open(os.path.join(ROOT, "outputs/cs2/cs2_results.json")))
FACTS = json.load(open(os.path.join(ROOT, "outputs/cs1/figure_facts.json")))
AGREE = json.load(open(os.path.join(ROOT, "outputs/cs2/agreement_frame.json")))
TEXT_W = 8.27 - 2.5
FONT = "Times New Roman"

doc = Document()
sec = doc.sections[0]
sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
sec.left_margin, sec.right_margin = Inches(1.5), Inches(1.0)
sec.top_margin, sec.bottom_margin = Inches(1.0), Inches(1.0)
sec.different_first_page_header_footer = True


def set_font(style, size, bold=False):
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    rpr = style.element.get_or_add_rPr()
    fonts = rpr.find(qn("w:rFonts"))
    if fonts is None:
        fonts = OxmlElement("w:rFonts")
        rpr.append(fonts)
    for attr in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
        fonts.set(qn(attr), FONT)
    for attr in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
        fonts.attrib.pop(qn(attr), None)


st = doc.styles
set_font(st["Normal"], 12)
pf = st["Normal"].paragraph_format
pf.line_spacing_rule = WD_LINE_SPACING.ONE_POINT_FIVE
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.first_line_indent = Inches(0.5)
pf.space_before, pf.space_after = Pt(0), Pt(6)

set_font(st["Heading 1"], 14, True)
h1 = st["Heading 1"].paragraph_format
h1.alignment = WD_ALIGN_PARAGRAPH.CENTER
h1.first_line_indent = Inches(0)
h1.space_before, h1.space_after = Pt(0), Pt(12)
h1.keep_with_next = True

set_font(st["Heading 2"], 12, True)
h2 = st["Heading 2"].paragraph_format
h2.alignment = WD_ALIGN_PARAGRAPH.LEFT
h2.first_line_indent = Inches(0)
h2.space_before, h2.space_after = Pt(10), Pt(4)
h2.keep_with_next = True

set_font(st["List Bullet"], 12)
lb = st["List Bullet"].paragraph_format
lb.first_line_indent = Inches(-0.25)
lb.left_indent = Inches(0.5)
lb.space_after = Pt(3)
lb.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
set_font(st["List Number"], 12)
ln = st["List Number"].paragraph_format
ln.first_line_indent = Inches(-0.25)
ln.left_indent = Inches(0.5)
ln.space_after = Pt(3)

PHOTO = {"cs1_references.png", "cs1_query_samples.png", "cs1_keypoints.png", "cs1_matches_same.png", "cs1_matches_cases.png",
         "cs1_failures.png", "cs2_highway.png", "cs2_urban.png", "cs2_agreement.png", "cs2_conditions_grid.png"}
counts = {"fig": 0, "tab": 0}
# Figure/table numbers are prefixed by chapter
chapter = {"n": 0}


def para(text, indent=True, align=None, bold=False, size=None, after=None, before=None, keep=False, spacing=None):
    p = doc.add_paragraph()
    add_runs(p, text, bold, size)
    if not indent:
        p.paragraph_format.first_line_indent = Inches(0)
    if align is not None:
        p.alignment = align
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    if before is not None:
        p.paragraph_format.space_before = Pt(before)
    if keep:
        p.paragraph_format.keep_with_next = True
    if spacing is not None:
        p.paragraph_format.line_spacing = spacing
    return p


def add_runs(p, text, bold=False, size=None):
    """**bold** markup inside text."""
    parts = text.split("**")
    for i, part in enumerate(parts):
        if not part:
            continue
        r = p.add_run(part)
        r.bold = bold or i % 2 == 1
        if size:
            r.font.size = Pt(size)


def bullets(items, style="List Bullet"):
    for it in items:
        p = doc.add_paragraph(style=style)
        add_runs(p, it)


def numbered(items):
    """Manual numbers so each list restarts at 1."""
    for i, it in enumerate(items, 1):
        p = doc.add_paragraph()
        p.paragraph_format.first_line_indent = Inches(-0.25)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.tab_stops.add_tab_stop(Inches(0.5))
        add_runs(p, f"{i}.\t{it}")


def chap(title):
    chapter["n"] += 1
    counts["fig"] = counts["tab"] = 0
    p = doc.add_heading(f"Chapter {chapter['n']}: {title}", level=1)
    if chapter["n"] > 1:
        p.paragraph_format.page_break_before = True
    return p


def sub(title):
    return doc.add_heading(title, level=2)


def figure(name, caption, width=TEXT_W, max_h=7.5):
    path = os.path.join(FIG, name)
    if name in PHOTO:
        jpg = os.path.join(ROOT, "figures_jpg", name.replace(".png", ".jpg"))   # made and verified by compress_figures.py
        if os.path.exists(jpg):
            path = jpg
    w, h = Image.open(path).size
    width = min(width, TEXT_W)
    if width * h / w > max_h:
        width = max_h * w / h
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    p.paragraph_format.line_spacing = 1.0
    p.paragraph_format.keep_with_next = True
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(path, width=Inches(width))
    counts["fig"] += 1
    c = para(f"Figure {chapter['n']}.{counts['fig']}: {caption}", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=10, spacing=1.0)
    return f"{chapter['n']}.{counts['fig']}"


def table(caption, header, rows, widths, size=11):
    counts["tab"] += 1
    para(f"Table {chapter['n']}.{counts['tab']}: {caption}", indent=False, align=WD_ALIGN_PARAGRAPH.CENTER, after=4, before=6, keep=True, spacing=1.0)
    t = doc.add_table(rows=1 + len(rows), cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    scale = TEXT_W / max(sum(widths), TEXT_W)
    for i, row in enumerate([header] + rows):
        tr = t.rows[i]
        if i < len(rows):
            tr._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
        if i == 0:
            tr._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
        for j, val in enumerate(row):
            cell = tr.cells[j]
            cell.width = Inches(widths[j] * scale)
            p = cell.paragraphs[0]
            p.paragraph_format.first_line_indent = Inches(0)
            p.paragraph_format.line_spacing = 1.0
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j == 0 else WD_ALIGN_PARAGRAPH.CENTER
            if i < len(rows):
                p.paragraph_format.keep_with_next = True
            r = p.add_run(str(val))
            r.font.size = Pt(size)
            r.bold = i == 0
            if i == 0:
                shd = OxmlElement("w:shd")
                shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), "E7E6E6")
                cell._tc.get_or_add_tcPr().append(shd)
    para("", indent=False, after=4, spacing=1.0)


def page_number_footer():
    p = sec.footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.first_line_indent = Inches(0)
    r = p.add_run()
    for kind, txt in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            e = OxmlElement("w:fldChar"); e.set(qn("w:fldCharType"), kind)
        else:
            e = OxmlElement("w:instrText"); e.set(qn("xml:space"), "preserve"); e.text = txt
        r._r.append(e)
    r.font.name = FONT
    r.font.size = Pt(12)


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


page_number_footer()
CENTER = WD_ALIGN_PARAGRAPH.CENTER

# ---------------------------------------------------------------- Title page
for text, size, bold, after in [
    ("A Case Study Report On", 16, True, 6),
    ("Advanced Computer Vision (CA-II)", 16, True, 14),
    ("Image Recognition Using Feature Matching", 14, True, 2),
    ("and", 12, False, 2),
    ("Vehicle Detection Using YOLO", 14, True, 16),
    ("Submitted by", 14, True, 4),
    ("Suraj Singh Bayas", 14, False, 0),
    ("Roll No. 13, B.Tech AI & DS", 14, False, 14),
    ("Under the guidance of", 12, False, 0),
    ("Prof. G. N. Chanderki", 14, True, 14),
    ("In partial fulfilment for the award of the degree of", 12, False, 0),
    ("Bachelor of Technology", 14, True, 0),
    ("in", 12, False, 0),
    ("Artificial Intelligence & Data Science Engineering", 14, True, 14),
]:
    para(text, indent=False, align=CENTER, bold=bold, size=size, after=after, spacing=1.15)
p = doc.add_paragraph(); p.alignment = CENTER
p.paragraph_format.first_line_indent = Inches(0); p.paragraph_format.space_after = Pt(12); p.paragraph_format.line_spacing = 1.0
p.add_run().add_picture(os.path.join(ROOT, "logo/word/media/image1.png"), width=Inches(1.3))
for text, size, after in [
    ("Pradnya Niketan Education Society, Pune.", 14, 2),
    ("Nagesh Karajagi Orchid College of Engineering & Technology, Solapur", 14, 8),
    ("2026-27", 14, 0),
]:
    para(text, indent=False, align=CENTER, bold=True, size=size, after=after, spacing=1.15)

# ---------------------------------------------------------------- Abstract
ms, mo = R1["mixed_sift"], R1["mixed_orb"]
th15 = next(t for t in ms["thresholds"] if t["th"] == 15)
orb15 = next(t for t in mo["thresholds"] if t["th"] == 15)
vids = R2["videos"]; HW, UR = vids["Highway (sparse)"], vids["Urban (dense)"]
m = lambda v, name, sz: next(r for r in v["models"] if r["model"] == name and r["imgsz"] == sz)
cond = lambda v, name: next(r for r in v["conditions"] if r["condition"] == name)

ab = doc.add_heading("Abstract", level=1); ab.paragraph_format.page_break_before = True
para(f"This report presents two computer vision systems developed in Python. The first identifies library book covers using SIFT features, KNN matching, Lowe's ratio test and RANSAC geometric verification. It was tested on seven real book covers using 140 simulated photographs that varied viewing angle, scale, rotation, blur and lighting, giving 980 reference and query pairs. With a threshold of {15} inliers fixed in advance, it achieved {pct(th15['precision'], 0)} precision, {pct(th15['recall'])} recall and {pct(ms['top1'], 0)} top-1 identification. Reliable recognition held up to about 55° of viewing angle and down to 0.4 times the reference scale. It failed beyond about 65°, under heavy blur and in very dark images.")
para(f"The second system detects vehicles in traffic videos using a COCO-pretrained YOLOv8 model, and draws bounding boxes, class labels and live counts. It was evaluated on a sparse highway video and a dense urban video. No annotated ground truth was available, so results were measured against a stronger tiled YOLOv8x reference that was partly checked by eye. YOLOv8n at 640 pixels found at least {pct(min(HW['by_size'][1]['recall'], UR['by_size'][1]['recall']), 0)} of medium and large vehicles but only {pct(HW['by_size'][0]['recall'], 0)} to {pct(UR['by_size'][0]['recall'], 0)} of small, distant ones, so it undercounted dense traffic by about {abs(UR['count_bias'] / UR['gt_mean']) * 100:.0f}%. Simulated low light and motion blur caused the largest losses. In both case studies, the main limits are understood and practical improvements are identified.")
para("**Keywords:** SIFT, KNN matching, RANSAC, homography, object detection, YOLOv8, traffic monitoring.", indent=False)

# ---------------------------------------------------------------- Contents (static, numbered from the rendered PDF)
toc_title = doc.add_heading("Table of Contents", level=1); toc_title.paragraph_format.page_break_before = True
TOC_ANCHOR = doc.add_paragraph(); TOC_ANCHOR.paragraph_format.first_line_indent = Inches(0)
TOC_ANCHOR.paragraph_format.space_after = Pt(0)

# ---------------------------------------------------------------- Chapter 1
c1 = chap("Introduction"); c1.paragraph_format.page_break_before = True
sub("1.1 Background")
para("Computer vision allows software to extract useful information from images and video. This report covers two of its central tasks. **Recognition by feature matching** decides whether two images show the same object by comparing local image patterns. **Object detection** finds and classifies every object of interest in an image. The first relies on hand-designed features and geometry, while the second relies on a deep neural network trained on a large dataset. Studying both shows how each approach performs and where each one fails.")
sub("1.2 Scope of the Report")
para("Each case study is reported in the same order: the problem, the theory, the implementation, the results, the analysis of those results and a critical evaluation of the algorithm. Both programs were executed, and every number, chart and output image in this report comes from those runs.")
sub("1.3 Mapping to the Evaluation Criteria")
para("Table 1.1 shows where each evaluation criterion is addressed. The criteria are the same for both case studies, apart from the name of the method.", keep=True)
table("Evaluation criteria and report sections",
      ["Evaluation criterion", "Marks", "CO", "Case Study 1", "Case Study 2"],
      [["Implementation of the method", 4, "CO5", "Section 2.4", "Section 3.4"],
       ["Execution and presentation of results", 2, "CO5", "Section 2.5", "Section 3.5"],
       ["Analysis and interpretation of results", 2, "CO4", "Section 2.6", "Section 3.6"],
       ["Critical evaluation of performance and limitations", 2, "CO4", "Section 2.7", "Section 3.7"]],
      [2.45, 0.65, 0.6, 1.05, 1.05])
sub("1.4 Experimental Setup")
para("All experiments used the software and hardware listed in Table 1.2. Timings are specific to this machine and will differ on other hardware.", keep=True)
table("Software and hardware used",
      ["Item", "Version or details"],
      [["Language", "Python 3.11.4"], ["Image processing", "OpenCV 4.13.0, NumPy"],
       ["Deep learning", "PyTorch 2.12.0, Ultralytics 8.4.68"],
       ["Hardware", "Apple M2, CPU and integrated GPU (MPS)"]],
      [1.9, 3.8])

# ---------------------------------------------------------------- Chapter 2
chap("Image Recognition Using Feature Matching")
sub("2.1 Problem Statement")
para("A college library wants to identify book covers from images captured under different angles and scales. A Python program using OpenCV is required to extract and match image features, and to determine whether two images represent the same book.")
sub("2.2 Tasks")
numbered(["Implement SIFT feature extraction and KNN-based feature matching using Python and OpenCV.",
          "Display the matched features between the reference and query images.",
          "Analyze the matching results and discuss the algorithm's performance and limitations."])
sub("2.3 Theory")
para("**Scale-Invariant Feature Transform (SIFT).** SIFT finds distinctive points that can be detected again when an image is scaled or rotated [1]. It builds a Difference-of-Gaussians scale space and selects local extrema as candidate keypoints. Low-contrast and edge-like candidates are removed. Each keypoint is assigned a dominant orientation, and a 128-dimensional descriptor is computed from gradient histograms around it, relative to that orientation. This gives invariance to scale and rotation, and partial tolerance to lighting and viewpoint changes.")
para("**K-Nearest-Neighbour matching and the ratio test.** For each query descriptor, the matcher finds its two nearest reference descriptors by Euclidean distance. On a repetitive cover, the closest match alone can be wrong. Lowe's ratio test therefore keeps a match only when the closest distance d1 is clearly smaller than the second closest d2, that is, d1 < 0.75 × d2 [1].")
para("**RANSAC homography.** A flat cover seen from two viewpoints is related by a 3 × 3 homography. RANSAC repeatedly fits a homography to random sets of four matches and keeps the one supported by the most matches, called inliers [2]. Random false matches rarely agree on one geometry, so the inlier count is a much stronger signal than the raw match count. The program reports “same book” when there are at least 15 inliers.")
figure("cs1_pipeline.png", "Processing pipeline of the book cover recognition system")

sub("2.4 Implementation")
para("The program is written as small functions in book_matcher.py, using OpenCV [7]. Figure 2.2 shows SIFT extraction and KNN matching, Figure 2.3 shows geometric verification and the complete pipeline, and Figure 2.4 shows how the matches are displayed.")
figure("code_cs1_a.png", "SIFT feature extraction and KNN matching with the ratio test")
figure("code_cs1_b.png", "RANSAC verification and the complete matching pipeline")
figure("code_cs1_c.png", "Drawing matched features and outlining the located cover")
para("**Dataset.** Seven real book covers were downloaded from the Open Library Covers API [10] to serve as the reference database (Figure 2.5). They range from highly textured to mostly plain designs. As shown in Table 2.1, this produced between "
     f"{min(R1['ref_keypoints'].values())} and {max(R1['ref_keypoints'].values())} keypoints per cover.")
figure("cs1_references.png", "The seven reference book covers")
para("**Query images.** The query images are simulated rather than photographed. Each cover was placed on a synthetic desk surface using a pinhole camera model, and lighting variation, sensor noise and JPEG compression were then added (Figure 2.6). Simulation has one major advantage: the true position of the cover is known exactly, so the location error can be measured. Its limitation is discussed in Section 2.7.")
figure("cs1_query_samples.png", "Examples of the simulated query images", max_h=4.2)

sub("2.5 Execution and Results")
facts = FACTS["demo"]
para(f"Figure 2.7 shows the program running on two pairs. For the same book, viewed at 35° with blur, it kept {facts['good']} good matches and {facts['inliers']} inliers. For a different book, it kept only 3 good matches and no inliers, and correctly rejected the pair.")
figure("term_cs1.png", "Program output for a same-book pair and a different-book pair")
para("Figure 2.8 shows the SIFT keypoints detected on a reference cover. The circle size shows the scale of each keypoint, and the radius line shows its orientation. Keypoints gather on the branches and flowers, whereas plain areas of the cover have few.")
figure("cs1_keypoints.png", "SIFT keypoints detected on the reference cover", max_h=3.2)
para("Figure 2.9 is the main output required by Task 2. It displays the matched features between the reference and query images, before and after RANSAC.")
figure("cs1_matches_same.png", "Matched features between the reference and query images", max_h=6.8)
p = FACTS["pairs"]
para(f"Figure 2.10 shows three further cases. The cover is recognised when it is much smaller than the reference ({p[0]['inliers']} inliers) and when it is rotated by 120° ({p[1]['inliers']} inliers). A different book produces only {p[2]['good']} matches, none of which agree on a geometry.")
figure("cs1_matches_cases.png", "Matching across scale, rotation and a different book", max_h=7.4)

sub("2.6 Analysis and Interpretation of Results")
para(f"**Overall recognition performance.** A test set of 140 queries was generated, 20 per book, with random yaw (±50°), pitch (±35°), rotation (±180°), scale (0.35 to 1.4), blur, brightness and noise. Each query was compared with all seven references, giving 980 pairs. Table 2.1 shows the results by book, and Table 2.2 shows the overall confusion matrix at the threshold of 15 inliers, which was fixed before the test was run.")
rows = [[n, R1["ref_keypoints"][n], f"{ms['per_book_median'][n]:.0f}", ms["per_book_min"][n]] for n in R1["ref_keypoints"]]
table("Reference keypoints and inliers on the mixed test set, by book", ["Book", "Reference keypoints", "Median inliers", "Lowest inliers"], rows, [2.6, 1.1, 1.0, 1.0])
table("Results at 15 inliers over 980 pairs", ["", "Predicted same book", "Predicted different book"],
      [["Actually same book (140)", th15["tp"], th15["fn"]], ["Actually different book (840)", th15["fp"], th15["tn"]]], [2.5, 1.6, 1.6])
acc = (th15["tp"] + th15["tn"]) / 980
para(f"This corresponds to {pct(th15['precision'], 0)} precision, {pct(th15['recall'])} recall and {pct(acc)} accuracy. When a query was assigned to the reference with the most inliers, top-1 identification was {pct(ms['top1'], 0)}. On accepted queries, the estimated cover corners were a median of {ms['corner_err_median']:.2f} pixels from their true positions, and 95% were within {ms['corner_err_p95']:.2f} pixels. The system therefore locates the cover as well as recognising it.")
para(f"**Separation between same and different books.** Figure 2.11 explains why the system makes so few errors. Same-book pairs had a median of {ms['genuine_median']:.0f} inliers, while different-book pairs had a median of {ms['impostor_median']:.0f} and never exceeded {ms['impostor_max']}. The threshold of 15 lies in the gap between them. Table 2.3 shows the trade-off for other thresholds. A threshold of 10 starts to admit false acceptances, while a threshold of 30 rejects more true matches.")
figure("cs1_distribution.png", "Inlier counts for same-book and different-book pairs")
rows = [[t["th"], t["tp"], t["fn"], t["fp"], pct(t["precision"]), pct(t["recall"]), f"{t['f1']:.3f}"] for t in ms["thresholds"] if t["th"] in (8, 10, 12, 15, 20, 30)]
table("Effect of the inlier threshold", ["Threshold", "TP", "FN", "FP", "Precision", "Recall", "F1"], rows, [0.9, 0.6, 0.6, 0.6, 1.0, 1.0, 0.8])
para(f"**Effect of the ratio threshold.** Figure 2.12 shows the trade-off in the ratio test. At 0.5, {pct(R1['ratio'][0]['match_precision'], 0)} of the kept matches are correct, but few are kept. At 1.0, which is equivalent to no ratio test, only {pct(R1['ratio'][-1]['match_precision'], 0)} are correct. The chosen value of 0.75 keeps {pct(R1['ratio'][3]['match_precision'], 0)} correct matches while retaining enough for RANSAC.")
figure("cs1_ratio.png", "Effect of the ratio threshold on match quality and quantity")
para("**Effect of individual conditions.** Each condition was then varied on its own, for all seven books (Figure 2.13). Table 2.4 summarises the range in which all seven covers were recognised. In the charts, a value of zero inliers is plotted at the bottom of the axis.")
figure("cs1_sweeps.png", "Mean RANSAC inliers as each condition is varied")
table("Operating range of the recogniser (out of seven books)", ["Condition", "All 7 accepted", "Partial", "None accepted"],
      [["Viewing angle (yaw)", "Up to 55°", "1 of 7 at 65°", "70° and above"],
       ["Scale", "0.4 to 3.0", "2 to 6 of 7 at 0.15 to 0.3", "Not reached"],
       ["In-plane rotation", "All, 0° to 180°", "None", "Not reached"],
       ["Gaussian blur (σ)", "Up to 3", "6 of 7 at 4, 1 of 7 at 6", "8"],
       ["Brightness gain", "Down to 0.3", "5 of 7 at 0.15", "0.08 and below"],
       ["Cover hidden", "Up to 60%", "6 of 7 at 80%, 2 of 7 at 90%", "Not reached"],
       ["Simulated glare", "All tested levels", "None", "Not reached"]], [1.6, 1.3, 1.9, 1.0])
para("Two conclusions stand out. First, rotation has almost no effect on the inlier count, which confirms the rotation invariance of SIFT. Second, scale and viewing angle behave differently: enlarging the cover causes no loss, but shrinking it steadily removes detail, and tilting it causes a sharp collapse after about 55°.")

sub("2.7 Critical Evaluation of Performance and Limitations")
f = FACTS["fails"]
para(f"**Strengths.** The method requires no training, and adding a book only needs one reference image. It is invariant to rotation and scale within the tested ranges, handles heavy occlusion, and produces a location as well as a decision. Inliers provide a clear and explainable basis for each decision.")
para(f"**Comparison with ORB.** ORB, a faster binary alternative to SIFT [3], was run on the same pipeline and test set (Table 2.5). ORB extracted features about {ms['ms_extract'] / mo['ms_extract']:.0f} times faster, which is an advantage for real-time use. With up to 5000 features, however, its matching stage was slower, and SIFT located the cover more accurately and had higher recall at the same threshold.", keep=True)
table("SIFT and ORB on the same 980 pairs", ["Metric", "SIFT", "ORB"],
      [["Precision at 15 inliers", pct(th15["precision"]), pct(orb15["precision"])], ["Recall at 15 inliers", pct(th15["recall"]), pct(orb15["recall"])],
       ["Top-1 identification", pct(ms["top1"]), pct(mo["top1"])], ["Median corner error (pixels)", f"{ms['corner_err_median']:.2f}", f"{mo['corner_err_median']:.2f}"],
       ["95th percentile corner error (pixels)", f"{ms['corner_err_p95']:.2f}", f"{mo['corner_err_p95']:.2f}"], ["Feature extraction per query (ms)", f"{ms['ms_extract']:.1f}", f"{mo['ms_extract']:.1f}"],
       ["Matching against 7 references (ms)", f"{ms['ms_match_db']:.1f}", f"{mo['ms_match_db']:.1f}"]], [3.0, 1.2, 1.2])
para(f"**Limitations.** Figure 2.14 shows three failure cases: strong tilt ({f[0]['inliers']} inliers), heavy blur ({f[1]['inliers']} inliers) and a very dark image ({f[2]['inliers']} inliers). The main limitations are:", keep=True)
figure("cs1_failures.png", "Failure cases, showing all matches that survived the ratio test", max_h=7.0)
bullets([
    "**Viewpoint.** SIFT is only partially robust to affine distortion. Recognition collapsed past about 55° to 65° of yaw.",
    f"**Low-texture covers.** Recognition depends on texture. Clean Code, a dark and simple design, had only {R1['ref_keypoints']['Clean Code']} reference keypoints and the lowest median inlier count ({ms['per_book_median']['Clean Code']:.0f}). Its weakest query had just {ms['per_book_min']['Clean Code']} inliers. Text-only or plain covers would be harder still.",
    "**Blur and darkness.** Blur destroys the fine detail that SIFT relies on, and very dark images leave too little contrast. At a brightness gain of 0.15, one cover was accepted but placed in the wrong position, showing that acceptance does not guarantee a correct location.",
    "**Non-planar and similar covers.** The homography assumes a flat cover, so a curved or bent book will lose inliers. Different editions of one book share artwork, so the system may not tell them apart.",
    "**Scalability.** Brute-force matching grows with the number of books. It took about "
    f"{ms['ms_match_db']:.0f} ms for seven books, so a full library would need approximate search such as FLANN with a visual vocabulary. Different-book scores may also rise with a much larger collection, so the threshold would need re-checking.",
    "**Colour.** SIFT uses grayscale only, so two covers with the same design in different colours would look alike.",
    "**Limits of this evaluation.** The queries are simulated and the database has only seven books. Real photographs add effects that the simulation does not fully capture, such as curved pages, uneven shadows, sleeve reflections and cluttered shelves. The simulated glare in particular is smooth, and real reflections may cause more damage. The figures here should therefore be read as an upper estimate until confirmed on real photographs.",
])
para("**Possible improvements.** Affine-SIFT could extend the viewpoint range [8]. Multiple reference views, contrast normalisation such as CLAHE, and a colour check would reduce failures. A larger system could shortlist candidates using a global image embedding, then apply SIFT and RANSAC for final verification.")
sub("2.8 Summary")
para(f"The program meets all three tasks. It correctly decided {th15['tp'] + th15['tn']} of 980 test pairs with no false acceptances, and displayed the matches clearly. It is well suited to a small library when covers are photographed within about 55° of frontal view, in reasonable light and with enough texture.")

# ---------------------------------------------------------------- Chapter 3
chap("Vehicle Detection Using YOLO")
sub("3.1 Problem Statement")
para("A traffic management department wants to automate vehicle detection from traffic videos. A Python program using a pretrained YOLO model is required to detect vehicles, and to display their bounding boxes and class labels.")
sub("3.2 Tasks")
numbered(["Implement vehicle detection using a pretrained YOLO model.", "Display detected vehicles with bounding boxes and labels.",
          "Evaluate the detection results and discuss limitations under different traffic conditions."])
sub("3.3 Theory")
para("**YOLO.** You Only Look Once (YOLO) treats detection as one regression problem: a single network pass over the image predicts all boxes and classes together, which makes it fast enough for video [4]. YOLOv8, used here, has a backbone that extracts features, a neck that combines features across scales, and an anchor-free head that predicts boxes and classes at three scales [5].")
para("**Pretrained weights.** The model was trained on the COCO dataset, which has 80 object classes [6]. No further training was needed because COCO includes five vehicle classes: bicycle, car, motorcycle, bus and truck. The program keeps only these classes.")
para("**Confidence and Non-Maximum Suppression.** Each prediction has a confidence score, and predictions below a threshold (0.25 here) are discarded. Non-Maximum Suppression (NMS) then removes overlapping boxes around the same object.")
para("**Evaluation measures.** A detection is correct when its Intersection over Union (IoU) with a reference box is at least 0.5. Precision is the share of detections that are correct, and recall is the share of reference vehicles that are found.")
figure("cs2_pipeline.png", "Processing pipeline of the vehicle detection system")

sub("3.4 Implementation")
para("The program vehicle_detector.py uses the Ultralytics library with OpenCV. Figure 3.2 shows the model call and vehicle classes, Figure 3.3 shows the drawing of boxes and labels, and Figure 3.4 shows the video loop, which also measures speed and counts detections.")
figure("code_cs2_a.png", "Vehicle classes and the pretrained YOLO detection call")
figure("code_cs2_b.png", "Drawing bounding boxes, class labels and the live count")
figure("code_cs2_c.png", "Processing the video and writing the annotated output")
para(f"**Test videos.** Two publicly available traffic videos from the Roboflow Supervision samples [11] represent different traffic conditions (Table 3.1). The evaluation used {R2['n_frames']} frames sampled evenly from each video.", keep=True)
table("Test videos", ["Property", "Highway", "Urban"],
      [["Traffic condition", "Sparse, free-flowing", "Dense, queued"], ["Resolution", f"{HW['w']} × {HW['h']}", f"{UR['w']} × {UR['h']}"],
       ["Frame rate", f"{HW['fps']:.0f} FPS", f"{UR['fps']:.0f} FPS"], ["Total frames", HW["frames"], UR["frames"]], ["Lighting", "Bright daylight", "Overcast, headlights on"]], [1.8, 1.8, 1.8])
para("**Reference for evaluation.** These videos have no annotated ground truth. A reference was therefore built by running the largest model, YOLOv8x, over overlapping high-resolution tiles of each frame and merging the results, as in slicing-based inference [9]. Precision and recall in this chapter measure agreement with that reference. They are not official mAP scores.")
para("**Checking the reference.** The first reference, a standard YOLOv8x pass at a confidence of 0.50, proved unreliable: it reported many “false positives” that were real cars when inspected. This is why the tiled version was adopted. The disagreements were then checked by eye using random samples of 40 from each group. All 40 sampled urban vehicles and all 40 highway vehicles that the small model missed were real vehicles. Of 40 urban detections absent from the reference, about 36 framed a real vehicle, and the rest covered part of a vehicle or a pair of vehicles. None was clearly a non-vehicle object. Even so, the reference is not complete, and some cars in the distant queue have no box.")

sub("3.5 Execution and Results")
s = open(os.path.join(ROOT, "outputs/term_cs2.txt")).read()
para("Figure 3.5 shows the program running on both videos, and writing an annotated output video for each. Figures 3.6 and 3.7 show the required output: vehicles with bounding boxes, class labels, confidence values and a live count banner.")
figure("term_cs2.png", "Program output for the urban and highway videos")
figure("cs2_highway.png", "Detections on the highway video", max_h=6.6)
figure("cs2_urban.png", "Detections on the urban video", max_h=6.8)

sub("3.6 Analysis and Interpretation of Results")
para(f"**Sparse and dense traffic.** Figure 3.8 compares vehicle counts over time. The reference averaged {HW['gt_mean']:.1f} vehicles per frame on the highway and {UR['gt_mean']:.1f} in the city. YOLOv8n at 640 pixels averaged {sum(HW['counts_nano']) / len(HW['counts_nano']):.1f} and {sum(UR['counts_nano']) / len(UR['counts_nano']):.1f}. It follows the trend but consistently undercounts, by an average of {abs(HW['count_bias']):.1f} and {abs(UR['count_bias']):.1f} vehicles per frame.")
figure("cs2_counts.png", "Vehicles per frame for YOLOv8n and the reference")
para(f"**Vehicle size is the main factor.** The undercount is not uniform. Figure 3.9 groups reference vehicles by their apparent size at the 640-pixel input. YOLOv8n found {pct(HW['by_size'][1]['recall'], 0)} and {pct(UR['by_size'][1]['recall'], 0)} of medium vehicles, and all of the few large ones, but only {pct(HW['by_size'][0]['recall'], 0)} and {pct(UR['by_size'][0]['recall'], 0)} of small ones. Small vehicles make up most of the reference: {HW['by_size'][0]['total']} of {sum(b['total'] for b in HW['by_size'])} on the highway and {UR['by_size'][0]['total']} of {sum(b['total'] for b in UR['by_size'])} in the city. This explains why the overall recall is low, even though the model is reliable on nearby vehicles.")
figure("cs2_size.png", "Recall of YOLOv8n by apparent vehicle size", width=4.8)
para(f"Figure 3.10 shows this on one dense frame. The model matched {AGREE['matched']} of {AGREE['ref']} reference vehicles, and most of the misses are distant or partly hidden. Dense traffic adds a further difficulty, since vehicles overlap and NMS may merge nearby boxes. On one urban frame, the same YOLOv8x model found {R2['density_probe']['standard_1280']} vehicles in a standard pass and {R2['density_probe']['tiled']} with tiling, so even the largest model is affected.")
figure("cs2_agreement.png", "Comparison on a dense frame: green is matched, red is missed by YOLOv8n, and blue is detected only by YOLOv8n")
para(f"**Model size and input resolution.** Table 3.2 compares configurations. Raising the input from 640 to 1280 pixels improved recall more than moving from the nano to the small model at the same resolution. With YOLOv8s at 1280 pixels, recall rose to {pct(m(HW, 'YOLOv8s', 1280)['recall'], 0)} on the highway and {pct(m(UR, 'YOLOv8s', 1280)['recall'], 0)} in the city. YOLOv8s also gave the best F1 score of the tested options on both videos, at roughly three times the 640-pixel nano time on this machine.", keep=True)
rows = []
for r_h, r_u in zip(HW["models"], UR["models"]):
    rows.append([r_h["model"], r_h["imgsz"], pct(r_h["precision"], 0), pct(r_h["recall"], 0), pct(r_u["precision"], 0), pct(r_u["recall"], 0), f"{(r_h['ms'] + r_u['ms']) / 2:.0f}"])
table("Model and input size (agreement with the reference)", ["Model", "Input (px)", "Highway precision", "Highway recall", "Urban precision", "Urban recall", "Time (ms, GPU)"], rows, [0.9, 0.6, 0.85, 0.75, 0.75, 0.7, 0.7])
para(f"The urban precision values should be read with care. Based on the manual check of the 640-pixel results in Section 3.4, most of the “extra” detections are real vehicles that the reference lacks, so true precision is higher than these figures. The lower values at 1280 pixels may have the same cause, as the model finds more distant vehicles at that size, but they were not checked by eye.")
para(f"**Speed.** Table 3.3 shows single-frame speed at 640 pixels. YOLOv8n reached about {R2['timing'][0]['fps']:.0f} FPS on CPU and {R2['timing'][1]['fps']:.0f} FPS on the GPU in this timed test. When running the complete loop, including decoding, drawing and writing on the CPU, it averaged 15 to 17 FPS (Figure 3.5), which is lower than the videos' 25 and 30 FPS. Real-time use therefore needs GPU acceleration or frame skipping.", keep=True)
rows = []
for name in ("YOLOv8n", "YOLOv8s", "YOLOv8x"):
    c = next(t for t in R2["timing"] if t["model"] == name and t["device"] == "cpu"); g = next(t for t in R2["timing"] if t["model"] == name and t["device"] != "cpu")
    rows.append([name, f"{c['ms']:.0f}", f"{c['fps']:.1f}", f"{g['ms']:.0f}", f"{g['fps']:.1f}"])
table("Inference speed at 640 pixels (median of repeated runs)", ["Model", "CPU (ms)", "CPU (FPS)", "GPU (ms)", "GPU (FPS)"], rows, [1.3, 1, 1, 1, 1])
para("**Confidence threshold.** Figure 3.11 shows that raising the threshold increases agreement precision but quickly lowers recall. The default of 0.25 is a reasonable balance for counting, while a monitoring system that must avoid false alarms could use a higher value.")
figure("cs2_conf.png", "Effect of the confidence threshold on precision and recall")
ct = UR["class_pairs"]; correct = sum(v for k, v in ct.items() if k.split("->")[0] == k.split("->")[1]); tot = sum(ct.values())
cth = HW["class_pairs"]; correcth = sum(v for k, v in cth.items() if k.split("->")[0] == k.split("->")[1]); toth = sum(cth.values())
para(f"**Class labels.** On matched boxes, the class label agreed with the reference in {correct} of {tot} urban cases ({pct(correct / tot, 0)}) and {correcth} of {toth} highway cases ({pct(correcth / toth, 0)}). Most disagreements were trucks labelled as buses or cars, as in Figure 3.12, where a truck is labelled as a bus under rain. Car labels are therefore more dependable than truck and bus labels. Neither video contained enough bicycles or motorcycles to evaluate those classes.")

sub("3.7 Critical Evaluation under Different Traffic Conditions")
para("Weather and lighting were simulated by applying image effects to the same frames, so that each condition could be compared directly against the clean result (Figures 3.12 and 3.13, Table 3.4). These effects approximate real conditions. Real night traffic, for example, has headlight glare, and real heavy rain also changes how vehicles appear.")
figure("cs2_conditions_grid.png", "YOLOv8n detections on one urban frame under simulated conditions", max_h=6.6)
figure("cs2_conditions_chart.png", "Recall against the clean-frame reference under each simulated condition")
rows = []
for name in [r["condition"] for r in HW["conditions"]]:
    h, u = cond(HW, name), cond(UR, name)
    h0, u0 = cond(HW, "Clean"), cond(UR, "Clean")
    rows.append([name, f"{h['count_mean']:.1f}", "-" if name == "Clean" else f"{100 * (h['count_mean'] / h0['count_mean'] - 1):+.0f}%", pct(h["recall"], 0), f"{u['count_mean']:.1f}", "-" if name == "Clean" else f"{100 * (u['count_mean'] / u0['count_mean'] - 1):+.0f}%", pct(u["recall"], 0)])
table("YOLOv8n under simulated conditions", ["Condition", "Highway count", "Change", "Highway recall", "Urban count", "Change", "Urban recall"], rows, [1.25, 0.75, 0.7, 0.75, 0.7, 0.7, 0.65])
lo = cond(UR, "Low light"); mb = cond(UR, "Motion blur"); lr = cond(UR, "Low resolution"); u0 = cond(UR, "Clean")
para(f"Low light caused the largest loss in the city, with the average count falling from {u0['count_mean']:.1f} to {lo['count_mean']:.1f}. Low resolution and motion blur followed. Fog and rain had smaller effects in these simulations. In every case, dense urban traffic lost a larger share of its detections than sparse highway traffic, because its small vehicles have less detail to lose.")
para("**Strengths.** The system needs no training and is simple to run. It gives boxes, labels and counts for every frame, and is dependable for nearby vehicles in good conditions. A department can also choose the balance of speed and accuracy by changing the model or input size.")
para("**Limitations.**", keep=True, after=2)
bullets([
    f"**Small and distant vehicles.** The 640-pixel input removes detail, especially from a 4K frame, which is reduced to one sixth of its width. This is the largest source of missed vehicles.",
    f"**Dense traffic and occlusion.** Queued vehicles hide one another. Counts in the urban video were about {abs(UR['count_bias'] / UR['gt_mean']) * 100:.0f}% below the reference, and the reference itself is incomplete, so the true shortfall is at least this large.",
    "**Lighting and weather.** Darkness and blur sharply reduce detection. A system tuned in daylight cannot be assumed to work at night or in bad weather.",
    "**Label confusion.** COCO's general classes do not distinguish autorickshaws, tempos or similar vehicles common on Indian roads. A pretrained COCO model may mislabel or miss them, and would need retraining on local data.",
    "**No tracking.** Each frame is processed separately, so per-frame counts fluctuate and one vehicle cannot be followed. Counting total traffic flow requires a tracker such as ByteTrack.",
    "**Limits of this evaluation.** The reference comes from a model rather than human annotation. It shares some weaknesses with the tested models and misses some vehicles. Only two videos were tested, with simulated weather. A proper mAP figure would need manually annotated frames from the department's own cameras.",
])
para("**Possible improvements.** Practical steps include using YOLOv8s at 1280 pixels when a GPU is available, tiling for distant queues, adding a tracker for stable counts, placing cameras higher to reduce occlusion, enhancing low-light images, and fine-tuning on local Indian traffic data.")
sub("3.8 Summary")
para(f"The program meets all three tasks. It detects vehicles in video and displays their boxes, labels and counts. It is reliable for medium and large vehicles, with {pct(HW['by_size'][1]['recall'], 0)} to {pct(UR['by_size'][1]['recall'], 0)} recall on medium vehicles at the default setting, but it undercounts small vehicles and dense queues. It should be treated as a working baseline that needs higher resolution, tracking and local tuning before operational use.")

# ---------------------------------------------------------------- Chapter 4
chap("Conclusion")
sub("4.1 Comparison of the Two Approaches")
para("The two case studies solve different problems using different methods. Table 4.1 compares them.", keep=True)
table("Feature matching and YOLO detection compared", ["Aspect", "SIFT with KNN and RANSAC", "YOLOv8"],
      [["Question answered", "Is this the same specific object?", "Which objects are in this image, and where?"], ["Training", "None; one reference image per item", "Pretrained on COCO"],
       ["Recognises", "Specific instances", "General classes"], ["Main weakness", "Low texture, strong tilt, blur", "Small, distant and overlapping objects"],
       ["Explainability", "High: visible matches and inliers", "Lower: internal to the network"], ["Speed in these tests", f"About {ms['ms_extract'] + ms['ms_match_db']:.0f} ms per query, CPU", f"About {R2['timing'][0]['ms']:.0f} ms per frame, CPU"]], [1.4, 2.2, 2.2])
sub("4.2 Conclusion")
para(f"Case Study 1 showed that SIFT, KNN matching and RANSAC can identify book covers across angle, scale and rotation changes. On simulated queries of seven real covers, it reached {pct(th15['precision'], 0)} precision and {pct(th15['recall'])} recall, and the RANSAC inlier count gave a clear separation between same and different books. It was limited by strong tilt, heavy blur, darkness and low-texture covers.")
para(f"Case Study 2 showed that a pretrained YOLOv8 model can provide a working traffic detection system with very little code. It performed well on nearby vehicles, but missed many small vehicles and undercounted dense traffic. Higher input resolution helped more than a larger model at the same resolution. Low light and motion blur caused the largest losses.")
para("An important lesson came from checking results by eye. An initial evaluation reported high false-positive rates, but inspection showed that these were real cars missing from a weak reference. The reliability of any evaluation depends on the quality of its ground truth.")
sub("4.3 Future Scope")
bullets(["Test the book recogniser on real photographs of a larger library collection, including shelves and damaged covers.",
         "Add approximate nearest-neighbour search and a global image embedding so that recognition scales to thousands of books.",
         "Annotate frames from local traffic cameras to obtain true mAP, and evaluate on real night and rain footage.",
         "Fine-tune YOLO for Indian vehicle classes, and add tracking for vehicle counting and flow analysis."])

refs = doc.add_heading("References", level=1); refs.paragraph_format.page_break_before = True
for i, r in enumerate([
    "D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91-110, 2004.",
    "M. A. Fischler and R. C. Bolles, “Random Sample Consensus: A Paradigm for Model Fitting with Applications to Image Analysis and Automated Cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381-395, 1981.",
    "E. Rublee, V. Rabaud, K. Konolige and G. Bradski, “ORB: An Efficient Alternative to SIFT or SURF,” in Proc. IEEE International Conference on Computer Vision, pp. 2564-2571, 2011.",
    "J. Redmon, S. Divvala, R. Girshick and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” in Proc. IEEE Conference on Computer Vision and Pattern Recognition, pp. 779-788, 2016.",
    "G. Jocher, A. Chaurasia and J. Qiu, “Ultralytics YOLOv8,” 2023. [Online]. Available: https://github.com/ultralytics/ultralytics",
    "T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” in Proc. European Conference on Computer Vision, pp. 740-755, 2014.",
    "G. Bradski, “The OpenCV Library,” Dr. Dobb's Journal of Software Tools, 2000.",
    "J.-M. Morel and G. Yu, “ASIFT: A New Framework for Fully Affine Invariant Image Comparison,” SIAM Journal on Imaging Sciences, vol. 2, no. 2, pp. 438-469, 2009.",
    "F. C. Akyon, S. O. Altinuc and A. Temizel, “Slicing Aided Hyper Inference and Fine-Tuning for Small Object Detection,” in Proc. IEEE International Conference on Image Processing, pp. 966-970, 2022.",
    "Internet Archive, “Open Library Covers API.” [Online]. Available: https://openlibrary.org/dev/docs/api/covers",
    "Roboflow, “Supervision: Reusable Computer Vision Tools,” sample videos. [Online]. Available: https://github.com/roboflow/supervision",
], 1):
    p = doc.add_paragraph()
    p.paragraph_format.first_line_indent = Inches(-0.4)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4))
    add_runs(p, f"[{i}]\t{r}")

# ---------------------------------------------------------------- Fill the contents page
HEADINGS = [p for p in doc.paragraphs if p.style.name in ("Heading 1", "Heading 2") and p.text != "Table of Contents"]
anchor = TOC_ANCHOR
for hp in HEADINGS:
    level = 1 if hp.style.name == "Heading 1" else 2
    new = OxmlElement("w:p"); anchor._p.addnext(new)
    from docx.text.paragraph import Paragraph
    q = Paragraph(new, anchor._parent)
    pf_ = q.paragraph_format
    pf_.alignment = WD_ALIGN_PARAGRAPH.LEFT
    pf_.first_line_indent = Inches(0)
    pf_.left_indent = Inches(0 if level == 1 else 0.3)
    pf_.line_spacing = 1.15
    pf_.space_after = Pt(1 if level == 2 else 2)
    pf_.space_before = Pt(5 if level == 1 else 0)
    pf_.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    r = q.add_run(f"{hp.text}\t{TOC.get(hp.text, '')}")
    r.bold = level == 1
    anchor = q

json.dump([p.text for p in HEADINGS], open(os.path.join(ROOT, "outputs/headings.json"), "w"))
doc.core_properties.author = "Suraj Singh Bayas"
doc.core_properties.title = "Advanced Computer Vision Case Study Report"
doc.save(OUT)
print("saved", OUT, "headings", len(HEADINGS))
