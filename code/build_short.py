"""10-page single-topic report. Usage: python build_short.py <cv_dir> <out.docx> [toc.json]"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "build_report.py")).read().split("# ---------------------------------------------------------------- Title page")[0]
exec(compile(src, "helpers", "exec"))

_chap = chap
def chap(title):
    h = _chap(title); h.paragraph_format.page_break_before = False; h.paragraph_format.space_before = Pt(12); return h

ms, mo = R1["mixed_sift"], R1["mixed_orb"]
th15 = next(t for t in ms["thresholds"] if t["th"] == 15); orb15 = next(t for t in mo["thresholds"] if t["th"] == 15)
acc = (th15["tp"] + th15["tn"]) / 980
demo = FACTS["demo"]

for text, size, bold, after in [
    ("A Case Study Report On", 16, True, 6), ("Advanced Computer Vision (CA-II)", 16, True, 16),
    ("Case Study 1:", 14, True, 2), ("Image Recognition Using Feature Matching", 14, True, 18),
    ("Submitted by", 14, True, 4), ("Suraj Singh Bayas", 14, False, 0), ("Roll No. 13, B.Tech AI & DS", 14, False, 16),
    ("Under the guidance of", 12, False, 0), ("Prof. G. N. Chanderki", 14, True, 16),
    ("In partial fulfilment for the award of the degree of", 12, False, 0), ("Bachelor of Technology", 14, True, 0), ("in", 12, False, 0),
    ("Artificial Intelligence & Data Science Engineering", 14, True, 16)]:
    para(text, indent=False, align=CENTER, bold=bold, size=size, after=after, spacing=1.15)
p = doc.add_paragraph(); p.alignment = CENTER
p.paragraph_format.first_line_indent = Inches(0); p.paragraph_format.space_after = Pt(14); p.paragraph_format.line_spacing = 1.0
p.add_run().add_picture(os.path.join(ROOT, "logo/word/media/image1.png"), width=Inches(1.3))
for text, after in [("Pradnya Niketan Education Society, Pune.", 2), ("Nagesh Karajagi Orchid College of Engineering & Technology, Solapur", 8), ("2026-27", 0)]:
    para(text, indent=False, align=CENTER, bold=True, size=14, after=after, spacing=1.15)

ab = doc.add_heading("Abstract", level=1); ab.paragraph_format.page_break_before = True
para(f"This report presents a Python and OpenCV program that decides whether two images show the same book, for use in a college library. It uses SIFT features, KNN matching, Lowe's ratio test and RANSAC geometric verification. The program was tested on seven real book covers using 140 simulated photographs, giving 980 reference and query pairs. With a threshold of 15 inliers fixed in advance, it achieved {pct(th15['precision'], 0)} precision and {pct(th15['recall'])} recall. Recognition was reliable up to about 55° of viewing angle and down to 0.4 times the reference scale, but failed under stronger tilt, heavy blur and very dark lighting.")
para("**Keywords:** SIFT, KNN matching, Lowe's ratio test, RANSAC, homography.", indent=False)
toc_title = doc.add_heading("Table of Contents", level=1); toc_title.paragraph_format.space_before = Pt(14)
TOC_ANCHOR = doc.add_paragraph(); TOC_ANCHOR.paragraph_format.first_line_indent = Inches(0); TOC_ANCHOR.paragraph_format.space_after = Pt(0); TOC_ANCHOR.paragraph_format.line_spacing = 1.0

c = chap("Introduction"); c.paragraph_format.page_break_before = True
sub("1.1 Problem Statement")
para("A college library wants to identify book covers from images captured under different angles and scales. A Python program using OpenCV is required to extract and match image features, and to determine whether two images represent the same book.")
sub("1.2 Tasks")
numbered(["Implement SIFT feature extraction and KNN-based feature matching using Python and OpenCV.", "Display the matched features between the reference and query images.", "Analyze the matching results and discuss the algorithm's performance and limitations."])
sub("1.3 Theory")
para("**SIFT.** The Scale-Invariant Feature Transform finds distinctive points that can be detected again when an image is scaled or rotated [1]. Keypoints are located as extrema in a Difference-of-Gaussians scale space, and low-contrast and edge-like candidates are removed. Each keypoint receives a dominant orientation and a 128-dimensional descriptor built from gradient histograms relative to that orientation.")
para("**KNN matching and the ratio test.** For each query descriptor, the matcher finds the two nearest reference descriptors by Euclidean distance. A match is kept only when the closest distance d1 is clearly smaller than the second closest d2, that is, d1 < 0.75 × d2 [1]. This removes ambiguous matches.")
para("**RANSAC homography.** A flat cover seen from two viewpoints is related by a 3 × 3 homography. RANSAC repeatedly fits one to random sets of four matches and keeps the one supported by the most matches, called inliers [2]. False matches rarely agree on one geometry, so the program reports “same book” when there are at least 15 inliers.")
figure("cs1_pipeline.png", "Processing pipeline of the recognition system")

chap("Implementation")
sub("2.1 Program Code")
para("Figures 2.1 to 2.3 show the main functions of the program, book_matcher.py, which uses OpenCV [4].")
figure("code_cs1_a.png", "SIFT feature extraction and KNN matching with the ratio test", width=4.05)
figure("code_cs1_b.png", "RANSAC verification and the complete matching pipeline", width=4.05)
figure("code_cs1_c.png", "Drawing matched features and outlining the located cover", width=4.05)
sub("2.2 Dataset")
para(f"Seven real book covers from the Open Library Covers API [6] form the reference database (Figure 2.4), with {min(R1['ref_keypoints'].values())} to {max(R1['ref_keypoints'].values())} keypoints per cover. The query images are simulated rather than photographed: each cover is placed on a synthetic desk using a pinhole camera model, with lighting variation, noise and JPEG compression added (Figure 2.5). Because the true cover position is known exactly, the location error can be measured.")
figure("cs1_references.png", "The seven reference book covers")
figure("cs1_query_samples.png", "Examples of simulated query images", max_h=2.3)

chap("Results and Analysis")
sub("3.1 Execution")
para(f"Figure 3.1 shows the program running. For the same book, viewed at 35° with blur, it kept {demo['good']} good matches and {demo['inliers']} inliers. For a different book, it kept only 3 good matches and no inliers, and correctly rejected the pair.")
figure("term_cs1.png", "Program output for a same-book pair and a different-book pair", width=3.9)
sub("3.2 Matched Features")
para("Figure 3.2 is the output required by Task 2, showing matches before and after RANSAC. Figure 3.3 shows recognition at a much smaller scale and the rejection of a different book.")
figure("cs1_matches_row.png", "Matched features between the reference and query images")
figure("cs1_cases_row.png", "A smaller cover is recognised, and a different book is rejected", width=5.4)
sub("3.3 Overall Performance")
para(f"A test set of 140 queries, 20 per book, used random yaw (±50°), pitch (±35°), rotation (±180°), scale (0.35 to 1.4), blur, brightness and noise. Each query was compared with all seven references, giving 980 pairs (Table 3.1). This gives {pct(th15['precision'], 0)} precision, {pct(th15['recall'])} recall and {pct(acc)} accuracy. Assigning each query to the reference with the most inliers gave {pct(ms['top1'], 0)} top-1 identification. The estimated cover corners were a median of {ms['corner_err_median']:.2f} pixels from their true positions.", keep=True)
table("Results at 15 inliers over 980 pairs", ["", "Predicted same book", "Predicted different book"], [["Actually same book (140)", th15["tp"], th15["fn"]], ["Actually different book (840)", th15["fp"], th15["tn"]]], [2.5, 1.6, 1.6])
para(f"Figure 3.4 explains these results. Same-book pairs had a median of {ms['genuine_median']:.0f} inliers, while different-book pairs had a median of {ms['impostor_median']:.0f} and never exceeded {ms['impostor_max']}, so the threshold of 15 lies in the gap between them.")
figure("cs1_distribution.png", "Inlier counts for same-book and different-book pairs", width=4.0)
sub("3.4 Effect of Individual Conditions")
para("Each condition was then varied on its own for all seven books (Figure 3.5, Table 3.2). Rotation has almost no effect, confirming the rotation invariance of SIFT. Enlarging the cover causes no loss, but shrinking it steadily removes detail, and tilt causes a sharp collapse after about 55°. A value of zero inliers is plotted at the bottom of the axis.")
figure("cs1_sweeps.png", "Mean RANSAC inliers as each condition is varied", width=4.9)
table("Operating range of the recogniser (out of seven books)", ["Condition", "All 7 accepted", "Partial", "None accepted"],
      [["Viewing angle (yaw)", "Up to 55°", "1 of 7 at 65°", "70° and above"], ["Scale", "0.4 to 3.0", "2 to 6 of 7 at 0.15 to 0.3", "Not reached"], ["In-plane rotation", "0° to 180°", "None", "Not reached"],
       ["Gaussian blur (σ)", "Up to 3", "6 of 7 at 4, 1 of 7 at 6", "8"], ["Brightness gain", "Down to 0.3", "5 of 7 at 0.15", "0.08 and below"], ["Cover hidden", "Up to 60%", "6 of 7 at 80%, 2 of 7 at 90%", "Not reached"]], [1.6, 1.2, 2.0, 1.0])

chap("Critical Evaluation")
sub("4.1 Strengths and Comparison with ORB")
para(f"The method needs no training, and adding a book requires only one reference image. It also tolerates heavy occlusion, gives the cover's location, and has an explainable basis for each decision. ORB [3], run on the same test, extracted features about {ms['ms_extract'] / mo['ms_extract']:.0f} times faster ({mo['ms_extract']:.1f} ms against {ms['ms_extract']:.1f} ms). However, its recall was lower ({pct(orb15['recall'])} against {pct(th15['recall'])}) and its median corner error was higher ({mo['corner_err_median']:.2f} against {ms['corner_err_median']:.2f} pixels).")
sub("4.2 Limitations")
figure("cs1_failures_row.png", "Failure cases: strong tilt, heavy blur and very dark lighting", width=4.2)
bullets([
    "**Viewpoint.** SIFT is only partly robust to affine distortion, so recognition collapsed past about 55° to 65° of yaw.",
    f"**Low-texture covers.** Clean Code, a dark and simple design, had only {R1['ref_keypoints']['Clean Code']} reference keypoints and the lowest median inlier count ({ms['per_book_median']['Clean Code']:.0f}). Plain covers would be harder still.",
    "**Blur and darkness.** Both remove the detail SIFT relies on. At a brightness gain of 0.15, one cover was accepted but placed in the wrong position, so acceptance does not guarantee a correct location.",
    "**Non-planar and similar covers.** The homography assumes a flat cover, and different editions that share artwork may not be separated.",
    f"**Scalability.** Brute-force matching took about {ms['ms_match_db']:.0f} ms for seven books. A full library would need approximate search, and the threshold would need re-checking.",
    "**Limits of this evaluation.** The queries are simulated and the database has only seven books. Real photographs add curved pages, uneven shadows and cluttered shelves, so these figures are an upper estimate until confirmed on real images.",
])
para("Affine-SIFT could extend the viewpoint range [5], while multiple reference views and contrast normalisation would reduce other failures.")

chap("Conclusion")
para(f"The program meets all three tasks. It correctly decided {th15['tp'] + th15['tn']} of 980 test pairs with no false acceptances, and displayed the matched features clearly. It is well suited to a small library when covers are photographed within about 55° of frontal view, in reasonable light and with enough texture. The next step is to test it on real photographs of a larger collection.")
refs = doc.add_heading("References", level=1); refs.paragraph_format.space_before = Pt(12)
for i, r in enumerate([
    "D. G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints,” International Journal of Computer Vision, vol. 60, no. 2, pp. 91-110, 2004.",
    "M. A. Fischler and R. C. Bolles, “Random Sample Consensus,” Communications of the ACM, vol. 24, no. 6, pp. 381-395, 1981.",
    "E. Rublee, V. Rabaud, K. Konolige and G. Bradski, “ORB: An Efficient Alternative to SIFT or SURF,” in Proc. IEEE ICCV, pp. 2564-2571, 2011.",
    "G. Bradski, “The OpenCV Library,” Dr. Dobb's Journal of Software Tools, 2000.",
    "J.-M. Morel and G. Yu, “ASIFT: A New Framework for Fully Affine Invariant Image Comparison,” SIAM Journal on Imaging Sciences, vol. 2, no. 2, pp. 438-469, 2009.",
    "Internet Archive, “Open Library Covers API.” [Online]. Available: https://openlibrary.org/dev/docs/api/covers"], 1):
    p = doc.add_paragraph(); p.paragraph_format.first_line_indent = Inches(-0.4); p.paragraph_format.left_indent = Inches(0.4); p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.tab_stops.add_tab_stop(Inches(0.4)); add_runs(p, f"[{i}]\t{r}")

from docx.text.paragraph import Paragraph
HEADINGS = [p for p in doc.paragraphs if p.style.name in ("Heading 1", "Heading 2") and p.text not in ("Table of Contents", "Abstract")]
anchor = TOC_ANCHOR
for hp in HEADINGS:
    level = 1 if hp.style.name == "Heading 1" else 2
    new = OxmlElement("w:p"); anchor._p.addnext(new); q = Paragraph(new, anchor._parent); f = q.paragraph_format
    f.alignment = WD_ALIGN_PARAGRAPH.LEFT; f.first_line_indent = Inches(0); f.left_indent = Inches(0 if level == 1 else .3); f.line_spacing = 1.0
    f.space_after = Pt(1); f.space_before = Pt(4 if level == 1 else 0)
    f.tab_stops.add_tab_stop(Inches(TEXT_W), WD_TAB_ALIGNMENT.RIGHT, WD_TAB_LEADER.DOTS)
    r = q.add_run(f"{hp.text}\t{TOC.get(hp.text, '')}"); r.bold = level == 1; anchor = q
json.dump([p.text for p in HEADINGS], open(os.path.join(ROOT, "outputs/headings_short.json"), "w"))
doc.core_properties.author = "Suraj Singh Bayas"; doc.core_properties.title = "Image Recognition Using Feature Matching"
doc.save(OUT); print("saved")
