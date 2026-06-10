"""Generate the Methods deck: Gradient Masking + Normality test.

Standalone PowerPoint covering two methodological pieces of
"Stabilizing Gradient Inversion in Federated Learning". Same clean DTU-red
theme as the ResNet results deck so the slides drop into the group deck.

Run:  python slides/make_methods_slides.py
Out:  slides/methods.pptx
"""
from pathlib import Path

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from PIL import Image

# ---------------------------------------------------------------- palette ----
DTU_RED = RGBColor(0x99, 0x00, 0x00)
INK = RGBColor(0x22, 0x22, 0x22)
SLATE = RGBColor(0x55, 0x55, 0x55)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
BLUE = RGBColor(0x2F, 0x6D, 0xB0)
FONT = "Calibri"

ASSETS = Path(__file__).resolve().parent / "assets"
OUT = Path(__file__).resolve().parent / "methods.pptx"

EMU_W = Inches(13.333)
EMU_H = Inches(7.5)


def _set_bg(slide, color):
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = color


def _box(slide, x, y, w, h, fill=None, line=None, line_w=None):
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    shp.shadow.inherit = False
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w or Pt(1)
    return shp


def _text(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
          space_after=6, line_spacing=1.0):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_after = Pt(space_after)
        p.space_before = Pt(0)
        p.line_spacing = line_spacing
        if isinstance(para, tuple):
            para = [para]
        for (txt, size, bold, color, *rest) in para:
            italic = rest[0] if rest else False
            r = p.add_run()
            r.text = txt
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.italic = italic
            r.font.name = FONT
            r.font.color.rgb = color
    return tb


def _bullets(slide, x, y, w, h, items, size=17, gap=10, color=INK):
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        lvl, txt, bold = item[0], item[1], item[2]
        c = item[3] if len(item) > 3 else color
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.space_before = Pt(0)
        p.line_spacing = 1.05
        p.level = lvl
        bullet = "—  " if lvl == 0 else "•  "
        r = p.add_run()
        r.text = bullet + txt
        r.font.size = Pt(size - lvl * 2)
        r.font.bold = bold
        r.font.name = FONT
        r.font.color.rgb = c
    return tb


def _img_fit(slide, path, x, y, max_w, max_h, align="center", valign="middle"):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(max_w / iw, max_h / ih)
    w = Emu(int(iw * scale))
    h = Emu(int(ih * scale))
    if align == "center":
        x = x + (max_w - w) // 2
    elif align == "right":
        x = x + (max_w - w)
    if valign == "middle":
        y = y + (max_h - h) // 2
    elif valign == "bottom":
        y = y + (max_h - h)
    slide.shapes.add_picture(str(path), x, y, w, h)
    return x, y, w, h


def _header(slide, kicker, title):
    _box(slide, 0, 0, EMU_W, Inches(0.12), fill=DTU_RED)
    _text(slide, Inches(0.6), Inches(0.32), Inches(12.1), Inches(0.35),
          [(kicker.upper(), 13, True, DTU_RED)])
    _text(slide, Inches(0.6), Inches(0.62), Inches(12.1), Inches(0.8),
          [(title, 30, True, INK)])
    _box(slide, Inches(0.62), Inches(1.42), Inches(1.1), Pt(3), fill=DTU_RED)


def _footer(slide, page):
    _text(slide, Inches(0.6), Inches(7.02), Inches(9.5), Inches(0.35),
          [("Stabilizing Gradient Inversion in Federated Learning · Methods",
            10, False, SLATE)])
    _text(slide, Inches(11.6), Inches(7.02), Inches(1.2), Inches(0.35),
          [(page, 10, False, SLATE)], align=PP_ALIGN.RIGHT)


def blank_slide(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s, WHITE)
    return s


# ============================================================== build deck ===
prs = Presentation()
prs.slide_width = EMU_W
prs.slide_height = EMU_H

# ---- Slide 1 : section divider ---------------------------------------------
s = blank_slide(prs)
_set_bg(s, DTU_RED)
_box(s, 0, Inches(2.55), EMU_W, Inches(2.4), fill=RGBColor(0x7A, 0x00, 0x00))
_text(s, Inches(0.9), Inches(2.0), Inches(11.5), Inches(0.5),
      [("METHODS", 16, True, RGBColor(0xF2, 0xC0, 0xC0))])
_text(s, Inches(0.9), Inches(2.5), Inches(11.5), Inches(1.3),
      [("Gradient masking & statistical evaluation", 44, True, WHITE)])
_text(s, Inches(0.9), Inches(4.0), Inches(11.5), Inches(1.0),
      [("How we remove gradient components — and how we compare "
        "reconstructions rigorously.", 22, False, RGBColor(0xF6, 0xDD, 0xDD))],
      line_spacing=1.1)
_text(s, Inches(0.9), Inches(6.5), Inches(11.5), Inches(0.5),
      [("Within-image paired design · leave-one-layer-out ablation · "
        "Shapiro–Wilk normality testing", 13, False, RGBColor(0xE8, 0xC8, 0xC8))])

# ---- Slide 2 : masking — concept & tradeoff --------------------------------
s = blank_slide(prs)
_header(s, "Methods — gradient masking", "What masking is, and why it might help")
_bullets(s, Inches(0.6), Inches(1.75), Inches(7.0), Inches(5.0), [
    (0, "Inversion matches the observed gradient by optimization — but not "
        "all gradient entries help equally.", True),
    (1, "Some carry useful input information; others add poorly conditioned "
        "optimization directions.", False),
    (0, "Masking = selectively removing subsets of the gradient before "
        "reconstruction.", True),
    (1, "Fewer gradients to match + potentially better numerical "
        "conditioning.", False),
    (0, "The central tradeoff.", True, DTU_RED),
    (1, "Removing entries can ease optimization, but also removes "
        "information — masking must drop redundant / poorly conditioned "
        "entries while preserving independent constraints.", False),
    (0, "Diagnostic use: if a smaller subset reconstructs better than the "
        "full gradient, the information was present but hard to extract.", True),
], size=16, gap=10)
# necessary-condition note card
_box(s, Inches(7.85), Inches(1.85), Inches(4.95), Inches(2.55), fill=LIGHT)
_box(s, Inches(7.85), Inches(1.85), Inches(0.1), Inches(2.55), fill=DTU_RED)
_text(s, Inches(8.1), Inches(2.0), Inches(4.55), Inches(2.3),
      [[("Necessary condition\n", 16, True, DTU_RED)],
       [("Reconstruction needs the gradient Jacobian J(x)=∂g/∂x to have "
         "full column rank (rank = d = 3,072 for 3×32×32).", 13.5, False, INK)],
       [("n ≥ d is necessary but not sufficient — entries can be linearly "
         "dependent. Rank is verified after masking (Sec. 5.1).", 13.5, False, INK)]],
      line_spacing=1.05, space_after=6)
_text(s, Inches(7.85), Inches(4.6), Inches(4.95), Inches(2.0),
      [("So observed PSNR differences reflect optimization conditioning, "
        "not lost information.", 13.5, False, SLATE, True)], line_spacing=1.05)
_footer(s, "2")

# ---- Slide 3 : masking — how we mask ---------------------------------------
s = blank_slide(prs)
_header(s, "Methods — gradient masking", "How we mask: paired design & two modes")
_img_fit(s, ASSETS / "masking_diagram.png",
         Inches(0.5), Inches(1.6), Inches(7.4), Inches(4.4))
_text(s, Inches(0.5), Inches(5.95), Inches(7.4), Inches(0.5),
      [("Leave-one-layer-out ablation: baseline retains all groups; each "
        "masked run excludes exactly one.", 11, False, SLATE)],
      align=PP_ALIGN.CENTER, line_spacing=1.0)
_bullets(s, Inches(8.05), Inches(1.7), Inches(4.85), Inches(5.0), [
    (0, "Paired, within-image design.", True),
    (1, "Same image, same network init: baseline vs. masked differ only "
        "by masking. Label fixed via iDLG, batch size 1.", False),
    (0, "Layer-wise ablation.", True, BLUE),
    (1, "prefix_topfrac_entries_layer, fraction 1.0 → binary leave-one-"
        "group-out. 30 CIFAR-100 images × 4 settings (2 loss/optimizer × "
        "2 weight regimes).", False),
    (0, "Masking-percentage sweep.", True, BLUE),
    (1, "gradsize_topfrac_entries_layer: keep the top fraction f of entries "
        "by |gradient| within each tensor; vary f. 100 CIFAR-100 images.", False),
], size=15.5, gap=10)
_footer(s, "3")

# ---- Slide 4 : normality — why & how ---------------------------------------
s = blank_slide(prs)
_header(s, "Methods — normality test", "Can we use parametric statistics?")
_bullets(s, Inches(0.6), Inches(1.7), Inches(6.0), Inches(5.0), [
    (0, "Paired per-image difference is the unit of analysis.", True),
    (1, "dᵢ = PSNR_masked − PSNR_baseline on the same image i.", False),
    (0, "A paired t-test would assume the dᵢ are normal — but "
        "reconstruction difficulty varies wildly across images.", True),
    (1, "Normality is assessed, not presumed.", False),
    (0, "Shapiro–Wilk test, α = 0.05.", True, DTU_RED),
    (1, "Applied to PSNR, MSE and SSIM differences per configuration; "
        "W ∈ (0,1], values near 1 = consistent with normality.", False),
    (0, "Metric choice matters (see right).", True),
    (1, "MSE ≥ 0 is right-skewed; PSNR’s log10 compresses the tail, so it "
        "is more likely to pass the test.", False),
], size=16, gap=9)
_img_fit(s, ASSETS / "normality_per_metric_slide.png",
         Inches(6.75), Inches(2.5), Inches(6.2), Inches(2.9))
_text(s, Inches(6.75), Inches(5.45), Inches(6.2), Inches(0.5),
      [("Real data: ResNet-18, mask = layer4, n = 30. Per-image differences "
        "(masked − baseline); reproduces Appendix C.", 10.5, False, SLATE, True)],
      align=PP_ALIGN.CENTER)
_footer(s, "4")

# ---- Slide 5 : normality — outcome -----------------------------------------
s = blank_slide(prs)
_header(s, "Methods — normality test", "Outcome and consequence")
_bullets(s, Inches(0.6), Inches(1.8), Inches(12.1), Inches(3.2), [
    (0, "Normality is rejected for almost all configurations.", True, DTU_RED),
    (1, "Per-condition Shapiro–Wilk p-values are tabulated in Appendix C; "
        "MSE differences fail most often (frequently p < 0.001), PSNR passes "
        "more often as expected from the log transform.", False),
    (0, "Small samples contribute.", True),
    (1, "With only 30 paired differences per condition, a few outlying images "
        "pull the distribution away from normality and the test has limited "
        "power against mild deviations.", False),
    (0, "Independence holds by construction.", True),
    (1, "Each dᵢ comes from a distinct image with an independently "
        "initialized network.", False),
], size=16, gap=9)
# consequence band
_box(s, Inches(0.6), Inches(5.35), Inches(12.13), Inches(1.35), fill=LIGHT)
_box(s, Inches(0.6), Inches(5.35), Inches(0.12), Inches(1.35), fill=DTU_RED)
_text(s, Inches(0.95), Inches(5.5), Inches(11.6), Inches(1.05),
      [[("Consequence:  ", 18, True, DTU_RED),
        ("the paired t-test assumption does not hold, so results are reported "
         "with ", 18, False, INK),
        ("descriptive statistics (mean ± SD) and box / violin plots", 18, True, INK),
        (" rather than parametric p-values.", 18, False, INK)]],
      anchor=MSO_ANCHOR.MIDDLE, line_spacing=1.1)
_footer(s, "5")

prs.save(str(OUT))
print(f"Wrote {OUT}  ({len(prs.slides._sldIdLst)} slides)")
