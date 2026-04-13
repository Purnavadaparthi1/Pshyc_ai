"""
PSYCH.AI — Knowledge Base PDF Generator
-----------------------------------------
Generates comprehensive psychology PDFs from the bundled content module.
These PDFs are automatically ingested into ChromaDB on first startup,
so ALL users get the same rich knowledge base without needing any files.

Run manually:  python scripts/generate_docs.py
Auto-run by:   main.py on first startup
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable,
    PageBreak, Table, TableStyle, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUTPUT_DIR = Path(__file__).parent.parent / "data" / "documents"

PURPLE = colors.HexColor("#6d28d9")
DARK   = colors.HexColor("#1e1b4b")
LIGHT  = colors.HexColor("#f5f3ff")
GREY   = colors.HexColor("#64748b")


def build_styles():
    base = getSampleStyleSheet()

    styles = {
        "cover_title": ParagraphStyle(
            "cover_title", fontSize=28, textColor=PURPLE,
            spaceAfter=8, alignment=TA_CENTER,
            fontName="Helvetica-Bold",
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub", fontSize=14, textColor=DARK,
            spaceAfter=6, alignment=TA_CENTER,
            fontName="Helvetica",
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta", fontSize=10, textColor=GREY,
            spaceAfter=4, alignment=TA_CENTER,
            fontName="Helvetica-Oblique",
        ),
        "h1": ParagraphStyle(
            "h1", fontSize=18, textColor=PURPLE,
            spaceBefore=20, spaceAfter=8,
            fontName="Helvetica-Bold", borderPad=4,
        ),
        "h2": ParagraphStyle(
            "h2", fontSize=14, textColor=DARK,
            spaceBefore=14, spaceAfter=6,
            fontName="Helvetica-Bold",
        ),
        "h3": ParagraphStyle(
            "h3", fontSize=12, textColor=PURPLE,
            spaceBefore=10, spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "body", fontSize=10, textColor=colors.black,
            spaceBefore=3, spaceAfter=3,
            fontName="Helvetica", leading=16,
            alignment=TA_JUSTIFY,
        ),
        "bullet": ParagraphStyle(
            "bullet", fontSize=10, textColor=colors.black,
            spaceBefore=2, spaceAfter=2,
            fontName="Helvetica", leading=14,
            leftIndent=18, bulletIndent=6,
        ),
        "case_study": ParagraphStyle(
            "case_study", fontSize=10, textColor=DARK,
            spaceBefore=4, spaceAfter=4,
            fontName="Helvetica-Oblique", leading=15,
            leftIndent=12, rightIndent=12,
            borderPad=8,
        ),
        "definition": ParagraphStyle(
            "definition", fontSize=10, textColor=colors.black,
            spaceBefore=3, spaceAfter=3,
            fontName="Helvetica-Bold", leading=14,
        ),
        "link": ParagraphStyle(
            "link", fontSize=9, textColor=colors.HexColor("#4f46e5"),
            spaceBefore=2, spaceAfter=2,
            fontName="Helvetica",
        ),
    }
    return styles


def parse_and_render(text: str, styles: dict) -> list:
    """Convert the plain-text content into ReportLab flowables."""
    flowables = []
    lines = text.strip().splitlines()
    i = 0
    in_case = False

    while i < len(lines):
        line = lines[i]

        # Section dividers → H1
        if line.startswith("===="):
            i += 1
            continue

        if line.startswith("----"):
            flowables.append(HRFlowable(width="100%", thickness=0.5, color=PURPLE, spaceAfter=4))
            i += 1
            continue

        # Heading detection
        stripped = line.strip()

        if stripped.startswith("UNIT ") and len(stripped) < 80 and stripped.isupper():
            flowables.append(Paragraph(stripped, styles["h1"]))
            i += 1
            continue

        if stripped.endswith(":") and stripped.isupper() and len(stripped) < 80:
            flowables.append(Paragraph(stripped[:-1], styles["h2"]))
            i += 1
            continue

        # Sub-headings: lines that are all-caps and short
        if stripped and stripped == stripped.upper() and len(stripped) < 70 and stripped.replace(" ", "").replace("(", "").replace(")", "").replace("/", "").replace("-", "").isalpha():
            flowables.append(Paragraph(stripped, styles["h3"]))
            i += 1
            continue

        # Case study blocks
        if stripped.upper().startswith("CASE STUDY:"):
            flowables.append(Spacer(1, 6))
            flowables.append(Paragraph("📋 " + stripped, styles["h3"]))
            i += 1
            # Collect case study text
            case_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("CASE STUDY:") and not lines[i].strip().startswith("UNIT ") and not lines[i].strip().startswith("====="):
                case_lines.append(lines[i].strip())
                i += 1
            if case_lines:
                case_text = " ".join(case_lines)
                flowables.append(Paragraph(case_text, styles["case_study"]))
            flowables.append(Spacer(1, 6))
            continue

        # Bullet points
        if stripped.startswith("•") or stripped.startswith("-"):
            text_content = stripped[1:].strip()
            if text_content:
                flowables.append(Paragraph("• " + text_content, styles["bullet"]))
            i += 1
            continue

        # Numbered list
        import re
        if re.match(r"^\d+\.", stripped):
            flowables.append(Paragraph(stripped, styles["bullet"]))
            i += 1
            continue

        # URL lines (resource links)
        if stripped.startswith("http") or "https://" in stripped:
            flowables.append(Paragraph(stripped, styles["link"]))
            i += 1
            continue

        # Empty line
        if not stripped:
            flowables.append(Spacer(1, 4))
            i += 1
            continue

        # Normal body text
        if stripped:
            # Check if it's a bold definition (contains ":" and short intro)
            if ":" in stripped and len(stripped.split(":")[0]) < 40:
                parts = stripped.split(":", 1)
                rendered = f"<b>{parts[0]}:</b>{parts[1]}" if len(parts) == 2 else stripped
                flowables.append(Paragraph(rendered, styles["body"]))
            else:
                flowables.append(Paragraph(stripped, styles["body"]))

        i += 1

    return flowables


def make_cover_page(doc_title: str, subtitle: str, styles: dict) -> list:
    flowables = [
        Spacer(1, 3 * cm),
        Paragraph("🧠 PSYCH.AI", styles["cover_title"]),
        Spacer(1, 0.3 * cm),
        Paragraph(doc_title, ParagraphStyle(
            "big", fontSize=22, textColor=DARK,
            alignment=TA_CENTER, fontName="Helvetica-Bold"
        )),
        Spacer(1, 0.4 * cm),
        Paragraph(subtitle, styles["cover_sub"]),
        Spacer(1, 0.6 * cm),
        HRFlowable(width="60%", thickness=2, color=PURPLE),
        Spacer(1, 0.4 * cm),
        Paragraph("IGNOU MA Psychology — Comprehensive Study Material", styles["cover_meta"]),
        Paragraph("For use with RAG Knowledge Base", styles["cover_meta"]),
        Paragraph("PSYCH.AI © 2025 | psych.ai", styles["cover_meta"]),
        PageBreak(),
    ]
    return flowables


def generate_pdf(filename: str, doc_title: str, subtitle: str, content: str):
    output_path = OUTPUT_DIR / filename
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
        title=doc_title,
        author="PSYCH.AI",
    )

    styles = build_styles()
    story = []

    # Cover page
    story.extend(make_cover_page(doc_title, subtitle, styles))

    # Content pages
    story.extend(parse_and_render(content, styles))

    doc.build(story)
    size_kb = output_path.stat().st_size // 1024
    print(f"  ✓ Generated: {filename} ({size_kb} KB)")
    return str(output_path)


def generate_all_pdfs() -> list[str]:
    """Generate all psychology PDFs. Returns list of generated file paths."""
    # Import content
    from data.seed.psychology_content import (
        PERSONALITY_THEORIES,
        COGNITIVE_PSYCHOLOGY,
        ABNORMAL_PSYCHOLOGY,
        SOCIAL_PSYCHOLOGY,
        DEVELOPMENTAL_PSYCHOLOGY,
        THERAPEUTIC_APPROACHES,
        RESEARCH_METHODS,
        ORGANIZATIONAL_PSYCHOLOGY,
        NEUROPSYCHOLOGY,
        COUNSELLING_PSYCHOLOGY,
        RESOURCE_LINKS,
    )

    docs = [
        ("01_Personality_Theories.pdf",
         "Personality Theories",
         "MPC-003 | Freud · Jung · Adler · Rogers · Maslow · Big Five",
         PERSONALITY_THEORIES),

        ("02_Cognitive_Psychology.pdf",
         "Cognitive Psychology",
         "MPC-001 | Perception · Attention · Memory · Thinking · Language · Intelligence",
         COGNITIVE_PSYCHOLOGY),

        ("03_Abnormal_Psychology.pdf",
         "Abnormal Psychology & Therapies",
         "MPC-007 | Anxiety · Mood · Schizophrenia · Personality · Eating Disorders",
         ABNORMAL_PSYCHOLOGY),

        ("04_Social_Psychology.pdf",
         "Advanced Social Psychology",
         "MPC-004 | Social Cognition · Attitudes · Conformity · Prejudice · Aggression",
         SOCIAL_PSYCHOLOGY),

        ("05_Developmental_Psychology.pdf",
         "Lifespan Developmental Psychology",
         "MPC-002 | Piaget · Vygotsky · Erikson · Kohlberg · Attachment Theory",
         DEVELOPMENTAL_PSYCHOLOGY),

        ("06_Therapeutic_Approaches.pdf",
         "Therapeutic Approaches & Counselling",
         "MPC-007/008 | CBT · DBT · ACT · Psychoanalysis · Humanistic",
         THERAPEUTIC_APPROACHES),

        ("07_Research_Methods_Statistics.pdf",
         "Research Methods & Statistics",
         "MPC-005/006 | Research Design · Sampling · Statistics · Ethics",
         RESEARCH_METHODS),

        ("08_Organisational_Psychology.pdf",
         "Organisational & Industrial Psychology",
         "MPCE | Motivation · Leadership · Culture · Stress · Selection",
         ORGANIZATIONAL_PSYCHOLOGY),

        ("09_Neuropsychology.pdf",
         "Neuropsychology & Biopsychology",
         "MPC-001/007 | Brain Structure · Neurotransmitters · Sleep · Assessment",
         NEUROPSYCHOLOGY),

        ("10_Counselling_Psychology.pdf",
         "Counselling Psychology",
         "MPC-008 | Skills · Approaches · Ethics · Diversity · Special Populations",
         COUNSELLING_PSYCHOLOGY),

        ("11_Resources_and_Links.pdf",
         "Online Resources & References",
         "Comprehensive Links | Free PDFs · Research Databases · Tools",
         RESOURCE_LINKS),
    ]

    generated = []
    print(f"\nGenerating {len(docs)} psychology knowledge base PDFs...")
    print(f"Output directory: {OUTPUT_DIR}\n")

    for filename, title, subtitle, content in docs:
        try:
            path = generate_pdf(filename, title, subtitle, content)
            generated.append(path)
        except Exception as e:
            print(f"  ✗ Failed: {filename} — {e}")

    print(f"\n✓ Generated {len(generated)}/{len(docs)} PDFs successfully.")
    return generated


def pdfs_already_exist() -> bool:
    """Check if knowledge base PDFs have already been generated."""
    marker = OUTPUT_DIR / "01_Personality_Theories.pdf"
    return marker.exists()


if __name__ == "__main__":
    generate_all_pdfs()
