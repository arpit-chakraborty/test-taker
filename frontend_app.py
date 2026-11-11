#!/usr/bin/env python3
"""
Pure-Python Frontend Web Application for the PDF Question & Answer Parser.

Features:
- Accepts a user-uploaded PDF via a browser form (`POST /upload`).
- Ensures `temp/image` directory exists (`os.makedirs("temp/image", exist_ok=True)`).
- Runs `pdf_question_parser.parse_pdf_questions` on the uploaded PDF, saving all
  extracted question and formula images into `temp/image` and generating the JSON.
- Renders each question with its images in exact reading order (0: above question,
  1: between/inline with question and answer, 2: in/after the answer options).
- Provides interactive inputs (radio buttons for MCQ, checkboxes for multichoice,
  and a text/number input for numeric questions) with a "Submit" button next to
  each question that evaluates the user's response in Python and reveals the
  correct answer.

Usage:
    python3 frontend_app.py [--host 127.0.0.1] [--port 8000] [--pdf path/to/file.pdf]
"""

import argparse
import html
import json
import mimetypes
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
import uvicorn

# Ensure the directory containing pdf_question_parser.py is importable
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from parse_pdf import parse_pdf_questions


BASE_STORAGE_DIR = Path("/tmp") if os.environ.get("VERCEL") else Path(".")
TEMP_DIR = BASE_STORAGE_DIR / "temp"
IMAGE_DIR = TEMP_DIR / "image"
JSON_OUTPUT_PATH = TEMP_DIR / "parsed_questions.json"
TEMP_JSON_COPY_PATH = TEMP_DIR / "parsed_questions_copy.json"
UPLOADED_PDF_PATH = TEMP_DIR / "uploaded_question_paper.pdf"


class ExamSessionState:
    """Holds server-side application state in Python for the loaded exam paper."""

    def __init__(self) -> None:
        self.pdf_filename: Optional[str] = None
        self.questions: List[Dict[str, Any]] = []
        self.revealed_indices: Set[int] = set()
        self.user_responses: Dict[int, Union[str, List[str]]] = {}
        self.error_message: Optional[str] = None

    def ensure_directories(self) -> None:
        """Create `temp` and `temp/image` directories if they do not already exist."""
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    def load_pdf(self, pdf_path: Union[str, Path], display_name: str) -> List[Dict[str, Any]]:
        """Run the PDF through `parse_pdf_questions`, saving images in `temp/image`."""
        self.ensure_directories()

        # Clean previous extracted images in temp/image
        for existing in IMAGE_DIR.iterdir():
            if existing.is_file():
                existing.unlink()

        parsed = parse_pdf_questions(
            pdf_path=str(pdf_path),
            output_json_path=str(JSON_OUTPUT_PATH),
            image_output_dir=str(IMAGE_DIR),
        )
        # Also save a copy of the JSON inside temp/parsed_questions.json
        with open(TEMP_JSON_COPY_PATH, "w", encoding="utf-8") as f:
            json.dump(parsed, f, indent=2, ensure_ascii=False)

        self.pdf_filename = display_name
        self.questions = parsed
        self.revealed_indices.clear()
        self.user_responses.clear()
        self.error_message = None
        return parsed


STATE = ExamSessionState()
app = FastAPI(title="PDF Question Paper Parser & Interactive Viewer")


def image_url_from_path(raw_path: str) -> str:
    """Convert a local image path (e.g. 'temp/image/q5_img1.jpg') to its served URL."""
    filename = Path(raw_path.strip()).name
    return f"/temp/image/{html.escape(filename)}"


def render_text_with_inline_images(text: str) -> str:
    """
    Escape HTML in text while converting `[IMAGE: <path>]` markers into `<img>` elements.
    Differentiates full-width block images (`.jpg` or large snippets) from inline math PNGs.
    """
    if not text:
        return ""

    pattern = re.compile(r"\[IMAGE:\s*([^\]]+)\]")
    parts: List[str] = []
    last_idx = 0

    for match in pattern.finditer(text):
        before = text[last_idx : match.start()]
        if before:
            parts.append(html.escape(before))
        img_path = match.group(1).strip()
        url = image_url_from_path(img_path)
        is_block = img_path.lower().endswith((".jpg", ".jpeg")) or (
            match.start() == 0 and match.end() == len(text.strip())
        )
        css_class = "block-img" if is_block else "inline-img"
        parts.append(
            f'<img src="{url}" alt="{html.escape(Path(img_path).name)}" class="{css_class}" />'
        )
        last_idx = match.end()

    tail = text[last_idx:]
    if tail:
        parts.append(html.escape(tail))

    return "".join(parts)


def check_user_answer(
    q_item: Dict[str, Any], user_val: Optional[Union[str, List[str]]]
) -> Optional[bool]:
    """
    Evaluate whether the user's submitted response matches the question's `Correct answer`.
    Returns True (correct), False (incorrect), or None (no answer selected/entered).
    """
    if user_val is None or user_val == "" or user_val == []:
        return None

    ans_type = q_item.get("Answer type", "MCQ")
    correct = q_item.get("Correct answer")

    if ans_type == "numeric":
        raw_str = str(user_val).strip()
        if not raw_str:
            return None
        corr_str = str(correct or "").strip()
        # Support range answers such as "19.1 to 19.3"
        range_match = re.match(
            r"^\s*([+-]?\d+(?:\.\d+)?)\s*to\s*([+-]?\d+(?:\.\d+)?)\s*$",
            corr_str,
            re.IGNORECASE,
        )
        try:
            val_num = float(raw_str)
            if range_match:
                low = float(range_match.group(1))
                high = float(range_match.group(2))
                return low <= val_num <= high
            return abs(val_num - float(corr_str)) < 1e-6
        except ValueError:
            return raw_str.lower() == corr_str.lower()

    if ans_type == "MCQ":
        selected = user_val[0] if isinstance(user_val, list) and user_val else str(user_val)
        if isinstance(correct, list):
            return selected in correct
        return selected == str(correct)

    # multichoice
    selected_list = [user_val] if isinstance(user_val, str) else list(user_val)
    correct_list = [correct] if isinstance(correct, str) else list(correct or [])
    return set(selected_list) == set(correct_list)


def render_question_card_html(idx: int, q_item: Dict[str, Any]) -> str:
    """Render a single question card with its inputs, Submit button, and revealed answer."""
    q_num = idx + 1
    q_text = q_item.get("Question") or ""
    ans_type = q_item.get("Answer type", "MCQ")
    options = q_item.get("Answer")
    correct_ans = q_item.get("Correct answer")
    images = q_item.get("Image")
    image_locs = q_item.get("Image location")

    is_revealed = idx in STATE.revealed_indices
    user_val = STATE.user_responses.get(idx)
    is_correct = check_user_answer(q_item, user_val) if is_revealed else None

    # Normalize image lists in case external JSON without inline [IMAGE: ...] tags is used
    img_list = [images] if isinstance(images, str) else (images or [])
    loc_list = [image_locs] if isinstance(image_locs, int) else (image_locs or [])

    above_imgs_html = ""
    between_imgs_html = ""
    after_imgs_html = ""

    for img_path, loc in zip(img_list, loc_list):
        marker = f"[IMAGE: {img_path}]"
        in_q = marker in q_text
        in_opts = any(marker in (opt or "") for opt in (options or []))
        if not in_q and not in_opts:
            tag = f'<div class="standalone-img-wrap"><img src="{image_url_from_path(img_path)}" class="block-img" alt="Question image" /></div>'
            if loc == 0:
                above_imgs_html += tag
            elif loc == 1:
                between_imgs_html += tag
            else:
                after_imgs_html += tag

    rendered_q_html = render_text_with_inline_images(q_text)

    # Build answer inputs based on Answer type
    inputs_html_parts: List[str] = []
    if ans_type == "numeric" or options is None:
        prev_val = html.escape(str(user_val)) if isinstance(user_val, str) else ""
        inputs_html_parts.append(
            f'<div class="numeric-input-row">'
            f'<label for="ans-{idx}" class="numeric-label">Your Answer (Numeric):</label>'
            f'<input type="text" id="ans-{idx}" name="answer" value="{prev_val}" '
            f'placeholder="Enter numeric value..." class="numeric-input" />'
            f"</div>"
        )
    else:
        input_type = "radio" if ans_type == "MCQ" else "checkbox"
        selected_set: Set[str] = set()
        if isinstance(user_val, str):
            selected_set.add(user_val)
        elif isinstance(user_val, list):
            selected_set.update(user_val)

        correct_set: Set[str] = set()
        if isinstance(correct_ans, str):
            correct_set.add(correct_ans)
        elif isinstance(correct_ans, list):
            correct_set.update(correct_ans)

        inputs_html_parts.append('<div class="options-list">')
        for opt_i, opt_text in enumerate(options):
            opt_id = f"q-{idx}-opt-{opt_i}"
            checked_attr = "checked" if opt_text in selected_set else ""
            opt_escaped_val = html.escape(opt_text, quote=True)
            opt_rendered = render_text_with_inline_images(opt_text)

            opt_row_class = "option-item"
            badge_html = ""
            if is_revealed:
                if opt_text in correct_set:
                    opt_row_class += " option-correct"
                    badge_html = '<span class="opt-badge correct-badge">&#10003; Correct</span>'
                elif opt_text in selected_set:
                    opt_row_class += " option-wrong"
                    badge_html = '<span class="opt-badge wrong-badge">&#10007; Your Selection</span>'

            inputs_html_parts.append(
                f'<label class="{opt_row_class}" for="{opt_id}">'
                f'<input type="{input_type}" id="{opt_id}" name="answer" value="{opt_escaped_val}" {checked_attr} />'
                f'<span class="option-text">{opt_rendered}</span>'
                f"{badge_html}"
                f"</label>"
            )
        inputs_html_parts.append("</div>")

    inputs_html = "".join(inputs_html_parts)

    # Build revealed answer panel if Submit was clicked for this question
    reveal_panel_html = ""
    if is_revealed:
        if isinstance(correct_ans, list):
            corr_items = "".join(
                f"<li>{render_text_with_inline_images(c)}</li>" for c in correct_ans
            )
            corr_display = f"<ul class='correct-list'>{corr_items}</ul>"
        else:
            corr_display = (
                f"<span class='correct-single'>"
                f"{render_text_with_inline_images(str(correct_ans))}"
                f"</span>"
            )

        if is_correct is True:
            status_banner = '<div class="eval-status status-pass">&#10003; Correct! Your response matches the answer key.</div>'
        elif is_correct is False:
            status_banner = '<div class="eval-status status-fail">&#10007; Incorrect. Check the correct answer below.</div>'
        else:
            status_banner = '<div class="eval-status status-neutral">Answer revealed:</div>'

        reveal_panel_html = (
            f'<div class="answer-reveal-box">'
            f"{status_banner}"
            f'<div class="correct-answer-body"><strong>Correct Answer:</strong> {corr_display}</div>'
            f"</div>"
        )

    type_badge_label = {
        "MCQ": "MCQ (Single Choice)",
        "multichoice": "Multichoice (Select All That Apply)",
        "numeric": "Numeric / Short Answer",
    }.get(ans_type, ans_type)

    return f"""
    <section class="question-card" id="q-{idx}">
      <form method="post" action="/submit/{idx}#q-{idx}" class="question-form">
        <div class="question-header">
          <div class="q-title-group">
            <span class="q-number">Question {q_num}</span>
            <span class="q-type-badge type-{html.escape(ans_type.lower())}">{html.escape(type_badge_label)}</span>
          </div>
          <div class="q-actions">
            <button type="submit" class="btn-submit">
              {"Re-Submit / Check" if is_revealed else "Submit &amp; Reveal Answer"}
            </button>
          </div>
        </div>
        {above_imgs_html}
        <div class="question-prompt">{rendered_q_html}</div>
        {between_imgs_html}
        {inputs_html}
        {after_imgs_html}
        {reveal_panel_html}
      </form>
    </section>
    """


def render_full_page() -> str:
    """Render the complete HTML page from Python state."""
    STATE.ensure_directories()

    error_banner = (
        f'<div class="alert-error">{html.escape(STATE.error_message)}</div>'
        if STATE.error_message
        else ""
    )

    if STATE.questions:
        total_q = len(STATE.questions)
        revealed_count = len(STATE.revealed_indices)
        extracted_imgs = sorted(p.name for p in IMAGE_DIR.glob("*") if p.is_file())
        summary_bar = f"""
        <div class="status-bar">
          <div>
            <strong>Loaded PDF:</strong> {html.escape(STATE.pdf_filename or "Uploaded PDF")}
            &bull; <strong>Questions Parsed:</strong> {total_q}
            &bull; <strong>Images Saved in <code>temp/image/</code>:</strong> {len(extracted_imgs)}
            &bull; <strong>Answers Revealed:</strong> {revealed_count} / {total_q}
          </div>
          <div class="status-links">
            <a href="/api/json" target="_blank" class="btn-secondary">View Parsed JSON</a>
            <form method="post" action="/reveal-all" style="display:inline;">
              <button type="submit" class="btn-secondary">Reveal All Answers</button>
            </form>
            <form method="post" action="/hide-all" style="display:inline;">
              <button type="submit" class="btn-secondary">Hide All Answers</button>
            </form>
          </div>
        </div>
        """
        cards_html = "\n".join(
            render_question_card_html(i, q) for i, q in enumerate(STATE.questions)
        )
    else:
        summary_bar = ""
        cards_html = """
        <div class="empty-state">
          <h2>Upload a Question Paper PDF to Begin</h2>
          <p>Select a PDF file above and click <strong>Upload &amp; Parse PDF</strong>. The parser will extract all questions, options, answer types, and images into <code>temp/image/</code> and display interactive questions below.</p>
        </div>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>PDF Question Paper Parser &amp; Quiz Viewer</title>
  <style>
    :root {{
      --primary: #1e3a8a;
      --primary-hover: #1d4ed8;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --border: #e2e8f0;
      --text: #0f172a;
      --muted: #475569;
      --green-bg: #f0fdf4;
      --green-border: #16a34a;
      --green-text: #15803d;
      --red-bg: #fef2f2;
      --red-border: #dc2626;
      --red-text: #b91c1c;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.55;
    }}
    header.top-bar {{
      background: var(--primary);
      color: #ffffff;
      padding: 20px 28px;
      box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
    }}
    .container {{
      max-width: 960px;
      margin: 0 auto;
      padding: 24px 20px 60px;
    }}
    header.top-bar .header-inner {{
      max-width: 960px;
      margin: 0 auto;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }}
    header.top-bar h1 {{
      margin: 0;
      font-size: 1.35rem;
      font-weight: 700;
    }}
    header.top-bar p {{
      margin: 4px 0 0;
      font-size: 0.9rem;
      color: #dbeafe;
    }}
    .upload-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 18px 22px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      justify-content: space-between;
      gap: 14px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .upload-form {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 12px;
      width: 100%;
    }}
    .file-input {{
      flex: 1;
      min-width: 240px;
      padding: 8px 10px;
      border: 1px solid var(--border);
      border-radius: 6px;
      background: #f8fafc;
      font-size: 0.92rem;
    }}
    .btn-primary {{
      background: var(--primary);
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 10px 18px;
      font-size: 0.92rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn-primary:hover {{ background: var(--primary-hover); }}
    .btn-secondary {{
      background: #f1f5f9;
      color: var(--text);
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 0.84rem;
      font-weight: 600;
      text-decoration: none;
      cursor: pointer;
    }}
    .btn-secondary:hover {{ background: #e2e8f0; }}
    .status-bar {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-radius: 8px;
      padding: 12px 16px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      font-size: 0.9rem;
    }}
    .status-links {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }}
    .question-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 20px 24px;
      margin-bottom: 18px;
      box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }}
    .question-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 10px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 12px;
      margin-bottom: 14px;
    }}
    .q-title-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .q-number {{
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--primary);
    }}
    .q-type-badge {{
      font-size: 0.78rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 999px;
      background: #e0e7ff;
      color: #1e40af;
    }}
    .type-multichoice {{ background: #fef3c7; color: #92400e; }}
    .type-numeric {{ background: #dcfce7; color: #166534; }}
    .btn-submit {{
      background: #0f766e;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 8px 16px;
      font-size: 0.88rem;
      font-weight: 600;
      cursor: pointer;
    }}
    .btn-submit:hover {{ background: #0d9488; }}
    .question-prompt {{
      font-size: 0.98rem;
      margin-bottom: 14px;
      overflow-wrap: break-word;
    }}
    .block-img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 10px 0;
      border: 1px solid var(--border);
      border-radius: 6px;
    }}
    .inline-img {{
      display: inline-block;
      vertical-align: middle;
      max-height: 28px;
      width: auto;
      margin: 0 2px;
    }}
    .options-list {{
      display: flex;
      flex-direction: column;
      gap: 8px;
    }}
    .option-item {{
      display: flex;
      align-items: flex-start;
      gap: 10px;
      padding: 10px 12px;
      border: 1px solid var(--border);
      border-radius: 7px;
      cursor: pointer;
      background: #f8fafc;
    }}
    .option-item:hover {{ background: #f1f5f9; }}
    .option-item input {{ margin-top: 4px; }}
    .option-text {{ flex: 1; font-size: 0.94rem; }}
    .option-correct {{
      background: var(--green-bg);
      border-color: var(--green-border);
    }}
    .option-wrong {{
      background: var(--red-bg);
      border-color: var(--red-border);
    }}
    .opt-badge {{
      font-size: 0.78rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      white-space: nowrap;
    }}
    .correct-badge {{ background: #dcfce7; color: var(--green-text); }}
    .wrong-badge {{ background: #fee2e2; color: var(--red-text); }}
    .numeric-input-row {{
      display: flex;
      align-items: center;
      gap: 12px;
      margin: 10px 0;
    }}
    .numeric-label {{ font-weight: 600; font-size: 0.92rem; }}
    .numeric-input {{
      padding: 8px 12px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      font-size: 0.95rem;
      width: 240px;
    }}
    .answer-reveal-box {{
      margin-top: 14px;
      padding: 12px 16px;
      border-radius: 8px;
      background: var(--green-bg);
      border: 1px solid var(--green-border);
    }}
    .eval-status {{
      font-weight: 700;
      font-size: 0.9rem;
      margin-bottom: 6px;
    }}
    .status-pass {{ color: var(--green-text); }}
    .status-fail {{ color: var(--red-text); }}
    .status-neutral {{ color: var(--primary); }}
    .correct-list {{ margin: 6px 0 0 20px; padding: 0; }}
    .empty-state {{
      text-align: center;
      padding: 48px 24px;
      background: var(--card-bg);
      border: 1px dashed #cbd5e1;
      border-radius: 10px;
      color: var(--muted);
    }}
    .alert-error {{
      background: var(--red-bg);
      border: 1px solid var(--red-border);
      color: var(--red-text);
      padding: 12px 16px;
      border-radius: 8px;
      margin-bottom: 16px;
    }}
  </style>
</head>
<body>
  <header class="top-bar">
    <div class="header-inner">
      <div>
        <h1>PDF Question Paper Parser &amp; Interactive Practice</h1>
        <p>Pure-Python Web Frontend &bull; Saves extracted images in <code>temp/image/</code></p>
      </div>
    </div>
  </header>
  <main class="container">
    {error_banner}
    <div class="upload-card">
      <form method="post" action="/upload" enctype="multipart/form-data" class="upload-form">
        <input type="file" name="pdf_file" accept=".pdf,application/pdf" required class="file-input" />
        <button type="submit" class="btn-primary">Upload &amp; Parse PDF</button>
      </form>
    </div>
    {summary_bar}
    {cards_html}
  </main>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    """Render the main quiz frontend page."""
    return HTMLResponse(content=render_full_page())


@app.post("/upload", response_class=HTMLResponse)
async def upload_pdf(pdf_file: UploadFile = File(...)) -> HTMLResponse:
    """
    Accept a PDF file from the user, save it, ensure `temp/image` exists, run
    `parse_pdf_questions` to extract the JSON and images into `temp/image`, and
    render the questions.
    """
    STATE.ensure_directories()
    try:
        content = await pdf_file.read()
        if not content:
            STATE.error_message = "Uploaded PDF file is empty."
            return HTMLResponse(content=render_full_page(), status_code=400)

        with open(UPLOADED_PDF_PATH, "wb") as f:
            f.write(content)

        STATE.load_pdf(UPLOADED_PDF_PATH, pdf_file.filename or "uploaded.pdf")
    except Exception as exc:
        STATE.error_message = f"Failed to parse PDF: {exc}"
        return HTMLResponse(content=render_full_page(), status_code=500)

    return HTMLResponse(content=render_full_page())


@app.post("/submit/{q_idx}", response_class=HTMLResponse)
async def submit_question_answer(request: Request, q_idx: int) -> HTMLResponse:
    """
    Handle the Submit button next to an individual question: record the user's
    selected/typed answer, mark the question's answer as revealed, and re-render.
    """
    if 0 <= q_idx < len(STATE.questions):
        form_data = await request.form()
        raw_answers = form_data.getlist("answer")
        q_type = STATE.questions[q_idx].get("Answer type", "MCQ")
        if q_type == "multichoice":
            STATE.user_responses[q_idx] = [str(v) for v in raw_answers]
        elif raw_answers:
            STATE.user_responses[q_idx] = str(raw_answers[0])
        else:
            STATE.user_responses[q_idx] = ""
        STATE.revealed_indices.add(q_idx)

    return HTMLResponse(content=render_full_page())


@app.post("/reveal-all", response_class=HTMLResponse)
async def reveal_all_answers() -> HTMLResponse:
    """Reveal answers for all currently loaded questions."""
    STATE.revealed_indices = set(range(len(STATE.questions)))
    return HTMLResponse(content=render_full_page())


@app.post("/hide-all", response_class=HTMLResponse)
async def hide_all_answers() -> HTMLResponse:
    """Hide all revealed answers."""
    STATE.revealed_indices.clear()
    return HTMLResponse(content=render_full_page())


@app.get("/api/json", response_class=JSONResponse)
async def get_parsed_json() -> JSONResponse:
    """Return the parsed questions JSON list."""
    return JSONResponse(content=STATE.questions)


@app.get("/temp/image/{filename}")
async def serve_extracted_image(filename: str):
    """Serve extracted question and formula images from `temp/image`."""
    STATE.ensure_directories()
    safe_name = Path(filename).name
    file_path = IMAGE_DIR / safe_name
    if not file_path.is_file():
        return JSONResponse({"error": "Image not found"}, status_code=404)
    media_type, _ = mimetypes.guess_type(str(file_path))
    return FileResponse(file_path, media_type=media_type or "image/png")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pure-Python PDF Question Paper Parser Web Frontend."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port number to bind")
    parser.add_argument(
        "--pdf",
        default=None,
        help="Optional initial PDF file to pre-load on startup",
    )
    args = parser.parse_args()

    STATE.ensure_directories()
    if args.pdf and Path(args.pdf).is_file():
        STATE.load_pdf(args.pdf, Path(args.pdf).name)
        print(
            f"Pre-loaded {len(STATE.questions)} questions and saved images in {IMAGE_DIR}"
        )

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()