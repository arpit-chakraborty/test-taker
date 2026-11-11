#!/usr/bin/env python3
"""
PDF Question & Answer Parser for TCS iON / IIT Madras BS Assessment PDFs.

Patterns exploited for accurate extraction:
1. Question Boundaries & Types:
   - Each question starts with `Question Number : <N> Question Id : <ID>`.
   - `Question Label :` directly identifies the question type:
     * "Multiple Choice Question" -> "MCQ"
     * "Multiple Select Question" -> "multichoice"
     * "Short Answer Question" (with `Response Type : Numeric`) -> "numeric"
2. Correct vs. Incorrect Answers (Dual Signal Verification):
   - Option lines begin with a numeric Option ID prefix (`<13-digit-id>. <option text>`).
   - Correct option IDs are rendered in green RGB `(0.0, 0.50196, 0.0)` and paired with
     a 16x16 green tick icon (`Im` XObject with mean G > R).
   - Incorrect option IDs are rendered in red RGB `(1.0, 0.0, 0.0)` and paired with
     a 16x16 red cross icon (`Im` XObject with mean R > G).
   - Numeric answers appear under `Possible Answers :` in green RGB `(0.0, 0.50196, 0.0)`.
3. Content Images vs. UI Icons & Exact Reading Order:
   - All 16x16 (`12x12` pt) tick/cross option icons are filtered out automatically.
   - Inline mathematical formula images (`height < 40` pt) are baseline-aligned with
     text lines (`abs(img.bottom - line.bottom) <= 8` pt) and spliced into the exact
     horizontal word order (`x0`) where they appear.
   - Full-width question/code-snippet block images (`height >= 40` pt) are ordered
     vertically (`top`) relative to surrounding text and options.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pdfplumber
from pypdf import PdfReader


def is_green_color(color: Any) -> bool:
    """Return True if a pdfplumber non_stroking_color tuple is green (correct answer)."""
    if not isinstance(color, (tuple, list)) or len(color) != 3:
        return False
    r, g, b = color
    return g > 0.35 and r < 0.2 and b < 0.2


def classify_icon_image(pil_img) -> Optional[str]:
    """Classify a 16x16 option icon image into 'tick' (correct) or 'cross' (incorrect)."""
    if pil_img.size != (16, 16):
        return None
    arr = np.array(pil_img.convert("RGBA"))
    mask = (arr[:, :, 3] > 50) & (
        (arr[:, :, 0] < 240) | (arr[:, :, 1] < 240) | (arr[:, :, 2] < 240)
    )
    if not mask.any():
        return None
    mean_rgb = arr[mask][:, :3].mean(axis=0)
    return "tick" if mean_rgb[1] > mean_rgb[0] else "cross"


def extract_page_layout_lines(
    pdf_path: str,
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[int, str], Any]]:
    """
    Extract ordered layout lines across all pages, merging inline formula images
    into their corresponding text lines by baseline (`bottom`) alignment and keeping
    block images as standalone layout elements.
    """
    reader = PdfReader(pdf_path)
    pypdf_images: Dict[Tuple[int, str], Any] = {}
    icon_types: Dict[Tuple[int, str], str] = {}

    for p_idx, page in enumerate(reader.pages):
        for img in page.images:
            base_name = img.name.split(".")[0]
            if img.image.size == (16, 16):
                icon_cls = classify_icon_image(img.image)
                if icon_cls:
                    icon_types[(p_idx, base_name)] = icon_cls
            else:
                pypdf_images[(p_idx, base_name)] = img

    all_lines: List[Dict[str, Any]] = []

    with pdfplumber.open(pdf_path) as pdf:
        for p_idx, page in enumerate(pdf.pages):
            words = page.extract_words(
                extra_attrs=["non_stroking_color", "fontname", "size"]
            )
            text_lines: List[Dict[str, Any]] = []
            for w in words:
                if not text_lines or abs(w["top"] - text_lines[-1]["top"]) > 4.5:
                    text_lines.append(
                        {
                            "type": "text_line",
                            "page": p_idx,
                            "top": w["top"],
                            "bottom": w["bottom"],
                            "tokens": [
                                {
                                    "kind": "word",
                                    "x0": w["x0"],
                                    "x1": w["x1"],
                                    "top": w["top"],
                                    "bottom": w["bottom"],
                                    "text": w["text"],
                                    "color": w.get("non_stroking_color"),
                                }
                            ],
                            "icon": None,
                        }
                    )
                else:
                    text_lines[-1]["tokens"].append(
                        {
                            "kind": "word",
                            "x0": w["x0"],
                            "x1": w["x1"],
                            "top": w["top"],
                            "bottom": w["bottom"],
                            "text": w["text"],
                            "color": w.get("non_stroking_color"),
                        }
                    )
                    text_lines[-1]["bottom"] = max(
                        text_lines[-1]["bottom"], w["bottom"]
                    )

            standalone_img_lines: List[Dict[str, Any]] = []

            for img_meta in page.images:
                img_name = img_meta.get("name", "")
                srcsize = img_meta.get("srcsize")
                # Filter out 16x16 tick/cross option icons and record their status
                if srcsize == (16, 16) or (
                    abs(img_meta["width"] - 12.0) < 1.0
                    and abs(img_meta["height"] - 12.0) < 1.0
                ):
                    icon_cls = icon_types.get((p_idx, img_name))
                    if icon_cls:
                        for tl in text_lines:
                            if abs(tl["top"] - img_meta["top"]) < 8.0:
                                tl["icon"] = icon_cls
                                break
                    continue

                # Non-icon content image (inline formula or standalone block image)
                is_inline = img_meta["height"] < 40.0
                attached = False
                if is_inline:
                    best_tl = None
                    best_diff = 999.0
                    for tl in text_lines:
                        diff = abs(img_meta["bottom"] - tl["bottom"])
                        if diff <= 8.0 and diff < best_diff:
                            best_diff = diff
                            best_tl = tl
                    if best_tl is not None:
                        best_tl["tokens"].append(
                            {
                                "kind": "inline_image",
                                "x0": img_meta["x0"],
                                "x1": img_meta["x1"],
                                "top": img_meta["top"],
                                "bottom": img_meta["bottom"],
                                "page": p_idx,
                                "name": img_name,
                                "width": img_meta["width"],
                                "height": img_meta["height"],
                            }
                        )
                        attached = True

                if not attached:
                    standalone_img_lines.append(
                        {
                            "type": "image_block",
                            "page": p_idx,
                            "top": img_meta["top"],
                            "bottom": img_meta["bottom"],
                            "x0": img_meta["x0"],
                            "x1": img_meta["x1"],
                            "name": img_name,
                            "width": img_meta["width"],
                            "height": img_meta["height"],
                        }
                    )

            for tl in text_lines:
                tl["tokens"].sort(key=lambda tok: tok["x0"])

            page_items = text_lines + standalone_img_lines
            page_items.sort(key=lambda item: item["top"])
            all_lines.extend(page_items)

    return all_lines, pypdf_images


def line_raw_text(line: Dict[str, Any]) -> str:
    """Return plain text of word tokens in a text_line."""
    if line["type"] != "text_line":
        return ""
    return " ".join(
        tok["text"] for tok in line["tokens"] if tok["kind"] == "word"
    ).strip()


def render_line_with_images(
    line: Dict[str, Any],
    q_num: int,
    section_stage: int,
    pypdf_images: Dict[Tuple[int, str], Any],
    image_output_dir: Path,
    image_rel_prefix: str,
    saved_images: List[Dict[str, Any]],
) -> str:
    """
    Convert a layout line (text_line or image_block) into text, saving any
    referenced images to `image_output_dir` and inserting their relative path
    in exact reading order.
    """
    if line["type"] == "image_block":
        key = (line["page"], line["name"])
        img_obj = pypdf_images.get(key)
        ext = Path(img_obj.name).suffix if img_obj else ".png"
        idx = len(saved_images) + 1
        filename = f"q{q_num}_img{idx}{ext}"
        out_path = image_output_dir / filename
        if img_obj is not None:
            img_obj.image.save(out_path)
        rel_path = f"{image_rel_prefix}/{filename}" if image_rel_prefix else filename
        saved_images.append(
            {
                "path": rel_path,
                "location": section_stage,
                "is_block": True,
            }
        )
        return f"[IMAGE: {rel_path}]"

    parts: List[str] = []
    for tok in line["tokens"]:
        if tok["kind"] == "word":
            parts.append(tok["text"])
        elif tok["kind"] == "inline_image":
            key = (tok["page"], tok["name"])
            img_obj = pypdf_images.get(key)
            ext = Path(img_obj.name).suffix if img_obj else ".png"
            idx = len(saved_images) + 1
            filename = f"q{q_num}_img{idx}{ext}"
            out_path = image_output_dir / filename
            if img_obj is not None:
                img_obj.image.save(out_path)
            rel_path = f"{image_rel_prefix}/{filename}" if image_rel_prefix else filename
            saved_images.append(
                {
                    "path": rel_path,
                    "location": section_stage,
                    "is_block": False,
                }
            )
            parts.append(f"[IMAGE: {rel_path}]")
    return " ".join(parts).strip()


def parse_question_block(
    q_num: int,
    q_id: str,
    block_lines: List[Dict[str, Any]],
    pypdf_images: Dict[Tuple[int, str], Any],
    image_output_dir: Path,
    image_rel_prefix: str,
    include_metadata: bool = False,
) -> Dict[str, Any]:
    """
    Parse all lines belonging to a single question block into the requested JSON schema:
    - Question
    - Answer
    - Correct answer
    - Answer type
    - Image
    - Image location (0: above the question, 1: between question and answer, 2: after the answer)
    """
    question_label = ""
    response_type = ""
    state = "HEADER"

    question_parts: List[str] = []
    has_question_text_before_image = False
    saved_images: List[Dict[str, Any]] = []

    options: List[Dict[str, Any]] = []
    numeric_answer_lines: List[str] = []

    option_pattern = re.compile(r"^(\d{8,15})\.\s*(.*)$")

    for line in block_lines:
        raw_text = line_raw_text(line)

        # Skip section/sub-section transition metadata at the end of a question block
        if any(
            raw_text.startswith(p)
            for p in (
                "Sub-Section Number :",
                "Sub-Section Id :",
                "Question Shuffling Allowed :",
                "Section Id :",
                "Section Number :",
                "Enable Mark as Answered",
                "Clear Response :",
            )
        ):
            continue

        if raw_text.startswith("Correct Marks :"):
            continue

        if raw_text.startswith("Question Label :"):
            question_label = raw_text.split("Question Label :", 1)[1].strip()
            state = "QUESTION"
            continue

        if raw_text == "Options :":
            state = "OPTIONS"
            continue

        if raw_text.startswith("Response Type :"):
            response_type = raw_text.split("Response Type :", 1)[1].strip()
            state = "NUMERIC_META"
            continue

        if raw_text == "Possible Answers :":
            state = "POSSIBLE_ANSWERS"
            continue

        if state == "NUMERIC_META":
            continue

        if state == "QUESTION":
            if line["type"] == "image_block":
                # 0: above/at start of question (before any question text)
                # 1: between question text and answer options
                loc = 1 if has_question_text_before_image else 0
                rendered = render_line_with_images(
                    line,
                    q_num,
                    loc,
                    pypdf_images,
                    image_output_dir,
                    image_rel_prefix,
                    saved_images,
                )
                question_parts.append(rendered)
            elif line["type"] == "text_line":
                if raw_text:
                    has_question_text_before_image = True
                rendered = render_line_with_images(
                    line,
                    q_num,
                    1,
                    pypdf_images,
                    image_output_dir,
                    image_rel_prefix,
                    saved_images,
                )
                if rendered:
                    question_parts.append(rendered)

        elif state == "OPTIONS":
            if line["type"] == "text_line":
                m = option_pattern.match(raw_text)
                if m:
                    opt_id = m.group(1)
                    first_word_color = line["tokens"][0].get("color")
                    is_correct = (
                        is_green_color(first_word_color) or line.get("icon") == "tick"
                    )
                    rendered_full = render_line_with_images(
                        line,
                        q_num,
                        2,
                        pypdf_images,
                        image_output_dir,
                        image_rel_prefix,
                        saved_images,
                    )
                    opt_text = re.sub(r"^\d{8,15}\.\s*", "", rendered_full).strip()
                    options.append(
                        {
                            "option_id": opt_id,
                            "text": opt_text,
                            "is_correct": is_correct,
                        }
                    )
                else:
                    if options:
                        rendered_cont = render_line_with_images(
                            line,
                            q_num,
                            2,
                            pypdf_images,
                            image_output_dir,
                            image_rel_prefix,
                            saved_images,
                        )
                        if rendered_cont:
                            options[-1]["text"] = (
                                f"{options[-1]['text']} {rendered_cont}".strip()
                            )
            elif line["type"] == "image_block":
                rendered_img = render_line_with_images(
                    line,
                    q_num,
                    2,
                    pypdf_images,
                    image_output_dir,
                    image_rel_prefix,
                    saved_images,
                )
                if options:
                    options[-1]["text"] = (
                        f"{options[-1]['text']} {rendered_img}".strip()
                    )

        elif state == "POSSIBLE_ANSWERS":
            if line["type"] == "text_line" and raw_text:
                numeric_answer_lines.append(raw_text)

    # Map Question Label / Response Type to required Answer type: [MCQ, multichoice or numeric]
    label_lower = question_label.lower()
    if "multiple choice" in label_lower:
        answer_type = "MCQ"
    elif "multiple select" in label_lower:
        answer_type = "multichoice"
    elif "short answer" in label_lower or response_type.lower() == "numeric":
        answer_type = "numeric"
    else:
        answer_type = "MCQ"

    question_text = " ".join(question_parts).strip()

    if answer_type == "numeric":
        all_answers = None
        correct_answer: Union[str, List[str]] = " ".join(numeric_answer_lines).strip()
    elif answer_type == "MCQ":
        all_answers = [opt["text"] for opt in options]
        correct_opts = [opt["text"] for opt in options if opt["is_correct"]]
        correct_answer = correct_opts[0] if len(correct_opts) == 1 else correct_opts
    else:  # multichoice
        all_answers = [opt["text"] for opt in options]
        correct_answer = [opt["text"] for opt in options if opt["is_correct"]]

    if not saved_images:
        image_field: Optional[Union[str, List[str]]] = None
        image_location_field: Optional[Union[int, List[int]]] = None
    elif len(saved_images) == 1:
        image_field = saved_images[0]["path"]
        image_location_field = saved_images[0]["location"]
    else:
        image_field = [img["path"] for img in saved_images]
        image_location_field = [img["location"] for img in saved_images]

    result: Dict[str, Any] = {}
    if include_metadata:
        result["Question_Number"] = q_num
        result["Question_Id"] = q_id

    result.update(
        {
            "Question": question_text,
            "Answer": all_answers,
            "Correct answer": correct_answer,
            "Answer type": answer_type,
            "Image": image_field,
            "Image location": image_location_field,
        }
    )
    return result


def parse_pdf_questions(
    pdf_path: str,
    output_json_path: str,
    image_output_dir: str,
    skip_zero_mark_confirmation: bool = False,
    include_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """
    Parse an assessment PDF into a list of question dictionaries, saving extracted
    question/option images to `image_output_dir` and writing the JSON output to
    `output_json_path`.
    """
    img_dir = Path(image_output_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    out_json = Path(output_json_path)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        image_rel_prefix = os.path.relpath(img_dir.resolve(), out_json.parent.resolve())
        if image_rel_prefix == ".":
            image_rel_prefix = ""
    except ValueError:
        image_rel_prefix = str(img_dir)

    all_lines, pypdf_images = extract_page_layout_lines(pdf_path)

    q_header_re = re.compile(
        r"^Question\s+Number\s*:\s*(\d+)\s+Question\s+Id\s*:\s*(\d+)"
    )
    question_blocks: List[Tuple[int, str, List[Dict[str, Any]]]] = []
    current_q: Optional[Tuple[int, str, List[Dict[str, Any]]]] = None

    for line in all_lines:
        raw = line_raw_text(line)
        m = q_header_re.match(raw)
        if m:
            if current_q is not None:
                question_blocks.append(current_q)
            current_q = (int(m.group(1)), m.group(2), [])
        else:
            if current_q is not None:
                current_q[2].append(line)

    if current_q is not None:
        question_blocks.append(current_q)

    parsed_questions: List[Dict[str, Any]] = []
    for q_num, q_id, b_lines in question_blocks:
        if skip_zero_mark_confirmation:
            marks_lines = [
                line_raw_text(l)
                for l in b_lines
                if line_raw_text(l).startswith("Correct Marks :")
            ]
            if marks_lines and marks_lines[0].split(":", 1)[1].strip() == "0":
                continue

        q_dict = parse_question_block(
            q_num=q_num,
            q_id=q_id,
            block_lines=b_lines,
            pypdf_images=pypdf_images,
            image_output_dir=img_dir,
            image_rel_prefix=image_rel_prefix,
            include_metadata=include_metadata,
        )
        parsed_questions.append(q_dict)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(parsed_questions, f, indent=2, ensure_ascii=False)

    return parsed_questions


def main():
    parser = argparse.ArgumentParser(
        description="Parse questions, answers, answer types, and images from an exam PDF."
    )
    parser.add_argument("pdf_path", help="Path to the input question paper PDF")
    parser.add_argument(
        "-o",
        "--output",
        default="parsed_questions.json",
        help="Path to the output JSON file (default: parsed_questions.json)",
    )
    parser.add_argument(
        "-i",
        "--images-dir",
        default="extracted_images",
        help="Directory to save extracted images (default: extracted_images)",
    )
    parser.add_argument(
        "--skip-confirmation",
        action="store_true",
        help="Skip 0-mark subject confirmation questions (e.g., Question 1)",
    )
    parser.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include Question_Number and Question_Id fields in each JSON element",
    )
    args = parser.parse_args()

    results = parse_pdf_questions(
        pdf_path=args.pdf_path,
        output_json_path=args.output,
        image_output_dir=args.images_dir,
        skip_zero_mark_confirmation=args.skip_confirmation,
        include_metadata=args.include_metadata,
    )
    print(f"Successfully parsed {len(results)} questions into {args.output}")


if __name__ == "__main__":
    main()