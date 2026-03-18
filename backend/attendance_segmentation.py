import os
import re
import csv
import cv2
import fitz
import numpy as np
import easyocr

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_DIR = os.path.join(BASE_DIR, "static")
ATTENDANCE_DIR = os.path.join(STATIC_DIR, "attendance")

OUTPUT_DIR = os.path.join(ATTENDANCE_DIR, "cropped_signatures")
TEMP_DIR = os.path.join(ATTENDANCE_DIR, "rendered_pages")
OCR_DEBUG_DIR = os.path.join(ATTENDANCE_DIR, "ocr_debug")

STUDENT_MAP_FILE = os.path.join(ATTENDANCE_DIR, "student_map.csv")
METADATA_CSV = os.path.join(ATTENDANCE_DIR, "signatures_metadata.csv")

FORCE_REBUILD_STUDENT_MAP = True

COURSE_NAME = "CNG213"
ATTENDANCE_SHEET_ID = "AS001"

ID_COLUMN_INDEX = 0
FIRST_SIGNATURE_CELL_INDEX = 3
OCR_DEBUG_DIR = os.path.join(OUTPUT_DIR, "ocr_debug")

EXPECTED_ID_LENGTHS = {7, 8}
MIN_FILLED_RATIO = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OCR_DEBUG_DIR, exist_ok=True)

reader = easyocr.Reader(['en'], gpu=False)


def clean_filename(text):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(text).strip())


def looks_like_student_id(text):
    text = re.sub(r"\D", "", str(text))
    return len(text) in EXPECTED_ID_LENGTHS


def render_page_to_image(page, zoom=2.0):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img_path = os.path.join(TEMP_DIR, f"page_{page.number + 1}.jpg")
    pix.save(img_path)
    img = cv2.imread(img_path)
    return img, zoom


def is_highlighted_absent(cell_img):
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    pink_lower = np.array([135, 20, 80])
    pink_upper = np.array([179, 255, 255])

    yellow_lower = np.array([15, 20, 80])
    yellow_upper = np.array([40, 255, 255])

    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    mask = cv2.bitwise_or(pink_mask, yellow_mask)
    ratio = np.sum(mask > 0) / (cell_img.shape[0] * cell_img.shape[1])

    return ratio > 0.03


def signature_ink_ratio_excluding_highlight(cell_img):
    hsv = cv2.cvtColor(cell_img, cv2.COLOR_BGR2HSV)

    pink_lower = np.array([135, 20, 80])
    pink_upper = np.array([179, 255, 255])

    yellow_lower = np.array([15, 20, 80])
    yellow_upper = np.array([40, 255, 255])

    pink_mask = cv2.inRange(hsv, pink_lower, pink_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    highlight_mask = cv2.bitwise_or(pink_mask, yellow_mask)

    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)

    _, ink = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    ink[highlight_mask > 0] = 0

    kernel = np.ones((3, 3), np.uint8)
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)

    ink_pixels = np.sum(ink > 0)
    total_pixels = ink.shape[0] * ink.shape[1]

    return ink_pixels / total_pixels


def group_rows(cells, y_tolerance=20):
    rows = []
    for cell in sorted(cells, key=lambda b: (b[1], b[0])):
        x, y, w, h = cell
        placed = False

        for row in rows:
            row_y = row[0][1]
            if abs(y - row_y) < y_tolerance:
                row.append(cell)
                placed = True
                break

        if not placed:
            rows.append([cell])

    for row in rows:
        row.sort(key=lambda c: c[0])

    rows.sort(key=lambda r: r[0][1])
    return rows


def detect_rows(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))

    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

    grid = cv2.add(horizontal, vertical)

    contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cells = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 100 < w < 350 and 40 < h < 120:
            cells.append((x, y, w, h))

    rows = group_rows(cells, y_tolerance=20)

    filtered_rows = []
    for row in rows:
        if len(row) < max(5, FIRST_SIGNATURE_CELL_INDEX + 1):
            continue
        filtered_rows.append(row)

    return filtered_rows


def preprocess_id_variants(cell_img):
    gray = cv2.cvtColor(cell_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    variants = []

    _, v1 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("otsu", v1))

    _, v2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    v2 = cv2.bitwise_not(v2)
    variants.append(("inv_otsu", v2))

    v3 = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 15
    )
    variants.append(("adaptive", v3))

    blur = cv2.GaussianBlur(gray, (0, 0), 3)
    sharp = cv2.addWeighted(gray, 1.8, blur, -0.8, 0)
    _, v4 = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(("sharp", v4))

    out = []
    for name, img in variants:
        img = cv2.copyMakeBorder(img, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        out.append((name, img))

    return out


def pick_best_id(texts):
    candidates = []

    for txt in texts:
        txt = re.sub(r"\D", "", txt)
        if len(txt) in EXPECTED_ID_LENGTHS:
            candidates.append(txt)

    if not candidates:
        return None

    freq = {}
    for c in candidates:
        freq[c] = freq.get(c, 0) + 1

    best = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return best


def extract_student_id_from_cell(cell_img, debug_tag):
    raw_path = os.path.join(OCR_DEBUG_DIR, f"{debug_tag}_raw.png")
    cv2.imwrite(raw_path, cell_img)

    variants = preprocess_id_variants(cell_img)
    ocr_texts = []

    for idx, (name, var_img) in enumerate(variants):
        dbg_path = os.path.join(OCR_DEBUG_DIR, f"{debug_tag}_{idx}_{name}.png")
        cv2.imwrite(dbg_path, var_img)

        results = reader.readtext(
            var_img,
            detail=0,
            paragraph=False,
            allowlist="0123456789"
        )
        if results:
            ocr_texts.extend(results)

    best = pick_best_id(ocr_texts)
    return best


def create_student_map_from_ocr(pdf_path, out_csv=STUDENT_MAP_FILE):
    doc = fitz.open(pdf_path)
    global_row_counter = 0
    detected_count = 0

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["global_row", "student_id"])

        for i in range(len(doc)):
            page = doc.load_page(i)
            img, _ = render_page_to_image(page, zoom=2.0)
            rows = detect_rows(img)

            for row in rows:
                student_id = ""

                if len(row) > ID_COLUMN_INDEX:
                    x_id, y_id, w_id, h_id = row[ID_COLUMN_INDEX]
                    id_crop = img[y_id:y_id + h_id, x_id:x_id + w_id]
                    id_crop = id_crop[5:h_id - 5, 5:w_id - 5]

                    if id_crop.size > 0:
                        student_id = extract_student_id_from_cell(
                            id_crop,
                            f"page{i+1}_candidate{global_row_counter+1}"
                        ) or ""

                if not looks_like_student_id(student_id):
                    continue

                global_row_counter += 1
                detected_count += 1
                writer.writerow([global_row_counter, student_id])

    doc.close()

    print(f"OCR-based map created: {out_csv}")
    print(f"Detected IDs: {detected_count}/{global_row_counter}")

    fill_ratio = detected_count / max(global_row_counter, 1)
    if fill_ratio < MIN_FILLED_RATIO:
        print("WARNING: OCR quality is poor.")
        print("Possible reasons:")
        print("- ID_COLUMN_INDEX is wrong")
        print("- OCR crop is noisy")
        print("- Student IDs have a different length than expected")
        print("- The row detection order is off")

    print("Check cropped_signatures/ocr_debug/ to verify raw ID-cell crops.")


def load_student_map(csv_path):
    if not os.path.exists(csv_path):
        return None

    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader_csv = csv.DictReader(f)
        for row in reader_csv:
            global_row_text = str(row.get("global_row", "")).strip()
            student_id = str(row.get("student_id", "")).strip()

            if not global_row_text:
                continue

            global_row = int(global_row_text)

            if student_id:
                mapping[global_row] = student_id

    return mapping


def append_metadata(file_path, student_id, page_number, row_number, cell_index, x, y):
    file_exists = os.path.exists(METADATA_CSV)

    with open(METADATA_CSV, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "file_name",
                "student_id",
                "course_name",
                "attendance_sheet_id",
                "page",
                "row",
                "cell_index",
                "x",
                "y"
            ])

        writer.writerow([
            file_path.replace("\\", "/"),
            student_id,
            COURSE_NAME,
            ATTENDANCE_SHEET_ID,
            page_number,
            row_number,
            cell_index,
            x,
            y
        ])


def process_page(page, page_number, student_map, global_row_counter):
    img, zoom = render_page_to_image(page, zoom=2.0)
    debug = img.copy()

    rows = detect_rows(img)
    exported = 0

    for local_row_idx, row in enumerate(rows):
        row_student_id = None

        if len(row) > ID_COLUMN_INDEX:
            x_id, y_id, w_id, h_id = row[ID_COLUMN_INDEX]
            id_crop = img[y_id:y_id + h_id, x_id:x_id + w_id]
            id_crop = id_crop[5:h_id - 5, 5:w_id - 5]

            if id_crop.size > 0:
                row_student_id = extract_student_id_from_cell(
                    id_crop,
                    f"page{page_number}_row{local_row_idx+1}"
                )

        if not looks_like_student_id(row_student_id):
            continue

        global_row_counter += 1
        student_id = student_map.get(global_row_counter, row_student_id)

        for idx, (cx, cy, cw, ch) in enumerate(row):
            cv2.rectangle(debug, (cx, cy), (cx + cw, cy + ch), (0, 255, 255), 1)
            cv2.putText(
                debug,
                str(idx),
                (cx + 5, cy + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 255),
                1
            )

        for cell_index, (x, y, w, h) in enumerate(row):
            if cell_index < FIRST_SIGNATURE_CELL_INDEX:
                continue

            crop = img[y:y+h, x:x+w]
            crop = crop[8:h-8, 8:w-8]

            if crop.size == 0:
                continue

            if is_highlighted_absent(crop):
                continue

            ink_ratio = signature_ink_ratio_excluding_highlight(crop)

            if ink_ratio > 0.003:
                student_folder = os.path.join(OUTPUT_DIR, clean_filename(student_id))
                os.makedirs(student_folder, exist_ok=True)

                base_name = (
                    f"{clean_filename(COURSE_NAME)}_"
                    f"{clean_filename(student_id)}_"
                    f"{clean_filename(ATTENDANCE_SHEET_ID)}_"
                    f"p{page_number}_r{global_row_counter}_c{cell_index}"
                )

                save_path = os.path.join(student_folder, f"{base_name}.png")
                cv2.imwrite(save_path, crop)

                relative_path = os.path.relpath(save_path, OUTPUT_DIR)
                append_metadata(
                    relative_path,
                    student_id,
                    page_number,
                    global_row_counter,
                    cell_index,
                    x,
                    y
                )

                exported += 1
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        first_x, first_y, _, _ = row[0]
        cv2.putText(
            debug,
            str(student_id),
            (first_x, max(20, first_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

        if len(row) > ID_COLUMN_INDEX:
            x_id, y_id, w_id, h_id = row[ID_COLUMN_INDEX]
            cv2.rectangle(debug, (x_id, y_id), (x_id + w_id, y_id + h_id), (255, 0, 0), 2)

    debug_path = os.path.join(OUTPUT_DIR, f"debug_page_{page_number}.jpg")
    cv2.imwrite(debug_path, debug)

    print(f"[Page {page_number}] Data rows used: {global_row_counter}")
    print(f"[Page {page_number}] Exported signatures: {exported}")

    return global_row_counter


def count_total_rows(pdf_path):
    doc = fitz.open(pdf_path)
    total = 0
    for i in range(len(doc)):
        page = doc.load_page(i)
        img, _ = render_page_to_image(page, zoom=2.0)
        rows = detect_rows(img)

        for row in rows:
            if len(row) <= ID_COLUMN_INDEX:
                continue

            x_id, y_id, w_id, h_id = row[ID_COLUMN_INDEX]
            id_crop = img[y_id:y_id + h_id, x_id:x_id + w_id]
            id_crop = id_crop[5:h_id - 5, 5:w_id - 5]

            if id_crop.size == 0:
                continue

            row_student_id = extract_student_id_from_cell(
                id_crop,
                f"count_page{i+1}_{total+1}"
            )

            if looks_like_student_id(row_student_id):
                total += 1

    doc.close()
    return total

def process_attendance_pdf(pdf_path):
    global PDF_PATH
    PDF_PATH = pdf_path

    total_rows = count_total_rows(PDF_PATH)

    if FORCE_REBUILD_STUDENT_MAP or not os.path.exists(STUDENT_MAP_FILE):
        create_student_map_from_ocr(PDF_PATH, STUDENT_MAP_FILE)

    student_map = load_student_map(STUDENT_MAP_FILE)

    if os.path.exists(METADATA_CSV):
        os.remove(METADATA_CSV)

    doc = fitz.open(PDF_PATH)
    global_row_counter = 0

    for i in range(len(doc)):
        page = doc.load_page(i)
        global_row_counter = process_page(page, i + 1, student_map, global_row_counter)

    doc.close()

    return {
        "output_dir": OUTPUT_DIR,
        "metadata_csv": METADATA_CSV
    }


def main():
    total_rows = count_total_rows(PDF_PATH)
    print("Expected total student rows:", total_rows)

    if FORCE_REBUILD_STUDENT_MAP or not os.path.exists(STUDENT_MAP_FILE):
        print("Rebuilding student_map.csv with OCR...")
        create_student_map_from_ocr(PDF_PATH, STUDENT_MAP_FILE)

    student_map = load_student_map(STUDENT_MAP_FILE)

    filled = len(student_map) if student_map else 0
    print("Loaded student IDs:", filled)

    if filled < max(1, int(total_rows * MIN_FILLED_RATIO)):
        print("student_map.csv is too incomplete.")
        print("Try changing ID_COLUMN_INDEX or EXPECTED_ID_LENGTHS and rerun.")
        return

    if os.path.exists(METADATA_CSV):
        os.remove(METADATA_CSV)

    doc = fitz.open(PDF_PATH)
    global_row_counter = 0

    for i in range(len(doc)):
        page = doc.load_page(i)
        global_row_counter = process_page(page, i + 1, student_map, global_row_counter)

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
