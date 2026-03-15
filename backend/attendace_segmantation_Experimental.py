import os
import re
import cv2
import fitz
import numpy as np

PDF_PATH = "SKM_554e26031121560.pdf"
OUTPUT_DIR = "cropped_signatures"
TEMP_DIR = "rendered_pages"

COURSE_NAME = "CNG213"
ATTENDANCE_SHEET_ID = "AS001"

# plain text file: one student ID per line, in table row order
STUDENT_ID_FILE = "student_ids.txt"

SIGNATURE_X_MIN = 480

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)


def clean_filename(text):
    return re.sub(r"[^A-Za-z0-9_-]", "", str(text).strip())


def load_student_ids(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Student ID file not found: {path}")

    ids = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            value = line.strip()
            if value:
                ids.append(value)

    return ids



def load_student_map(csv_path):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Mapping file not found: {csv_path}")

    mapping = {}
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            global_row = int(row["global_row"])
            student_id = row["student_id"].strip()
            mapping[global_row] = student_id

    if not mapping:
        raise ValueError(f"{csv_path} is empty.")

    return mapping
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

    # keep only real data rows
    filtered_rows = []
    for row_idx, row in enumerate(rows):
        if len(row) < 5:
            continue
        if row_idx == 0:
            continue
        filtered_rows.append(row)

    return filtered_rows


def process_page(page, page_number, student_ids, row_offset):
    img, zoom = render_page_to_image(page, zoom=2.0)
    debug = img.copy()

    rows = detect_rows(img)
    exported = 0

    for local_row_idx, row in enumerate(rows):
        global_row_idx = row_offset + local_row_idx

        if global_row_idx < len(student_ids):
            student_id = student_ids[global_row_idx]
        else:
            student_id = f"UNKNOWN_row{page_number}_{local_row_idx + 1}"

        for (x, y, w, h) in row:
            if x <= SIGNATURE_X_MIN:
                continue

            crop = img[y:y+h, x:x+w]
            crop = crop[8:h-8, 8:w-8]

            if crop.size == 0:
                continue

            if is_highlighted_absent(crop):
                continue

            ink_ratio = signature_ink_ratio_excluding_highlight(crop)

            if ink_ratio > 0.003:
                base_name = f"{clean_filename(COURSE_NAME)}_{clean_filename(student_id)}_{clean_filename(ATTENDANCE_SHEET_ID)}"
                save_path = os.path.join(OUTPUT_DIR, f"{base_name}.png")

                if os.path.exists(save_path):
                    save_path = os.path.join(
                        OUTPUT_DIR,
                        f"{base_name}_p{page_number}_x{x}_y{y}.png"
                    )

                cv2.imwrite(save_path, crop)
                exported += 1
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # draw row label
        first_x, first_y, first_w, first_h = row[0]
        cv2.putText(
            debug,
            str(student_id),
            (first_x, max(20, first_y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )

    debug_path = os.path.join(OUTPUT_DIR, f"debug_page_{page_number}.jpg")
    cv2.imwrite(debug_path, debug)

    print(f"[Page {page_number}] Data rows: {len(rows)}")
    print(f"[Page {page_number}] Exported signatures: {exported}")

    return len(rows)


def main():
    student_ids = load_student_ids(STUDENT_ID_FILE)
    print("Loaded student IDs:", len(student_ids))

    doc = fitz.open(PDF_PATH)

    row_offset = 0
    for i in range(len(doc)):
        page = doc.load_page(i)
        rows_on_page = process_page(page, i + 1, student_ids, row_offset)
        row_offset += rows_on_page

    doc.close()
    print("Done.")


if __name__ == "__main__":
    main()
