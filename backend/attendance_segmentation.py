import cv2
import numpy as np
import os

IMAGE_PATH = "synth_sheet.png"
OUTPUT_DIR = "cropped_signatures"

os.makedirs(OUTPUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH)

if img is None:
    print("Image not found")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# table detection
_, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)

vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)

grid = cv2.add(horizontal, vertical)

contours, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cells = []

for cnt in contours:

    x,y,w,h = cv2.boundingRect(cnt)

    # gerçek hücre boyutları
    if 100 < w < 350 and 40 < h < 120:
        cells.append((x,y,w,h))

cells = sorted(cells, key=lambda b:(b[1],b[0]))

signature_cells = []

for (x,y,w,h) in cells:

    # sadece signature columns
    if x > 480:

        crop = img[y:y+h, x:x+w]

        # BORDERLARI KES
        crop = crop[8:h-8, 8:w-8]

        crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # threshold
        _, thresh = cv2.threshold(
            crop_gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # küçük gürültü temizliği
        kernel = np.ones((3,3),np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        ink_pixels = np.sum(thresh > 0)

        total_pixels = thresh.shape[0] * thresh.shape[1]

        ink_ratio = ink_pixels / total_pixels

        # SILIK IMZA ICIN DAHA DUSUK THRESHOLD
        if ink_ratio > 0.003:

            signature_cells.append((x,y,w,h))

print("Detected signatures:", len(signature_cells))

# crop signatures
for i,(x,y,w,h) in enumerate(signature_cells):

    crop = img[y:y+h, x:x+w]

    cv2.imwrite(f"{OUTPUT_DIR}/signature_{i}.png", crop)

# visualization
debug = img.copy()

for (x,y,w,h) in signature_cells:
    cv2.rectangle(debug,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("Detected Signatures",debug)

key = cv2.waitKey(0)

if key == ord("q"):
    cv2.destroyAllWindows()