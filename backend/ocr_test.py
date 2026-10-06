import cv2
import pytesseract
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Load image
img = cv2.imread(r"D:\\coin\\img3.jpeg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect edges
edges = cv2.Canny(gray, 100, 200)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)

    # Ignore very small contours (noise)
    if w*h < 5000:
        continue

    # Crop coin
    coin = gray[y:y+h, x:x+w]

    # Preprocess
    blur = cv2.GaussianBlur(coin, (5,5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Run OCR only for digits
    text = pytesseract.image_to_string(thresh, config="--psm 11 -c tessedit_char_whitelist=0123456789")
    print(f"Coin {i+1} OCR Text:", text)

    # Find year
    year_match = re.findall(r"\b(19[5-9]\d|20[0-2]\d)\b", text)
    if year_match:
        print(f"Detected Year(s): {year_match}")
    else:
        print("No year detected")

    # Optional: save cropped coin for debugging
    cv2.imwrite(f"coin_{i+1}.png", coin)
