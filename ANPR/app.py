from flask import Flask, render_template, request, redirect, url_for
import cv2
import numpy as np
import pytesseract
import imutils
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Set Tesseract Path (adjust based on your OS)
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

def process_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (600, 400))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 13, 15, 15)
    edged = cv2.Canny(gray, 30, 200)

    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screenCnt = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        if len(approx) == 4:
            screenCnt = approx
            break

    if screenCnt is None:
        return None, "No license plate detected"

    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [screenCnt], 0, 255, -1)
    new_image = cv2.bitwise_and(img, img, mask=mask)

    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = gray[topx:bottomx + 1, topy:bottomy + 1]

    plate_number = pytesseract.image_to_string(cropped, config='--psm 7').strip()

    plate_path = os.path.join(app.config['UPLOAD_FOLDER'], 'plate.jpg')
    cv2.imwrite(plate_path, cropped)
    
    return plate_path, plate_number

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(image_path)
            plate_image, plate_number = process_image(image_path)
            return render_template('index.html', plate_image=plate_image, plate_number=plate_number, uploaded_image=image_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
