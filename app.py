import streamlit as st
import pytesseract
import tempfile
import os
from PIL import Image
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from googletrans import Translator
import cv2
import numpy as np

# If Tesseract is not on your PATH, specify its location here:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Arabic OCR & Translation", layout="wide")

# Inject some custom CSS to style text in a larger box
st.markdown(
    """
    <style>
    .highlight-box {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 5px;
        background-color: #f9f9f9;
        font-size: 1.2rem; /* Increase font size */
        line-height: 1.6;
        white-space: pre-wrap; /* preserve line breaks */
        overflow-wrap: break-word;
    }
    .title-text {
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def deskew_image(gray_image):
    """
    Optional deskew function if your image is rotated.
    Using minAreaRect approach for demonstration.
    """
    coords = np.column_stack(np.where(gray_image > 0))
    angle = cv2.minAreaRect(coords)[-1]

    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = gray_image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray_image, M, (w, h),
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def preprocess_image(input_image, do_deskew=False):
    """Grayscale, (optional) deskew, Otsu threshold, morphological closing."""
    if not isinstance(input_image, Image.Image):
        input_image = Image.open(input_image)

    # Convert PIL -> NumPy (OpenCV BGR)
    open_cv_image = np.array(input_image.convert("RGB"))
    img_bgr = open_cv_image[:, :, ::-1].copy()

    # Grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Deskew if needed
    if do_deskew:
        gray = deskew_image(gray)

    # Otsu threshold
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Convert back to PIL
    preprocessed_pil_image = Image.fromarray(closed)
    return preprocessed_pil_image

def extract_text_from_image(img_file, lang="ara"):
    """OCR for images, with morphological preprocessing."""
    original_img = Image.open(img_file)
    pre_img = preprocess_image(original_img, do_deskew=False)
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    text = pytesseract.image_to_string(pre_img, lang=lang, config=config)
    return text

def extract_text_from_pdf(pdf_file, lang="ara"):
    """OCR for PDFs, converting each page to an image, with the same preprocessing."""
    text_pages = []
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_pages = convert_from_path(pdf_file, dpi=300, output_folder=tmpdir)
        for page_num, page_image in enumerate(pdf_pages, start=1):
            pre_img = preprocess_image(page_image, do_deskew=False)
            config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
            page_text = pytesseract.image_to_string(pre_img, lang=lang, config=config)
            text_pages.append(f"--- Page {page_num} ---\n{page_text}")
    return "\n".join(text_pages)

def main():
    st.title("Arabic OCR & Translation App")
    st.write("Upload an image (PNG/JPG) or PDF. We’ll do Arabic OCR and let you review/correct before translating.")
    
    uploaded_file = st.file_uploader("Upload a file", type=["png", "jpg", "jpeg", "pdf"])
    if uploaded_file is not None:
        file_type = uploaded_file.type

        if file_type in ["image/png", "image/jpeg", "image/jpg"]:
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
            extracted_text = extract_text_from_image(uploaded_file, lang="ara")
        elif file_type == "application/pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_filepath = tmp.name
            extracted_text = extract_text_from_pdf(tmp_filepath, lang="ara")
            os.remove(tmp_filepath)
        else:
            st.error("Unsupported file type. Please upload PNG, JPG, JPEG, or PDF.")
            return

        # Display the extracted text in a read-only box or text_area for corrections:
        st.subheader("Extracted Arabic Text")
        corrected_text = st.text_area(
            "You can correct any OCR errors here before translation:",
            value=extracted_text,
            height=200
        )

        # Real-time translation option
        auto_translate = st.checkbox("Enable real-time translation")
        translator = Translator()

        # Display the text in a highlight box
        st.markdown("<div class='title-text'>Arabic Text (Final):</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='highlight-box'>{corrected_text}</div>", unsafe_allow_html=True)

        if auto_translate:
            # Automatically translate whenever corrected_text changes
            if corrected_text.strip():
                translation = translator.translate(corrected_text, dest='en').text
                st.markdown("<div class='title-text'>English Translation (Auto):</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='highlight-box'>{translation}</div>", unsafe_allow_html=True)
        else:
            # Manual translation button
            if st.button("Translate to English"):
                translation = translator.translate(corrected_text, dest='en').text
                st.markdown("<div class='title-text'>English Translation:</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='highlight-box'>{translation}</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
