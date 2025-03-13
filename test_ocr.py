import pytesseract
from PIL import Image
from googletrans import Translator

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def main():
    image_path = "arabic_sample.png"
    img = Image.open(image_path)
    arabic_text = pytesseract.image_to_string(img, lang='ara')
    print("Extracted Arabic Text:\n", arabic_text)

    # Translate the extracted text to English
    translator = Translator()
    translation = translator.translate(arabic_text, dest='en')
    print("English Translation:\n", translation.text)

if __name__ == "__main__":
    main()
