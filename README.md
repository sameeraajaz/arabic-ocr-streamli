# Arabic OCR & Translation App

A Streamlit application that performs OCR on Arabic text in images or PDFs using Tesseract, with optional preprocessing (OpenCV) for improved accuracy, manual text correction, and English translation (via `googletrans`).

## Live Demo

[Arabic OCR Streamlit App](https://arabic-ocr-streamli-8lswm5dhgh253qzpxsraat.streamlit.app/)

*(Click the link to open the deployed Streamlit application.)*

---

## Features

- **Upload Images/PDFs**: Drag-and-drop or file select.
- **Arabic OCR**: Utilizes Tesseract (`lang='ara'`).
- **Preprocessing**: Binarization, morphological ops, optional deskew.
- **Manual Correction**: Edit recognized text if needed.
- **Real-Time or Manual Translation**: Translate to English using a checkbox or a button.

---

## Requirements

### 1. System Packages

In order for the app to run on Ubuntu/Debian (such as Streamlit Cloud), these are needed:

- `tesseract-ocr`
- `tesseract-ocr-ara`
- `poppler-utils`
- **OpenCV dependencies**: `libgl1`, `libglib2.0-0`, `libsm6`, `libxrender1`, `libxext6`

Add them to your **`packages.txt`** so Streamlit Cloud installs them automatically.

### 2. Python Packages

See [requirements.txt](./requirements.txt). Key libraries include:
- `streamlit`
- `pytesseract`
- `opencv-python`
- `numpy`
- `pdf2image`
- `PyMuPDF`
- `googletrans==4.0.0-rc1`
- `pillow`

Install via:
```bash
pip install -r requirements.txt

MIT License

Copyright (c) 2023 ...

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the “Software”), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

(… truncated for brevity—see full MIT text if needed …)

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND …
