# 🎙️ F5-TTS Telugu Voice Cloning Project

## 📌 Overview

This project is based on F5-TTS (Text-to-Speech) and is customized for generating speech using Telugu voice samples.
It allows you to clone voices and generate high-quality speech outputs from text.

---

## 🚀 Features

* 🎤 Voice cloning using reference audio
* 🗣️ Telugu text-to-speech generation
* ⚡ Fast inference using optimized models
* 📁 Simple input/output workflow

---

## 📂 Project Structure

```
F5-TTS/
│── app.py
│── main.py
│── model.py
│── whisper.py
│── requirements.txt
│── config.json
│
├── f5_tts_local/        # Core TTS model files
├── input/               # Place input audio files here
├── output/              # Generated audio will be saved here
├── checkpoints/         # (Not included) Model weights
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository

```
git clone https://github.com/UTADurgalakshminirmala/F5-TTS-.git
cd F5-TTS-
```

---

### 2️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 3️⃣ Download Checkpoints

⚠️ Checkpoints are NOT included in this repo due to size limitations.

👉 Download from: Google Drive Link
**(https://drive.google.com/drive/folders/1P6up6Ej_Khq2gMpuIt13L7IW4zQAaj1b?usp=drive_link)**

After downloading, place them inside:

```
checkpoints/
```

---

### 4️⃣ Run the Project

```
python whisper.py
```

---

## 📥 Input & Output

### 🔹 Input

* Place reference audio files inside:

```
input/
```

### 🔹 Output

* Generated speech will be saved in:

```
output/
```

---

## 🧠 How It Works

1. Provide a reference voice sample
2. Input text
3. Model generates speech in the same voice

---

## ⚠️ Notes

* Ensure checkpoints are correctly placed before running
* Large files (models/datasets) are excluded from GitHub
* Keep input/output files small for better repo performance

---

## 🔮 Future Improvements

* 🌐 Web UI for easier interaction
* 🎯 Better Telugu dataset training
* ⚡ Faster inference optimization

---

## 🙌 Credits

* Based on F5-TTS architecture
* Modified and customized for Telugu voice synthesis
