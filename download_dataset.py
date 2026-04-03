import os
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

OUTPUT_DIR = "dataset/wavs"
METADATA_FILE = "dataset/metadata.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("📥 Downloading IndicTTS Telugu dataset (safe method)...")

# Load dataset normally (let HF handle download)
ds = load_dataset("SPRINGLab/IndicTTS_Telugu")

print("✅ Dataset loaded successfully")

metadata_lines = []
counter = 1

for item in tqdm(ds["train"]):

    # Get audio as numpy array directly
    audio_array = item["audio"]["array"]
    sr = item["audio"]["sampling_rate"]

    file_name = f"{counter:05d}.wav"
    save_path = os.path.join(OUTPUT_DIR, file_name)

    # Save wav file
    sf.write(save_path, audio_array, sr)

    text = item["text"]
    metadata_lines.append(f"{file_name}|speaker1|{text}\n")

    counter += 1

with open(METADATA_FILE, "w", encoding="utf-8") as f:
    f.writelines(metadata_lines)

print("🎉 Dataset downloaded successfully!")
print(f"Total files: {counter-1}")