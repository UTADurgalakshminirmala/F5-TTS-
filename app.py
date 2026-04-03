import os

os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\ffmpeg\bin\ffprobe.exe"

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

import torch
import soundfile as sf
from pathlib import Path
import re
from faster_whisper import WhisperModel
from f5_tts_local.api import F5TTS

print("🚀 THIS IS GPU APP.PY RUNNING")
# ==============================
# CONFIGURATION
# ==============================

TEXT ="తెలుగు కవితలు, తెలుగు పద్యాలు, తెలుగు మాటలు, పాటలు, తెలుగుతనం అంటేనే అమృతం తాగినంత అనుభూతి."
REFERENCE_AUDIO = r"C:\Users\PMF\Documents\Indicf5\input\v1.wav"
CHECKPOINT_PATH = "checkpoints/model_250000.pt"
VOCAB_PATH = "checkpoints/vocab.txt"

# ==============================
# FORCE GPU
# ==============================

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA is not available! Install GPU version of PyTorch.")

device = "cuda"
print(f"🔥 Using device: {device}")
print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================
# LOAD FASTER-WHISPER (BEST FOR TELUGU)
# ==============================

print("🔎 Loading Faster-Whisper large-v3 model...")
asr_model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

def transcribe_reference(audio_path):
    print("📝 Transcribing reference audio...")

    segments, info = asr_model.transcribe(
        audio_path,
        language="te",
        beam_size=5,
        temperature=0
    )

    text = ""
    for segment in segments:
        text += segment.text

    text = text.strip()

    print("📄 Extracted Reference Text:", text)
    return text


# ==============================
# CHECK REFERENCE AUDIO
# ==============================

if not Path(REFERENCE_AUDIO).exists():
    raise FileNotFoundError(f"⚠️ Reference audio not found: {REFERENCE_AUDIO}")

ref_text = transcribe_reference(REFERENCE_AUDIO)

if len(ref_text) < 3:
    raise ValueError("❌ Reference text extraction failed. Check audio quality.")

# ==============================
# LOAD TELUGU F5 MODEL
# ==============================

print("📦 Loading Telugu F5-TTS model...")

model = F5TTS(
    model_type="F5-TTS",
    ckpt_file=CHECKPOINT_PATH,
    vocab_file=VOCAB_PATH,
    device=device
)

print("✅ Model loaded successfully!")
print("🎤 Generating Telugu speech... please wait.")

# ==============================
# GENERATE SPEECH
# ==============================

wav, sr, spect = model.infer(
    ref_file=REFERENCE_AUDIO,
    ref_text=ref_text,
    gen_text=TEXT,
    remove_silence=True,
    speed=0.8 # 🔥 SPEED CONTROL HERE
)

# ==============================
# SAVE OUTPUT (AUTO INCREMENT)
# ==============================

output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

existing_files = list(output_dir.glob("generated_*.wav"))

max_number = 0
for file in existing_files:
    match = re.search(r"generated_(\d+)\.wav", file.name)
    if match:
        num = int(match.group(1))
        max_number = max(max_number, num)

next_number = max_number + 1
output_file = output_dir / f"generated_{next_number}.wav"

sf.write(str(output_file), wav, sr)

print(f"✅ Telugu TTS complete! Saved to: {output_file}")