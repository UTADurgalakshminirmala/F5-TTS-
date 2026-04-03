import os

# ==============================
# SET FFMPEG PATH
# ==============================

os.environ["FFMPEG_BINARY"] = r"C:\ffmpeg\bin\ffmpeg.exe"
os.environ["FFPROBE_BINARY"] = r"C:\ffmpeg\bin\ffprobe.exe"

from pydub import AudioSegment
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# ==============================
# IMPORTS
# ==============================

import torch
import soundfile as sf
from pathlib import Path
import re
from f5_tts_local.api import F5TTS

print("🚀 THIS IS GPU APP.PY RUNNING")

# ==============================
# CONFIGURATION
# ==============================

# Text you want the model to speak
TEXT = "మూడు వందల ఏళ్ళ క్రితం ఒక పెద్ద మర్రి చెట్టు కింద ఒక చిన్న విలేజ్ లో జరిగిన కథ ఇది."

# Reference audio (voice you want to clone)
REFERENCE_AUDIO = r"C:\Users\PMF\Documents\Indicf5\input\v1.wav"

# ⚠️ IMPORTANT
# This text must match EXACTLY what is spoken in the reference audio
REFERENCE_TEXT = "తెలుగు కవితలు, తెలుగు పద్యాలు, తెలుగు మాటలు, పాటలు, తెలుగుతనం అంటేనే అమృతం తాగినంత అనుభూతి."

# Model checkpoint
CHECKPOINT_PATH = "checkpoints/model_250000.pt"

# Vocabulary
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
# CHECK REFERENCE AUDIO
# ==============================

if not Path(REFERENCE_AUDIO).exists():
    raise FileNotFoundError(f"⚠️ Reference audio not found: {REFERENCE_AUDIO}")

if len(REFERENCE_TEXT) < 3:
    raise ValueError("❌ Reference text is too short.")

# ==============================
# LOAD F5-TTS MODEL
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

ref_text = REFERENCE_TEXT

wav, sr, _ = model.infer(
    ref_file=REFERENCE_AUDIO,
    ref_text=ref_text,
    gen_text=" "+TEXT,
    remove_silence=True,
    speed=0.5
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