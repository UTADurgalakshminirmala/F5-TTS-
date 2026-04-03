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
from faster_whisper import WhisperModel
from f5_tts_local.api import F5TTS

print("🚀 MULTILINGUAL GPU APP RUNNING")

# ==============================
# CONFIGURATION
# ==============================

TEXT = "हिंदी दिवस पर एक प्रभावशाली भाषण के लिए शुरुआत श्रोताओं के अभिवादन और हिंदी भाषा के महत्व से करें।"

REFERENCE_AUDIO = r"C:\Users\PMF\Documents\Indicf5\input\modihindi.wav"

CHECKPOINT_PATH = "checkpoints/model_250000.pt"
VOCAB_PATH = "checkpoints/vocab.txt"

# ==============================
# FORCE GPU
# ==============================

if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA is not available!")

device = "cuda"

print(f"🔥 Using device: {device}")
print(f"🎮 GPU Name: {torch.cuda.get_device_name(0)}")

# ==============================
# LOAD WHISPER MODEL
# ==============================

print("🔎 Loading Faster-Whisper model...")

asr_model = WhisperModel(
    "large-v3",
    device=device,
    compute_type="float16"
)

# ==============================
# TRANSCRIBE AUDIO
# ==============================

def transcribe_reference(audio_path):

    print("📝 Transcribing reference audio...")

    segments, info = asr_model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True
    )

    detected_lang = info.language

    language_names = {
        "te": "Telugu",
        "hi": "Hindi",
        "en": "English"
    }

    supported_languages = ["te", "hi", "en"]

    if detected_lang not in supported_languages:
        raise ValueError(
            f"❌ Unsupported language detected: {detected_lang}. "
            "Supported languages: Telugu, Hindi, English"
        )

    print(f"🌎 Detected language: {language_names[detected_lang]} ({detected_lang})")

    text = " ".join([seg.text for seg in segments]).strip()

    print("📄 Whisper Raw Output:")
    print(text)

    corrected_text = text

    print("✏️ Corrected Reference Text:")
    print(corrected_text)

    return corrected_text, detected_lang

# ==============================
# CHECK REFERENCE AUDIO
# ==============================

if not Path(REFERENCE_AUDIO).exists():
    raise FileNotFoundError(f"⚠️ Reference audio not found: {REFERENCE_AUDIO}")

ref_text, detected_lang = transcribe_reference(REFERENCE_AUDIO)

if len(ref_text) < 3:
    raise ValueError("❌ Reference text extraction failed.")

# ==============================
# LOAD F5-TTS MODEL
# ==============================

print("📦 Loading F5-TTS model...")

model = F5TTS(
    model_type="F5-TTS",
    ckpt_file=CHECKPOINT_PATH,
    vocab_file=VOCAB_PATH,
    device=device
)

print("✅ Model loaded successfully!")

# ==============================
# GENERATE SPEECH
# ==============================

print(f"🎤 Generating speech using {detected_lang} voice reference...")

wav, sr, _ = model.infer(
    ref_file=REFERENCE_AUDIO,
    ref_text=ref_text,
    gen_text=TEXT,
    remove_silence=True,
    speed=1.0
)

# ==============================
# SAVE OUTPUT
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

output_file = output_dir / f"generated_{max_number + 1}.wav"

sf.write(str(output_file), wav, sr)

print(f"✅ Speech generation complete! Saved to: {output_file}")