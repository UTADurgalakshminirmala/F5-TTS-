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

print("🚀 THIS IS GPU APP.PY RUNNING")

# ==============================
# CONFIGURATION
# ==============================

#TEXT = 'ఒక గ్రామంలో అందరూ స్వార్థంగా మారారు, సమస్యలు పెరిగాయి. ఒక గురువు కలిసి పని చేయమని నేర్పించాడు. అందరూ సమష్టిగా పనిచేయడంతో సమస్యలు తగ్గిపోయాయి. స్వార్థం కాదు, ఐక్యతే విజయం.'
#TEXT='జీవితంలో కష్టాలు సహజం. కానీ మనం ధైర్యంగా ముందుకు సాగితే విజయాన్ని సాధించగలం. ప్రయత్నమే మన విజయానికి మొదటి అడుగు.'
#TEXT='ఒక్కరికే కాదు, అందరికీ కలిసి పనిచేస్తేనే మంచి ఫలితాలు వస్తాయి. ఐక్యతలోనే శక్తి ఉంది. మనం కలిసి ఉంటే ఏ సమస్యనైనా జయించగలం.'
#TEXT='జీవితంలో ఆనందం అంటే బాహ్య వస్తువులలో లేదు, మనసు ప్రశాంతతలో ఉంది.'
TEXT='ఐక్యతే శక్తి. కలిసి ఉంటే విజయం ఖాయం.'
REFERENCE_AUDIO = r"C:\Users\PMF\Documents\Indicf5\input\garika3.wav"

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

    segments, _ = asr_model.transcribe(
        audio_path,
        language="te",
        beam_size=5,
        vad_filter=True
    )

    text = " ".join([seg.text for seg in segments]).strip()

    print("📄 Whisper Raw Output:")
    print(text)

    # ⚠️ You can manually correct Telugu here
    corrected_text = text

    print("✏️ Corrected Reference Text:")
    print(corrected_text)

    return corrected_text


# ==============================
# CHECK REFERENCE AUDIO
# ==============================

if not Path(REFERENCE_AUDIO).exists():
    raise FileNotFoundError(f"⚠️ Reference audio not found: {REFERENCE_AUDIO}")

ref_text = transcribe_reference(REFERENCE_AUDIO)

if len(ref_text) < 3:
    raise ValueError("❌ Reference text extraction failed.")

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

# ==============================
# GENERATE SPEECH
# ==============================

print("🎤 Generating Telugu speech...")

ref_text = "మన చరిత్రలో ఇది చాలా ముఖ్యమైనది. మిత్రులారా అందరూ వినండి."

wav, sr, _ = model.infer(
    ref_file=REFERENCE_AUDIO,
    ref_text=ref_text,
    gen_text=TEXT,
    remove_silence=False,
    speed=0.95
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

print(f"✅ Telugu TTS complete! Saved to: {output_file}")