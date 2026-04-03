import whisper

print("Loading model...")
model = whisper.load_model("medium")  # use medium for better accuracy

audio_path = "prabas.wav"

print("Transcribing (forcing Telugu)...")

result = model.transcribe(
    audio_path,
    language="te"   # Force Telugu
)

print("\n===== TRANSCRIPTION RESULT =====")
print(result["text"])