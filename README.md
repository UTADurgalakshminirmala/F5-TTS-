# **IndicF5: A Multilingual TTS Fairyteller**


We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **Rasa, IndicTTS, LIMMITS, and IndicVoices-R**.  

IndicF5 supports **11 Indian languages**:  
**Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  

---

## 🚀 Installation
```bash
conda create -n indicf5 python=3.10 -y
conda activate indicf5
pip install git+https://github.com/ai4bharat/indicf5.git
```


## 🎙 Usage
```python
from transformers import AutoModel
import numpy as np
import soundfile as sf

# Load INF5 from Hugging Face
repo_id = "ai4bharat/indicf5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Generate speech
audio = model(
    "नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है, बस इसे सही ताल में जीना आना चाहिए.",
    ref_audio_path="prompts/PAN_F_HAPPY_00001.wav",
    ref_text="ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।"
)

# Normalize and save output
if audio.dtype == np.int16:
    audio = audio.astype(np.float32) / 32768.0
sf.write("samples/namaste.wav", np.array(audio, dtype=np.float32), samplerate=24000)
```

## References

1. https://github.com/SWivid/F5-TTS