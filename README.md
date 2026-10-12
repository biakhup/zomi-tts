# Zomi (Tedim) Text-to-Speech

Free text-to-speech for **Tedim (Zomi)** — a low-resource language spoken by the Zomi people in Myanmar, India and the worldwide diaspora, including Malaysia.

Commercial TTS tools do not support Tedim. This project gives the Zomi community a free way to hear written Tedim spoken aloud — for Bible reading, news, education and media content creation.

**Maintainer:** Bia Khup — Zomi content creator (YouTube: @biakhup, 8.9K subscribers; @biakhup1 Tedim Bible readings; Facebook: Bia Khup, 12K followers)

## Features

- Real Tedim speech using Meta's MMS model (`facebook/mms-tts-ctd`)
- Normal voice and female-style voice (pitch/formant shift)
- Mobile-friendly web page — type Tedim, tap Speak, download MP3
- Simple API: `POST /api/speak` with `{"text": "...", "voice": "normal" | "female"}`
- Runs free on a CPU (Google Colab, Hugging Face Spaces, Docker)

## Run locally

```bash
pip install -r requirements.txt
python server.py
# open http://localhost:7860
```

First run downloads the model (~145 MB).

## Why this matters

Tedim is an underserved language in AI. There is no Tedim voice in Google, OpenAI or other commercial TTS services. Tools like this help preserve the language and help creators produce Bible, news and educational content for thousands of Zomi speakers.

## License

The code in this repository is released under the MIT License.
The speech model (`facebook/mms-tts-ctd`) is Meta's MMS model and is used under its own license (CC BY-NC 4.0) — it is not covered by this repository's MIT License.
