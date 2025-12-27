import logging
import warnings
import os
import torch
import gc
import uuid
import json
import requests
import numpy as np
import re
import threading
import socket
import subprocess
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from pydub import AudioSegment

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# --- PATHS ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TUTOR_DIR = os.path.join(BASE_DIR, "tutors")
CACHE_DIR = os.path.join(BASE_DIR, "voice_cache")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

for d in [AUDIO_DIR, TUTOR_DIR, CACHE_DIR, STATIC_DIR, TEMPLATES_DIR]: 
    os.makedirs(d, exist_ok=True)

tts_abort_event = threading.Event()
gpu_lock = threading.Lock()

def startup_cleanup():
    logger.info("🧹 STARTUP: Cleaning audio directory...")
    for f in os.listdir(AUDIO_DIR):
        try: os.unlink(os.path.join(AUDIO_DIR, f))
        except: pass
startup_cleanup()

def wav_to_mp3(wav_path, mp3_path):
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, 
                       "-vn", "-acodec", "libmp3lame", "-ab", "160k", mp3_path], check=True)
    except Exception as e:
        logger.error(f"⚠️ FFmpeg Error: {e}")

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ENGINE LOAD ---
logger.info(f"--- 🚀 LOADING ENGINE ({device.upper()}) ---")
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel

stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="int8_float16")
tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

def load_tutor_memory(tutor_name):
    cache_path = os.path.join(CACHE_DIR, f"{tutor_name}.pt")
    ref_wav_path = os.path.join(TUTOR_DIR, f"{tutor_name}.wav")
    if os.path.exists(cache_path): return torch.load(cache_path, map_location=device)
    if os.path.exists(ref_wav_path):
        logger.info(f"🎙️ Encoding Voice: {tutor_name}")
        tts.prepare_conditionals(ref_wav_path, exaggeration=0.7)
        torch.save(tts.conds, cache_path)
        return tts.conds
    return None

# --- ROUTES ---

@app.route("/")
def index(): return send_from_directory(STATIC_DIR, 'index.html')

@app.route("/get_tutors")
def get_tutors(): return jsonify({"tutors": [f.replace('.wav', '') for f in os.listdir(TUTOR_DIR) if f.endswith('.wav')]})

@app.route("/get_templates")
def get_templates(): return jsonify({"templates": [f for f in os.listdir(TEMPLATES_DIR) if f.endswith('.txt')]})

@app.route("/template/<path:filename>")
def get_template(filename):
    with open(os.path.join(TEMPLATES_DIR, filename), 'r', encoding='utf-8') as f: return jsonify({"content": f.read()})

@app.route("/abort_tts", methods=["POST"])
def abort_tts():
    tts_abort_event.set()
    logger.warning("🛑 ABORT: TTS Generation Cancelled")
    return jsonify({"status": "aborted"})

@app.route("/clear_audio", methods=["POST"])
def clear_audio():
    for f in os.listdir(AUDIO_DIR):
        try: os.unlink(os.path.join(AUDIO_DIR, f))
        except: pass
    return jsonify({"success": True})

@app.route("/stt", methods=["POST"])
def speech_to_text():
    temp = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.webm")
    try:
        request.files['audio'].save(temp)
        lang = request.form.get('language', 'it')
        logger.info(f"👂 STT Processing ({lang})...")
        segments, _ = stt_model.transcribe(temp, language=lang, beam_size=1)
        text = "".join([s.text for s in segments]).strip()
        logger.info(f"📝 Heard: \"{text}\"")
        return jsonify({"text": text})
    finally:
        if os.path.exists(temp): os.remove(temp)

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    payload = {"model": "local-model", "messages": [{"role": "system", "content": data.get("system", "")}, {"role": "user", "content": data.get("text", "")}], "temperature": 0.8, "stream": True}
    @stream_with_context
    def generate():
        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True)
            for line in r.iter_lines():
                if line: yield f"{line.decode('utf-8')}\n\n"
        except: yield "data: [DONE]\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route("/tts_sentence", methods=["POST"])
def tts_sentence():
    tts_abort_event.clear()
    data = request.json
    raw_text = data.get("text", "")
    tutor = data.get("tutor", "sofia")
    lang = data.get("language", "it")

    # Cleaning: Removing likely hallucinated translation headers
    clean = re.split(r'\(|Translation:|\nTłumaczenie:|\nEnglish:|\nItalian:', raw_text)[0].strip()
    final = re.sub(r'\s+', ' ', re.sub(r"[^\w\s',.\?!\-]", ' ', clean)).strip()

    if not final: return jsonify({"error": "Empty text"}), 400
    
    try:
        with gpu_lock:
            if tts_abort_event.is_set(): return jsonify({"error": "Aborted"}), 499
            
            logger.info(f"🔊 Generating ({lang}): \"{final[:40]}...\"")
            tts.conds = load_tutor_memory(tutor)
            
            wav = tts.generate(final, language_id=lang, cfg_weight=0.3, exaggeration=0.6)
            
            if tts_abort_event.is_set(): return jsonify({"error": "Aborted"}), 499
            
            wav_norm = (wav.squeeze().detach().cpu().numpy() * 32767).astype(np.int16)
            seg = AudioSegment(wav_norm.tobytes(), frame_rate=24000, sample_width=2, channels=1)

        fname = f"{uuid.uuid4()}"
        wav_path = os.path.join(AUDIO_DIR, f"{fname}.wav")
        mp3_path = os.path.join(AUDIO_DIR, f"{fname}.mp3")
        
        seg.export(wav_path, format="wav")
        wav_to_mp3(wav_path, mp3_path)
        
        logger.info(f"✅ Audio Ready: {fname}.mp3")
        return jsonify({"audio_url": f"/audio/{fname}.wav"})
    except Exception as e:
        logger.error(f"❌ TTS Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/download_mp3/<path:filename>")
def download_mp3(filename):
    # Filename comes in as .wav, swap to .mp3
    mp3_file = os.path.join(AUDIO_DIR, filename.replace(".wav", ".mp3"))
    if not os.path.exists(mp3_file): return "File not found", 404
    return send_file(mp3_file, as_attachment=True, download_name=f"Lesson_{filename.replace('.wav','.mp3')}")

@app.route("/audio/<path:fname>")
def serve_audio(fname): return send_from_directory(AUDIO_DIR, fname)

if __name__ == "__main__":
    ip = socket.gethostbyname(socket.gethostname())
    print(f"\n🌍 SERVER RUNNING AT: http://{ip}:5000\n")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)