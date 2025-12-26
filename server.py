import logging
import warnings
import os
import torch
import gc
import uuid
import json
import requests
import soundfile as sf
import numpy as np
import re
import threading
import socket
import shutil
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from pydub import AudioSegment

# --- SETUP & LOGGING ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# --- ABSOLUTE PATH DEFINITIONS ---
# This ensures we always find folders relative to where server.py is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STATIC_DIR = os.path.join(BASE_DIR, 'static')       # Holds index.html, voiceconf.json
AUDIO_DIR = os.path.join(BASE_DIR, "audio")         # Holds generated .wavs
TUTOR_DIR = os.path.join(BASE_DIR, "tutors")        # Holds source .wav voices
CACHE_DIR = os.path.join(BASE_DIR, "voice_cache")   # Holds .pt embeddings
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates") # Holds .txt prompt templates

# Create missing directories automatically
for d in [AUDIO_DIR, TUTOR_DIR, CACHE_DIR, STATIC_DIR, TEMPLATES_DIR]: 
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_lock = threading.Lock()

# --- MODELS ---
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel

logger.info(f"--- 🚀 ENGINE STARTING ON {device.upper()} ---")
stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="int8_float16")
tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

# --- UTILS ---
def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try: s.connect(('8.8.8.8', 1)); IP = s.getsockname()[0]
    except Exception: IP = '127.0.0.1'
    finally: s.close()
    return IP

def clear_vram():
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()

def load_tutor_memory(tutor_name):
    cache_path = os.path.join(CACHE_DIR, f"{tutor_name}.pt")
    ref_wav_path = os.path.join(TUTOR_DIR, f"{tutor_name}.wav")
    if os.path.exists(cache_path): return torch.load(cache_path, map_location=device)
    if os.path.exists(ref_wav_path):
        tts.prepare_conditionals(ref_wav_path, exaggeration=0.7)
        torch.save(tts.conds, cache_path)
        return tts.conds
    return None

def get_voice_params(tutor_name):
    # Loads voiceconf.json from the STATIC folder
    config_path = os.path.join(STATIC_DIR, "voiceconf.json")
    defaults = {"exag": 0.6, "cfg": 0.3}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f: return json.load(f).get(tutor_name.lower(), defaults)
        except: pass
    return defaults

# --- ROUTES ---

@app.route("/")
def index():
    """Serves the main UI."""
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route("/get_tutors")
def get_tutors():
    """Returns list of available voice files."""
    files = [f.replace('.wav', '') for f in os.listdir(TUTOR_DIR) if f.endswith('.wav')]
    return jsonify({"tutors": files})

# --- TEMPLATE ROUTES (FIXED) ---
@app.route("/get_templates")
def get_templates():
    """Returns list of .txt files in the templates folder."""
    try:
        files = [f for f in os.listdir(TEMPLATES_DIR) if f.endswith('.txt')]
        return jsonify({"templates": files})
    except Exception as e:
        logger.error(f"Template Error: {e}")
        return jsonify({"templates": []})

@app.route("/template/<path:filename>")
def get_template_content(filename):
    """Reads the content of a specific template file."""
    try:
        safe_path = os.path.join(TEMPLATES_DIR, filename)
        # Security check to ensure the file is actually inside TEMPLATES_DIR
        if os.path.abspath(safe_path).startswith(os.path.abspath(TEMPLATES_DIR)):
            with open(safe_path, 'r', encoding='utf-8') as f:
                return jsonify({"content": f.read()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
# -------------------------------

@app.route("/clear_audio", methods=["POST"])
def clear_audio():
    """Wipes audio folder (RESET button)."""
    try:
        count = 0
        for f in os.listdir(AUDIO_DIR):
            try: os.unlink(os.path.join(AUDIO_DIR, f)); count += 1
            except: pass
        return jsonify({"success": True})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/stt", methods=["POST"])
def speech_to_text():
    temp_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.webm")
    try:
        audio_file = request.files['audio']
        audio_file.save(temp_path)
        segments, _ = stt_model.transcribe(temp_path, language=request.form.get('language', 'it'), beam_size=1, temperature=0.0)
        text = "".join([s.text for s in segments]).strip()
        return jsonify({"text": text})
    except Exception as e: return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    payload = {
        "model": "local-model",
        "messages": [{"role": "system", "content": data.get("system", "")}, {"role": "user", "content": data.get("text", "")}],
        "temperature": 0.8, "stream": True
    }
    @stream_with_context
    def generate():
        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True, timeout=15)
            for line in r.iter_lines():
                if line: yield f"{line.decode('utf-8')}\n\n"
        except: yield "data: [DONE]\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route("/tts_sentence", methods=["POST"])
def tts_sentence():
    data = request.json
    raw_text = data.get("text", "").strip()
    tutor_name = data.get("tutor", "sofia")
    params = get_voice_params(tutor_name)

    # Clean text
    it_text = re.split(r'\(|Translation:|\nEnglish:', raw_text)[0].strip()
    it_text = re.sub(r"[^a-zA-Z0-9àèéìòùÀÈÉÌÒÙ',.\?!\s]", ' ', it_text)
    final_text = re.sub(r'\s+', ' ', it_text).strip()

    if not final_text: return jsonify({"error": "No text"}), 400
    try:
        with gpu_lock:
            tts.conds = load_tutor_memory(tutor_name)
            wav = tts.generate(final_text, language_id="it", cfg_weight=params["cfg"], exaggeration=params["exag"])
            if hasattr(wav, "detach"): wav = wav.detach().cpu().numpy()
            wav_norm = (wav.squeeze() * 32767).astype(np.int16)
            seg = AudioSegment(wav_norm.tobytes(), frame_rate=24000, sample_width=2, channels=1)
            final_audio = AudioSegment.silent(duration=100) + seg + AudioSegment.silent(duration=100)
        
        filename = f"{uuid.uuid4()}.wav"
        fpath = os.path.join(AUDIO_DIR, filename)
        final_audio.export(fpath, format="wav")
        clear_vram()
        return jsonify({"audio_url": f"/audio/{filename}"})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/audio/<path:fname>")
def serve_audio(fname): return send_from_directory(AUDIO_DIR, fname)

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    print(f"\n🌍 NETWORK ACCESS: http://{local_ip}:{port}\n")
    try:
        import qrcode
        qr = qrcode.QRCode(); qr.add_data(f"http://{local_ip}:{port}"); qr.print_ascii()
    except ImportError: pass
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)