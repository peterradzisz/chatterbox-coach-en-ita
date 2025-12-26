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
import qrcode  # Ensure you ran: pip install qrcode
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from pydub import AudioSegment

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_lock = threading.Lock()

# --- MODELS INITIALIZATION ---
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel

logger.info(f"--- 🚀 STARTING ENGINE ON {device.upper()} ---")
stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="int8_float16")
tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

AUDIO_DIR = "audio"
TUTOR_DIR = "tutors"
CACHE_DIR = "voice_cache" 
for d in [AUDIO_DIR, TUTOR_DIR, CACHE_DIR]: os.makedirs(d, exist_ok=True)

# --- UTILS ---
def get_local_ip():
    """Retrieves the local network IP for mobile access."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_tutor_memory(tutor_name):
    """Memory Hack: Caches speaker embeddings to disk."""
    cache_path = os.path.join(CACHE_DIR, f"{tutor_name}.pt")
    ref_wav_path = os.path.join(TUTOR_DIR, f"{tutor_name}.wav")
    
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)
    
    if os.path.exists(ref_wav_path):
        logger.info(f"🧠 Memory Miss: Learning {tutor_name} reference voice...")
        tts.prepare_conditionals(ref_wav_path, exaggeration=0.7)
        torch.save(tts.conds, cache_path)
        return tts.conds
    return None

# --- FLASK ROUTES ---

@app.route("/")
def index():
    return send_from_directory('.', 'index.html')

@app.route("/get_tutors")
def get_tutors():
    files = [f.replace('.wav', '') for f in os.listdir(TUTOR_DIR) if f.endswith('.wav')]
    return jsonify({"tutors": files})

@app.route("/stt", methods=["POST"])
def speech_to_text():
    temp_path = f"temp_{uuid.uuid4()}.webm"
    try:
        lang = request.form.get('language', 'it')
        audio_file = request.files['audio']
        audio_file.save(temp_path)
        
        segments, _ = stt_model.transcribe(
            temp_path,
            language=lang,
            beam_size=1,
            temperature=0.0,
            condition_on_previous_text=False,
            vad_filter=False
        )
        
        text = "".join([s.text for s in segments]).strip()
        logger.info(f"STT Result: [{text}]")
        return jsonify({"text": text})
    except Exception as e:
        logger.error(f"STT Error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path): os.remove(temp_path)

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    payload = {
        "model": "local-model",
        "messages": [{"role": "system", "content": data.get("system", "")}, {"role": "user", "content": data.get("text", "")}],
        "temperature": 0.8,
        "repeat_penalty": 1.1,
        "stream": True
    }
    @stream_with_context
    def generate():
        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True, timeout=15)
            for line in r.iter_lines():
                if line: yield f"{line.decode('utf-8')}\n\n"
        except Exception as e:
            logger.error(f"LLM Connection Error: {e}")
            yield "data: [DONE]\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route("/tts_sentence", methods=["POST"])
def tts_sentence():
    data = request.json
    raw_text = data.get("text", "").strip()
    tutor_name = data.get("tutor", "sofia")
    
    # --- CUSTOM SOUND CONFIGS (YOUR TUNING) ---
    tuning = {
        "anna": {"exag": 0.4, "cfg": 0.22},
        "sofia": {"exag": 0.7, "cfg": 0.3},
        "froncz": {"exag": 0.4, "cfg": 0.22}
    }
    params = tuning.get(tutor_name.lower(), {"exag": 0.6, "cfg": 0.3})
    
    logger.info(f"TTS Request: [{raw_text}] | Tutor: {tutor_name}")

    it_text = re.split(r'\(|Translation:|\nEnglish:', raw_text)[0].strip()
    it_text = re.sub(r"[^a-zA-Z0-9àèéìòùÀÈÉÌÒÙ',.\?!\s]", ' ', it_text)
    final_text = re.sub(r'\s+', ' ', it_text).strip()

    if not final_text:
        return jsonify({"error": "No translatable text"}), 400
    
    try:
        with gpu_lock:
            tts.conds = load_tutor_memory(tutor_name)
            wav = tts.generate(final_text, language_id="it", cfg_weight=params["cfg"], exaggeration=params["exag"])
            
            if hasattr(wav, "detach"): wav = wav.detach().cpu().numpy()
            wav_norm = (wav.squeeze() * 32767).astype(np.int16)
            
            seg = AudioSegment(wav_norm.tobytes(), frame_rate=24000, sample_width=2, channels=1)
            padding = AudioSegment.silent(duration=100)
            final_audio = padding + seg + padding

        filename = f"{uuid.uuid4()}.wav"
        fpath = os.path.join(AUDIO_DIR, filename)
        final_audio.export(fpath, format="wav")
        
        clear_vram()
        return jsonify({"audio_url": f"/audio/{filename}"})
    except Exception as e:
        logger.error(f"TTS Error: {e}")
        clear_vram()
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<path:fname>")
def serve_audio(fname):
    return send_from_directory(AUDIO_DIR, fname)

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    
    # PRINT QR CODE FOR NETWORK ACCESS
    qr = qrcode.QRCode()
    qr.add_data(f"http://{local_ip}:{port}")
    print("\n" + "█"*50)
    print(f"🌍 SERVER LIVE AT: http://{local_ip}:{port}")
    print("Scan this with your phone to start practicing:")
    qr.print_ascii()
    print("█"*50 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)