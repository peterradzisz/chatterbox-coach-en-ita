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
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from pydub import AudioSegment

# --- CONFIGURATION & LOGGING ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)

# Absolute Paths (Crucial for Windows)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TUTOR_DIR = os.path.join(BASE_DIR, "tutors")
CACHE_DIR = os.path.join(BASE_DIR, "voice_cache")

# Ensure directories exist
for d in [AUDIO_DIR, TUTOR_DIR, CACHE_DIR, STATIC_DIR]: 
    os.makedirs(d, exist_ok=True)

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
gpu_lock = threading.Lock()

# --- LOAD AI MODELS ---
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel

logger.info(f"--- 🚀 ENGINE STARTING ON {device.upper()} ---")
stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="int8_float16")
tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

# --- UTILITY FUNCTIONS ---
def get_local_ip():
    """Finds the computer's IP address on the local network."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Doesn't actually connect, just gets the route
        s.connect(('8.8.8.8', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def clear_vram():
    """Forces garbage collection to prevent VRAM fragmentation."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def load_tutor_memory(tutor_name):
    """
    Checks for a cached .pt voice embedding. 
    If not found, generates it from the .wav file and saves it.
    """
    cache_path = os.path.join(CACHE_DIR, f"{tutor_name}.pt")
    ref_wav_path = os.path.join(TUTOR_DIR, f"{tutor_name}.wav")
    
    if os.path.exists(cache_path):
        return torch.load(cache_path, map_location=device)
    
    if os.path.exists(ref_wav_path):
        logger.info(f"🧠 Learning Voice Profile: {tutor_name}")
        tts.prepare_conditionals(ref_wav_path, exaggeration=0.7)
        torch.save(tts.conds, cache_path)
        return tts.conds
    return None

# --- API ROUTES ---

@app.route("/")
def index():
    """Serves the main interface from the /static folder."""
    return send_from_directory(STATIC_DIR, 'index.html')

@app.route("/get_tutors")
def get_tutors():
    """Lists available voices in the /tutors folder."""
    files = [f.replace('.wav', '') for f in os.listdir(TUTOR_DIR) if f.endswith('.wav')]
    return jsonify({"tutors": files})

@app.route("/clear_audio", methods=["POST"])
def clear_audio():
    """RESET BUTTON: Deletes all generated audio files."""
    try:
        count = 0
        for f in os.listdir(AUDIO_DIR):
            file_path = os.path.join(AUDIO_DIR, f)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path) # unlink is safer than remove on Windows
                    count += 1
            except Exception as e:
                logger.error(f"Failed to delete {f}: {e}")
        logger.info(f"🧹 Wiped {count} audio files.")
        return jsonify({"success": True, "deleted": count})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/stt", methods=["POST"])
def speech_to_text():
    """Raw STT transcription (No normalization)."""
    temp_path = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.webm")
    try:
        audio_file = request.files['audio']
        audio_file.save(temp_path)
        
        # Beam_size=1 prevents the model from hallucinating corrections
        segments, _ = stt_model.transcribe(
            temp_path, 
            language=request.form.get('language', 'it'), 
            beam_size=1, 
            temperature=0.0
        )
        text = "".join([s.text for s in segments]).strip()
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    """Proxies the request to LM Studio."""
    data = request.json
    payload = {
        "model": "local-model",
        "messages": [
            {"role": "system", "content": data.get("system", "")},
            {"role": "user", "content": data.get("text", "")}
        ],
        "temperature": 0.8,
        "stream": True
    }
    @stream_with_context
    def generate():
        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True, timeout=15)
            for line in r.iter_lines():
                if line: yield f"{line.decode('utf-8')}\n\n"
        except Exception:
            yield "data: [DONE]\n\n"
    return Response(generate(), mimetype='text/event-stream')

@app.route("/tts_sentence", methods=["POST"])
def tts_sentence():
    """Generates audio with voice-specific pitch tuning."""
    data = request.json
    raw_text = data.get("text", "").strip()
    tutor_name = data.get("tutor", "sofia")
    
    # --- VOICE TUNING ---
    tuning = {
        "anna": {"exag": 0.4, "cfg": 0.22},   # Deeper, less metallic
        "sofia": {"exag": 0.7, "cfg": 0.3},   # Standard rich voice
        "froncz": {"exag": 0.4, "cfg": 0.22}
    }
    params = tuning.get(tutor_name.lower(), {"exag": 0.6, "cfg": 0.3})

    # Clean text: remove translations and non-Italian symbols
    it_text = re.split(r'\(|Translation:|\nEnglish:', raw_text)[0].strip()
    # Updated Regex to allow apostrophes
    it_text = re.sub(r"[^a-zA-Z0-9àèéìòùÀÈÉÌÒÙ',.\?!\s]", ' ', it_text)
    final_text = re.sub(r'\s+', ' ', it_text).strip()

    if not final_text:
        return jsonify({"error": "No text to speak"}), 400
    
    try:
        with gpu_lock:
            tts.conds = load_tutor_memory(tutor_name)
            
            # Generate audio
            wav = tts.generate(
                final_text, 
                language_id="it", 
                cfg_weight=params["cfg"], 
                exaggeration=params["exag"]
            )
            
            # Convert float32 -> int16
            if hasattr(wav, "detach"): wav = wav.detach().cpu().numpy()
            wav_norm = (wav.squeeze() * 32767).astype(np.int16)
            
            # Add silence padding to prevent "clicks"
            seg = AudioSegment(wav_norm.tobytes(), frame_rate=24000, sample_width=2, channels=1)
            final_audio = AudioSegment.silent(duration=100) + seg + AudioSegment.silent(duration=100)

        filename = f"{uuid.uuid4()}.wav"
        fpath = os.path.join(AUDIO_DIR, filename)
        final_audio.export(fpath, format="wav")
        
        clear_vram()
        return jsonify({"audio_url": f"/audio/{filename}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<path:fname>")
def serve_audio(fname):
    return send_from_directory(AUDIO_DIR, fname)

if __name__ == "__main__":
    local_ip = get_local_ip()
    port = 5000
    
    print("\n" + "█"*60)
    print(f"🌍 NETWORK ACCESS URL: http://{local_ip}:{port}")
    print("Scan the QR code below to connect via Mobile:")
    
    try:
        import qrcode
        qr = qrcode.QRCode()
        qr.add_data(f"http://{local_ip}:{port}")
        qr.print_ascii()
    except ImportError:
        print("(Install 'qrcode' library to see a QR code here)")
    
    print("█"*60 + "\n")
    
    # host='0.0.0.0' makes it accessible on the network
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)