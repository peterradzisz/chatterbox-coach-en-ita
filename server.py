import logging
import warnings
import os
import torch
import uuid
import json
import requests
import numpy as np
import re
import threading
import socket
import subprocess
import shutil
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, send_file
from pydub import AudioSegment

# --- SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
AUDIO_DIR = os.path.join(BASE_DIR, "audio")
TUTOR_DIR = os.path.join(BASE_DIR, "tutors")
CACHE_DIR = os.path.join(BASE_DIR, "voice_cache")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
SESSIONS_DIR = os.path.join(BASE_DIR, "sessions")

for d in [AUDIO_DIR, TUTOR_DIR, CACHE_DIR, STATIC_DIR, TEMPLATES_DIR, SESSIONS_DIR]: 
    os.makedirs(d, exist_ok=True)

tts_abort_event = threading.Event()
gpu_lock = threading.Lock()
active_session = {"id": None, "path": None, "audio_dir": None, "history_file": None}

def startup_cleanup():
    logger.info("🧹 STARTUP: Cleaning temp audio...")
    for f in os.listdir(AUDIO_DIR):
        try: os.unlink(os.path.join(AUDIO_DIR, f))
        except: pass
    # Create dummy history.js for the main interface so it doesn't 404
    with open(os.path.join(STATIC_DIR, "history.js"), 'w') as f:
        f.write("window.SESSION_HISTORY = [];")

startup_cleanup()

def wav_to_mp3(wav_path, mp3_path):
    try:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_path, 
                       "-vn", "-acodec", "libmp3lame", "-ab", "160k", mp3_path], check=True)
    except Exception as e:
        logger.error(f"FFmpeg Error: {e}")

def sanitize_filename(text):
    return re.sub(r'[^\w\s-]', '', text).strip().replace(' ', '_')[:30]

def write_history_files(session_path, history_data):
    """Writes both JSON (server) and JS (offline viewer) files."""
    try:
        # 1. JSON
        with open(os.path.join(session_path, "history.json"), 'w', encoding='utf-8') as f:
            json.dump(history_data, f, indent=2)
        # 2. JS (This allows index.html to read history without a running server)
        with open(os.path.join(session_path, "history.js"), 'w', encoding='utf-8') as f:
            f.write(f"window.SESSION_HISTORY = {json.dumps(history_data, indent=2)};")
    except Exception as e:
        logger.error(f"History Save Failed: {e}")

def update_history(session_path, append_entry=None, update_last_text=None, update_last_audio=None):
    """Robust history updater."""
    json_path = os.path.join(session_path, "history.json")
    history = []
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f: history = json.load(f)
        except: pass
    
    if append_entry:
        history.append(append_entry)
    
    if history and (update_last_text is not None or update_last_audio is not None):
        last_msg = history[-1]
        if update_last_text is not None:
            last_msg["text"] = update_last_text
        if update_last_audio:
            if "audio_map" not in last_msg: last_msg["audio_map"] = {}
            for k, v in update_last_audio.items():
                last_msg["audio_map"][k] = v # Key: Text snippet, Value: Relative path
                
    write_history_files(session_path, history)

app = Flask(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from faster_whisper import WhisperModel
stt_model = WhisperModel("large-v3-turbo", device=device, compute_type="int8_float16")
tts = ChatterboxMultilingualTTS.from_pretrained(device=device)

def load_tutor_memory(tutor_name):
    cache_path = os.path.join(CACHE_DIR, f"{tutor_name}.pt")
    ref_wav_path = os.path.join(TUTOR_DIR, f"{tutor_name}.wav")
    if os.path.exists(cache_path): return torch.load(cache_path, map_location=device)
    if os.path.exists(ref_wav_path):
        tts.prepare_conditionals(ref_wav_path, exaggeration=0.7)
        torch.save(tts.conds, cache_path)
        return tts.conds
    return None

# --- ROUTES ---

@app.route("/")
def index(): return send_from_directory(STATIC_DIR, 'index.html')

@app.route("/history.js")
def serve_history_js(): return send_from_directory(STATIC_DIR, 'history.js')

@app.route("/get_tutors")
def get_tutors(): return jsonify({"tutors": [f.replace('.wav', '') for f in os.listdir(TUTOR_DIR) if f.endswith('.wav')]})

@app.route("/get_templates")
def get_templates(): return jsonify({"templates": [f for f in os.listdir(TEMPLATES_DIR) if f.endswith('.txt')]})

@app.route("/template/<path:filename>")
def get_template(filename):
    try:
        with open(os.path.join(TEMPLATES_DIR, filename), 'r', encoding='utf-8') as f:
            return jsonify({"content": f.read()})
    except: return jsonify({"content": ""})

@app.route("/open_session_folder", methods=["POST"])
def open_session_folder():
    target = active_session["path"] if active_session["path"] else SESSIONS_DIR
    target = os.path.abspath(target)
    if os.path.exists(target):
        if os.name == 'nt': os.startfile(target)
        else:
            cmd = 'open' if 'darwin' in os.sys.platform else 'xdg-open'
            subprocess.call([cmd, target])
        return jsonify({"status": "opened"})
    return jsonify({"error": "Path not found"}), 404

@app.route("/reset_session", methods=["POST"])
def reset_session():
    global active_session
    active_session = {"id": None, "path": None, "audio_dir": None, "history_file": None}
    startup_cleanup()
    return jsonify({"status": "reset"})

@app.route("/abort_tts", methods=["POST"])
def abort_tts():
    tts_abort_event.set()
    return jsonify({"status": "aborted"})

@app.route("/stt", methods=["POST"])
def speech_to_text():
    temp = os.path.join(BASE_DIR, f"temp_{uuid.uuid4()}.webm")
    try:
        request.files['audio'].save(temp)
        segments, _ = stt_model.transcribe(temp, language=request.form.get('language', 'it'))
        return jsonify({"text": "".join([s.text for s in segments]).strip()})
    finally:
        if os.path.exists(temp): os.remove(temp)

@app.route("/chat_stream", methods=["POST"])
def chat_stream():
    data = request.json
    user_text = data.get("text", "")
    save_mode = data.get("save_mode", False)

    # 1. Start Session if in Save Mode and not active
    if save_mode and active_session["id"] is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        snippet = sanitize_filename(user_text)
        session_id = f"{ts}_{snippet}"
        session_path = os.path.join(SESSIONS_DIR, session_id)
        session_audio = os.path.join(session_path, "audio")
        os.makedirs(session_audio, exist_ok=True)
        
        # Copy UI
        shutil.copy(os.path.join(STATIC_DIR, "index.html"), os.path.join(session_path, "index.html"))
        # Init JS History
        with open(os.path.join(session_path, "history.js"), 'w') as f: f.write("window.SESSION_HISTORY = [];")
        
        active_session.update({
            "id": session_id, "path": session_path, 
            "audio_dir": session_audio, "history_file": os.path.join(session_path, "history.json")
        })

    # 2. Append User Message
    if save_mode and active_session["path"]:
        update_history(active_session["path"], append_entry={
            "role": "user", "text": user_text, "audio_map": {}
        })
        # 3. Create Placeholder for Assistant (So TTS has a slot to update)
        update_history(active_session["path"], append_entry={
            "role": "assistant", "text": "", "audio_map": {}
        })

    payload = {"model": "local-model", "messages": [{"role": "system", "content": data.get("system", "")}, {"role": "user", "content": user_text}], "temperature": 0.8, "stream": True}
    
    @stream_with_context
    def generate():
        full_response = ""
        try:
            r = requests.post("http://localhost:1234/v1/chat/completions", json=payload, stream=True)
            for line in r.iter_lines():
                if line:
                    decoded = line.decode('utf-8')
                    yield f"{decoded}\n\n"
                    try:
                        json_line = json.loads(decoded.replace('data: ', ''))
                        token = json_line['choices'][0]['delta'].get('content', '')
                        full_response += token
                    except: pass
        except: yield "data: [DONE]\n\n"
        
        # 4. Finalize the Assistant Text (Audio map was populated concurrently)
        if save_mode and active_session["path"] and full_response:
            update_history(active_session["path"], update_last_text=full_response)

    return Response(generate(), mimetype='text/event-stream')

@app.route("/tts_sentence", methods=["POST"])
def tts_sentence():
    tts_abort_event.clear()
    data = request.json
    raw_text = data.get("text", "")
    tutor = data.get("tutor", "sofia")
    lang = data.get("language", "it")

    clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
    clean = re.sub(r'(###|@@@)?\s*(START|STOP|END|END_)\s*(AUDIO|ENGLISH)\s*(###|@@@)?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'[*_#`|>\\-]', ' ', clean)
    final = re.sub(r'\s+', ' ', re.sub(r"[^\w\s',.\?!\-]", ' ', clean)).strip()

    if not final or len(final) < 2: return jsonify({"error": "No content"}), 400
    
    try:
        with gpu_lock:
            if tts_abort_event.is_set(): return jsonify({"error": "Aborted"}), 499
            tts.conds = load_tutor_memory(tutor)
            wav = tts.generate(final, language_id=lang, cfg_weight=0.3, exaggeration=0.6)
            if tts_abort_event.is_set(): return jsonify({"error": "Aborted"}), 499
            
            wav_norm = (wav.squeeze().detach().cpu().numpy() * 32767).astype(np.int16)
            seg = AudioSegment(wav_norm.tobytes(), frame_rate=24000, sample_width=2, channels=1)

        fid = str(uuid.uuid4())
        
        if active_session["audio_dir"]:
            # SAVE MODE
            target_dir = active_session["audio_dir"]
            audio_url = f"/get_session_audio/{active_session['id']}/{fid}.wav"
            offline_path = f"audio/{fid}.mp3" # Relative path for HTML
            
            # Map this specific text block to this audio file
            update_history(active_session["path"], update_last_audio={
                raw_text.strip(): offline_path
            })
        else:
            # LIVE MODE
            target_dir = AUDIO_DIR
            audio_url = f"/audio/{fid}.wav"

        wav_path = os.path.join(target_dir, f"{fid}.wav")
        mp3_path = os.path.join(target_dir, f"{fid}.mp3")
        seg.export(wav_path, format="wav")
        wav_to_mp3(wav_path, mp3_path)

        return jsonify({"audio_url": audio_url})
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route("/get_session_audio/<session_id>/<filename>")
def get_session_audio(session_id, filename):
    return send_from_directory(os.path.join(SESSIONS_DIR, session_id, "audio"), filename)

@app.route("/audio/<path:fname>")
def serve_audio(fname): return send_from_directory(AUDIO_DIR, fname)

@app.route("/download_mp3/<path:filename>")
def download_mp3(filename):
    fname = filename.replace(".wav", ".mp3")
    if active_session["audio_dir"]:
        p = os.path.join(active_session["audio_dir"], fname)
        if os.path.exists(p): return send_file(p, as_attachment=True, download_name=fname)
    p = os.path.join(AUDIO_DIR, fname)
    if os.path.exists(p): return send_file(p, as_attachment=True, download_name=fname)
    return "File not found", 404

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)