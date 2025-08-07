import os
import cv2
import sqlite3
import re
import numpy as np
from datetime import datetime
from playsound import playsound
from flask import Flask, render_template, Response, jsonify, request, send_from_directory
import pytesseract
from ultralytics import YOLO
import subprocess
import threading # Import modul threading
import time
import sys # Import sys for platform detection

# --- KONSTANTA & FUNGSI UTAMA ---
COOLDOWN_SECONDS = 30
DB_TARGET_PLATS = "target_plats.db"
DB_LOGS = "detection_logs.db"
screenshot_folder = "captured_plates"
last_detected_time = {}

# Inisialisasi lock untuk melindungi akses ke last_detected_time
last_detected_time_lock = threading.Lock()

# Muat model YOLO sekali di awal program
# Ganti dengan 'yolov8n.pt' jika belum melatih model kustom
yolo_model = YOLO('runs/detect/train/weights/best.pt')

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Global dictionary to manage active camera streams
# Stores {camera_index: {'cap': cv2.VideoCapture object, 'stop_event': threading.Event object}}
active_camera_streams = {}
camera_lock = threading.Lock() # Global lock for accessing active_camera_streams

def init_target_db():
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS target_plats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plat_nomor TEXT NOT NULL UNIQUE
        )
    ''')
    target_plats_example = ['B 1001 ZZZ', 'B 2156 TOR', 'F 9012 HIJ']
    for plat in target_plats_example:
        try:
            cursor.execute("INSERT INTO target_plats (plat_nomor) VALUES (?)", (plat,))
        except sqlite3.IntegrityError:
            pass # Plat already exists
    conn.commit()
    conn.close()

def init_log_db():
    conn = sqlite3.connect(DB_LOGS)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detection_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME NOT NULL,
            plat_nomor TEXT NOT NULL,
            is_target BOOLEAN NOT NULL,
            screenshot_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_detected_plat(plat_nomor, is_target, screenshot_path=None):
    try:
        conn = sqlite3.connect(DB_LOGS)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT INTO detection_logs (timestamp, plat_nomor, is_target, screenshot_path) VALUES (?, ?, ?, ?)",
                       (timestamp, plat_nomor, is_target, screenshot_path))
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        print(f"Error saat menyimpan log ke database: {e}")

def format_plat(plat_text):
    if not plat_text:
        return ""
    
    plat_text_clean = plat_text.replace(" ", "").upper().strip()
    
    match = re.search(r'([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})', plat_text_clean)
    if match:
        prefix = match.group(1)
        angka = match.group(2)
        suffix = match.group(3)
        suffix = suffix.replace('2', 'Z') # Common OCR error correction
        return f"{prefix} {angka} {suffix}".strip()
    
    if len(plat_text_clean) > 5:
        corrected_plat = ""
        for char in plat_text_clean:
            if char.isdigit() and char != '0' and char != '1': # Assuming 2 might be Z
                corrected_plat += char.replace('2', 'Z')
            else:
                corrected_plat += char
        plat_text_clean = corrected_plat
        
        match = re.search(r'([A-Z]+)(\d+)([A-Z]+)', plat_text_clean)
        if match:
            prefix = match.group(1)
            angka = match.group(2)
            suffix = match.group(3)
            return f"{prefix} {angka} {suffix}".strip()
            
    return ""

def detect_and_recognize_plate(frame):
    results = yolo_model(frame, verbose=False)
    plat_text = ""
    bbox = None
    frame_height, frame_width, _ = frame.shape
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            aspect_ratio = w / float(h) if h > 0 else 0
            min_aspect, max_aspect = 2.0, 5.0 # Typical aspect ratio for license plates
            min_width, min_height = frame_width * 0.1, frame_height * 0.05 # Minimum size for a plate
            if min_aspect <= aspect_ratio <= max_aspect and w >= min_width and h >= min_height:
                cropped_plate = frame[y1:y2, x1:x2]
                # Tesseract configuration for license plates
                custom_config = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(cropped_plate, config=custom_config)
                if text:
                    plat_text = text.strip()
                    bbox = (x1, y1, w, h)
                    return plat_text, bbox
    return "", None

def get_cameras():
    """
    Discovers available cameras and attempts to open them with different backends.
    Returns a list of strings like "Kamera 0", "Kamera 1" for working cameras.
    """
    index = 0
    arr = []
    
    # Release any lingering camera instances before checking
    # This part should ideally not clear active_camera_streams if they are in use by video feeds.
    # For now, we'll keep it as is, but note that it might interrupt active streams.
    with camera_lock:
        for cam_info in list(active_camera_streams.values()): # Iterate over a copy to allow modification
            if cam_info['cap'].isOpened():
                cam_info['stop_event'].set() # Signal to stop
                cam_info['cap'].release()
        active_camera_streams.clear() # Clear the dictionary

    print("Mencari kamera yang tersedia...")
    # List of backends to try, in order of preference
    backends_to_try = [cv2.CAP_DSHOW] if sys.platform == "win32" else [] # DirectShow for Windows
    backends_to_try.append(cv2.CAP_ANY) # Default/Any backend

    # Limit search to avoid infinite loop on some systems or trying too many non-existent cameras
    max_camera_index_to_check = 5 

    while index < max_camera_index_to_check:
        found_at_index = False
        for backend in backends_to_try:
            cap = None
            try:
                cap = cv2.VideoCapture(index + backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        arr.append(f"Kamera {index}")
                        print(f"Kamera {index} ditemukan (backend: {backend}).")
                        found_at_index = True
                        break # Found a working backend for this index, move to next index
                    else:
                        print(f"Kamera {index} ditemukan tapi tidak bisa membaca frame (backend: {backend}).")
                else:
                    print(f"Kamera {index} tidak bisa dibuka (backend: {backend}).")
            except Exception as e:
                print(f"Error saat mencoba Kamera {index} (backend: {backend}): {e}")
            finally:
                if cap is not None:
                    cap.release()
                cap = None # Reset cap for next try
        
        index += 1
    print(f"Ditemukan {len(arr)} kamera yang berfungsi.")
    return arr

# --- Fungsi untuk generator frame ---
def generate_frames(camera_index):
    """
    Generator function to capture frames from the camera and process them.
    Manages camera lifecycle using active_camera_streams and stop_event.
    """
    print(f"Mencoba memulai stream untuk Kamera {camera_index}...")

    # Ensure any previous stream for this index is stopped before starting a new one
    with camera_lock:
        if camera_index in active_camera_streams:
            print(f"Menghentikan stream lama untuk Kamera {camera_index}...")
            active_camera_streams[camera_index]['stop_event'].set()
            time.sleep(0.2) # Give it a moment to stop
            # Clean up the old entry if it hasn't already by its own loop
            if camera_index in active_camera_streams: # Check again if it's still there
                try:
                    active_camera_streams[camera_index]['cap'].release()
                    del active_camera_streams[camera_index]
                except KeyError:
                    pass # Already removed

        cap = None
        stop_event = threading.Event()

        # Try opening camera with different backends if on Windows
        backends_to_try = [cv2.CAP_DSHOW] if sys.platform == "win32" else []
        backends_to_try.append(cv2.CAP_ANY)

        for backend in backends_to_try:
            try:
                cap = cv2.VideoCapture(camera_index + backend)
                if cap.isOpened():
                    break # Successfully opened with this backend
                else:
                    if cap is not None: cap.release() # Release if failed to open
                    cap = None
            except Exception as e:
                print(f"Error saat mencoba membuka Kamera {camera_index} dengan backend {backend}: {e}")
                if cap is not None: cap.release()
                cap = None

        if cap is None or not cap.isOpened():
            print(f"Error: Tidak bisa membuka Kamera {camera_index} dengan backend apapun.")
            # Set stop event immediately if camera can't be opened
            stop_event.set()
            yield (b'--frame\r\n' # Send an empty frame to indicate no video
                    b'Content-Type: image/jpeg\r\n\r\n' + b'' + b'\r\n')
            return # Exit generator

        active_camera_streams[camera_index] = {'cap': cap, 'stop_event': stop_event}
        print(f"Stream untuk Kamera {camera_index} berhasil dimulai.")
    
    # Load target plates from database (same as before)
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute("SELECT plat_nomor FROM target_plats")
    target_plats = {plat_nomor.upper() for (plat_nomor,) in cursor.fetchall()}
    conn.close()

    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)

    while not stop_event.is_set(): # Loop until stop event is set
        success, frame = cap.read()
        if not success:
            if stop_event.is_set(): # Break if stop event was set during read
                break
            # If read fails but not stopped, try to re-open camera (can happen if camera disconnects)
            print(f"Peringatan: Gagal membaca frame dari Kamera {camera_index}. Mencoba membuka kembali...")
            cap.release() # Release current capture object
            cap = None # Reset cap
            
            # Try to re-open with preferred backends
            for backend in backends_to_try:
                try:
                    cap = cv2.VideoCapture(camera_index + backend)
                    if cap.isOpened():
                        print(f"Kamera {camera_index} berhasil dibuka kembali dengan backend {backend}.")
                        break
                    else:
                        if cap is not None: cap.release()
                        cap = None
                except Exception as e:
                    print(f"Error saat mencoba membuka kembali Kamera {camera_index} dengan backend {backend}: {e}")
                    if cap is not None: cap.release()
                    cap = None
            
            if cap is None or not cap.isOpened():
                print(f"Error: Gagal membuka kembali Kamera {camera_index}. Menghentikan stream.")
                stop_event.set() # Force stop if re-opening fails
                break
            time.sleep(0.5) # Wait a bit before retrying read
            continue
        
        #frame = cv2.flip(frame, 1) # Flip frame horizontally for mirror effect

        plat_text, bbox = detect_and_recognize_plate(frame)
        display_text = "Mencari plat nomor..."
        display_color = (0, 255, 255) # Yellow color
        formatted_plat = ""

        if plat_text:
            formatted_plat = format_plat(plat_text)
            if formatted_plat in target_plats:
                current_time = datetime.now()
                # Check cooldown period
                # Gunakan last_detected_time_lock saat membaca dan menulis last_detected_time
                with last_detected_time_lock:
                    if formatted_plat not in last_detected_time or \
                       (current_time - last_detected_time.get(formatted_plat, datetime.min)).total_seconds() > COOLDOWN_SECONDS:
                        display_text = f"*** COCOK! {formatted_plat} ***"
                        display_color = (0, 255, 0) # Green color for match
                        timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(screenshot_folder, f"{formatted_plat.replace(' ', '_')}_{timestamp}.jpg")
                        cv2.imwrite(filename, frame) # Save screenshot
                        log_detected_plat(formatted_plat, True, screenshot_path=filename) # Log detection
                        try:
                            playsound('alert.wav') # Play alert sound
                        except Exception as e:
                            print(f"Error memutar suara: {e}")
                        last_detected_time[formatted_plat] = current_time
                # End of lock usage
            else:
                display_text = f"Plat terdeteksi: {formatted_plat}"
                display_color = (0, 255, 255) # Yellow color for detected but not target
                # log_detected_plat(formatted_plat, False) # Disabled to reduce log spam for non-target plates

        # Draw bounding box and text on frame
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), display_color, 2)
            cv2.putText(frame, formatted_plat if formatted_plat else display_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        else:
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2, cv2.LINE_AA)

        # Encode frame to JPEG and yield it
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            if stop_event.is_set():
                break
            continue
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    
    # Clean up when the loop breaks
    with camera_lock:
        if camera_index in active_camera_streams:
            active_camera_streams[camera_index]['cap'].release()
            del active_camera_streams[camera_index]
            print(f"Stream untuk Kamera {camera_index} dihentikan dan dilepaskan.")


app = Flask(__name__)

# --- Rute Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_cameras')
def get_cameras_route():
    cameras = get_cameras()
    return jsonify(cameras)

@app.route('/video_feed/<int:camera_index>')
def video_feed(camera_index):
    # This route now directly calls generate_frames which handles stopping previous streams
    return Response(generate_frames(camera_index), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video_feed/<int:camera_index>', methods=['POST'])
def stop_video_feed(camera_index):
    """Signals the specified camera stream to stop."""
    with camera_lock:
        if camera_index in active_camera_streams:
            active_camera_streams[camera_index]['stop_event'].set()
            print(f"Sinyal berhenti dikirim ke Kamera {camera_index}.")
            return jsonify({'message': f'Sinyal berhenti dikirim ke Kamera {camera_index}.'}), 200
        else:
            print(f"Kamera {camera_index} tidak aktif atau sudah dihentikan.")
            return jsonify({'message': f'Kamera {camera_index} tidak aktif.'}), 404

@app.route('/open_screenshots_folder')
def open_folder():
    if os.path.exists(screenshot_folder):
        try:
            # For Windows
            os.startfile(screenshot_folder) 
            return 'Folder dibuka', 200
        except AttributeError:
            # For macOS/Linux
            subprocess.Popen(['open', screenshot_folder]) if sys.platform == "darwin" else subprocess.Popen(['xdg-open', screenshot_folder])
            return 'Folder dibuka', 200
    else:
        return 'Folder tidak ditemukan', 404

# --- Rute untuk Halaman Log ---
@app.route('/logs')
def logs_page():
    """Renders the log history page."""
    return render_template('log.html')

@app.route('/get_logs')
def get_logs():
    """Fetches all detection logs from the database."""
    conn = sqlite3.connect(DB_LOGS)
    cursor = conn.cursor()
    # Select all relevant columns, ordered by timestamp descending
    cursor.execute("SELECT id, timestamp, plat_nomor, is_target, screenshot_path FROM detection_logs ORDER BY timestamp DESC")
    logs = cursor.fetchall()
    conn.close()

    log_list = []
    for log in logs:
        log_dict = {
            'id': log[0],
            'timestamp': log[1],
            'plat_nomor': log[2],
            'is_target': bool(log[3]), # Convert integer to boolean
            'screenshot_path': log[4]
        }
        log_list.append(log_dict)
    
    return jsonify(log_list)

@app.route('/delete_logs', methods=['POST'])
def delete_logs():
    """Deletes selected logs and their associated screenshots."""
    data = request.json # Get JSON data from the request body
    ids_to_delete = data.get('ids', [])
    
    if not ids_to_delete:
        return jsonify({'message': 'Tidak ada log yang dipilih untuk dihapus.'}), 400

    conn = sqlite3.connect(DB_LOGS)
    cursor = conn.cursor()
    
    # First, retrieve screenshot paths for the logs to be deleted
    # Using a parameterized query to prevent SQL injection
    cursor.execute("SELECT screenshot_path FROM detection_logs WHERE id IN ({})".format(','.join('?' for _ in ids_to_delete)), ids_to_delete)
    screenshot_paths = cursor.fetchall()
    
    # Delete the actual screenshot files from the server
    for path in screenshot_paths:
        if path[0] and os.path.exists(path[0]):
            try:
                os.remove(path[0])
                print(f"Screenshot dihapus: {path[0]}")
            except OSError as e:
                print(f"Error menghapus screenshot {path[0]}: {e}")
            
    # Then, delete the entries from the database
    cursor.execute("DELETE FROM detection_logs WHERE id IN ({})".format(','.join('?' for _ in ids_to_delete)), ids_to_delete)
    conn.commit()
    conn.close()
    
    return jsonify({'message': f'{len(ids_to_delete)} log berhasil dihapus.'}), 200

@app.route('/captured_plates/<path:filename>')
def serve_screenshot(filename):
    """Serves screenshot files from the 'captured_plates' folder."""
    return send_from_directory(screenshot_folder, filename)

# --- Rute untuk Manajemen Plat Target ---
@app.route('/manage_targets')
def manage_targets_page():
    """Renders the target plate management page."""
    return render_template('manage_targets.html')

@app.route('/get_target_plats')
def get_target_plats():
    """Fetches all target plates from the database."""
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute("SELECT id, plat_nomor FROM target_plats ORDER BY plat_nomor ASC")
    plats = cursor.fetchall()
    conn.close()

    plat_list = []
    for plat in plats:
        plat_dict = {
            'id': plat[0],
            'plat_nomor': plat[1]
        }
        plat_list.append(plat_dict)
    
    return jsonify(plat_list)

@app.route('/add_target_plat', methods=['POST'])
def add_target_plat():
    """Adds a new target plate to the database."""
    data = request.json
    plat_nomor = data.get('plat_nomor', '').strip().upper()

    if not plat_nomor:
        return jsonify({'message': 'Plat Nomor tidak boleh kosong.'}), 400

    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO target_plats (plat_nomor) VALUES (?)", (plat_nomor,))
        conn.commit()
        return jsonify({'message': f'Plat "{plat_nomor}" berhasil ditambahkan.'}), 201
    except sqlite3.IntegrityError:
        return jsonify({'message': f'Plat "{plat_nomor}" sudah ada dalam daftar.'}), 409 # Conflict
    except sqlite3.Error as e:
        return jsonify({'message': f'Error saat menambahkan plat: {e}'}), 500
    finally:
        conn.close()

@app.route('/edit_target_plat', methods=['POST'])
def edit_target_plat():
    """Edits an existing target plate in the database."""
    data = request.json
    plat_id = data.get('id')
    new_plat_nomor = data.get('plat_nomor', '').strip().upper()

    if not plat_id or not new_plat_nomor:
        return jsonify({'message': 'ID Plat dan Plat Nomor baru tidak boleh kosong.'}), 400

    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    try:
        # Check if the new plate number already exists for another ID
        cursor.execute("SELECT id FROM target_plats WHERE plat_nomor = ? AND id != ?", (new_plat_nomor, plat_id))
        if cursor.fetchone():
            return jsonify({'message': f'Plat "{new_plat_nomor}" sudah ada untuk plat lain.'}), 409

        cursor.execute("UPDATE target_plats SET plat_nomor = ? WHERE id = ?", (new_plat_nomor, plat_id))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'message': 'Plat tidak ditemukan atau tidak ada perubahan.'}), 404
        return jsonify({'message': f'Plat dengan ID {plat_id} berhasil diperbarui menjadi "{new_plat_nomor}".'}), 200
    except sqlite3.Error as e:
        return jsonify({'message': f'Error saat mengedit plat: {e}'}), 500
    finally:
        conn.close()

@app.route('/delete_target_plat', methods=['POST'])
def delete_target_plat():
    """Deletes a specific target plate by ID."""
    data = request.json
    plat_id = data.get('id')

    if not plat_id:
        return jsonify({'message': 'ID Plat tidak boleh kosong.'}), 400

    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM target_plats WHERE id = ?", (plat_id,))
        conn.commit()
        if cursor.rowcount == 0:
            return jsonify({'message': 'Plat tidak ditemukan.'}), 404
        return jsonify({'message': f'Plat dengan ID {plat_id} berhasil dihapus.'}), 200
    except sqlite3.Error as e:
        return jsonify({'message': f'Error saat menghapus plat: {e}'}), 500
    finally:
        conn.close()

@app.route('/delete_all_target_plats', methods=['POST'])
def delete_all_target_plats():
    """Deletes all target plates from the database."""
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM target_plats")
        conn.commit()
        return jsonify({'message': 'Semua plat target berhasil dihapus.'}), 200
    except sqlite3.Error as e:
        return jsonify({'message': f'Error saat menghapus semua plat: {e}'}), 500
    finally:
        conn.close()

if __name__ == '__main__':
    init_target_db()
    init_log_db()
    app.run(debug=True)
