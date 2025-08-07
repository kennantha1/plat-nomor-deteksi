import re
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import numpy as np
import sqlite3
from datetime import datetime
import os # Untuk memeriksa file database
import imutils
from playsound import playsound

# --- Konfigurasi Awal ---
# !!! PENTING: Sesuaikan PATH Tesseract jika di Windows !!!
# Contoh: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Hapus tanda '#' di awal baris di atas dan sesuaikan path-nya jika kamu pakai Windows.
# Jika kamu pakai macOS/Linux dan sudah terinstal via brew/apt, baris ini mungkin tidak perlu.

COOLDOWN_SECONDS = 30  # Atur waktu cooldown dalam detik

# --- Database Setup ---
DB_TARGET_PLATS = 'target_plat_nomor.db'
DB_DETECTED_PLATS = 'detected_plat_nomor.db'
DB_LOGS = "detection_logs.db"

import re

import re

def format_plat(plat_text):
    """
    Mengambil plat nomor dari teks yang lebih panjang.
    Mengoreksi kesalahan umum seperti Z vs 2 secara selektif.
    """
    if not plat_text:
        return ""
    
    # Hapus spasi, karakter baris baru, dan ubah ke kapital
    plat_text_clean = plat_text.replace(" ", "").upper().strip()
    
    # Pola Regex yang lebih spesifik: (huruf depan)(angka)(huruf belakang)
    match = re.search(r'([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})', plat_text_clean)
    if match:
        prefix = match.group(1) # Huruf depan (misalnya 'B')
        angka = match.group(2)  # Bagian angka (misalnya '2156')
        suffix = match.group(3) # Huruf belakang (misalnya 'TOR')
        
        # Koreksi kesalahan OCR hanya pada bagian huruf belakang
        # Ganti angka '2' di suffix menjadi 'Z'
        suffix = suffix.replace('2', 'Z')
        
        return f"{prefix} {angka} {suffix}".strip()
    
    # Koreksi untuk kasus yang lebih sederhana, misal 'B1001Z22'
    if len(plat_text_clean) > 5:
        # Coba perbaiki plat_text_clean. Contoh: B1001Z22 -> B1001ZZZ
        corrected_plat = ""
        for char in plat_text_clean:
            if char.isdigit() and char != '0' and char != '1':
                corrected_plat += char.replace('2', 'Z')
            else:
                corrected_plat += char
        plat_text_clean = corrected_plat
        
        # Coba cocokkan lagi dengan regex setelah perbaikan
        match = re.search(r'([A-Z]+)(\d+)([A-Z]+)', plat_text_clean)
        if match:
            prefix = match.group(1)
            angka = match.group(2)
            suffix = match.group(3)
            return f"{prefix} {angka} {suffix}".strip()
            
    return ""

def init_target_db():
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS target_plats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plat_nomor TEXT NOT NULL UNIQUE
        )
    ''')

    # Daftar plat nomor target
    target_plats_example = [
        'B 1001 ZZZ',
        'B 2156 TOR',  # <-- Tambahkan plat nomor baru di sini, pastikan itu string
        'F 9012 HIJ',  # <-- Atau tambahkan lebih banyak lagi
    ]

    for plat in target_plats_example:
        try:
            cursor.execute("INSERT INTO target_plats (plat_nomor) VALUES (?)", (plat,))
        except sqlite3.IntegrityError:
            print(f"Plat nomor '{plat}' sudah ada di database. Dilewati.")

    conn.commit()
    conn.close()
    print(f"Plat nomor target contoh ditambahkan ke {DB_TARGET_PLATS}.")
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
    print(f"Database log diinisialisasi di {DB_LOGS}.")

def init_detected_db():
    conn = sqlite3.connect(DB_DETECTED_PLATS)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS detected_plats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plat_nomor_terdeteksi TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            is_match INTEGER NOT NULL DEFAULT 0 -- 0 for no match, 1 for match
        )
    ''')
    conn.commit()
    conn.close()

def get_target_plats():
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute("SELECT plat_nomor FROM target_plats")
    plats = [row[0] for row in cursor.fetchall()]
    conn.close()
    return set(plats) # Menggunakan set untuk pencarian yang lebih cepat

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

# --- Fungsi untuk Melakukan OCR pada Area Plat Nomor ---
def recognize_plate(image_roi):
    """
    Melakukan pra-pemrosesan gambar ROI (Region of Interest) dan OCR.
    image_roi adalah gambar yang hanya berisi plat nomor.
    """
    if image_roi is None or image_roi.size == 0: # Pastikan ROI tidak kosong
        return ""

    try:
        # Resize gambar untuk meningkatkan akurasi OCR (opsional)
        # Faktor scaling bisa disesuaikan. Memperbesar gambar dapat membantu.
        scale_percent = 200 # Resize ke 200% dari ukuran asli
        width = int(image_roi.shape[1] * scale_percent / 100)
        height = int(image_roi.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_roi = cv2.resize(image_roi, dim, interpolation = cv2.INTER_LINEAR)

        gray = cv2.cvtColor(resized_roi, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better character separation
        # cv2.ADAPTIVE_THRESH_GAUSSIAN_C seringkali lebih baik dari OTSU untuk teks yang tidak rata
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction (optional)
        # thresh = cv2.medianBlur(thresh, 3)
        
        # Konfigurasi Tesseract:
        # --oem 3: Menggunakan model OCR Engine Mode neural network & legacy
        # --psm 7: Page Segmentation Mode untuk single text line (cocok untuk plat)
        # -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789: Hanya izinkan huruf dan angka
        config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        
        text = pytesseract.image_to_string(thresh, config=config)
        
        # Bersihkan teks yang terbaca (hapus spasi, karakter non-alphanumeric, dll.)
        cleaned_text = "".join(filter(str.isalnum, text)).upper()
        return cleaned_text
    except Exception as e:
        # print(f"Error during OCR: {e}") # Debugging
        return ""

# --- Fungsi Deteksi Plat Nomor (Simulasi/Sederhana) ---
# Ini adalah bagian yang paling menantang dan merupakan PENYEDERHANAAN BESAR.
# Untuk ANPR sesungguhnya, ini akan melibatkan deteksi objek berbasis ML (Haar Cascade, YOLO, SSD).
# Untuk demo ini, kita akan mencoba mendekati lokasi plat nomor secara heuristik
# atau menggunakan detektor yang sangat dasar.

# Opsi 1: Menggunakan Haar Cascade (jika kamu punya file XML-nya)
# Contoh file XML bisa dicari di internet: "haarcascade_licence_plate_rus.xml"
# plate_cascade = cv2.CascadeClassifier('haarcascade_licence_plate_rus.xml')
# Jika tidak punya, komentar/hapus baris ini dan gunakan Opsi 2.

def detect_and_recognize_plate(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Pra-pemrosesan gambar untuk meningkatkan akurasi
    # 1. Terapkan Gaussian Blur untuk mengurangi noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Ambil garis tepi menggunakan Canny
    edged = cv2.Canny(gray, 100, 200)
    
    # 3. Temukan kontur plat nomor
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]

    bbox = None
    plate_text = ""
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            
            # Kriteria rasio aspek plat nomor (biasanya 2.5 hingga 5.0)
            if aspect_ratio > 2.5 and aspect_ratio < 5.0:
                plate_roi = gray[y:y + h, x:x + w]
                
                # Tambahkan pra-pemrosesan tambahan pada plat nomor yang sudah dipotong
                # 4. Terapkan thresholding untuk membuat gambar biner (hitam-putih)
                ret, threshold = cv2.threshold(plate_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # 5. Lakukan OCR
                plate_text = pytesseract.image_to_string(threshold, config='--psm 8')
                
                # Hapus karakter yang tidak valid
                plate_text = "".join(e for e in plate_text if e.isalnum() or e.isspace())
                
                bbox = (x, y, w, h)
                break
    
    return plate_text, bbox

# --- Fungsi Utama Sistem ANPR ---
def run_anpr_system(video_source=0):
    conn = sqlite3.connect(DB_TARGET_PLATS)
    cursor = conn.cursor()
    cursor.execute("SELECT plat_nomor FROM target_plats")
    target_plats = {plat_nomor.upper() for (plat_nomor,) in cursor.fetchall()}
    conn.close()

    print(f"Plat nomor target yang akan dicari: {target_plats}")

    last_detected_time_formatted = {}
    screenshot_folder = "captured_plates"
    if not os.path.exists(screenshot_folder):
        os.makedirs(screenshot_folder)
        print(f"Folder screenshot dibuat: {screenshot_folder}")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Error: Tidak bisa membuka kamera.")
        return

    print("Sistem ANPR berjalan. Dekatkan plat nomor ke kamera. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        #frame = cv2.flip(frame, 1)

        display_text = "Mencari plat nomor..."
        display_color = (0, 255, 255)
        bbox = None
        formatted_plat = ""

        plat_text, bbox_from_detection = detect_and_recognize_plate(frame)
        
        if plat_text:
            formatted_plat = format_plat(plat_text)
            
            print(f"Plat terdeteksi (raw): '{plat_text}', diformat: '{formatted_plat}'")

            if bbox_from_detection:
                bbox = bbox_from_detection

            if formatted_plat in target_plats:
                current_time = datetime.now()
                if formatted_plat not in last_detected_time_formatted or \
                   (current_time - last_detected_time_formatted.get(formatted_plat, datetime.min)).total_seconds() > COOLDOWN_SECONDS:

                    display_text = f"*** COCOK! {formatted_plat} ***"
                    display_color = (0, 255, 0)
                    
                    # Log deteksi ke database dengan jalur screenshot
                    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(screenshot_folder, f"{formatted_plat.replace(' ', '_')}_{timestamp}.jpg")
                    cv2.imwrite(filename, frame)
                    print(f"Screenshot disimpan: {filename}")
                    log_detected_plat(formatted_plat, True, screenshot_path=filename)
                    
                    print(f"NOTIFIKASI: Plat nomor cocok: {formatted_plat} pada {current_time.strftime('%H:%M:%S')}")
                    try:
                        playsound('alert.wav')
                    except Exception as e:
                        print(f"Error memutar suara: {e}")
                    last_detected_time_formatted.update({formatted_plat: current_time})
            else:
                display_text = f"Plat terdeteksi: {formatted_plat}"
                display_color = (0, 255, 255)
                # Log deteksi ke database tanpa jalur screenshot
                log_detected_plat(formatted_plat, False)
        
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), display_color, 2)
            cv2.putText(frame, formatted_plat if formatted_plat else display_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        else:
            cv2.putText(frame, display_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, display_color, 2, cv2.LINE_AA)

        cv2.imshow('ANPR System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Sistem ANPR berhenti.")

# Perbarui pemanggilan fungsi inisialisasi
if __name__ == "__main__":
    init_target_db()
    init_log_db() # <-- Panggil fungsi inisialisasi log
    run_anpr_system()