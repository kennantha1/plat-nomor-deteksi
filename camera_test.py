import cv2

def test_camera():
    # Coba buka kamera default (biasanya indeks 0)
    # Jika kamu punya beberapa kamera, mungkin perlu mencoba 1, 2, dst.
    cap = cv2.VideoCapture(0)

    # Periksa apakah kamera berhasil dibuka
    if not cap.isOpened():
        print("Error: Tidak dapat membuka kamera. Pastikan kamera terhubung dan tidak digunakan oleh aplikasi lain.")
        print("Coba ganti angka 0 menjadi 1 atau 2 jika ada kamera eksternal.")
        return

    print("Kamera berhasil dibuka. Menekan 'q' untuk keluar.")

    while True:
        # Baca satu frame (gambar) dari kamera
        ret, frame = cap.read()

        # Jika frame tidak berhasil dibaca, hentikan loop
        if not ret:
            print("Gagal mengambil frame dari kamera.")
            break

        # (Opsional) Balik frame secara horizontal jika gambar terbalik
        # frame = cv2.flip(frame, 1)

        # Tampilkan frame di jendela baru
        cv2.imshow('Camera Test (Tekan q untuk keluar)', frame)

        # Tunggu 1 milidetik, dan jika tombol 'q' ditekan, keluar dari loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Setelah loop selesai, bebaskan sumber daya kamera
    cap.release()
    # Tutup semua jendela OpenCV
    cv2.destroyAllWindows()
    print("Kamera ditutup.")

if __name__ == "__main__":
    test_camera()