from ultralytics import YOLO
import cv2

# Load model hasil training
model = YOLO('runs/detect/train10/weights/best.pt')

# Inisialisasi kamera (0 untuk webcam laptop)
cap = cv2.VideoCapture(0)

# Cek kamera berhasil terbuka atau tidak
if not cap.isOpened():
    print("Gagal membuka kamera")
    exit()

while True:
    # Baca frame dari kamera
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi objek di frame
    results = model.predict(source=frame, save=False, show=False, conf=0.2)

    # Ambil hasil deteksi (bounding box, class, confidence)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # koordinat kotak
            conf = box.conf[0]  # confidence
            cls = int(box.cls[0])  # class id
            label = f"{model.names[cls]} {conf:.2f}"

            # Gambar bounding box dan label ke frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Tampilkan frame hasil deteksi
    cv2.imshow('YOLOv11 Webcam Detection', frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Tutup kamera dan window
cap.release()
cv2.destroyAllWindows()