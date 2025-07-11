from ultralytics import YOLO
import cv2

# IP kamera dari IP Webcam Android
ip_cam_url = 'http://10.2.147.145:8080/video'  # Ganti sesuai IP kamu
cap = cv2.VideoCapture(ip_cam_url)

# Load model hasil training
model = YOLO('runs/detect/train10/weights/best.pt')

if not cap.isOpened():
    print("Gagal membuka stream dari IP Webcam")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame")
        break

    # Ukuran frame asli
    height, width, _ = frame.shape

    # ROI: area tengah (40% dari frame)
    box_width, box_height = int(width * 0.4), int(height * 0.4)
    x1 = (width - box_width) // 2
    y1 = (height - box_height) // 2
    x2 = x1 + box_width
    y2 = y1 + box_height

    # Ambil ROI dan lakukan deteksi
    roi = frame[y1:y2, x1:x2]
    results = model.predict(source=roi, conf=0.2, show=False, verbose=False)

    # Gambar hasil deteksi pada ROI
    annotated_roi = results[0].plot()
    frame[y1:y2, x1:x2] = annotated_roi

    # Gambar kotak ROI untuk panduan
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, "Area Deteksi", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Perkecil ukuran tampilan jendela
    display_frame = cv2.resize(frame, (1000, 800))
    cv2.imshow("Deteksi Objek di Tengah via IP Webcam", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
