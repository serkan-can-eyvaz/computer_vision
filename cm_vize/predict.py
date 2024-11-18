import cv2
import torch
from ultralytics import YOLO

# YOLO modelini yükleyin
model = YOLO('C:\\Users\\serka\\PycharmProjects\\cm_vize\\FinalSonuc\\train\\weights\\best.pt')  # Kendi model yolunuza göre değiştirin

# Webcam başlat
cap = cv2.VideoCapture(0)  # 0, varsayılan webcam'ı kullanır. Başka bir kamera varsa 1 veya 2 deneyin.

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Pencere oluştur
cv2.namedWindow("YOLO Object Tracking", cv2.WINDOW_NORMAL)

while True:
    # Kameradan kareyi al
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # YOLO modeliyle tahmin yap
    results = model.predict(source=frame, conf=0.45, iou=0.45, show=False)

    # Tespit edilen nesneler ve kutucuklar
    detections = results[0]
    if detections:
        boxes = detections.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 koordinatları
        scores = detections.boxes.conf.cpu().numpy()  # Güven skoru
        classes = detections.boxes.cls.cpu().numpy().astype(int)  # Sınıf id'leri

        # Sınıf id'leri sınıf isimlerine dönüştürmek
        class_names = model.names

        # Her tespit edilen nesne için kutucuk ve etiket çiz
        for box, score, cls in zip(boxes, scores, classes):
            x1, y1, x2, y2 = map(int, box)
            class_name = class_names[cls] if cls in class_names else str(cls)
            label = f"{class_name}: {score:.2f}"

            # Kutucuk çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Etiket ekle
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Görüntüyü göster
    cv2.imshow("YOLO Object Tracking", frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
