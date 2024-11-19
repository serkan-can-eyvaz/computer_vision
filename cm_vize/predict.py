import cv2
import torch
from ultralytics import YOLO

# YOLO modelini yükledik
model = YOLO('C:\\Users\\serka\\Desktop\\Bilgisayar Görmesi\\cm_vize\\FinalSonuc\\train\\weights\\best.pt')  # Kendi model yolunuza göre değiştirdik

# Webcam başlat
cap = cv2.VideoCapture(0)

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
    results = model.predict(source=frame, conf=0.20, iou=0.45, show=False)

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

            # Reduce bounding box size by a scaling factor
            scale = 0.9
            width = x2 - x1
            height = y2 - y1
            x1 = int(x1 + (1 - scale) * width / 2)
            x2 = int(x2 - (1 - scale) * width / 2)
            y1 = int(y1 + (1 - scale) * height / 2)
            y2 = int(y2 - (1 - scale) * height / 2)

            class_name = class_names[cls] if cls in class_names else str(cls)
            label = f"{class_name}: {score:.2f}"

            # Kutucuk çiz
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Add a background rectangle
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            label_x1 = x1
            label_y1 = y1 - 15
            label_x2 = x1 + label_size[0]
            label_y2 = y1
            cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)


            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)  # Red text

    # Increase the frame size for display
    display_frame = cv2.resize(frame, None, fx=1.5, fy=1.5)  # 1.5x scaling

    # Görüntüyü göster
    cv2.imshow("YOLO Object Tracking", display_frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()