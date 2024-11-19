import cv2
import torch
from ultralytics import YOLO

# Kendi modelimizi  ve YOLO'nun hazır COCO modeliyle eğitilmiş modelini yükledik
custom_model = YOLO('C:\\Users\\serka\\Desktop\\Bilgisayar Görmesi\\cm_vize\\FinalSonuc\\train\\weights\\best.pt')  # Kendi model yolunuza göre değiştirdik
coco_model = YOLO('yolov8n.pt')  # YOLO'nun COCO dataset modeli (yolov8n, yolov8s, yolov8m vb. olabilir)

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

    # İki model ile tahmin yap
    custom_results = custom_model.predict(source=frame, conf=0.30, iou=0.45, show=False)
    coco_results = coco_model.predict(source=frame, conf=0.30, iou=0.45, show=False)

    # Tespit edilen nesneler ve kutucuklar
    detections = []

    # Kendi modelinizin sonuçlarını işledik
    if custom_results[0]:
        custom_boxes = custom_results[0].boxes.xyxy.cpu().numpy()
        custom_scores = custom_results[0].boxes.conf.cpu().numpy()
        custom_classes = custom_results[0].boxes.cls.cpu().numpy().astype(int)
        custom_class_names = custom_model.names

        for box, score, cls in zip(custom_boxes, custom_scores, custom_classes):
            detections.append((box, score, custom_class_names.get(cls, str(cls)), (0, 255, 0)))  # Yeşil kutucuk

    # COCO modelinin sonuçlarını işledik
    if coco_results[0]:
        coco_boxes = coco_results[0].boxes.xyxy.cpu().numpy()
        coco_scores = coco_results[0].boxes.conf.cpu().numpy()
        coco_classes = coco_results[0].boxes.cls.cpu().numpy().astype(int)
        coco_class_names = coco_model.names

        for box, score, cls in zip(coco_boxes, coco_scores, coco_classes):
            detections.append((box, score, coco_class_names.get(cls, str(cls)), (255, 0, 0)))  # Mavi kutucuk

    # Her tespit edilen nesne için kutucuk ve etiket çiz
    for box, score, class_name, color in detections:
        x1, y1, x2, y2 = map(int, box)

        # Sınırlayıcı kutucuk çiz
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Etiketi çiz
        label = f"{class_name}: {score:.2f}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        label_x1 = x1
        label_y1 = y1 - 15
        label_x2 = x1 + label_size[0]
        label_y2 = y1
        cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Görüntüyü ölçeklendirin ve gösterin
    display_frame = cv2.resize(frame, None, fx=1.5, fy=1.5)  # 1.5x ölçeklendirme
    cv2.imshow("YOLO Object Tracking", display_frame)

    # 'q' tuşuna basarak çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()