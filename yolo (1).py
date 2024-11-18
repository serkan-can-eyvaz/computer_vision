import cv2
import numpy as np
import requests
from ultralytics import YOLO

# YOLOv8 modelini yükle (yolunu doğru belirt)
model = YOLO('yolov8n.pt')  # Model dosyasının yolunu doğru yazın

# Görüntü URL'si
image_url = 'https://www.gazetekadikoy.com.tr/Uploads/gazetekadikoy.com.tr/202208121522221-img.jpg'
# Görüntüyü URL'den indir ve OpenCV formatına dönüştür
response = requests.get(image_url)
if response.status_code == 200:
    # URL'den indirilen veriyi numpy dizisine dönüştür
    image_np = np.asarray(bytearray(response.content), dtype="uint8")
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
else:
    print(f"Error: Unable to fetch image from {image_url}")
    exit()

# YOLO ile tahmin yap
results = model.predict(image)

# Sonuçları kullanarak bounding box ve etiketleri görüntü üzerine çiz
annotated_frame = results[0].plot()

# BGR formatına gerek varsa (RGB'den BGR'ye dönüşüm)
annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

# Pencereyi manuel olarak oluşturun ve yeniden boyutlandırılabilir hale getirin
cv2.namedWindow('YOLOv8 Detection', cv2.WINDOW_NORMAL)

# Pencere boyutunu görüntü boyutuna göre ayarlayın
cv2.resizeWindow('YOLOv8 Detection', annotated_frame.shape[1], annotated_frame.shape[0])

# Sonuçları görüntüle
cv2.imshow('YOLOv8 Detection', annotated_frame)

# Bir tuşa basılmasını bekleyin ve pencereyi kapatın
cv2.waitKey(0)
cv2.destroyAllWindows()
