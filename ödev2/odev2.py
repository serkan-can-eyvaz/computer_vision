import cv2
import numpy as np
import requests
from ultralytics import YOLO

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# URL'den fotografları indirip inceleyen fonksiyon
def analyze_profile_image(profile_url):
    try:
        # URL'den profil resmini çek
        response = requests.get(profile_url, stream=True)

        if response.status_code == 200:
            img_data = response.content
            img_array = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                return "Resim yüklenemedi"

            # YOLO ile insan olup olmadığını kontrol et
            results = model(img)
            confidence_threshold = 0.5  # Minimum güven eşiği

            # İnsan tespit etme
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    class_id = int(box.cls[0])  # Sınıf kimliği
                    confidence = box.conf[0]  # Güven oranı
                    if model.names[class_id] == 'person' and confidence > confidence_threshold:
                        return f"İNSAN (Güven: {confidence:.2f})"
            return "BASKA BİR VARLIK"
        else:
            return "Resim indirilemedi"
    except Exception as e:
        return f"Hata: {str(e)}"

# Fotograf URL'leri
photo_urls = [
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRhOGMjim8g23tPD_YbNKMQslaVh5Az-8M-YQ&s",
    "https://www.haber49.net/wp-content/uploads/2024/07/AW249240_01.jpg",
    "https://cdn.ntvspor.net/807e92d628324f4fb3d6240f8891619c.jpg?mode=crop&w=940&h=626",
    "https://soihotel.com/wp-content/uploads/2021/09/vodafone-park-otel-soi-hotel.jpg",
    "https://www.indyturk.com/sites/default/files/thumbnails/image/2023/06/25/1161416-1231986484.jpg",
    "https://robbreport.com/wp-content/uploads/2020/02/classic-recreations-hit-man-mustang-03.gif",
    "hhttps://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQmUt4G-_uqxzpcnw5idZ9vI4kutyKxPLml2A&s",
    "https://assets.goal.com/images/v3/blt31afa9776895bf11/9d77b4eaba99bc478f4791011dc4e66d7791ea55.jpg?auto=webp&format=pjpg&width=3840&quality=60",
    "https://gezginyuzlersitesi.com/wp-content/uploads/cin-seddi-1_640x428.jpg",
    "https://media.istockphoto.com/id/491411894/tr/foto%C4%9Fraf/aerial-view-of-capetown-south-africa.jpg?s=612x612&w=0&k=20&c=HjHrqJvD5SV25RKgThPilwj--IHOKZxoJQnhQmeB42Y=",
]

# Her bir fotograf için analiz yap
for url in photo_urls:
    result = analyze_profile_image(url)
    print(f"{url}: {result}")
