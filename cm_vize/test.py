import os
from ultralytics import YOLO
from sklearn.metrics import classification_report
import numpy as np

# Kendi eğittiğiniz modeli yükleyin
model = YOLO('C:\\Users\\serka\\PycharmProjects\\cm_vize\\FinalSonuc\\train\\weights\\best.pt')  # Eğitilmiş modelin yolunu belirtin

# Test klasörünün yolunu belirleyin
test_images_folder = "C:\\Users\\serka\\PycharmProjects\\cm_vize\\cv_vize2-1\\test\\images"
labels_folder = "C:\\Users\\serka\\PycharmProjects\\cm_vize\\cv_vize2-1\\test\\labels"  # Gerçek etiketlerin olduğu klasör
results_folder = "results"
os.makedirs(results_folder, exist_ok=True)

# Gerçek etiketler ve tahmin edilen etiketleri tutmak için listeler
true_labels = []
predicted_labels = []

# Test klasöründeki her bir görüntüde tahmin yap ve kaydet
for idx, image_file in enumerate(os.listdir(test_images_folder), start=1):
    image_path = os.path.join(test_images_folder, image_file)
    label_path = os.path.join(labels_folder, image_file.replace(".jpg", ".txt"))  # Etiket dosyası

    # Gerçek etiket dosyasının varlığını kontrol edin
    if not os.path.exists(label_path):
        print(f"Warning: Label file for {image_file} not found, skipping this file.")
        continue

    # Gerçek etiketi oku (tek sınıflı örnekler için)
    with open(label_path, 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            print(f"Warning: Empty label file for {image_file}, skipping this file.")
            continue

        # İlk satırdaki sınıf id'sini alıyoruz
        true_label = int(lines[0].split()[0])  # Her satırda ilk değer sınıf id'sidir

    # Tahminleri yap
    results = model.predict(source=image_path, conf=0.45, iou=0.45)
    predicted_classes = results[0].boxes.cls.cpu().numpy().astype(int)

    # Tahmin edilen sınıf varsa
    if len(predicted_classes) > 0:
        predicted_label = predicted_classes[0]  # En yüksek güvene sahip olan sınıf
        # Sadece gerçek etiket ve tahmin varsa listeye ekliyoruz
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
    else:
        print(f"Warning: No prediction for {image_file}, skipping.")

    # Tahmin edilen görüntüyü kaydet
    results[0].save(os.path.join(results_folder, f"prediction_{idx}.jpg"))
    print(f"Processed {image_file} - Results saved as prediction_{idx}.jpg in 'results' folder")

# Gerçek ve tahmin listelerinin uzunluklarını kontrol edin
print(f"Total True Labels: {len(true_labels)}")
print(f"Total Predicted Labels: {len(predicted_labels)}")

# Classification report oluştur
if len(true_labels) == len(predicted_labels):
    print("\nClassification Report:")
    print(classification_report(true_labels, predicted_labels, zero_division=1))
else:
    print("Error: Length mismatch between true labels and predicted labels.")
