import numpy as np
import cv2
from matplotlib import pyplot as plt

class EdgeDetector:
    def __init__(self):
        # Yatay ve dikey kenar tespiti için filtreler
        self.horizontal_filter = np.array([[-1, 1]])
        self.vertical_filter = np.array([[-1], [1]])

    def apply_filters(self, gray_image):
        # Filtreleri uygulayıp sonuçları topluyoruz
        horizontal_edges = cv2.filter2D(gray_image, -1, self.horizontal_filter)
        vertical_edges = cv2.filter2D(gray_image, -1, self.vertical_filter)
        return horizontal_edges + vertical_edges

#PNG dosyasını oku ve gri tonlamaya çevir
image_path = "deneme.png"
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Kenar tespiti yap
edge_detector = EdgeDetector()
edges = edge_detector.apply_filters(gray_image)

# Sonucu görselleştir
plt.imshow(edges, cmap='gray')
plt.title("Kenar Tespit Sonucu")
plt.axis("off")
plt.show()
