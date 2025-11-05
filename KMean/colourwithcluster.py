
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load image
img = cv2.imread('building.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR → RGB for matplotlib
h, w, c = img.shape

# Use only color features
features = img.reshape((-1, 3))

# Cluster counts to test
cluster_list = [2, 3, 7, 10]

plt.figure(figsize=(12, 8))

for i, k in enumerate(cluster_list, 1):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(features)
    labels = kmeans.labels_.reshape((h, w))
    
    # Map each cluster to its centroid color (so we see colors instead of gray)
    segmented_img = kmeans.cluster_centers_[labels].reshape(h, w, 3).astype(np.uint8)
    
    plt.subplot(2, 2, i)
    plt.imshow(segmented_img)
    plt.title(f"KMeans Segmentation (k={k})")
    plt.axis("off")

plt.tight_layout()
plt.show()