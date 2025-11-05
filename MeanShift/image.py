
import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# -------- Step 1: Load Image --------
# Make sure "building.jpg" is in the same folder OR give full path
img = cv2.imread('building.jpg')
if img is None:
    raise FileNotFoundError("Image not found. Place 'building.jpg' in the project folder.")

# Convert BGR (OpenCV default) to RGB (for matplotlib display)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, c = img.shape

# -------- Step 2: Create Feature Space [R,G,B,X,Y] --------
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([
    img.reshape((-1, 3)),   # Color features (R,G,B)
    X.reshape(-1, 1),       # X-coordinate
    Y.reshape(-1, 1)        # Y-coordinate
], axis=1)

# -------- Step 3: Estimate bandwidth and apply Mean-Shift --------
bandwidth = estimate_bandwidth(features, quantile=0.1, n_samples=500)
print(f"Estimated Bandwidth = {bandwidth:.2f}")

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(features)

labels = ms.labels_
cluster_centers = ms.cluster_centers_

# -------- Step 4: Reconstruct Segmented Image --------
segmented_img = cluster_centers[labels][:, :3].reshape(h, w, 3).astype(np.uint8)

# -------- Step 4.5: Get Number of Clusters --------
num_clusters = len(np.unique(labels))
print(f"Number of clusters found: {num_clusters}")

# -------- Step 5: Show Results --------
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(img)
plt.axis("off")

plt.subplot(1, 2, 2)
plt.title(f"Mean-Shift Segmentation\nClusters: {num_clusters}")
plt.imshow(segmented_img)
plt.axis("off")

plt.tight_layout()
plt.show()