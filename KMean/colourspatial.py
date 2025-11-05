
import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys

# Load image
img = cv2.imread('building.jpg')
h, w, c = img.shape

# Prepare features (color + position)
X, Y = np.meshgrid(np.arange(w), np.arange(h))
features = np.concatenate([img.reshape((-1, 3)), X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)

# Run KMeans
kmeans = KMeans(n_clusters=5, random_state=42).fit(features)
labels = kmeans.labels_.reshape((h, w))

# Show segmented image
cv2.imshow('Segmented with XY', labels.astype('uint8') * 50)
cv2.waitKey(0)   # <-- Must press a key inside the image window
cv2.destroyAllWindows()

# Print cluster centers after window is closed
print("\nCluster centers (B, G, R, X, Y):")
for idx, center in enumerate(kmeans.cluster_centers_):
    b, g, r, x, y = center
    print(f"Cluster {idx+1}: Color = (B={b:.2f}, G={g:.2f}, R={r:.2f}), Position ≈ (X={x:.2f}, Y={y:.2f})")

sys.stdout.flush()