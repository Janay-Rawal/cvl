import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Read image
img = cv2.imread('building.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Reshape into 2D array of pixels (single channel)
data = gray.reshape((-1, 1))

# Apply KMeans clustering
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
labels = kmeans.fit_predict(data)

# Replace pixel values with their cluster center values
segmented_img = kmeans.cluster_centers_[labels]
segmented_img = segmented_img.reshape(gray.shape).astype(np.uint8)

# Show results
cv2.imshow("Original Gray", gray)
cv2.imshow("Segmented Gray", segmented_img)

# Plot KMeans clustering in grayscale
plt.figure(figsize=(8, 4))
sample = data[::100].flatten()  # Sample for speed
sample_labels = labels[::100]
plt.scatter(np.arange(len(sample)), sample, c=sample_labels, cmap='tab10', s=10)
plt.xlabel('Pixel Index (sampled)')
plt.ylabel('Gray Value')
plt.title('KMeans Clustering in Grayscale')
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()