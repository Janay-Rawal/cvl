import cv2
import numpy as np
from sklearn.cluster import KMeans

img = cv2.imread('building.jpg')
h, w, c = img.shape
# Use only BGR color features
features = img.reshape((-1, 3))
# Apply KMeans
kmeans = KMeans(n_clusters=5, random_state=42).fit(features)
labels = kmeans.labels_.reshape((h, w))
# Display result
cv2.imshow('Segmented with Colors only', labels.astype('uint8') * 50)
cv2.waitKey(0)
cv2.destroyAllWindows()