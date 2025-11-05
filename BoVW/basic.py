import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from skimage.feature import local_binary_pattern, hog
from tensorflow.keras.datasets import cifar10

# Parameters
n_clusters = 50  # Size of visual vocabulary
radius = 1
n_points = 8 * radius

# Load CIFAR-10 data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train[:500]  # Use a subset for speed
y_train = y_train[:500].flatten()
X_test = X_test[:100]
y_test = y_test[:100].flatten()

# Step 1: Extract descriptors (LBP + HOG) from patches
def extract_descriptors(img):
    img_gray = np.mean(img, axis=2).astype(np.uint8)
    # LBP
    lbp = local_binary_pattern(img_gray, n_points, radius, method='uniform')
    lbp_desc = lbp.flatten()
    # HOG
    hog_desc = hog(img_gray, pixels_per_cell=(8, 8), cells_per_block=(1, 1), feature_vector=True)
    # Combine
    return np.concatenate([lbp_desc, hog_desc])

# Build descriptor list for all images
desc_list = []
for img in X_train:
    desc = extract_descriptors(img)
    desc_list.append(desc)
desc_list = np.array(desc_list)

# Step 2: Cluster descriptors to form visual vocabulary
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(desc_list)

# Step 3: Represent each image as histogram of visual words
def img_to_bovw(img):
    desc = extract_descriptors(img)
    word = kmeans.predict([desc])[0]
    hist = np.zeros(n_clusters)
    hist[word] += 1
    return hist

X_train_bovw = np.array([img_to_bovw(img) for img in X_train])
X_test_bovw = np.array([img_to_bovw(img) for img in X_test])

# Step 4: Train SVM classifier
clf = SVC(kernel='linear')
clf.fit(X_train_bovw, y_train)

# Step 5: Evaluate
y_pred = clf.predict(X_test_bovw)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))