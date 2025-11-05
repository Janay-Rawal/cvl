import cv2
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

# ---------- Load video ----------
cap = cv2.VideoCapture("Walk.mp4")  # change to 0 for webcam

if not cap.isOpened():
    raise FileNotFoundError("Video not found or unable to open.")

# ---------- Setup Video Writer ----------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter("Walk_segmented.mp4", fourcc, fps, (width, height))

# ---------- Estimate bandwidth once using first frame ----------
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read first frame for bandwidth estimation.")

frame_small = cv2.resize(frame, (120, 120))
features = frame_small.reshape(-1, 3)  # color features only
bandwidth = estimate_bandwidth(features, quantile=0.2, n_samples=500)
if bandwidth is None or bandwidth <= 0:
    bandwidth = 20

print(f"Estimated bandwidth: {bandwidth:.2f}")

# Reset video to first frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    # Resize for faster processing
    frame_small = cv2.resize(frame, (120, 120))
    features = frame_small.reshape(-1, 3)
    
    # Apply Mean Shift
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(features)
    labels = ms.labels_
    cluster_centers = np.uint8(ms.cluster_centers_)
    
    # Map pixels to cluster centers
    segmented = cluster_centers[labels].reshape(frame_small.shape)
    
    # Resize back to original size
    segmented_big = cv2.resize(segmented, (frame.shape[1], frame.shape[0]))
    
    # Calculate number of clusters
    num_clusters = len(np.unique(labels))
    print(f"Frame {frame_count}: Clusters = {num_clusters}")
    
    # Display results
    cv2.putText(segmented_big, f"Bandwidth: {bandwidth:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(segmented_big, f"Clusters: {num_clusters}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Original Video", frame)
    cv2.imshow("Mean-Shift Segmentation", segmented_big)
    
    # Save segmented frame
    out.write(segmented_big)
    
    # Quit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Segmentation video saved as 'Walk_segmented.mp4' with {frame_count} frames processed.")