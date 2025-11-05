import cv2

# Open the video
cap = cv2.VideoCapture('traffic.mp4')

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object
out = cv2.VideoWriter('output_motion.avi', cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))

# Read two initial frames
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Compute absolute difference between frames
    diff = cv2.absdiff(frame1, frame2)

    # Convert to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply threshold to detect motion
    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)

    # Find contours (regions of motion)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw bounding boxes around moving objects
    for contour in contours:
        if cv2.contourArea(contour) < 500:  # filter small noise
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Resize and display
    display = cv2.resize(frame1, (800, 600))
    cv2.imshow("Motion Segmentation", display)

    # Write the frame to output video
    out.write(frame1)

    # Update frames
    frame1 = frame2
    ret, frame2 = cap.read()
    if not ret or cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()