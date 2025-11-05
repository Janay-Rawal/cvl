import cv2
import numpy as np

IMAGE_PATH = "coins.png"            
orig = cv2.imread(IMAGE_PATH)
if orig is None:
    raise FileNotFoundError(f"Cannot open {IMAGE_PATH}")

gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (9, 9), sigmaX=2, sigmaY=2)
edges = cv2.Canny(blurred, threshold1=50, threshold2=150)


MIN_R, MAX_R = 10, 80
circles = cv2.HoughCircles(
    blurred,                 
    cv2.HOUGH_GRADIENT,  
    dp=1.2,                  
    minDist=MAX_R*0.8,       
    param1=100,              
    param2=30,               
    minRadius=MIN_R,
    maxRadius=MAX_R
)

output = orig.copy()

if circles is not None:
    circles = np.uint16(np.around(circles[0]))
    for (x, y, r) in circles:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)  
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)  
else:
    print("No circles detected.")
cv2.imshow("Input", orig)
cv2.imshow("Edges", edges)
cv2.imshow("Detected Circles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()