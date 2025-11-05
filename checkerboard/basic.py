import cv2
import numpy as np
import glob
import os

CHECKERBOARD = (7, 6)  
image_folder = 'checkerboard_images'  
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

objpoints = [] 
imgpoints = [] 

images = glob.glob(os.path.join(image_folder, '*.jpg'))

if not images:
    print("❌ No images found in the folder. Please check the path or add images.")
    exit()

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Detected Corners', img)
        cv2.waitKey(200000)

cv2.destroyAllWindows()

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None)


print("\n === CAMERA CALIBRATION RESULTS ===")
print("\n Intrinsic Matrix (Camera Matrix):\n", camera_matrix)
print("\n Distortion Coefficients:\n", dist_coeffs)

for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
    print(f"\n Image {i+1} Extrinsic Parameters:")
    print("  Rotation Vector:\n", rvec)
    print("  Translation Vector:\n", tvec)


mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2)/len(imgpoints2)
    mean_error += error
print(f"\n Mean Reprojection Error: {mean_error/len(objpoints):.4f}")