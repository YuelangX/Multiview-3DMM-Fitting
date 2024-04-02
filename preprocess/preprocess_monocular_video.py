import os
import cv2
import numpy as np

video_path = 'demo_dataset/sample_video.mp4'
save_folder = 'demo_dataset/sample_video'

capture = cv2.VideoCapture(video_path)
frame = 0
while(True):
    _, image = capture.read()
    if image is None:
        break
    image_size = min(image.shape[:2])
    margin_y = int((image.shape[0] - image_size) / 2)
    margin_x = int((image.shape[1] - image_size) / 2)
    image = image[margin_y: image.shape[0]-margin_y, margin_x: image.shape[1]-margin_x]

    os.makedirs(os.path.join(save_folder, 'images', '%04d' % frame), exist_ok=True)
    cv2.imwrite(os.path.join(save_folder, 'images', '%04d/image_00.jpg' % frame), image)

    extrinsic = np.array([[1.0,  0.0,  0.0,  0.0],
                          [0.0, -1.0,  0.0,  0.0],
                          [0.0,  0.0, -1.0,  4.0]])
    intrinsic = np.array([[5000.0,   0.0,      256.0],
                          [0.0,      5000.0,   256.0],
                          [0.0,      0.0,      1.0]])
    os.makedirs(os.path.join(save_folder, 'cameras', '%04d' % frame), exist_ok=True)
    np.savez(os.path.join(save_folder, 'cameras', '%04d/camera_00.npz' % frame), extrinsic=extrinsic, intrinsic=intrinsic)

    frame += 1