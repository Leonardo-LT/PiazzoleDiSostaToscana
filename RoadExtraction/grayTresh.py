import cv2
import numpy as np
from matplotlib import pyplot as plt

img_bgr = cv2.imread("/home/leon/Documenti/TestTesi/datasets/smallTest/1/43.9037093814093.10.96658646840066.png")
hsv_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
filtered = cv2.bilateralFilter(hsv_img, d=11, sigmaColor=80, sigmaSpace=80)

lower_bound = np.array([0, 0, 20])
upper_bound = np.array([179, 49, 80])

color_mask = cv2.inRange(filtered, lower_bound, upper_bound)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
clean_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)

plt.figure(figsize=(10, 5))
plt.subplot(121), plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(122), plt.imshow(clean_mask, cmap='gray'), plt.title('HSV Color Mask')
plt.show()