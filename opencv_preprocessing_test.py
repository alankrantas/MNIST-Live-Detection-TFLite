TEST_FILE = './test.jpg'  # test image file
IMG_BORDER = 40  # image border width (won't be used for finding contours)
CONTOUR_COLOR = (0, 255, 255)  # digit frame color

import cv2
import numpy as np

# kernel for morphological closing
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# load image and get width, height
img = cv2.imread(TEST_FILE, cv2.IMREAD_COLOR)
IMG_W, IMG_H = img.shape[1], img.shape[0]

# convert to gray
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('./01-gray.jpg', img_gray)

# image thresholding (to black and white)
_, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imwrite('./02-binary.jpg', img_binary)

# do morphological closing to filter out noise
img_binary = cv2.morphologyEx(img_binary, cv2.MORPH_CLOSE, MORPH_KERNEL)
cv2.imwrite('./03-binary-morph.jpg', img_binary)

# find contours (possible digits area) in the frame
contours, _ = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# iterate all contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # if the area is overlapping the border, ignore it
    if x < IMG_BORDER or x + w > (IMG_W - 1) - IMG_BORDER or y < IMG_BORDER or y + h > (IMG_H - 1) - IMG_BORDER:
        continue
    
    # draw rectangle around the binary image area
    cv2.rectangle(img_binary, (x, y), (x + w, y + h), (255, 255, 255), 2)
    # draw rectangle around the original image area
    cv2.rectangle(img, (x, y), (x + w, y + h), CONTOUR_COLOR, 2)

# display results

cv2.imshow('Contours on binary image', img_binary)
cv2.imwrite('./04-binary-contours.jpg', img_binary)
cv2.imshow('Contours on original image', img)
cv2.imwrite('./05-original-contours.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
