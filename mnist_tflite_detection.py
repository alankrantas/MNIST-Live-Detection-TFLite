TEST_FILE = './test.jpg'  # test image file
TF_LITE_MODEL = './mnist.tflite'  # TF Lite model
IMG_W = 640  # video capture width
IMG_H = 480  # video capture height
IMG_BORDER = 40  # video capture border width (won't be used for detection)
DETECT_THRESHOLD = 0.7  # only display digits with 70%+ probability 
LABEL_SIZE = 0.7  # digit label size (70%)
RUNTIME_ONLY = True  # use TF Lite runtime instead of Tensorflow

import cv2
import numpy as np

# load TF Lite model
if RUNTIME_ONLY:
    from tflite_runtime.interpreter import Interpreter
    interpreter = Interpreter(model_path=TF_LITE_MODEL)
else:
    import tensorflow as tf
    interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)

# prepare model
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# get input shape
INPUT_SHAPE = input_details[0]['shape'][1:3]

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

img_binary_copy = img_binary.copy()
img_binary_result_copy = img_binary.copy()

# iterate all contours
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    # draw rectangle and text label around the binary image area
    cv2.rectangle(img_binary_copy, (x, y), (x + w, y + h), (255, 255, 255), 2)
    
    # if the area is overlapping the border, ignore it
    if x < IMG_BORDER or x + w > (IMG_W - 1) - IMG_BORDER or y < IMG_BORDER or y + h > (IMG_H - 1) - IMG_BORDER:
        continue
    
    # if the area is too small or too large, ignore it
    if w < INPUT_SHAPE[0] // 2 or h < INPUT_SHAPE[1] // 2 or w > IMG_W // 2 or h > IMG_H // 2:
        continue
    
    # get the image from the area
    img_digit = img_binary[y: y + h, x: x + w]
        
    # add padding to make the image square with extra border
    r = max(w, h)
    y_pad = ((w - h) // 2 if w > h else 0) + r // 5
    x_pad = ((h - w) // 2 if h > w else 0) + r // 5
    img_digit = cv2.copyMakeBorder(img_digit, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
    # resize image to input size
    img_digit = cv2.resize(img_digit, INPUT_SHAPE, interpolation=cv2.INTER_AREA)

    # make prediction
    interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img_digit, axis=0))
    interpreter.invoke()
    predicted = interpreter.get_tensor(output_details[0]['index']).flatten()
        
    # get label and probability
    label = predicted.argmax(axis=0)
    prob = predicted[label]
        
    # ignore it if probability is below the threshold
    if prob < DETECT_THRESHOLD:
        continue
    
    print(f'Detected digit: [{label}] at x={x}, y={y}, w={w}, h={h} ({prob*100:.3f}%)')
    
    # draw rectangle and text label around the (copied) original image area
    cv2.rectangle(img_binary_result_copy, (x, y), (x + w, y + h), (255, 255, 255), 1)
    cv2.putText(img_binary_result_copy, str(label), (x + w // 5, y - h // 5), cv2.FONT_HERSHEY_COMPLEX, LABEL_SIZE, (255, 255, 255), 1)

# display and save results
cv2.imshow('Contours on binary image', img_binary_copy)
cv2.imwrite('./04-binary-contours.jpg', img_binary_copy)
cv2.imshow('MNIST detection result', img_binary_result_copy)
cv2.imwrite('./05-mnist-detection.jpg', img_binary_result_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()
