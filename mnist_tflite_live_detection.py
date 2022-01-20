TF_LITE_MODEL = './mnist.tflite'  # TF Lite model
IMG_W = 640  # video capture width
IMG_H = 480  # video capture height
IMG_BORDER = 40  # video capture border width (won't be used for detection)
DETECT_THRESHOLD = 0.7  # only display digits with 70%+ probability
CONTOUR_COLOR = (0, 255, 255)  # digit frame color (BGR)
LABEL_COLOR = (255, 255, 0)  # digit label color (BGR)
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
INPUT_SHAPE = (input_details[0]['shape'][2], input_details[0]['shape'][1])

# kernel for morphological closing
MORPH_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMG_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMG_H)

while cap.isOpened():
    # get one frame
    success, frame = cap.read()
    
    # convert to gray
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # image thresholding (to black and white)
    _, frame_binary = cv2.threshold(frame_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    # do morphological closing to filter out noise
    frame_binary = cv2.morphologyEx(frame_binary, cv2.MORPH_CLOSE, MORPH_KERNEL)
    
    # find contours (possible digits area) in the frame
    contours, _ = cv2.findContours(frame_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # iterate all contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        # if the area is overlapping the border, ignore it
        if x < IMG_BORDER or x + w > (IMG_W - 1) - IMG_BORDER or y < IMG_BORDER or y + h > (IMG_H - 1) - IMG_BORDER:
            continue
        
        # if the area is too small or too large, ignore it
        if w < INPUT_SHAPE[0] // 2 or h < INPUT_SHAPE[1] // 2 or w > IMG_W // 2 or h > IMG_H // 2:
            continue
        
        # get the image from the area
        img = frame_binary[y: y + h, x: x + w]
        
        # add padding to make the image square with extra border
        r = max(w, h)
        y_pad = ((w - h) // 2 if w > h else 0) + r // 5
        x_pad = ((h - w) // 2 if h > w else 0) + r // 5
        img = cv2.copyMakeBorder(img, top=y_pad, bottom=y_pad, left=x_pad, right=x_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        
        # resize image to input size
        img = cv2.resize(img, INPUT_SHAPE, interpolation=cv2.INTER_AREA)

        # make prediction
        interpreter.set_tensor(input_details[0]['index'], np.expand_dims(img, axis=0))
        interpreter.invoke()
        predicted = interpreter.get_tensor(output_details[0]['index']).flatten()
        
        # get label and probability
        label = predicted.argmax(axis=0)
        prob = predicted[label]
        
        # ignore it if probability is below the threshold
        if prob < DETECT_THRESHOLD:
            continue
        
        # draw rectangle and text label around the image area
        cv2.rectangle(frame, (x, y), (x + w, y + h), CONTOUR_COLOR, 2)
        cv2.putText(frame, str(label), (x + w // 5, y - h // 5), cv2.FONT_HERSHEY_COMPLEX, LABEL_SIZE, LABEL_COLOR, 2)
    
    # display the frame
    cv2.imshow('MNIST Live Detection', frame)
    
    # exit video capture if user press 'q'
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
