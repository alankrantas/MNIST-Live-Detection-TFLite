# MNIST Still/Live Detection using OpenCV, Tensorflow Lite and AutoKeras

![mnist_live_screenshot](https://user-images.githubusercontent.com/44191076/149617350-2b805e9f-4204-4108-a3bf-6e5c6b2eaaee.png)

This example use [AutoKeras](https://autokeras.com/) to train a CNN model with the [MNIST handwriting dataset](https://www.tensorflow.org/datasets/catalog/mnist), convert it to [Tensorflow Lite](https://www.tensorflow.org/lite) version, and use [OpenCV](https://opencv.org/) to do multiple-digits detection/classification, either using a still image or live video. Tested on PC and Raspberry Pi 3B+/4B.

Be noted that the training dataset are consisted of handwritten numbers with certain features, etc. So it's better to use a sharpie on white papers and make the digits as square and clear as possible. Long, thin numbers are likely to get incorrect results. Also the scripts will ignore anything on the border of the image/video and digits that are too big or too small (can be adjusted in the code).

### Testing environment

* Python 3.9.9 (PC); Pytho 3.7.3 (Raspberry Pis)
* AutoKeras 1.0.16 post1 (will install Numpy, Pandas, scikit-learn and Tensorflow etc. if you don't have them)
* Tensorflow 2.5.2
* [TF Lite runtime](https://github.com/google-coral/pycoral/releases/) 2.5.0 post1 (both PC and RPis)
* [OpenCV](https://pypi.org/project/opencv-python/) 4.5.5
* USB/laptop webcam

If you have GPU and installed CUDA, AutoKeras will use it for training.

### Files

```mnist_tflite_trainer.py``` is the model trainer and ```mnist.tflite``` was my result generated from it.

The output of the trainer is as follows (with GPU):

```
Trial 1 Complete [00h 04m 14s]
val_loss: 0.03911824896931648

Best val_loss So Far: 0.03911824896931648
Total elapsed time: 00h 04m 14s
Epoch 1/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.1584 - accuracy: 0.9513
Epoch 2/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0735 - accuracy: 0.9778
Epoch 3/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0616 - accuracy: 0.9809
Epoch 4/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0503 - accuracy: 0.9837
Epoch 5/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0441 - accuracy: 0.9860
Epoch 6/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0414 - accuracy: 0.9864
Epoch 7/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0383 - accuracy: 0.9872
Epoch 8/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0331 - accuracy: 0.9893
Epoch 9/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0325 - accuracy: 0.9893
Epoch 10/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0307 - accuracy: 0.9901 
Epoch 11/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0305 - accuracy: 0.9899
Epoch 12/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0287 - accuracy: 0.9906
Epoch 13/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0258 - accuracy: 0.9917
Epoch 14/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0243 - accuracy: 0.9920
Epoch 15/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0254 - accuracy: 0.9915
Epoch 16/21
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0243 - accuracy: 0.9920 
Epoch 17/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0231 - accuracy: 0.9922 
Epoch 18/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0218 - accuracy: 0.9924 
Epoch 19/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0213 - accuracy: 0.9932
Epoch 20/21
1875/1875 [==============================] - 9s 5ms/step - loss: 0.0226 - accuracy: 0.9927 
Epoch 21/21
1875/1875 [==============================] - 8s 4ms/step - loss: 0.0197 - accuracy: 0.9938

313/313 [==============================] - 1s 3ms/step - loss: 0.0387 - accuracy: 0.9897

Prediction loss: 0.039, accurcy: 98.970%

Model: "model"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28)]          0         
_________________________________________________________________
cast_to_float32 (CastToFloat (None, 28, 28)            0         
_________________________________________________________________
expand_last_dim (ExpandLastD (None, 28, 28, 1)         0         
_________________________________________________________________
normalization (Normalization (None, 28, 28, 1)         3         
_________________________________________________________________
conv2d (Conv2D)              (None, 26, 26, 32)        320       
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 24, 24, 64)        18496     
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 12, 12, 64)        0         
_________________________________________________________________
dropout (Dropout)            (None, 12, 12, 64)        0         
_________________________________________________________________
flatten (Flatten)            (None, 9216)              0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 9216)              0         
_________________________________________________________________
dense (Dense)                (None, 10)                92170     
_________________________________________________________________
classification_head_1 (Softm (None, 10)                0         
=================================================================
Total params: 110,989
Trainable params: 110,986
Non-trainable params: 3
_________________________________________________________________
```

My model is 611 KB and its Lite version is 432 KB.

## Model test

```mnist_tflite_model_test.py``` can be used to test the TF lite model (using the original MNIST test dataset):

```
Loading ./mnist.tflite ...
input shape: [ 1 28 28]
output shape: [ 1 10]
new input shape: [10000    28    28]
new output shape: [10000    10]

Predicting...
Prediction accuracy: 0.9897
Prediction MSE: 0.1737

              precision    recall  f1-score   support

           0       0.99      1.00      0.99       980
           1       0.99      1.00      0.99      1135
           2       0.99      0.98      0.99      1032
           3       0.99      0.99      0.99      1010
           4       0.98      0.99      0.99       982
           5       0.99      0.99      0.99       892
           6       0.99      0.99      0.99       958
           7       0.98      0.99      0.99      1028
           8       0.99      0.99      0.99       974
           9       1.00      0.98      0.99      1009

    accuracy                           0.99     10000
   macro avg       0.99      0.99      0.99     10000
weighted avg       0.99      0.99      0.99     10000
```

![mnist-model-test](https://user-images.githubusercontent.com/44191076/149889067-6f124477-0d80-4b80-972b-136e49f10ab9.jpg)

## Digit detection

```mnist_tflite_detection.py``` is for a single still photo: it will save images to demostrate different preprocessing step, print out digit labels/positions and show the final black-and-white result:

```
Detected digit: [5] at x=71, y=292, w=43, h=54 (100.000%)
Detected digit: [6] at x=176, y=288, w=32, h=47 (100.000%)
Detected digit: [8] at x=373, y=282, w=42, h=43 (99.861%)
Detected digit: [7] at x=267, y=282, w=36, h=52 (99.974%)
Detected digit: [9] at x=473, y=271, w=32, h=57 (99.852%)
Detected digit: [2] at x=279, y=133, w=38, h=52 (99.997%)
Detected digit: [1] at x=186, y=130, w=29, h=60 (99.874%)
Detected digit: [4] at x=471, y=129, w=52, h=55 (100.000%)
Detected digit: [3] at x=378, y=126, w=29, h=55 (100.000%)
Detected digit: [0] at x=79, y=125, w=56, h=56 (100.000%)
```

![05-mnist-detection](https://user-images.githubusercontent.com/44191076/149882061-d969a6cd-912d-46d9-bf13-62c61b385509.jpg)

```mnist_tflite_live_detection.py``` is the live video version using a webcam, which draws the result directly on the original images.

Both script can either use Tensorflow Lite from the standard Tensorflow package or pure TF Lite runtime.

### Note on image thresholding

In the detection script OpenCV will do automatic image thresholding to convert the video frame to black and white, in order to get clean images of digits. For most of the time it works well, as long as you provide a bright and evenly-lighted surface, but you may want to manually control the threshold when needed:

```python
_, frame_binary = cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY_INV)
```
