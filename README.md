# MNIST Live Detection using OpenCV, Tensorflow Lite and AutoKeras

![mnist_live_screenshot](https://user-images.githubusercontent.com/44191076/149617350-2b805e9f-4204-4108-a3bf-6e5c6b2eaaee.png)

This example use [AutoKeras](https://autokeras.com/) to train a CNN model with the [MNIST handwriting dataset](https://www.tensorflow.org/datasets/catalog/mnist), convert it to [Tensorflow Lite](https://www.tensorflow.org/lite) version, and use [OpenCV](https://opencv.org/) to do multiple-digits live detection/classification. Tested on PC and Raspberry Pi 3B+/4B.

Due to the training dataset your digits has to be as square as possible. Long, thin numbers are more likely to get incorrect results.

### Testing environment

* Python 3.9.9 (PC); Pytho 3.7.3 (Raspberry Pis)
* AutoKeras 1.0.16 post1 (will install Numpy, Pandas, scikit-learn and Tensorflow etc. if you don't have them)
* Tensorflow 2.5.2
* [TF Lite runtime](https://github.com/google-coral/pycoral/releases/) 2.5.0 post1 (both PC and RPis)
* [OpenCV](https://pypi.org/project/opencv-python/) 4.5.5

If you have GPU and installed CUDA, AutoKeras will use it for training.

### Files

```opencv_preprocessing_test.py``` use a still image to demostrate the preprocessing effects used in the live detection script. It will display both the original image and the binary image with contours (boxes around possible digits). It also saves images in different step of the process.

![2](https://user-images.githubusercontent.com/44191076/149666600-3eb9e977-34cf-4d1a-8c42-3bd556ffe4e5.png)

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

```mnist_tflite_live_detection.py``` is the main detection script. It can use either Tensorflow Lite from standard Tensorflow package or the pure TF Lite runtime.

### Note on image thresholding

In the detection script OpenCV will do automatic image thresholding to convert the video frame to black and white, in order to get clean images of digits. For most of the time it works well, as long as you provide a bright and evenly-lighted surface, but you may want to manually control the threshold when needed:

```python
_, frame_binary = cv2.threshold(frame_gray, 160, 255, cv2.THRESH_BINARY_INV)
```
