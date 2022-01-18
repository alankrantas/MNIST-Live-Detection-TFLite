TF_LITE_MODEL = './mnist.tflite'

import tensorflow as tf
from tensorflow.keras.datasets import mnist

# load MNIST test dataset
(_, _), (x_test, y_test) = mnist.load_data()

# load TF Lite model and inspect input/output shape
print('Loading', TF_LITE_MODEL, '...')
interpreter = tf.lite.Interpreter(model_path=TF_LITE_MODEL)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('input shape:', input_details[0]['shape'])
print('output shape:', output_details[0]['shape'])

# resize the input/output shape to fit the test dataset (so we can do batch prediction)
interpreter.resize_tensor_input(input_details[0]['index'], x_test.shape)
interpreter.resize_tensor_input(output_details[0]['index'], y_test.shape)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print('new input shape:', input_details[0]['shape'])
print('new output shape:', output_details[0]['shape'])
print('')

# make prediction
print('Predicting...')
interpreter.set_tensor(input_details[0]['index'], x_test)
interpreter.invoke()
predicted = interpreter.get_tensor(output_details[0]['index']).argmax(axis=1)
print('')

# inspect metrics
from sklearn.metrics import accuracy_score, mean_squared_error
print('Prediction accuracy:', accuracy_score(y_test, predicted).round(4))
print('Prediction MSE:', mean_squared_error(y_test, predicted).round(4))
print('')

# compare prediction to real labels
from sklearn.metrics import classification_report
print(classification_report(y_test, predicted))

# draw first 40 test digits and their predicted labels
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(6, 3))
for i in range(40):
    ax = fig.add_subplot(4, 10, i + 1)
    ax.set_axis_off()
    ax.set_title(f'{predicted[i]}')
    plt.imshow(x_test[i], cmap='gray')
plt.tight_layout()
plt.savefig('./mnist-model-test.jpg')
plt.show()