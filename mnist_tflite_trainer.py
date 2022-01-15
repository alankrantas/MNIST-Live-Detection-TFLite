TF_LITE_MODEL = './mnist.tflite'
SAVE_KERAS_MODEL = True

import autokeras as ak
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# train a AutoKeras model
clf = ak.ImageClassifier(max_trials=1, overwrite=True)
clf.fit(x_train, y_train)

# evaluate model
loss, accuracy = clf.evaluate(x_test, y_test)
print(f'\nPrediction loss: {loss:.3f}, accurcy: {accuracy*100:.3f}%\n')

# export model
model = clf.export_model()
model.summary()

# save Keras model if needed
if SAVE_KERAS_MODEL:
    model.save('./mnist_model')

# convert to TF Lite model
converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter = tf.lite.TFLiteConverter.from_saved_model('./mnist_model')
tflite_model = converter.convert()

# save TF Lite model
with open(TF_LITE_MODEL, 'wb') as f:
    f.write(tflite_model)