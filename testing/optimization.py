"""
Practice activity: Implementing optimization techniques
From Building Intelligent Troubleshooting Agents by Microsoft on Coursera
"""

import os
# Fix for macOS ARM architecture threading issues
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow_model_optimization as tfmot
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist

# Load dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build a simple model using tf_keras directly
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Apply pruning to the model
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000)
}
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the pruned model to finalize pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=callbacks)

# Strip pruning wrappers to remove pruning-specific layers and metadata
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

# Convert the pruned model to a TensorFlow Lite quantized model
converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

# Measure accuracy of the quantized model using the test set
interpreter = tf.lite.Interpreter(model_content=quantized_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

# Evaluate accuracy
correct_predictions = 0
for i in range(len(x_test)):
    input_data = x_test[i:i+1].astype('float32')
    interpreter.set_tensor(input_index, input_data)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    predicted_label = output.argmax()
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / len(x_test)
print(f'Quantized model accuracy: {accuracy * 100:.2f}%')
