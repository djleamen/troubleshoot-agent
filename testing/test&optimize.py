import tensorflow as tf 
from tensorflow.keras.datasets import mnist
from sklearn.metrics import accuracy_score, precision_score

# y_true: actual labels, y_pred: agent’s predictions
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='binary')

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')

import time

start_time = time.time()
agent.predict(input_data)
end_time = time.time()

response_time = end_time - start_time
print(f'Response Time: {response_time} seconds')

import tensorflow_model_optimization as tfmot

# Define pruning parameters
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0.0, final_sparsity=0.5, begin_step=0, end_step=1000
    )
}

# Apply pruning to the Sequential model
pruned_model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

# Compile the pruned model
pruned_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Retrain the pruned model to finalize pruning
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
pruned_model.fit(x_train, y_train, epochs=2, validation_data=(x_test, y_test), callbacks=callbacks)

# Strip pruning wrappers to remove pruning-specific layers and metadata
pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

converter = tf.lite.TFLiteConverter.from_keras_model(pruned_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model = converter.convert()

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
rfe = RFE(model, n_features_to_select=5)
rfe = rfe.fit(X_train, y_train)

# Train the model with selected features
X_train_selected = rfe.transform(X_train)
model.fit(X_train_selected, y_train)

for i in range(1000):
    agent.predict(input_data)