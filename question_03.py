import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers
import numpy as np
import random

# Set random seeds for reproducibility
random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to the range [0, 1] and cast to float32
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Flatten the images
x_train_flat = x_train.reshape((-1, 28 * 28))
x_test_flat = x_test.reshape((-1, 28 * 28))

# Define the neural network architecture
def create_model(neurons_per_layer=64, activation_function='relu', dropout_rate=0.2, weight_decay=0.001):
    model = Sequential([
        Dense(neurons_per_layer, activation=activation_function, input_shape=(28 * 28,),
              kernel_regularizer=regularizers.l2(weight_decay)),
        Dropout(dropout_rate),
        Dense(10, activation='softmax', kernel_regularizer=regularizers.l2(weight_decay))  # 10 output units for 10 classes
    ])
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model

# Create KerasClassifier for GridSearchCV
model = KerasClassifier(build_fn=create_model, verbose=0)

# Define hyperparameters to tune
param_grid = {
    'neurons_per_layer': [64, 128, 256],
    'activation_function': ['relu', 'tanh', 'sigmoid'],
    'dropout_rate': [0.2, 0.3, 0.4],
    'weight_decay': [0.001, 0.0001]
}

# Learning rate schedule function
def lr_schedule(epoch):
    initial_lr = 0.001
    decay_factor = 0.1
    if epoch < 10:
        return initial_lr
    else:
        return initial_lr * decay_factor ** (epoch // 10)

# Learning rate scheduler callback
lr_scheduler = LearningRateScheduler(lr_schedule)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, verbose=2)
grid_result = grid_search.fit(x_train_flat, y_train, callbacks=[lr_scheduler])

# Summarize results
print("Overall Results:")
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

# Separate results for each activation function
activation_functions = ['relu', 'tanh', 'sigmoid']
for activation_function in activation_functions:
    activation_results = [(mean, std, param) for mean, std, param in zip(means, stds, params) if param['activation_function'] == activation_function]
    print(f"\nResults for {activation_function.capitalize()} activation:")
    for mean, std, param in activation_results:
        print("%f (%f) with: %r" % (mean, std, param))

# Evaluate the best model
test_acc = grid_search.score(x_test_flat, y_test)
print('\nTest accuracy of the best model:', test_acc)
