import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras import *
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt
from likelihood_regression import normal_loss, normal_rmse
from application import *




learning_rate_init = 0.001 #0.0002
n_epochs = 200
batch_size = 64
model_name = "fullvar_dw_noisy_v5"


def decay_schedule(epoch, lr):
    if (epoch % 1 == 0) and (epoch != 0):
        lr = lr * 1
    return lr


# Load training data
training_path = '../DataGeneration/TrainingData/trainingdata_fullvar_dw_noisy.mat'
inputs, targets = load_thermometry_data(training_path)
#inputs, targets = transform_profiles(inputs, targets)


# Load test data
test_path = '../DataGeneration/TrainingData/testdata_fullvar_dw_noisy.mat'
x_test, y_test = load_thermometry_data(test_path)


# Setup model
n_shots = inputs.shape[2]
n_gridpoints = inputs.shape[1]
input_shape = (n_gridpoints, n_shots)

model = Sequential()
model.add(Conv1D(filters=8, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2', input_shape=input_shape))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Conv1D(filters=24, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
#model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Flatten())

model.add(Dropout(0.5))
model.add(Dense(300, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Dropout(0.5))
model.add(Dense(300, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))


# For the output we use a Weibull distribution (continuous, positive output),
# which has two parameters, k and lambda. Both parameters are positive, 
# whereby we employ a softplus activation. If they had mixed requiredments,
# we would need a lambda-layer for multiple activation functions.
model.add(Dense(2, activation='softplus')) 

# Since we are optimizing a distribution, we employ a custom loss function, 
# where one minimizes the negative log-likelyhood of the distribution.
opt = optimizers.Adam(learning_rate=learning_rate_init)
model.compile(loss=normal_loss, optimizer=opt, metrics=normal_rmse)

print(model.summary())

lr_scheduler = LearningRateScheduler(decay_schedule)

# Train model
print("Fit model on training data")
history = model.fit(
    inputs,
    targets,
    batch_size=batch_size,
    epochs=n_epochs,
    verbose=1,
    callbacks=lr_scheduler,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_split=0.1,
)


save_model(model, "saved_models/" + model_name)

evaluate_model(x_test, y_test, model=model, report_name="saved_models/eval_report_" + model_name, binsize=20)
save_training_history(history.history, "saved_models/training_hist_" + model_name)


# Plot losses
plt.figure(66)

plt.plot(history.history['loss'][1:-1], label='training data')
plt.plot(history.history['val_loss'][1:-1], label='validation data')
plt.grid(True)
plt.ylim((0, 15))
plt.ylabel('loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")

plt.show()