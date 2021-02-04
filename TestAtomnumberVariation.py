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




learning_rate_init = 0.0001
n_epochs = 50
batch_size = 64
model_name = "Nvardw_L100"


def decay_schedule(epoch, lr):
    if (epoch % 1 == 0) and (epoch != 0):
        lr = lr * 1
    return lr


# Load training data
training_path = '../DataGeneration/TrainingData/trainingdata_Nvar_L100.mat'
inputs, targets = load_thermometry_data(training_path)

# Setup model
n_shots = inputs.shape[2]
n_gridpoints = inputs.shape[1]
input_shape = (n_gridpoints, n_shots)

model = Sequential()
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2', input_shape=input_shape))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Conv1D(filters=16, kernel_size=3, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))

model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Conv1D(filters=16, kernel_size=5, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(MaxPooling1D(pool_size=2, strides=1))
model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(200, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))
model.add(Dropout(0.2))
model.add(Dense(200, activation='relu', kernel_initializer='glorot_uniform', kernel_regularizer='l2'))


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
save_training_history(history.history, "saved_models/training_hist_" + model_name)




# Load test data
test_path = '../DataGeneration/TrainingData/testdata_N6500dw_L100.mat'
x_test1, y_test1 = load_thermometry_data(test_path)

test_path = '../DataGeneration/TrainingData/testdata_N8000dw_L100.mat'
x_test2, y_test2 = load_thermometry_data(test_path)

test_path = '../DataGeneration/TrainingData/testdata_N10500dw_L100.mat'
x_test3, y_test3 = load_thermometry_data(test_path)


evaluate_model(x_test1, y_test1, model=model, report_name="saved_models/eval_report_" + model_name)
evaluate_model(x_test2, y_test2, model=model, report_name="saved_models/eval_report_" + model_name)
evaluate_model(x_test3, y_test3, model=model, report_name="saved_models/eval_report_" + model_name)


plt.figure(66)

plt.plot(history.history['loss'][1:-1], label='training data')
plt.plot(history.history['val_loss'][1:-1], label='validation data')
plt.grid(True)
plt.ylabel('loss')
plt.xlabel('No. epoch')
plt.legend(loc="upper right")

plt.show()