# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 08:11:41 2020

@author: Amir
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from math import floor
import seaborn as sns
from datetime import datetime
import tensorflow as tf

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['font.size'] = 14
mpl.rcParams['axes.grid'] = False


def lstm_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
    
    # This function converts the data to a form usable by the LSTM model
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
      end_index = len(dataset) - target_size
      
    for i in range(start_index, end_index):
      indices = range(i-history_size, i, step)
      data.append(np.reshape(dataset[indices], (history_size, 1)))
      if single_step:
        labels.append(target[i+target_size])
      else:
        labels.append(target[i:i+target_size])
      
    return np.array(data), np.array(labels)


def multi_step_plot(history, true_future, prediction):
    
    # This function plots predictions of the models versus actual values
    step = 1  
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.plot(num_in, np.array(history), '.-', label='Past')
    plt.plot(np.arange(num_out)/step, np.array(true_future), 'bo',
             label='Actual Future')
    if prediction.any():
      plt.plot(np.arange(num_out)/step, np.array(prediction), 'ro',
               label='Predicted Future')
    plt.legend(loc='upper left')
    plt.ylabel('Glucose value')
    plt.xticks(np.arange(-len(history) + 1, num_out + 1, step = num_out), ('-1.5 h', '-1 h', '-30 min', '0', '+30 min'))
    plt.show()


def create_time_steps(length):
    return list(range(-length, 0))


def baseline(history):
    return history[-1]

#%% Data preparation

file = 'HDeviceCGM.txt'
i = 0

df = []
# reading data line by line
with open(file,'r') as f:
    for line in f :
        if i < 1000000:
            line = line.strip().split('|')
#            print(line)
            df.append(line)
        i += 1

# Total number of data points
print(i)
df = pd.DataFrame(df)

# cleaning data and creating datetime objects to be used for plotting
new_header = df.iloc[0]
df = df[1:]
df.columns = new_header
df = df[df['DexInternalDtTmDaysFromEnroll'].apply(lambda x: x.isnumeric())]
df['DexInternalDtTmDaysFromEnroll'] = df['DexInternalDtTmDaysFromEnroll'].astype(int)
df['DexInternalTm'] = pd.to_datetime(df['DexInternalTm'],format= '%H:%M:%S' ).dt.time
df['GlucoseValue'] = pd.to_numeric(df['GlucoseValue'])

# Identifying unique device IDs
device_ids = df['ParentHDeviceUploadsID'].unique().tolist()   #find unique values
d = {}
k = 0
for j in device_ids:
    d[j] = df[df['ParentHDeviceUploadsID']==device_ids[k]]
    k = k + 1

for j in d:
    d[j] = d[j].sort_values(['DexInternalDtTmDaysFromEnroll','DexInternalTm'], ascending=[True, True])

# Device selection. Any device ID within "d" may be selected
device_id = '10470'

t = d[device_id]['DexInternalDtTmDaysFromEnroll']
t = t.reset_index()
t = t.drop(['index'], axis=1)
day_ref = t['DexInternalDtTmDaysFromEnroll'][0] - 1  # creating day reference for the day column
d[device_id]['DexInternalDtTmDaysFromEnroll'] = ['2000-01-'+ str(x-day_ref) for x in d[device_id]['DexInternalDtTmDaysFromEnroll']]
data=d[device_id]['GlucoseValue']
data.index = pd.to_datetime(d[device_id]['DexInternalDtTmDaysFromEnroll'].astype(str)+' '+d[device_id]['DexInternalTm'].astype(str))

dat = data.to_frame()

# Plotting entire dataset for one device
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)
plt.plot(dat, label='')
ax.xaxis.set_major_formatter(mpl.dates.DateFormatter('%d'))
plt.ylabel('Glucose value')
plt.xlabel('Days')
plt.title('Device ID: ' + device_id + ', Data points: ' + str(len(t)))
plt.show()

data = data.values
# train_split = floor(len(t)*0.6)
train_split = 5000
data_mean   = data[:train_split].mean()
data_std    = data[:train_split].std()
data        = (data - data_mean)/data_std   # Data standardization

#%% LSTM prediction

# Network parameters
n_past          = 20
n_future        = 6
step            = 1
batch_size      = 200
buffer_size     = 10000
eval_interval   = 200
Epochs          = 12

# Converting data to form usable by the LSTM model
x_train, y_train    = lstm_data(data, data, 0, train_split, n_past, n_future, step)
x_val, y_val        = lstm_data(data, data, train_split, None, n_past, n_future, step)

# print ('Single window of past : {}'.format(x_train[0].shape))  # Internal check for data shape
# print ('\n Targets to predict : {}'.format(y_train[0].shape))

# Data shuffling and batching
train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train = train.cache().shuffle(buffer_size).batch(batch_size).repeat()

val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val = val.batch(batch_size).repeat()

# Creating LSTM model:
lstm_model = tf.keras.models.Sequential()
lstm_model.add(tf.keras.layers.LSTM(20,
                                          return_sequences=True,
                                          input_shape=(n_past, 1))) # hidden layer
# multi_step_model.add(tf.keras.layers.LSTM(50, activation='relu')) # additional hidden layer
lstm_model.add(tf.keras.layers.Dropout(0.5))                        # network dropout
lstm_model.add(tf.keras.layers.Flatten())                           # Flattening before dense layer
lstm_model.add(tf.keras.layers.Dense(6))                            # Output layer

print(lstm_model.summary())

lstm_model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=1.0), loss='mean_squared_error')

# for x, y in val.take(1):
#   print (lstm_model.predict(x).shape)

# Model fitting
fit_data = lstm_model.fit(train, epochs=Epochs,
                    steps_per_epoch=eval_interval,
                    validation_data=val,
                    validation_steps=50
                    )

# Generating validation plots
for x, y in val.take(1):
   # multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])
   x_real = (x[0] * data_std) + data_mean
   y_real = (y[0] * data_std) + data_mean
   y_pred = (lstm_model.predict(x)[0] * data_std) + data_mean
   multi_step_plot(x_real, y_real, y_pred)

# score = multi_step_model.evaluate(val_mul, steps = 100) % Manual validation check

# Generating loss plot
plt.plot(fit_data.history['loss'], label='Training loss')
plt.plot(fit_data.history['val_loss'], label='Validation loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()
