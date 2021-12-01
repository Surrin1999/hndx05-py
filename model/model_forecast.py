import tensorflow as tf
from tensorflow import keras
from data.data_process import dataProcess
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
from data.data_process import get_date, get_hour

batch_size = 2048


def test_handle(test_path):
    tourist_data = pd.read_csv(test_path)
    pd.set_option('display.float_format', lambda x: '%d' % x)
    tourist_data = tourist_data.loc[tourist_data['set_time'].notnull(), :]
    tourist_data.sort_values(by='set_time', ascending=True, inplace=True)
    tourist_data.loc[:, 'set_time'] = tourist_data['set_time'].astype(str).str.replace('\\.0', '', regex=True).str[
                                      4:].str[:-2]
    tourist_data.loc[:, 'date'] = tourist_data.apply(get_date, axis=1)
    tourist_data.loc[:, 'hour'] = tourist_data.apply(get_hour, axis=1)

    tourist_data.drop(['id', 'start_port', 'ship'], axis=1, inplace=True)

    group_result = tourist_data.groupby(['date', 'hour'])

    count_result = pd.DataFrame(columns=['date', 'hour', 'people'])
    for name, group in group_result:
        count_result = count_result.append(pd.Series([name[0], name[1], len(group)], index=['date', 'hour', 'people']),
                                           ignore_index=True)
    peoples = count_result['people']
    test_data = peoples[:240].tolist()
    return np.array(test_data).reshape(1, 240)


class Hndx05Net(keras.Model):

    def __init__(self):
        super(Hndx05Net, self).__init__()
        self.fc = keras.Sequential([
            keras.layers.Dense(64),
            keras.layers.Activation('relu'),
            keras.layers.Dense(32),
            keras.layers.Activation('relu'),
            keras.layers.Dropout(0.4),
            keras.layers.Dense(1),
        ])

    def call(self, inputs, training=None, mask=None):
        return self.fc(inputs)

    def train(self, dataset_train):
        train_data, label = dataset_train
        train_data = train_data.astype(np.float32)
        label = label.astype(np.float32)
        print(train_data)
        print(label)
        length = len(label)
        boundary = length - 500
        (x_train, y_train), (x_valid, y_valid) = (train_data[:boundary], label[:boundary]), (
            train_data[boundary:], label[boundary:])
        (x_train, y_train), (x_valid, y_valid) = (tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train)), (
            tf.convert_to_tensor(x_valid), tf.convert_to_tensor(y_valid))
        train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
        valid_db = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(batch_size)

        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=500,
            decay_rate=0.8)

        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr_schedule), loss=tf.losses.mean_squared_error,
                      metrics=['mae'])
        model.fit(train_db, epochs=1500, validation_data=valid_db, validation_freq=1)

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        self.save_weights(filepath)

    def load_model(self, model):
        network = Hndx05Net()
        network.build((1, 240))
        network.load_weights(model)
        return network

    def test(self, path):
        return self.predict(test_handle(path))


model = Hndx05Net()
