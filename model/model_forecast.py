import tensorflow as tf
import numpy as np
import pandas as pd

batch_size = 2048


def test_handle(test_path):
    tourist_data = pd.read_csv(test_path)
    tourist_data = tourist_data.drop(axis=1, labels=['ship', 'id', 'start_port'])
    pd.set_option('display.float_format', lambda x: '%d' % x)
    tourist_data.loc[:, 'set_time'] = pd.to_datetime(tourist_data.loc[:, 'set_time'].astype(str))
    grouper = pd.Grouper(freq='H', key='set_time')
    test_data = tourist_data.groupby(grouper)
    test_data = test_data.size().reset_index(name='people')
    return np.array(test_data.loc[:239, "people"].tolist()).reshape(1, 240)


class Hndx05Net(tf.keras.Model):

    def __init__(self):
        super(Hndx05Net, self).__init__()
        self.full_connected = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1),
        ])

    def train(self, dataset_train):
        train_data, label = dataset_train
        train_data, label = train_data.astype(np.float32), label.astype(np.float32)
        length = len(label)
        boundary = length - 500
        train_x, train_y = train_data[:boundary], label[:boundary]
        valid_x, valid_y = train_data[boundary:], label[boundary:]

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-4,
            decay_steps=500,
            decay_rate=0.8)

        model.compile(loss=tf.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule))
        model.fit(x=tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size),
                  validation_data=tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(batch_size),
                  epochs=1000)

    def call(self, inputs, mask=None, training=None):
        return self.full_connected(inputs)

    def save(self,
             filepath,
             overwrite=True,
             include_optimizer=True,
             save_format=None,
             signatures=None,
             options=None):
        self.save_weights(filepath)

    def load_model(self, model_name):
        new_model = Hndx05Net()
        new_model.load_weights(model_name)
        return new_model

    def test(self, path):
        return self.predict(test_handle(path))


model = Hndx05Net()
