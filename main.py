import numpy
import numpy as np
from model.model_forecast import model
from data.data_process import dataProcess
import sys

train_file_path = sys.argv[1]
dataset_train = dataProcess(train_file_path)

model.train(dataset_train)

model.save("my_model")

saved_model = model.load_model("my_model")

res_data = saved_model.test('/home/data/test_data.csv')


def get_res(data):
    res_data = saved_model.predict(data)

    return str(np.round_(res_data)).replace("[", "").replace("]", "").replace(".", "")
