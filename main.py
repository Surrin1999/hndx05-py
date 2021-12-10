from model.model_forecast import model
from data.data_process import dataProcess
import sys

train_file_path = 'data/datasets_final.csv'
dataset_train = dataProcess(train_file_path)

model.train(dataset_train)

model.save("my_model")

saved_model = model.load_model("my_model")

res_data = saved_model.test('data/datasets_final.csv')

print(res_data)
