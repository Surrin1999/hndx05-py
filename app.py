"""
Author: Surrin1999
Date: 2021-11-25
"""
from main import get_res
import numpy as np
from flask import Flask
from flask import request
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)


def set_eureka():
    server_host = "software-forecastservice"
    server_port = 8085
    eureka_client.init(eureka_server="http://software-eurekaservice:8086/eureka/",
                       app_name="software-forecastservice",
                       instance_host=server_host,
                       instance_port=server_port)


@app.route('/predict')
def predict():
    data = request.args.get("data")
    return get_res(np.array(data.split(',')).reshape(1, 240).astype(np.int32))


set_eureka()
app.run(port=8085, host="0.0.0.0")
