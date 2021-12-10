"""
海南大学高级软件工程第五组

Author: Surrin1999
Date: 2021-11-25
"""
from flask import Flask
from flask import request
import py_eureka_client.eureka_client as eureka_client

app = Flask(__name__)


def eureka_config():
    eureka_client.init(eureka_server="http://software-eurekaservice:8086/eureka/",
                       app_name="software-forecastservice",
                       instance_host="software-forecastservice",
                       instance_port=8085)


@app.route('/predictFlow')
def predict():
    date = request.args.get("date")
    print(date)
    return "666"


eureka_config()
app.run(host="0.0.0.0", port=8085)
