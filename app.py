from logging import debug
import os
import joblib

import yaml
import numpy as np
from flask import Flask, render_template, request, jsonify

params_path = "params.yaml"
webapp_root = "webapp"

static_dir = os.path.join(webapp_root, "static")
template_dir = os.path.join(webapp_root, "templates")

app = Flask(__name__, static_folder=static_dir, template_folder=template_dir)

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def predict(data):
    config = read_params(params_path) 
    model_dir_path = config["webapp_model_dir"]
    model = joblib.load(model_dir_path)
    prediction = model.predict(data)
    print(prediction)
    if prediction[0] == 1:
        prediction_text = "Success"
    else:
        prediction_text = "Failure"
    return prediction_text
    # prediction = model.predict(data).tolist()[0]
    # try:
    #     if 3 <= prediction <= 8:
    #         return prediction
    #     else:
    #         raise NotInRange
    # except NotInRange:
    #     return "Unexpected result"

def api_response(request):
    try:
        data = np.array([list(request.json.value())])
        response = predict(data)
        response = {"response":response}
        return 
    except Exception as e:
        print(e)
        error = {"error": "Something went wrong!! Try again"}
        return error

@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        try: 
            if request.form:
                data = dict(request.form).values()
                data = np.array([list(map(int, data))])
                response = predict(data)
                return render_template("index.html", response=response)

            elif request.json:
                response = api_response(request)
                return jsonify(response)

        except Exception as e:
            print(e)
            print("Check Point 2 Reached")
            error = {"error": "Something went wrong!! Try again"}
            return render_template("404.html", error=error)
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)