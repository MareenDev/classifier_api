from flask import Flask, Response, request
from flask_cors import CORS
import json
import helpers 
import torch

app = Flask(__name__)
CORS(app)

paths = helpers.paths()
filepath = paths.get_path_file_model("M_cnn_D_fmnist")
model = helpers.get_object_from_pkl(filepath)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    model = model.cuda()


@app.route('/', methods=['GET'])
def main():
    string = "Willkommen, schön, dass Sie hier sind! \n\r Für eine Klassifikation Ihres Bildes, nutzen Sie bitte einen Aufruf entsprechend unserer API siehe-> TBD"
    result = Response(content_type='application/json')
    result.set_data(string)

    return result


@app.route('/check', methods=['GET'])
def ckeck():
    result = Response(content_type='application/json')

    if model is None: 
        string = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut!"
    else:
        string = "Der Service zur Klassifizierung Ihrer Bilder steht bereit."
    result.set_data(string)

    return result


@app.route('/classifiy', methods=['POST'])
def predict():
    result = Response(content_type='application/json')

    if model is None: 
        result.status_code = 500
        string = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut!"
    else: 
        model_input = helpers.convertWSRequestToTensor(json.loads(request.data))
        model_input = model_input.to(device)
        prediction = model(model_input)
        

        _, label = prediction.max(1)
        label = label.item()
        string = "{\"label\":"+helpers.outputMapping(ds='fmnist', label=label)+"}"

    result.set_data(string)

    return result
