from flask import Flask, Response, request
from flask_cors import CORS
import json
from helpers import paths, get_object_from_pkl, RequstHandler
import torch

app = Flask(__name__)
CORS(app)


paths = paths()

modelNames = {"M_pynet_D_fmnist","M_resnet18_D_fmnist","M_resnet50_D_fmnist,M_resnext50_D_fmnist"}#"M_cnn_D_fmnist", "M_cnn_D_mnist", "M_cnn_D_cifar10","M_pynet_D_mnist","M_pynet_D_cifar10",
        #"M_pynetSoftmax_D_fmnist", "M_pynetSoftmax_D_mnist", "M_pynetSoftmax_D_cifar10",}
models = dict.fromkeys(modelNames)

try:
    for name in models:
        models[name] = get_object_from_pkl(paths.get_path_model(name))
except :
    pass


device = "cuda" if torch.cuda.is_available() else "cpu"

#--------------------------------------------
#Routen

#--- Route zur Prüfung der API-Erreichbarkeit


@app.route('/', methods=['GET'])
def main():
    string = "Willkommen, schön, dass Sie hier sind! \n\r Für eine Klassifikation Ihres Bildes, nutzen Sie bitte einen Aufruf entsprechend unserer API siehe-> TBD"
    result = Response(content_type='application/json')
    result.set_data(string)

    return result

#--- Routen für decision-based Models

@app.route('/cnn/decision', methods=['POST'])
def getDecisionByM_pynet_D_fmnist():
    return getResponse(modelName="M_pynet_D_fmnist",decisionbased=True)

@app.route('/ResNet50/decision', methods=['POST'])
def getDecisionByM_reset50_D_fmnist():
    return getResponse(modelName="M_resnet50_D_fmnist",decisionbased=True)

@app.route('/ResNet18/decision', methods=['POST'])
def getDecisionByM_reset18_D_fmnist():
    return getResponse(modelName="M_resnet18_D_fmnist",decisionbased=True)

@app.route('/ResNext50/decision', methods=['POST'])
def getDecisionByM_resNext50_D_fmnist():
    return getResponse(modelName="M_resnext50_D_fmnist",decisionbased=True)


#--- Routen für score-based Models

@app.route('/cnn/score', methods=['POST'])
def getScoreByM_pynet_D_fmnist():
    return getResponse(modelName="M_pynet_D_fmnist",decisionbased=False)

@app.route('/ResNet50/score', methods=['POST'])
def getScoreByM_reset50_D_fmnist():
    return getResponse(modelName="M_resnet50_D_fmnist",decisionbased=False)

@app.route('/ResNet18/score', methods=['POST'])
def getScoreByM_reset18_D_fmnist():
    return getResponse(modelName="M_resnet18_D_fmnist",decisionbased=False)


#-----------------------------------------
# Methoden 

def createResponse(reqHandler:RequstHandler, checkRequest:bool, 
                   decisionbased:bool, prediction:torch.Tensor ):
    result = Response(content_type='application/json')
    if reqHandler is None:
        result.status_code = 500
        string = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut!"
    elif not checkRequest:
        result.status_code = 400
        string = "Fehler"
    elif prediction is not None:
        try:
            result.status_code = 200
            if  decisionbased:
                value, label = prediction.max(1)
                label=label.item()
                string = "{\"data\":\""+reqHandler.outputMapping(label=label)+"\"}"
            else:
                prediction_lst = prediction.tolist()[0]
                prediction_dic = {"data":prediction_lst}
                string = json.dumps(prediction_dic)
        except:
            result.status_code = 500
            string = "Es ist ein Fehler aufgetreten!"
    else: 
        result.status_code = 500
        string = "Es ist ein Fehler aufgetreten!"

    result.set_data(string)
    return result

def getResponse(modelName, decisionbased:bool):
    requestHandler = RequstHandler(models[modelName], device)
    if requestHandler is None:
        checkRequest = False
        prediction = None
    else:
        checkRequest = requestHandler.valideRequest(request=request)
        if checkRequest:
            payload = json.loads(request.data)
            model_input = requestHandler.convertWSRequestToTensor(payload)
            prediction = requestHandler.predict(data=model_input)

    result = createResponse(reqHandler=requestHandler, checkRequest=checkRequest, decisionbased=decisionbased, prediction=prediction)

    return result


"""
from flask import Flask, Response, request
from flask_cors import CORS
import json
import helpers 
import torch

app = Flask(__name__)
CORS(app)


paths = helpers.paths()
modelNames = {"M_pynet_D_fmnist"}#"M_cnn_D_fmnist", "M_cnn_D_mnist", "M_cnn_D_cifar10","M_pynet_D_mnist","M_pynet_D_cifar10",
        #"M_pynetSoftmax_D_fmnist", "M_pynetSoftmax_D_mnist", "M_pynetSoftmax_D_cifar10","M_resnet18_D_fmnist","M_resnet50_D_fmnist"}
models = dict.fromkeys(modelNames)

try:
    for name in models:
        models[name] = helpers.get_object_from_pkl(paths.get_path_file_model(name))
except :
    pass


device = "cuda" if torch.cuda.is_available() else "cpu"


@app.route('/', methods=['GET'])
def main():
    string = "Willkommen, schön, dass Sie hier sind! \n\r Für eine Klassifikation Ihres Bildes, nutzen Sie bitte einen Aufruf entsprechend unserer API siehe-> TBD"
    result = Response(content_type='application/json')
    result.set_data(string)

    return result

#-----------------------------------------
# 1. Decision-based Models

#--- array-basierter Aufruf

@app.route('/ResNet50', methods=['POST'])
def getValueByM_reset50_D_fmnist():
    return predict("M_resnet50_D_fmnist")

@app.route('/ResNet18', methods=['POST'])
def getValueByM_reset_D_fmnist():
    return predict("M_resnet18_D_fmnist")

@app.route('/classifiy', methods=['POST'])
def getValueByM_pynet_D_fmnist():
    return predict("M_pynet_D_fmnist")



def predict(modelName):
    result = Response(content_type='application/json')
    
    if models[modelName] is None: 
        result.status_code = 500
        string = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut!"
    else: 
        ok =  helpers.valideRequest(request)
        print(request.headers)

        if ok:
            #ggf. Aufruf loggen: timestamp, 
            model_input = helpers.convertWSRequestToTensor(json.loads(request.data))
            model_input = model_input.to(device)
            if device == "cuda":
                model = models[modelName].cuda()
            prediction = model(model_input)
            
            value, label = prediction.max(1)
            
            # TBD: Prüfung auf Schwellwert für aussagekräftige Klassenzuweisung 
            # (abhängig von Funktion in AusgabeSchicht)
            threshold = 0.0
            
            #if value >= threshold: 
            #     label = label.item()
            label=label.item()
            string = "{\"label\":\""+helpers.outputMapping(ds='fmnist', label=label)+"\"}"
        else:
            result.status_code = 400
            string = "Fehler"

    result.set_data(string)

    return result


#--- Bild-basierter Aufruf

@app.route('/img', methods=['POST'])
def getValueByM_pynet_D_fmnistIMG():
    return predict2("M_pynet_D_fmnist")

def predict2(modelName):
    result = Response(content_type='application/json')
    
    if models[modelName] is None: 
        result.status_code = 500
        string = "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut!"
    else: 
        ok =  helpers.valideRequest2(request)
        print(request.headers)

        if ok:
            #ggf. Aufruf loggen: timestamp, 
            payload = json.loads(request.data)
            model_input = helpers.convertWSRequestToTensor2(payload)
            model_input = model_input.to(device)
            if device == "cuda":
                model = models[modelName].cuda()
            prediction = model(model_input)
            
            value, label = prediction.max(1)
            
            # TBD: Prüfung auf Schwellwert für aussagekräftige Klassenzuweisung 
            # (abhängig von Funktion in AusgabeSchicht)
            threshold = 0.0
            
            #if value >= threshold: 
            #     label = label.item()
            label=label.item()
            string = "{\"label\":\""+helpers.outputMapping(ds='fmnist', label=label)+"\"}"
        else:
            result.status_code = 400
            string = "Fehler"

    result.set_data(string)

    return result

#-----------------------------------------
# 2. Score-based Models
"""