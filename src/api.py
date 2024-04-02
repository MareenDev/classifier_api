from flask import Flask, Response, request
from flask_cors import CORS
import json
from helpers import paths, get_object_from_pkl, RequstHandler
import torch
from defenseByInputTransformation import Randomization,GaussBlur
app = Flask(__name__)
CORS(app)


paths = paths()

modelNames = {"0","1","2","3","4","5","6",
              "0_1","1_1","2_1","3_1","4_1","5_1","6_1",
              "0_2","1_2","2_2","3_2","4_2","5_2","6_2"}
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

#--- Routen für decision-based responses
#Gruppe 0
@app.route('/0/decision', methods=['POST'])
def getDecisionByM_0_D_fmnist():
    return getResponse(modelName="0",decisionbased=True)

@app.route('/0/1/decision', methods=['POST'])
def getDecisionByM_0_1_D_fmnist():
    return getResponse(modelName="0_1",decisionbased=True)

@app.route('/0/2/decision', methods=['POST'])
def getDecisionByM_0_2_D_fmnist():
    return getResponse(modelName="0_2",decisionbased=True)

@app.route('/0/3/decision', methods=['POST'])
def getDecisionByM_0_3_D_fmnist():
    processor = Randomization(device=device)
    return getResponseForPreprocessedData(modelName="0",decisionbased=True, preprocessor=processor)

@app.route('/0/4/decision', methods=['POST'])
def getDecisionByM_0_4_D_fmnist():
    processor = GaussBlur(device=device)
    return getResponseForPreprocessedData(modelName="0",decisionbased=True, preprocessor=processor)
"""#Gruppe 1
@app.route('/1/decision', methods=['POST'])
def getDecisionByM_1_D_fmnist():
    return getResponse(modelName="1",decisionbased=True)

@app.route('/1/1/decision', methods=['POST'])
def getDecisionByM_1_1_D_fmnist():
    return getResponse(modelName="1_1",decisionbased=True)

@app.route('/1/2/decision', methods=['POST'])
def getDecisionByM_1_2_D_fmnist():
    return getResponse(modelName="1_2",decisionbased=True)
"""

#Gruppe 2
@app.route('/2/decision', methods=['POST'])
def getDecisionByM_2_D_fmnist():
    return getResponse(modelName="2",decisionbased=True)

@app.route('/2/1/decision', methods=['POST'])
def getDecisionByM_2_1_D_fmnist():
    return getResponse(modelName="2_1",decisionbased=True)

@app.route('/2/2/decision', methods=['POST'])
def getDecisionByM_2_2_D_fmnist():
    return getResponse(modelName="2_2",decisionbased=True)

@app.route('/2/3/decision', methods=['POST'])
def getDecisionByM_2_3_D_fmnist():
    processor = Randomization(device=device)
    return getResponseForPreprocessedData(modelName="2",decisionbased=True, preprocessor=processor)
@app.route('/2/4/decision', methods=['POST'])
def getDecisionByM_2_4_D_fmnist():
    processor = GaussBlur(device=device)
    return getResponseForPreprocessedData(modelName="2",decisionbased=True, preprocessor=processor)
"""
#Gruppe 3
@app.route('/3/decision', methods=['POST'])
def getDecisionByM_3_D_fmnist():
    return getResponse(modelName="3",decisionbased=True)

@app.route('/3/1/decision', methods=['POST'])
def getDecisionByM_3_1_D_fmnist():
    return getResponse(modelName="3_1",decisionbased=True)

@app.route('/3/2/decision', methods=['POST'])
def getDecisionByM_3_2_D_fmnist():
    return getResponse(modelName="3_2",decisionbased=True)

#Gruppe 4
@app.route('/4/decision', methods=['POST'])
def getDecisionByM_4_D_fmnist():
    return getResponse(modelName="4",decisionbased=True)

@app.route('/4/1/decision', methods=['POST'])
def getDecisionByM_4_1_D_fmnist():
    return getResponse(modelName="4_1",decisionbased=True)

@app.route('/4/2/decision', methods=['POST'])
def getDecisionByM_4_2_D_fmnist():
    return getResponse(modelName="4_2",decisionbased=True)
"""
#Gruppe 5
@app.route('/5/decision', methods=['POST'])
def getDecisionByM_5_D_fmnist():
    return getResponse(modelName="5",decisionbased=True)

@app.route('/5/1/decision', methods=['POST'])
def getDecisionByM_5_1_D_fmnist():
    return getResponse(modelName="5_1",decisionbased=True)

@app.route('/5/2/decision', methods=['POST'])
def getDecisionByM_5_2_D_fmnist():
    return getResponse(modelName="5_2",decisionbased=True)

@app.route('/5/3/decision', methods=['POST'])
def getDecisionByM_5_3_D_fmnist():
    processor = Randomization(device=device)
    return getResponseForPreprocessedData(modelName="5",decisionbased=True, preprocessor=processor)

@app.route('/5/4/decision', methods=['POST'])
def getDecisionByM_5_4_D_fmnist():
    processor = GaussBlur(device=device)
    return getResponseForPreprocessedData(modelName="5",decisionbased=True, preprocessor=processor)

#Gruppe 6
@app.route('/6/decision', methods=['POST'])
def getDecisionByM_6_D_fmnist():
    return getResponse(modelName="6",decisionbased=True)

@app.route('/6/1/decision', methods=['POST'])
def getDecisionByM_6_1_D_fmnist():
    return getResponse(modelName="6_1",decisionbased=True)

@app.route('/6/2/decision', methods=['POST'])
def getDecisionByM_6_2_D_fmnist():
    return getResponse(modelName="6_2",decisionbased=True)

@app.route('/6/3/decision', methods=['POST'])
def getDecisionByM_6_3_D_fmnist():
    processor = Randomization(device=device)
    return getResponseForPreprocessedData(modelName="6",decisionbased=True, preprocessor=processor)

@app.route('/6/4/decision', methods=['POST'])
def getDecisionByM_6_4_D_fmnist():
    processor = GaussBlur(device=device)
    return getResponseForPreprocessedData(modelName="6",decisionbased=True, preprocessor=processor)

#--- Routen für score-based responses
"""
@app.route('/0/score', methods=['POST'])
def getScoreByM_0_D_fmnist():
    return getResponse(modelName="0",decisionbased=False)

@app.route('/1/score', methods=['POST'])
def getScoreByM_1_D_fmnist():
    return getResponse(modelName="1",decisionbased=False)

@app.route('/2/score', methods=['POST'])
def getScoreByM_2_D_fmnist():
    return getResponse(modelName="2",decisionbased=False)

@app.route('/3/score', methods=['POST'])
def getScoreByM_3_D_fmnist():
    return getResponse(modelName="3",decisionbased=False)

@app.route('/4/score', methods=['POST'])
def getScoreByM_4_D_fmnist():
    return getResponse(modelName="4",decisionbased=False)

@app.route('/5/score', methods=['POST'])
def getScoreByM_5_D_fmnist():
    return getResponse(modelName="5",decisionbased=False)
"""

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
    baseModelID = int(modelName.split("_")[0])
    mapping = getMappingByID(baseModelID)
        
    requestHandler = RequstHandler(models[modelName], device, mapping=mapping)
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

def getResponseForPreprocessedData(modelName, decisionbased:bool, preprocessor):
    baseModelID = int(modelName.split("_")[0])
    mapping = getMappingByID(baseModelID)
        
    requestHandler = RequstHandler(models[modelName], device, mapping=mapping)
    if requestHandler is None:
        checkRequest = False
        prediction = None
    else:
        checkRequest = requestHandler.valideRequest(request=request)
        if checkRequest:
            payload = json.loads(request.data)
            model_input = requestHandler.convertWSRequestToTensor(payload)
            model_input = preprocessor(model_input)
            prediction = requestHandler.predict(data=model_input)

    result = createResponse(reqHandler=requestHandler, checkRequest=checkRequest, decisionbased=decisionbased, prediction=prediction)

    return result

def getMappingByID(id:int):
    if id < 6: #FashionMNIST-Modelle
        mapping = {"0": "T-shirt/Top", "1": "Hosen", "2": "Pullover", 
                "3": "Kleid", "4": "Mantel", "5": "Sandalen", "6": "Shirt",
                "7": "Sneaker", "8": "Rucksack", "9": "Ankle boot"} 
    else:#MNIST-Modelle
        mapping =   {"0": "Null", "1": "Eins", "2": "Zwei", 
                "3": "Drei", "4": "Vier", "5": "Fünf", "6": "Sechs",
                "7": "Sieben", "8": "Acht", "9": "Neun"} 
    return mapping
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