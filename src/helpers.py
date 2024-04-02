import pickle as pkl
import os
import numpy as np
import torch
from easydict import EasyDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json
from PIL import Image
import torchvision.transforms.functional as F
import io
import base64
from model1 import NetCifar,PyNetSoftmax_MNIST, PyNet_MNIST, resnet18MNIST, resnet50MNIST, resnext50_32x4d_MNIST, resnet32_Cifar
from vit_pytorch import ViT


from enum import Enum
class DS(Enum):
    F_MNIST = "f_mnist"
    MNIST = "mnist"
    CIFAR10 = "cifar10"

def get_object_from_pkl(object_path):
    result = None
    try:
        file = open(object_path, 'rb')
        try:
            result = pkl.load(file)
        except Exception as e:
            print("Object konnte nicht geladen werden.", object_path, e)
        finally:
            file.close()
    except Exception as e:
        print("Datei unter", object_path, "kann nicht geöffnet werden.", e)
    return result


def save_object_to_pkl(obj, object_path):
    folder, _ = os.path.split(object_path)

    if not os.path.exists(folder):
        os.mkdir(folder)
    try:
        file = open(object_path, 'wb')
        try:
            pkl.dump(obj, file)
        except Exception as e:
            print("Fehler beim Schreiben einer Datei unter", object_path, e)
        finally:
            file.close()
    except Exception as e:
        print("Fehler beim Öffnen/Erzeugen einer Datei unter", object_path, e)


class paths:
    def __init__(self) -> None:
        # folder-path
        self._folder_data = os.path.abspath("data")
        self._folder_model = os.path.abspath("model")

    def get_path_folder_data(self):
        return self._folder_data

    def get_path_dataset(self, datasetname:DS, train:bool):
        if train:
            filename = str(datasetname)+"_train.pkl"
        else:
            filename = str(datasetname)+"_test.pkl"

        return os.path.join(self._folder_data,filename)

    def get_path_model(self, prefix="1"):
        result = os.path.join(self._folder_model, prefix+".pkl")
        return result

class RequstHandler:

    def __init__(self, model:torch.nn.Module, device,mapping:dict) -> None:
        if model is not None:
            self.model = model
            self.batch_size = 1
            self.channel = 1 #aus Model auslesen!
            self.image_size_target = (28, 28) #aus Model auslesen!
            self.input_size_model = (self.batch_size, self.channel, 28, 28)
            self.mapping = mapping
            self.device = device
            self.model = self.model.to(self.device)

        else:
            raise Exception("Modell nicht verfügbar, Verarbeitung von Anfragen nicht möglich.")
        
    def valideRequest(self,request):
        result = False
            
        try: 
            req = json.loads(request.data)          
            channel =  int(req["channel"])
            
            height =  int(req["image_size"][0])
            width =  int(req["image_size"][1])
            content = req["image"]
            content = base64.b64decode(content)

            if channel == 1 or channel == 3:
                result = True

        except: 
            raise ValueError("Übertragene Daten können nicht als Bild interpretiert werden.")
        return result

    def convertWSRequestToTensor(self,req):
        image_size = tuple(req["image_size"])
        channel =  int(req["channel"])
        data = base64.b64decode(req["image"])
        if channel == 1:
            image = Image.frombytes("L", image_size, data, "raw")
        else:
            image = Image.frombytes("RGB", image_size, data, "raw")
        #Bild in Tensordaten überführen
        transformer = transforms.Compose([transforms.PILToTensor(),transforms.Grayscale()])
        tens = transformer(image)

        #Wenn Größe des übertragenen Bildes nicht der Zielgröße entspricht 
        #wird die Größe des übertragenen Bildes angepasst
        if self.image_size_target != image_size:
            transformer = transforms.Compose([transforms.Resize(size=self.image_size_target, 
                                                                interpolation=F.InterpolationMode.BILINEAR),
                                                                transforms.CenterCrop(self.image_size_target)])
            tens = transformer(tens)

        model_input = tens.reshape(shape=self.input_size_model)/255
        return model_input

    def outputMapping(self, label):
        try:
            result = self.mapping[str(label)]
        except:
            result = ""
        return result

    def predict(self, data):
        data = data.to(self.device)
        return self.model(data)

def getFileName(dataset:DS, model)-> str:
	# Instantiate model, loss, and optimizer for training
    if dataset == "fmnist" and model == "pynet_mnist":
        name = "0"
    elif dataset == "fmnist" and model == "pynetSoftmax_mnist":
        name = "1"
    elif dataset == "fmnist" and model == "resnet18_mnist":
        name = "2"
    elif dataset == "fmnist"and model == "resnet50_mnist":
        name = "3"
    elif dataset == "fmnist" and model == "resnext50_mnist":
        name = "4"
    elif dataset == "fmnist" and  model == "ViT_mnist":
        name = "5"
    elif dataset == "mnist" and model == "pynet_mnist":
        name = "6"
    elif dataset == "mnist" and model == "pynetSoftmax_mnist":
        name = "7"
    elif dataset == "mnist" and model == "resnet18_mnist":
        name = "8"
    elif dataset == "mnist"and model == "resnet50_mnist":
        name = "9"
    elif dataset == "mnist" and model == "resnext50_mnist":
        name = "10"
    elif dataset == "mnist" and  model == "ViT_mnist":
        name = "11"
    elif dataset == "cifar10" and  model == "PyNet_cifar10":
        name = "12"
    elif dataset == "cifar10" and model == "resNet32_cifar10":
        name = "13"
    else:
        raise NotImplementedError
    return name

def getModel(name):
    net = None
    if name == "pynet_mnist":
        net = PyNet_MNIST(in_channels=1)
    elif name == "pynetSoftmax_mnist":
        net = PyNetSoftmax_MNIST(in_channels=1)
    elif name == "resnet18_mnist":
        net = resnet18MNIST(in_channel=1)
    elif name == "resnet50_mnist":
        net = resnet50MNIST(in_channel=1)
    elif name == "resnext50_mnist":
        net = resnext50_32x4d_MNIST(in_channel=1)
    elif name == "ViT_mnist":
        # konfig entsprechend https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0
        net = ViT(image_size = 28, patch_size = 7, num_classes = 10, dim = 64,
                  depth = 6, heads = 8,mlp_dim = 128, dropout = 0.1, channels=1)
    elif name == "PyNet_cifar10":
        net = NetCifar()
    elif name == "resNet32_cifar10":
        net = resnet32_Cifar()
    else:
        raise NotImplementedError

    return net

def ld(ds='mnist', reduced=False, batchsize=None, num_workers=0) :
    beginn = time.time()
    # ggf. normalisierung
    # durch Min-/Max-Scaling oder Z-Score-Normalisierung
    """
    train_transforms = transforms.Compose([transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    if ds == 'mnist':
        train_dataset = datasets.MNIST(
            root=paths().get_path_dataset(datasetname='mnist',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(
            root=paths().get_path_dataset(datasetname='mnist',train=False),
            train=False, download=True, transform=test_transforms)
    elif ds == 'fmnist':
        train_dataset = datasets.FashionMNIST(
            root=paths().get_path_dataset(datasetname='fmnist',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.FashionMNIST(
            root=paths().get_path_dataset(datasetname='fmnist', train=False),
            train=False, download=True, transform=test_transforms)
    elif ds == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=paths().get_path_dataset(datasetname='cifar10',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(
            root=paths().get_path_dataset(datasetname='cifar10',train=False),
            train=False, download=True, transform=test_transforms)"""
    train_dataset, test_dataset = loadDS(ds=ds)

    if reduced is True:
        len_tr = len(train_dataset)//1000
        len_te = len(test_dataset)//1000

        print("Länge Trainingsdatensatz", len_tr)
        print("Länge Testdatensatz", len_te)
        train_dataset = torch.utils.data.random_split(train_dataset,
                                                      [len_tr, len(train_dataset)-len_tr])[0]
        test_dataset = torch.utils.data.random_split(test_dataset,
                                                     [len_te, len(test_dataset)-len_te])[0]


    if batchsize == None:
        bs_train = len(train_dataset)
        bs_test = len(test_dataset)
    else:
        bs_test = batchsize
        bs_train = batchsize
        
    train_loader = DataLoader(train_dataset, batch_size=bs_train,
                              num_workers=num_workers, shuffle=True
                              )
    test_loader = DataLoader(test_dataset, batch_size=bs_test,
                             num_workers=num_workers, shuffle=False
                             )

    dauer = time.time() - beginn
    print("Dauer Datenladen: ", dauer, "Sekunden")

    return EasyDict(train=train_loader, test=test_loader)


def loadDS(ds):
    train_transforms = transforms.Compose([transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    if ds == 'mnist':
        train_dataset = datasets.MNIST(
            root=paths().get_path_dataset(datasetname='mnist',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(
            root=paths().get_path_dataset(datasetname='mnist',train=False),
            train=False, download=True, transform=test_transforms)
    elif ds == 'fmnist':
        train_dataset = datasets.FashionMNIST(
            root=paths().get_path_dataset(datasetname='fmnist',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.FashionMNIST(
            root=paths().get_path_dataset(datasetname='fmnist', train=False),
            train=False, download=True, transform=test_transforms)
    elif ds == 'cifar10':
        train_dataset = datasets.CIFAR10(
            root=paths().get_path_dataset(datasetname='cifar10',train=True),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(
            root=paths().get_path_dataset(datasetname='cifar10',train=False),
            train=False, download=True, transform=test_transforms)
    else:
        raise NotImplementedError()

    return train_dataset,test_dataset 


"""
import pickle as pkl
import os
import numpy as np
import torch
from easydict import EasyDict
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import math
import json
from PIL import Image
import torchvision.transforms.functional as F
import io

def get_object_from_pkl(object_path):
    result = None
    try:
        file = open(object_path, 'rb')
        try:
            result = pkl.load(file)
        except Exception as e:
            print("Object konnte nicht geladen werden.", object_path, e)
        finally:
            file.close()
    except Exception as e:
        print("Datei unter", object_path, "kann nicht geöffnet werden.", e)
    return result


def save_object_to_pkl(obj, object_path):
    folder, _ = os.path.split(object_path)

    if not os.path.exists(folder):
        os.mkdir(folder)
    try:
        file = open(object_path, 'wb')
        try:
            pkl.dump(obj, file)
        except Exception as e:
            print("Fehler beim Schreiben einer Datei unter", object_path, e)
        finally:
            file.close()
    except Exception as e:
        print("Fehler beim Öffnen/Erzeugen einer Datei unter", object_path, e)


class paths:
    def __init__(self) -> None:
        # folder-path
        self._folder = os.path.abspath("model")

    def get_path_folder_data(self):
        return self._folder

    def get_path_file_mnist_train(self):
        return os.path.join(self._folder, "mnist_train.pkl")

    def get_path_file_mnist_test(self):
        return os.path.join(self._folder, "mnist_test.pkl")

    def get_path_file_fmnist_train(self):
        return os.path.join(self._folder, "fmnist_train.pkl")

    def get_path_file_fmnist_test(self):
        return os.path.join(self._folder, "fmnist_test.pkl")

    def get_path_file_cifar10_train(self):
        return os.path.join(self._folder, "cifar10_train.pkl")

    def get_path_file_cifar10_test(self):
        return os.path.join(self._folder, "cifar10_test.pkl")

    def get_path_file_model(self, prefix="1"):
        result = os.path.join(self._folder, prefix+".pkl")
        return result

    def get_path_file_modelverteidigung(self):
        return os.path.join(self._folder, "model1vs.pkl")

def valideRequest(request):
    result = False
        
    try: 
        req = json.loads(request.data)
        channel =  int(req["channel"])
        height =  int(req["img_size"][0])
        width =  int(req["img_size"][1])
        content = req["data"]
        # print("Data",len(content),"Prod",width*height)

        if channel == 3: # Prüfung der data struktur auf r,g,b Attribute
            content_r = req["data"]["r"]
            content_g = req["data"]["g"]
            content_b = req["data"]["b"]
                # print("R",len(content_r),"G",len(content_g) ,"B",len(content_b),"Prod",width*height)
            if (len(content_r) == len(content_g) == len(content_b)) and (len(content_r) == width*height):
                result = True
        elif channel== 1 and width*height == len(content):
            result = True
    except: 
        pass
    return result

def valideRequest2(request):
    result = False
        
    try: 
        req = json.loads(request.data)
        content = req["image"]
        img_size = req["image_size"]

        result = True
    except: 
        raise ValueError("Übertragene Daten können nicht als Bild interpretiert werden.")
    return result

def convertWSRequestToTensor(req):
    channel =  int(req["channel"])
    height =  int(req["img_size"][0])
    width =  int(req["img_size"][1])
    print("Kanal:",channel,"Höhe:",height, "Breite:",width)

    image_size_req = (height, width)
    input_size_req = (channel, height, width)

    image_size_target = (28, 28)
    input_size_model = (1, 1, 28, 28)

    # Create Tensor by request with shape like 'input_size_req'

    if channel == 1:
        arr = np.array(req["data"], dtype='f')
        tens = torch.from_numpy(arr).reshape(input_size_req)
    else: #channel ==3  Reduce to one channel
        arr=  np.array([req["data"]["r"],req["data"]["g"],req["data"]["b"]], dtype='f')
        tens = torch.from_numpy(arr).reshape(input_size_req)
        transformer = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        tens = transformer(tens)

    # Bilddaten auf einen 28*28 bringen

    if image_size_target != image_size_req:
        transformer = transforms.Compose([transforms.Resize(size=image_size_target, interpolation=F.InterpolationMode.BILINEAR),transforms.CenterCrop(image_size_target)])
        
        tens = transformer(tens)

    # Überführe Daten in 
    model_input = tens.reshape(shape=input_size_model)/255
    print("Größe des Tensors:",model_input.size())

    return model_input

def convertWSRequestToTensor2(req):
    import base64
    image_size_target = (28, 28)
    batch_size = 1
    channel = 1
    input_size_model = (batch_size, channel, 28, 28)

    data = req["image"]
    data_des = base64.b64decode(data)
    image_size = tuple(req["image_size"])

    # 1.Daten als Bild interpretieren!
    image = Image.frombytes("L", image_size, data_des, "raw")
    # 2.a Bild in Tensordaten überführen
    transformer = transforms.Compose([transforms.PILToTensor(),transforms.Grayscale()])
    tens = transformer(image)

    # 2.b Wenn Größe des übertragenen Bildes nicht der Zielgröße entspricht wird das die Größe des übertragenen Bildes angepasst
    if image_size_target != image_size:
        transformer = transforms.Compose([transforms.Resize(size=image_size_target, interpolation=F.InterpolationMode.BILINEAR),transforms.CenterCrop(image_size_target)])
        tens = transformer(tens)

    # Überführe Daten in 
    model_input = tens.reshape(shape=input_size_model)/255
    print("Größe des Tensors:",model_input.size())

    return model_input


def ld(ds='mnist', reduced=False, batchsize=128, num_workers=0) -> EasyDict():
    beginn = time.time()
    # ggf. normalisierung
    # durch Min-/Max-Scaling oder Z-Score-Normalisierung

    train_transforms = transforms.Compose([transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.ToTensor()])

    if ds == 'mnist':
        train_dataset = datasets.MNIST(
            root=paths().get_path_file_mnist_train(),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.MNIST(
            root=paths().get_path_file_mnist_test(),
            train=False, download=True, transform=test_transforms)
    elif ds == 'fmnist':
        train_dataset = datasets.FashionMNIST(
            root=paths().get_path_file_fmnist_train(),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.FashionMNIST(
            root=paths().get_path_file_fmnist_test(),
            train=False, download=True, transform=test_transforms)
    else:
        train_dataset = datasets.CIFAR10(
            root=paths().get_path_file_fmnist_train(),
            train=True, download=True, transform=train_transforms)
        test_dataset = datasets.CIFAR10(
            root=paths().get_path_file_fmnist_test(),
            train=False, download=True, transform=test_transforms)

    if reduced is True:
        len_tr = len(train_dataset)//1000
        len_te = len(test_dataset)//1000

        print("Länge Trainingsdatensatz", len_tr)
        print("Länge Testdatensatz", len_te)
        train_dataset = torch.utils.data.random_split(train_dataset,
                                                      [len_tr, len(train_dataset)-len_tr])[0]
        test_dataset = torch.utils.data.random_split(test_dataset,
                                                     [len_te, len(test_dataset)-len_te])[0]

    train_loader = DataLoader(train_dataset, batch_size=batchsize,
                              num_workers=num_workers, shuffle=True
                              )
    test_loader = DataLoader(test_dataset, batch_size=batchsize,
                             num_workers=num_workers, shuffle=False
                             )

    dauer = time.time() - beginn
    print("Dauer Datenladen: ", dauer, "Sekunden")

    return EasyDict(train=train_loader, test=test_loader)


def outputMapping(ds, label):
    if ds == 'mnist':
        mapping = {"0": "T-shirt/Top", "1": "Hosen", "2": "Pullover", 
                   "3": "Kleid", "4": "Mantel", "5": "Sandalen", "6": "Shirt",
                   "7": "Sneaker", "8": "Rucksack", "9": "Ankle boot"}
    elif ds == 'fmnist':
        mapping = {"0": "T-shirt/Top", "1": "Hosen", "2": "Pullover", 
                   "3": "Kleid", "4": "Mantel", "5": "Sandalen", "6": "Shirt",
                   "7": "Sneaker", "8": "Rucksack", "9": "Ankle boot"}    
    else:
        mapping = {"0": "T-shirt/Top", "1": "Hosen", "2": "Pullover", 
                   "3": "Kleid", "4": "Mantel", "5": "Sandalen", "6": "Shirt",
                   "7": "Sneaker", "8": "Rucksack", "9": "Ankle boot"}
    if label in range(10):
        result = mapping[str(label)]
    else:
        result = ""
    
    return result
"""