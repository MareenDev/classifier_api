import pickle as pkl
import os
import numpy as np
import torch


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
        self._folder = "data"

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
        result = os.path.join(self._folder, prefix, ".pkl")
        return result

    def get_path_file_modelverteidigung(self):
        return os.path.join(self._folder, "model1vs.pkl")


def convertWSRequestToTensor(req):
    content = req["data"]
    arr = np.array(content, dtype='f')/255
    model_input = torch.from_numpy(arr)
    model_input = model_input.reshape(shape=(1, 1, 28, 28))

    return model_input
