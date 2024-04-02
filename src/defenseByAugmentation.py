from absl import app, flags
import time
import helpers as helpers
import torch
import numpy as np
from skimage.util import random_noise


FLAGS = flags.FLAGS

# Für erste Tests wird die Augmentation durch ein Retraining mit geänderten Bildern durchgeführt.
# TBD: Neues Torchvision-Dataset erstellen, das alle Bilder des Ursprungsdatasets beinhaltet + welche mit Gauss-Noise

def main(_):
    beginn = time.time()
    # Load training and test data
    data = helpers.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
                              batchsize=FLAGS.bs)
    
    
    # Get already trainded model 
    paths = helpers.paths()
    filepath = paths.get_path_model(FLAGS.filename)
    model = helpers.get_object_from_pkl(filepath)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        model = model.cuda()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learningRate)

    # Retrain  model
    model.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            #x, y = x.to(device), y.to(device)
            x_gn = torch.tensor(random_noise(x, mode='gaussian', mean=0, var=0.05, clip=True),dtype=torch.float32) 
            x_gn = x_gn.to(device)
            y =  y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x_gn), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(
            "epoch: {}/{}, train loss: {:.3f}".format(
                epoch, FLAGS.nb_epochs, train_loss
            )
        )
    dauer = time.time() - beginn
    print("Das Training ist nach: ", dauer, "Sekunden beendet")
    model.eval()
    
    filepath = paths.get_path_model(prefix=FLAGS.filename+"_2")
    helpers.save_object_to_pkl(model, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "mnist", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 100, "Number of epochs.")
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_enum("filename", "6", [ "0", "1", "2","3","4","5","6", "7", "8","9","10","11","12","13"], "Filename for model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.001, "Learning rate.")

    app.run(main)
