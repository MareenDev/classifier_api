from absl import app, flags
import time
import dataPreparation
import src.helpers as helpers
# from easydict import EasyDict
import numpy as np
import torch
# from torchvision import datasets, transforms
from src.model1 import CNN, PyNet

FLAGS = flags.FLAGS


def main(_):
    beginn = time.time()
    # Load training and test data
    data = dataPreparation.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
                              batchsize=FLAGS.bs)
    inChannel = 1
    # Instantiate model, loss, and optimizer for training
    if FLAGS.model == "cnn":
        net = CNN(in_channels=inChannel)
    elif FLAGS.model == "pynet":
        net = PyNet(in_channels=inChannel)
#    elif FLAGS.model == "net":
#        net = Net(in_channels=inChannel)
    else:
        raise NotImplementedError

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        net = net.cuda()
    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    optimizer = torch.optim.Adam(net.parameters(), lr=FLAGS.learningRate)

    # Train vanilla model
    net.train()
    for epoch in range(1, FLAGS.nb_epochs + 1):
        train_loss = 0.0
        for x, y in data.train:
            x, y = x.to(device), y.to(device)
            if FLAGS.adv_train:
                # Replace clean example with adversarial example for 
                # adversarial training
                if FLAGS.adv_attack == 'pgd':
                    x = projected_gradient_descent(net, x, FLAGS.eps, 0.01, 40, 
                                                   np.inf)
                else:
                    x = fast_gradient_method(net, x, FLAGS.eps, np.inf)

            optimizer.zero_grad()
            loss = loss_fn(net(x), y)
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
    net.eval()
    paths = helpers.paths()
    prefix = "M_" + FLAGS.model + "_D_" + FLAGS.dataset
    filepath = paths.get_path_file_model(prefix)
    helpers.save_object_to_pkl(net, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "fmnist", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 5, "Number of epochs.")
    flags.DEFINE_float("eps", 0.3, "Total epsilon for FGM and PGD attacks.")
    flags.DEFINE_bool(
        "adv_train", False, "Use adversarial training (with PGD examples)."
    )
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_enum("adv_attack", "pgd", ["pgd", "fgsm"],
                      "Choose adv. attack for adv.  Example Generation.")
    flags.DEFINE_enum("model", "cnn", ["cnn", "pynet", "net"], "Choose model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.001, "Learning rate.")
    app.run(main)
