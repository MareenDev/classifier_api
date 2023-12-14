from absl import app, flags
import time
import helpers as helpers
import torch
from model1 import CNN, PyNet

FLAGS = flags.FLAGS


def main(_):
    beginn = time.time()
    # Load training and test data
    data = helpers.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
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
    filepath = paths.get_path_file_model(FLAGS.filename)
    helpers.save_object_to_pkl(net, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "fmnist", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 25, "Number of epochs.")
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_enum("model", "cnn", ["cnn", "pynet", "net"], "Choose model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.001, "Learning rate.")
    flags.DEFINE_enum("filename", "M_cnn_D_fmnist", [
        "M_cnn_D_fmnist", "M_cnn_D_mnist", "M_cnn_D_cifar10"],
        "Filename for model.")

    app.run(main)
