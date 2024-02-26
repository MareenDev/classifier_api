from absl import app, flags
import time
import helpers as helpers
import torch
from model1 import CNN, PyNet, PyNetSoftmax, resnet18MNIST, resnet50MNIST, resnext50_32x4d_MNIST
from vit_pytorch import ViT

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
    elif FLAGS.model == "pynetSoftmax":
        net = PyNetSoftmax(in_channels=inChannel)
    elif FLAGS.model == "resnet18":
        net = resnet18MNIST(in_channel=inChannel)
    elif FLAGS.model == "resnet50":
        net = resnet50MNIST(in_channel=inChannel)
    elif FLAGS.model == "resnext50":
        net = resnext50_32x4d_MNIST(in_channel=inChannel)
    elif FLAGS.model == "ViTMNIST":
        # konfig entsprechend https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0
        net = ViT(image_size = 28, patch_size = 7, num_classes = 10, dim = 64,
                  depth = 6, heads = 8,mlp_dim = 128, dropout = 0.1, channels=1)
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
    filepath = paths.get_path_model(prefix=FLAGS.filename)
    helpers.save_object_to_pkl(net, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "fmnist", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 30, "Number of epochs.")
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_enum("model", "resnext50", ["cnn", "pynet", "net","pynetSoftmax", "resnet18","resnet50","resnext50","ViTMNIST"], "Choose model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.003, "Learning rate.")
    flags.DEFINE_enum("filename", "M_resnext50_D_fmnist", [
        "M_cnn_D_fmnist", "M_cnn_D_mnist", "M_pynet_D_mnist","M_pynet_D_fmnist","M_ViT_D_fmnist","M_ViT_D_mnist",
        "M_pynetSoftmax_D_fmnist", "M_pynetSoftmax_D_mnist", "M_pynetSoftmax_D_cifar10",
        "M_resnet18_D_fmnist","M_resnet50_D_fmnist","M_resnext50_D_fmnist",
        "M_resnet18_D_mnist","M_resnet50_D_mnist","M_resnext50_D_mnist"],
        "Filename for model.")

    app.run(main)
