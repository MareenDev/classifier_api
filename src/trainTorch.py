from absl import app, flags
import time
import helpers as helpers
import torch

FLAGS = flags.FLAGS


def main(_):
    beginn = time.time()
    # Load training and test data
    data = helpers.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
                              batchsize=FLAGS.bs)
    
    
    # Instantiate model, loss, and optimizer for training
    net = helpers.getModel(FLAGS.model)
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
    filepath = paths.get_path_model(prefix=helpers.getFileName(dataset=FLAGS.dataset, model= FLAGS.model))
    helpers.save_object_to_pkl(net, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "cifar10", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 150, "Number of epochs.")
    flags.DEFINE_integer("bs", 130, "Batchsize")
    flags.DEFINE_enum("model", "PyNet_cifar10",
                      ["pynet_mnist", "pynetSoftmax_mnist", "resnet18_mnist","resnet50_mnist","resnext50_mnist","ViT_mnist","PyNet_cifar10","resNet32_cifar10"], "Choose model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.0075, "Learning rate.")

    app.run(main)
