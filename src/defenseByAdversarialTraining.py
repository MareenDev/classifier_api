from absl import app, flags
import time
import helpers as helpers
import torch
import numpy as np
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent


FLAGS = flags.FLAGS


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
            x, y = x.to(device), y.to(device)
            x_adv = projected_gradient_descent(model, x, FLAGS.eps, FLAGS.learningRate, 40, np.inf)
            x_adv = x_adv.to(device)

            optimizer.zero_grad()
            loss = loss_fn(model(x_adv), y)
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
    
    filepath = paths.get_path_model(prefix=FLAGS.filename+"_1")
    helpers.save_object_to_pkl(model, filepath)


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "mnist", ["mnist", "fmnist", "cifar10"], 
                      "Used Dataset.")
    flags.DEFINE_integer("nb_epochs", 20, "Number of epochs.")
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_enum("filename", "6", [ "0", "1", "2","3","4","5","6", "7", "8","9","10","11","12","13"], "Filename for model.")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset")
    flags.DEFINE_float("learningRate", 0.001, "Learning rate.")
    flags.DEFINE_float("eps", 0.3, "pperturbation budget.")

    app.run(main)
