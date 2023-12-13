import src.helpers as helpers
import src.dataPreparation as dataPreparation
from absl import app, flags
from easydict import EasyDict
import torch
import time

FLAGS = flags.FLAGS


def main(_):
    # Load training and test data
    beginn = time.time()
    paths = helpers.paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = dataPreparation.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
                              batchsize=FLAGS.bs)
    filepath = paths.get_path_file_model(FLAGS.filename)
    model = helpers.get_object_from_pkl(filepath)
    if device == "cuda":
        model = model.cuda()

    model.eval()

    report = EasyDict(nb_test=0, correct=0)

    for i, (x, y) in enumerate(data.test):
        x, y = x.to(device), y.to(device)
        _, y_pred = model(x).max(1)  # model prediction on clean examples

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )
    dauer = time.time() - beginn
    print("Das Programm läuft: ", dauer, "Sekunden")


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "fmnist", [
                      "mnist", "fmnist", "cifar10"], "Used Dataset.")
    flags.DEFINE_integer("bs", 100, "Batchsize")
    flags.DEFINE_bool(
        "reduceDataset", False, "Reduce Dataset testing the implementation"
    )
    flags.DEFINE_enum("filename", "M_cnn_D_fmnist", [
                      "M_cnn_D_fmnist", "M_cnn_D_mnist", "M_cnn_D_cifar10"], 
                      "Filename for model.")

    app.run(main)
