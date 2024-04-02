import helpers as helpers
from absl import app, flags
from easydict import EasyDict
import torch
import time
import json

FLAGS = flags.FLAGS


def main(_):
    # Load training and test data
    beginn = time.time()
    paths = helpers.paths()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = helpers.ld(ds=FLAGS.dataset, reduced=FLAGS.reduceDataset,
                              batchsize=FLAGS.bs)
    filepath = paths.get_path_model(FLAGS.filename)
    model = helpers.get_object_from_pkl(filepath)
    if device == "cuda":
        model = model.cuda()

    model.eval()

    report = EasyDict(nb_test=0, correct=0)
    output = list()
    for i, (x, y) in enumerate(data.test):
        x, y = x.to(device), y.to(device)
        pred = model(x)
        _, y_pred = pred.max(1)  # model prediction on clean examples

        p = pred.squeeze().tolist()
        
        instance = {"prediction": p,"label": y.item()}
        output.append(instance)

        report.nb_test += y.size(0)
        report.correct += y_pred.eq(y).sum().item()
    print(
        "test acc on clean examples (%): {:.3f}".format(
            report.correct / report.nb_test * 100.0
        )
    )

    data = {"DS": output}# ,"meta_information": {"class_names_model": list(),"dataset_type":"in-distribution"}}
    # Save json to file (data.json)
    with open(FLAGS.filename+".json", "w") as output:
        json.dump(data, output, sort_keys=False, indent=4)
    dauer = time.time() - beginn
    print("Das Programm läuft: ", dauer, "Sekunden")


if __name__ == "__main__":
    flags.DEFINE_enum("dataset", "cifar10", ["mnist", "fmnist", "cifar10"], "Used Dataset.")
    flags.DEFINE_integer("bs", 1, "Batchsize")
    flags.DEFINE_bool("reduceDataset", False, "Reduce Dataset testing the implementation")
    flags.DEFINE_enum("filename", "12", [ "0", "1", "2","3","4","5","6","7", "8","9","10","11","12","13",
                                        "0_1","1_1","2_1","3_1","4_1","5_1","6_1",
                                        "0_2","1_2","2_2","3_2","4_2","5_2","6_2"], "Filename for model.")
    app.run(main)
