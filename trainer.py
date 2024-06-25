import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}_126".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random()
    _set_device(args)
    print_args(args)
    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        # args["target_index"],
    )
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve, fecam_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {"top1": [], "top5": []}
    for task in range(data_manager.nb_tasks):
        model.incremental_train(data_manager)
        cnn_accy, nme_accy, fecam_accy = model.eval_task()
        model.after_task()
        #save model 
        # source_index = args["source_index"]
        # target_index = args["target_index"]
        # domain = ["clipart", "painting", "real", "sketch"]
        # domain = ["Art","Clipart","Product","World"]
        # domain = ["amazon","dslr","webcam"]
        # torch.save(model._network.state_dict(), logs_name + "/model_{}_{}2{}.pth".format(task,domain[source_index],domain[target_index]))#need to changed

        if nme_accy is not None and fecam_accy is None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))
            logging.info("No FeCAM accuracy.")

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))
        elif fecam_accy is not None and cnn_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("No NME accuracy")
            logging.info("FeCAM: {}".format(fecam_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            fecam_curve["top1"].append(fecam_accy["top1"])
            fecam_curve["top5"].append(fecam_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("FeCAM top1 curve: {}".format(fecam_curve["top1"]))
            logging.info("FeCAM top5 curve: {}\n".format(fecam_curve["top5"]))
        elif fecam_accy is not None:
            logging.info("No CNN accuracy")
            logging.info("No NME accuracy")
            logging.info("FeCAM: {}".format(fecam_accy["grouped"]))

            fecam_curve["top1"].append(fecam_accy["top1"])
            fecam_curve["top5"].append(fecam_accy["top5"])

            logging.info("FeCAM top1 curve: {}".format(fecam_curve["top1"]))
            logging.info("FeCAM top5 curve: {}\n".format(fecam_curve["top5"]))
        else:
            logging.info("No NME accuracy.")
            logging.info("CNN: {}".format(cnn_accy["grouped"]))

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))


def _set_device(args):
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device_type == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
