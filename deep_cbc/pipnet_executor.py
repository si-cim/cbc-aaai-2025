import logging
import random
from copy import deepcopy
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

from deep_cbc.configs.model_config import ModelConfig
from deep_cbc.models.lib import get_optimizers_nn, init_weights_xavier
from deep_cbc.models.pipnet import PIPNet, get_network
from deep_cbc.models.test_pipnet import eval_ood, eval_pipnet, get_thresholds
from deep_cbc.models.train_pipnet import train_pipnet
from deep_cbc.utils.args import get_args, save_args
from deep_cbc.utils.data import get_dataloaders
from deep_cbc.utils.eval_purity import (
    eval_prototypes_cub_parts_csv,
    get_proto_patches_cub,
    get_topk_cub,
)
from deep_cbc.utils.logging import Log
from deep_cbc.utils.visualize_data import visualize_data, visualize_top_k
from deep_cbc.utils.visualize_predictions import visualize_predictions

# Creating a logger for the loading the execution run of the experiment.
logger = logging.getLogger()
logger.setLevel(logging.INFO)


# Creating a ConfigStore for loading the configuration through structured dataclass.
cs = ConfigStore.instance()
cs.store(name="pipnet_convnext_config_base", node=ModelConfig)


@hydra.main(
    version_base=None,
    config_path="configs/cub_configs",
    config_name="pipnet_convnext_config_base",
)
def run_pipnet(args: DictConfig) -> None:
    # Setting up experiment log file.
    logging.basicConfig(filename=args.log_dir + "/execution_log.txt", filemode="w")
    # Fixing all the execution related seeds.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Setting up logging for the PIPNet executor script.
    if not Path(args.log_dir).is_dir():
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.log_dir + "/execution_log.txt", filemode="w")

    args = args or get_args()
    assert args.batch_size > 1

    # Create a logger.
    log = Log(args.log_dir)
    logging.info("Log dir: " + str(args.log_dir))
    # Log the run arguments.
    save_args(args, log.metadata_dir)

    gpu_list = args.gpu_ids.split(",")
    device_ids = []
    if args.gpu_ids != "":
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not args.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device("cuda:{}".format(args.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device("cuda")
            logging.info("CUDA device set without id specification")
            device_ids.append(torch.cuda.current_device())
        else:
            logging.info(
                "This implementation is successfully tested for single GPU execution."
            )
            device_str = ""
            for d in device_ids:
                device_str += str(d)
                device_str += ","
            device = torch.device("cuda:" + str(device_ids[0]))
    else:
        device = torch.device("cpu")

    # Log which device was actually used.
    logging.info("Device used: " + str(device) + " with id " + str(device_ids))

    # Obtain the dataset and dataloaders.
    (
        trainloader,
        trainloader_pretraining,
        trainloader_normal,
        trainloader_normal_augment,
        projectloader,
        testloader,
        test_projectloader,
        classes,
    ) = get_dataloaders(args, device)
    if len(classes) <= 20:
        if args.validation_size == 0.0:
            logging.info("Classes: " + str(testloader.dataset.class_to_idx))
        else:
            logging.info("Classes: " + str(classes))

    # Create a convolutional network based on arguments and add 1x1 conv layer.
    (
        feature_net,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes,
    ) = get_network(len(classes), args)

    # Create the PIPNet model.
    net = PIPNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=args,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    net = net.to(device=device)
    net = nn.DataParallel(net, device_ids=device_ids)

    (
        optimizer_net,
        optimizer_classifier,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizers_nn(net, args, "separate")

    # Initialize or load model.
    with torch.no_grad():
        if args.state_dict_dir_net != "":
            epoch = 0
            checkpoint = torch.load(args.state_dict_dir_net, map_location=device)
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            logging.info("Pretrained network loaded.")
            net.module._multiplier.requires_grad = False
            try:
                optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])
            except:
                pass

            if (
                torch.mean(net.module._classification.weight).item() > 1.0
                and torch.mean(net.module._classification.weight).item() < 3.0
                and torch.count_nonzero(
                    torch.relu(net.module._classification.weight - 1e-5)
                )
                .float()
                .item()
                > 0.8 * (num_prototypes * len(classes))
            ):
                # Assuming that the linear classification layer is not yet trained
                # (e.g. when loading a pretrained backbone only).
                logging.info(
                    "We assume that the classification layer is not yet trained. We re-initialize it..."
                )
                torch.nn.init.normal_(
                    net.module._classification.weight, mean=1.0, std=0.1
                )
                torch.nn.init.constant_(net.module._multiplier, val=2.0)
                logging.info(
                    "Classification layer initialized with mean "
                    + str(torch.mean(net.module._classification.weight).item())
                )
                if args.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            # else: #uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        else:
            net.module._add_on.apply(init_weights_xavier)
            torch.nn.init.normal_(net.module._classification.weight, mean=1.0, std=0.1)
            if args.bias:
                torch.nn.init.constant_(net.module._classification.bias, val=0.0)
            torch.nn.init.constant_(net.module._multiplier, val=2.0)
            net.module._multiplier.requires_grad = False

            logging.info(
                "Classification layer initialized with mean "
                + str(torch.mean(net.module._classification.weight).item())
            )

    # Define classification loss function and scheduler.
    criterion = nn.NLLLoss(reduction="mean").to(device)
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(trainloader_pretraining) * args.epochs_pretrain,
        eta_min=args.lr_block / 100.0,
        last_epoch=-1,
    )

    # Forward one batch through the backbone to get the latent output size.
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        args.wshape = wshape  # Needed for calculating image patch size.
        logging.info("Output shape: " + str(proto_features.shape))

    if net.module._num_classes == 2:
        # Create a csv log for storing the test accuracy, F1-score, mean train accuracy and mean loss for each epoch.
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_f1",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )
        logging.info(
            "Your dataset only has two classes. Is the number of samples per class similar? If the data is imbalanced, we recommend to use the --weighted_loss flag to account for the imbalance."
        )
    else:
        # Create a csv log for storing the test accuracy (top 1 and top 5), mean train accuracy and mean loss for each epoch.
        log.create_log(
            "log_epoch_overview",
            "epoch",
            "test_top1_acc",
            "test_top5_acc",
            "almost_sim_nonzeros",
            "local_size_all_classes",
            "almost_nonzeros_pooled",
            "num_nonzero_prototypes",
            "mean_train_acc",
            "mean_train_loss_during_epoch",
        )

    lrs_pretrain_net = []
    # Prototypes pretraining phase.
    for epoch in range(1, args.epochs_pretrain + 1):
        for param in params_to_train:
            param.requires_grad = True
        for param in net.module._add_on.parameters():
            param.requires_grad = True
        for param in net.module._classification.parameters():
            param.requires_grad = False
        # Can be set to False when you want to freeze more layers.
        for param in params_to_freeze:
            param.requires_grad = True
        # Can be set to True when you want to train whole backbone (e.g. if dataset is very different from ImageNet).
        for param in params_backbone:
            param.requires_grad = False

        logging.info(
            "Pretrain Epoch "
            + str(epoch)
            + " with batch size "
            + str(trainloader_pretraining.batch_size)
        )

        # Pretrain prototypes.
        train_info = train_pipnet(
            net,
            trainloader_pretraining,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            None,
            criterion,
            epoch,
            args.epochs_pretrain,
            device,
            pretrain=True,
            finetune=False,
        )
        lrs_pretrain_net += train_info["lrs_net"]
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(Path(args.log_dir) / Path("lr_pretrain_net.png"))
        log.log_values(
            "log_epoch_overview",
            epoch,
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            "n.a.",
            train_info["loss"],
        )

    if args.state_dict_dir_net == "":
        net.eval()
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_net_state_dict": optimizer_net.state_dict(),
            },
            Path(args.log_dir) / Path("checkpoints") / Path("net_pretrained"),
        )
        net.train()
    with torch.no_grad():
        if "convnext" in args.net and args.epochs_pretrain > 0:
            if args.visualize_data:
                topks = visualize_top_k(
                    net,
                    projectloader,
                    len(classes),
                    device,
                    "visualised_pretrained_prototypes_topk",
                    args,
                )

    # Second training phase.
    # Reinitialize optimizers and schedulers for second training phase.
    (
        optimizer_net,
        optimizer_classifier,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizers_nn(net, args, "separate")
    # Scheduler for the classification layer is with restarts, such that the model can reactivate zeroed-out prototypes.
    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net, T_max=len(trainloader) * args.epochs, eta_min=args.lr_net / 100.0
    )

    if args.epochs <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=5, eta_min=0.001, T_mult=1, verbose=False
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_classifier, T_0=10, eta_min=0.001, T_mult=1, verbose=False
        )
    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True

    frozen = True
    lrs_net = []
    lrs_classifier = []

    for epoch in range(1, args.epochs + 1):
        # During fine-tuning, only train classification layer and freeze rest.
        # Usually done for a few epochs (at least 1, more depending on size of dataset).
        epochs_to_finetune = 3
        if epoch <= epochs_to_finetune and (
            args.epochs_pretrain > 0 or args.state_dict_dir_net != ""
        ):
            for param in net.module._add_on.parameters():
                param.requires_grad = False
            for param in params_to_train:
                param.requires_grad = False
            for param in params_to_freeze:
                param.requires_grad = False
            for param in params_backbone:
                param.requires_grad = False
            finetune = True

        else:
            finetune = False
            if frozen:
                # Unfreeze backbone.
                if epoch > (args.freeze_epochs):
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = True
                    frozen = False
                # Freeze first layers of backbone, train rest.
                else:
                    # Can be set to False if you want to train fewer layers of backbone.
                    for param in params_to_freeze:
                        param.requires_grad = True
                    for param in net.module._add_on.parameters():
                        param.requires_grad = True
                    for param in params_to_train:
                        param.requires_grad = True
                    for param in params_backbone:
                        param.requires_grad = False

        logging.info("Epoch" + str(epoch) + " frozen: " + str(frozen))
        if (epoch == args.epochs or epoch % 30 == 0) and args.epochs > 1:
            # Set small weight to zero, again a manual constraint for sparsity.
            with torch.no_grad():
                torch.set_printoptions(profile="full")
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 0.001, min=0.0)
                )
                logging.info(
                    "Classifier weights: "
                    + str(
                        net.module._classification.weight[
                            net.module._classification.weight.nonzero(as_tuple=True)
                        ]
                    )
                    + " "
                    + str(
                        (
                            net.module._classification.weight[
                                net.module._classification.weight.nonzero(as_tuple=True)
                            ]
                        ).shape
                    )
                )
                if args.bias:
                    logging.info(
                        "Classifier bias: " + str(net.module._classification.bias)
                    )
                torch.set_printoptions(profile="default")

        train_info = train_pipnet(
            net,
            trainloader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            args.epochs,
            device,
            pretrain=False,
            finetune=finetune,
        )
        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]
        # Evaluate model.
        eval_info = eval_pipnet(net, testloader, epoch, device, log)
        log.log_values(
            "log_epoch_overview",
            epoch,
            eval_info["top1_accuracy"],
            eval_info["top5_accuracy"],
            eval_info["almost_sim_nonzeros"],
            eval_info["local_size_all_classes"],
            eval_info["almost_nonzeros"],
            eval_info["num non-zero prototypes"],
            train_info["train_accuracy"],
            train_info["loss"],
        )

        with torch.no_grad():
            net.eval()
            torch.save(
                {
                    "model_state_dict": net.state_dict(),
                    "optimizer_net_state_dict": optimizer_net.state_dict(),
                    "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
                },
                Path(args.log_dir) / Path("checkpoints") / Path("net_trained"),
            )

            if epoch % 30 == 0:
                net.eval()
                torch.save(
                    {
                        "model_state_dict": net.state_dict(),
                        "optimizer_net_state_dict": optimizer_net.state_dict(),
                        "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
                    },
                    Path(args.log_dir)
                    / Path("checkpoints")
                    / Path("net_trained_%s" % str(epoch)),
                )

            # Save learning rate in figure.
            plt.clf()
            plt.plot(lrs_net)
            plt.savefig(Path(args.log_dir) / Path("lr_net.png"))
            plt.clf()
            plt.plot(lrs_classifier)
            plt.savefig(Path(args.log_dir) / Path("lr_class.png"))

    net.eval()
    torch.save(
        {
            "model_state_dict": net.state_dict(),
            "optimizer_net_state_dict": optimizer_net.state_dict(),
            "optimizer_classifier_state_dict": optimizer_classifier.state_dict(),
        },
        Path(args.log_dir) / Path("checkpoints") / Path("net_trained_final"),
    )

    if args.visualize_data:
        topks = visualize_top_k(
            net, projectloader, len(classes), device, "visualised_prototypes_topk", args
        )

    # Set weights of prototypes that are never really found in projection set to 0.
    set_to_zero = []

    if args.visualize_data:
        if topks:
            for prot in topks.keys():
                found = False
                for (i_id, score) in topks[prot]:
                    if score > 0.1:
                        found = True
                if not found:
                    torch.nn.init.zeros_(net.module._classification.weight[:, prot])
                    set_to_zero.append(prot)
            logging.info(
                "Weights of prototypes "
                + str(set_to_zero)
                + " are set to zero because it is never detected with similarity>0.1 in the training set"
            )
            eval_info = eval_pipnet(
                net, testloader, "notused" + str(args.epochs), device, log
            )
            log.log_values(
                "log_epoch_overview",
                "notused" + str(args.epochs),
                eval_info["top1_accuracy"],
                eval_info["top5_accuracy"],
                eval_info["almost_sim_nonzeros"],
                eval_info["local_size_all_classes"],
                eval_info["almost_nonzeros"],
                eval_info["num non-zero prototypes"],
                "n.a.",
                "n.a.",
            )

    logging.info("classifier weights: " + str(net.module._classification.weight))
    logging.info(
        "Classifier weights nonzero: "
        + str(
            net.module._classification.weight[
                net.module._classification.weight.nonzero(as_tuple=True)
            ]
        )
        + " "
        + str(
            (
                net.module._classification.weight[
                    net.module._classification.weight.nonzero(as_tuple=True)
                ]
            ).shape
        )
    )
    logging.info("Classifier bias: " + str(net.module._classification.bias))
    # Print weights and relevant prototypes per class.
    for c in range(net.module._classification.weight.shape[0]):
        relevant_ps = []
        proto_weights = net.module._classification.weight[c, :]
        for p in range(net.module._classification.weight.shape[1]):
            if proto_weights[p] > 1e-3:
                relevant_ps.append((p, proto_weights[p].item()))
        if args.validation_size == 0.0:
            logging.info(
                "Class "
                + str(c)
                + " ( "
                + str(
                    list(testloader.dataset.class_to_idx.keys())[
                        list(testloader.dataset.class_to_idx.values()).index(c)
                    ]
                )
                + " ): "
                + " has "
                + str(len(relevant_ps))
                + " relevant prototypes: "
                + str(relevant_ps)
            )

    # Evaluate prototype purity.
    if args.dataset == "CUB-200-2011":
        projectset_img0_path = projectloader.dataset.samples[0][0]
        project_path = Path(
            str(Path(projectset_img0_path).parent.parent).split("dataset")[0]
        )

        parts_loc_path = project_path / Path("parts/part_locs.txt")
        parts_name_path = project_path / Path("parts/parts.txt")
        imgs_id_path = project_path / Path("images.txt")
        cubthreshold = 0.5

        net.eval()
        logging.info("Evaluating cub prototypes for training set.")
        csvfile_topk = get_topk_cub(
            net, projectloader, 10, "train_" + str(epoch), device, args
        )
        eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "train_topk_" + str(epoch),
            args,
            log,
        )

        csvfile_all = get_proto_patches_cub(
            net,
            projectloader,
            "train_all_" + str(epoch),
            device,
            args,
            threshold=cubthreshold,
        )
        eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "train_all_thres" + str(cubthreshold) + "_" + str(epoch),
            args,
            log,
        )

        logging.info("Evaluating cub prototypes for test set.")
        csvfile_topk = get_topk_cub(
            net, test_projectloader, 10, "test_" + str(epoch), device, args
        )
        eval_prototypes_cub_parts_csv(
            csvfile_topk,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "test_topk_" + str(epoch),
            args,
            log,
        )
        cubthreshold = 0.5
        csvfile_all = get_proto_patches_cub(
            net,
            test_projectloader,
            "test_" + str(epoch),
            device,
            args,
            threshold=cubthreshold,
        )
        eval_prototypes_cub_parts_csv(
            csvfile_all,
            parts_loc_path,
            parts_name_path,
            imgs_id_path,
            "test_all_thres" + str(cubthreshold) + "_" + str(epoch),
            args,
            log,
        )

    # Visualize predictions.
    if args.visualize_data:
        visualize_data(
            net, projectloader, len(classes), device, "visualised_prototypes", args
        )
        testset_img0_path = test_projectloader.dataset.samples[0][0]
        test_path = Path(testset_img0_path).parent.parent
        visualize_predictions(net, test_path, classes, device, args)
        if args.extra_test_image_folder != "":
            if Path(args.extra_test_image_folder).exists():
                visualize_predictions(
                    net, args.extra_test_image_folder, classes, device, args
                )

    # Evaluate OOD detection.
    ood_datasets = ["CARS", "CUB-200-2011", "PETS"]
    for percent in [95.0]:
        logging.info(
            "OOD Evaluation for epoch "
            + str(epoch)
            + " with percent of "
            + str(percent)
        )
        _, _, _, class_thresholds = get_thresholds(
            net, testloader, epoch, device, percent, log
        )
        logging.info("Thresholds: " + str(class_thresholds))
        # Evaluate with in-distribution data.
        id_fraction = eval_ood(net, testloader, epoch, device, class_thresholds)
        logging.info(
            "ID class threshold ID fraction (TPR) with percent "
            + str(percent)
            + " : "
            + str(id_fraction)
        )

        # Evaluate with out-of-distribution data.
        for ood_dataset in ood_datasets:
            if ood_dataset != args.dataset:
                logging.info("OOD dataset: " + str(ood_dataset))
                ood_args = deepcopy(args)
                ood_args.dataset = ood_dataset
                _, _, _, _, _, ood_testloader, _, _ = get_dataloaders(ood_args, device)

                id_fraction = eval_ood(
                    net, ood_testloader, epoch, device, class_thresholds
                )
                logging.info(
                    str(args.dataset)
                    + " - OOD"
                    + str(ood_dataset)
                    + " class threshold ID fraction (FPR) with percent "
                    + str(percent)
                    + " : "
                    + str(id_fraction)
                )

    logging.info("Script execution completed.")


if __name__ == "__main__":
    run_pipnet()
