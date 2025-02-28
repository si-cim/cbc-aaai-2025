import logging
import random
from copy import deepcopy
from pathlib import Path
from typing import Union

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, open_dict

from deep_cbc.configs.model_config import ModelConfig
from deep_cbc.models.cbcnet import CBCNet, get_network
from deep_cbc.models.lib import get_optimizers_nn, init_weights_xavier
from deep_cbc.models.margin_loss import MarginLoss
from deep_cbc.models.test_cbcnet import eval_cbcnet, eval_ood_cbc, get_thresholds_cbc
from deep_cbc.models.train_cbcnet import train_cbcnet
from deep_cbc.utils.args import save_config
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
cs.store(name="cub_convnext_config_base", node=ModelConfig)


@hydra.main(
    version_base=None,
    config_path="configs/cub_configs",
    config_name="cub_convnext_config_base",
)
def scbc_baseline_trainer(model_config: Union[OmegaConf, DictConfig]) -> None:
    # Setting up experiment log file.
    logging.basicConfig(
        filename=model_config.log_dir + "/execution_log.txt", filemode="w"
    )
    # Create a logger.
    log = Log(model_config.log_dir)
    logging.info("Log dir: " + str(model_config.log_dir))
    # Log the run arguments.
    save_config(model_config, log.metadata_dir)

    # Handling the parsed GPU ID values.
    gpu_list = model_config.gpu_ids.split(",")
    device_ids = []
    if model_config.gpu_ids != "":
        for m in range(len(gpu_list)):
            device_ids.append(int(gpu_list[m]))

    global device
    if not model_config.disable_cuda and torch.cuda.is_available():
        if len(device_ids) == 1:
            device = torch.device("cuda:{}".format(model_config.gpu_ids))
        elif len(device_ids) == 0:
            device = torch.device("cuda")
            logging.info("CUDA device set without id specification.")
            device_ids.append(torch.cuda.current_device())
        else:
            logging.info(
                "This code should work with multiple GPU's but we didn't test that, so we recommend to use only 1 GPU."
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
    ) = get_dataloaders(model_config, device)
    if len(classes) <= 20:
        if model_config.validation_size == 0.0:
            print("Classes: ", testloader.dataset.class_to_idx, flush=True)
        else:
            print("Classes: ", str(classes), flush=True)

    # Create a convolutional network based on arguments and add 1x1 conv layer.
    (
        feature_net,
        add_on_layers,
        pool_layer,
        classification_layer,
        num_prototypes,
    ) = get_network(len(classes), model_config)

    # Create a CBCNet with its removed detection probability layer,
    # which is replaced by MaxPool layer's output.
    net = CBCNet(
        num_classes=len(classes),
        num_prototypes=num_prototypes,
        feature_net=feature_net,
        args=model_config,
        add_on_layers=add_on_layers,
        pool_layer=pool_layer,
        classification_layer=classification_layer,
    )
    net = net.to(device=device)
    # Note: The current implementation is only tested with single GPU only.
    net = nn.DataParallel(net, device_ids=device_ids)
    net.eval()

    (
        optimizer_net,
        optimizer_classifier,
        optimizer_classifier_only,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizers_nn(net, model_config, "complete")

    # Initialize or load model.
    with torch.no_grad():
        if model_config.state_dict_dir_net != "":
            epoch = 0
            checkpoint = torch.load(
                model_config.state_dict_dir_net, map_location=device
            )
            net.load_state_dict(checkpoint["model_state_dict"], strict=True)
            net.module._multiplier.requires_grad = False
            logging.info("Pretrained network loaded.")
            try:
                optimizer_net.load_state_dict(checkpoint["optimizer_net_state_dict"])
            except:
                pass
            if model_config.head_type in ("pipnet_head", "rbf_head"):
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
                    if model_config.bias:
                        torch.nn.init.constant_(
                            net.module._classification.bias, val=0.0
                        )
            # else:  # uncomment these lines if you want to load the optimizer too
            #     if 'optimizer_classifier_state_dict' in checkpoint.keys():
            #         optimizer_classifier.load_state_dict(checkpoint['optimizer_classifier_state_dict'])

        else:
            net.module._add_on.apply(init_weights_xavier)
            if model_config.head_type in ("pipnet_head", "rbf_head"):
                torch.nn.init.normal_(
                    net.module._classification.weight, mean=1.0, std=0.1
                )
                if model_config.bias:
                    torch.nn.init.constant_(net.module._classification.bias, val=0.0)
                logging.info(
                    "Classification layer initialized with mean "
                    + str(torch.mean(net.module._classification.weight).item())
                )
            torch.nn.init.constant_(net.module._multiplier, val=2.0)
            net.module._multiplier.requires_grad = False

    # Define classification loss function and scheduler.
    criterion = MarginLoss(similarity=True, margin=0.025).to(device)
    if model_config.head_type in ("pipnet_head", "rbf_head"):
        criterion = nn.NLLLoss(reduction="mean").to(device)

    scheduler_net = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_net,
        T_max=len(trainloader_pretraining) * model_config.epochs_pretrain,
        eta_min=model_config.lr_block / 100.0,
        last_epoch=-1,
    )

    # Forward one batch through the backbone to get the latent output size.
    with torch.no_grad():
        xs1, _, _ = next(iter(trainloader))
        xs1 = xs1.to(device)
        proto_features, _, _ = net(xs1)
        wshape = proto_features.shape[-1]
        with open_dict(model_config):
            # Needed for calculating image patch size.
            model_config.wshape = wshape
        print("Output shape: ", proto_features.shape, flush=True)

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
            "lr_net",
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
            "lr_net",
        )

    lrs_pretrain_net = []
    # Prototypes pretraining phase.
    for epoch in range(1, model_config.epochs_pretrain + 1):
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

        print(
            "Pretrain Epoch "
            + str(epoch)
            + " with batch size "
            + str(trainloader_pretraining.batch_size)
        )

        # Pretrain prototypes.
        train_info = train_cbcnet(
            net,
            trainloader_pretraining,
            optimizer_net,
            optimizer_classifier_only,
            scheduler_net,
            None,
            criterion,
            epoch,
            model_config.epochs_pretrain,
            device,
            pretrain=True,
            finetune=False,
            tensor_log_flag=False,
        )
        lrs_pretrain_net += train_info["lrs_net"]
        plt.clf()
        plt.plot(lrs_pretrain_net)
        plt.savefig(Path(model_config.log_dir) / Path("lr_pretrain_net.png"))
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
            train_info["lrs_net"][0],
        )

    # evaluate model
    eval_info = eval_cbcnet(net, testloader, epoch, device, log)
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
        train_info["lrs_net"][0],
    )

    # Note: Only for the single seed the models are currently.
    # Feel free to update this condition to save multiple models.
    if model_config.state_dict_dir_net == "" and model_config.seed == 1:
        net.eval()
        torch.save(
            {
                "model_state_dict": net.state_dict(),
                "optimizer_net_state_dict": optimizer_net.state_dict(),
            },
            Path(model_config.log_dir) / Path("checkpoints") / Path("net_pretrained"),
        )
        net.train()
    with torch.no_grad():
        if "convnext" in model_config.net and model_config.epochs_pretrain > 0:
            if model_config.visualize_data:
                topks = visualize_top_k(
                    net,
                    projectloader,
                    len(classes),
                    device,
                    "visualised_pretrained_prototypes_topk",
                    model_config,
                )

    # First Classifier Training Phase: Fine-tuning Only
    # Reinitialize optimizers and schedulers for this training phase.
    model_config.lr = model_config.lr_classifier_fine_tune
    (
        optimizer_net,
        _,
        optimizer_classifier_only,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizers_nn(net, model_config, "complete")

    if optimizer_net is not None:
        scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_net, mode="min", patience=10, factor=0.5
        )
    else:
        scheduler_net = None
    # Scheduler are changed with respect to the PIPNet or RBFNet models.
    # Since, there is no artificial constraints to clip zeroed out prototypes.
    # Therefore, "CosineAnnealingWarmRestarts" is not effective here compared to "ReduceLROnPlateau".
    if model_config.epochs_fine_tune <= 30:
        scheduler_classifier_only = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_classifier_only, mode="max", patience=10, factor=0.5
        )
    else:
        scheduler_classifier_only = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_classifier_only, mode="max", patience=10, factor=0.5
        )

    for param in net.module.parameters():
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True

    # Training with MarginLoss, directly using Maxpool output as detection probability from the PIPNet.
    frozen = False
    lrs_net = []
    lrs_classifier = []
    for epoch in range(1, model_config.epochs_fine_tune + 1):
        logging.info("Epoch " + str(epoch) + " frozen: " + str(frozen))
        if scheduler_net is not None:
            logging.info(f"Lr net: {optimizer_net.param_groups[0]['lr']}")

        logging.info(
            f"Lr classifier: {optimizer_classifier_only.param_groups[0]['lr']}"
        )

        if (
            epoch == model_config.epochs_fine_tune or epoch % 30 == 0
        ) and model_config.epochs_fine_tune > 1:
            # Set small weights to zero for PIPNet and RBFNet.
            with torch.no_grad():
                if model_config.head_type in ("pipnet_head", "rbf_head"):
                    torch.set_printoptions(profile="full")
                    net.module._classification.weight.copy_(
                        torch.clamp(
                            net.module._classification.weight.data - 0.001, min=0.0
                        )
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
                                    net.module._classification.weight.nonzero(
                                        as_tuple=True
                                    )
                                ]
                            ).shape
                        )
                    )
                    if model_config.bias:
                        logging.info(
                            "Classifier bias: " + str(net.module._classification.bias)
                        )
                    torch.set_printoptions(profile="default")

        train_info = train_cbcnet(
            net,
            trainloader,
            optimizer_net,
            optimizer_classifier_only,
            scheduler_net,
            scheduler_classifier_only,
            criterion,
            epoch,
            model_config.epochs_fine_tune,
            device,
            pretrain=False,
            finetune=True,
            tensor_log_flag=False,
        )

        if isinstance(
            optimizer_classifier_only, torch.optim.lr_scheduler.ReduceLROnPlateau
        ):
            scheduler_classifier_only.step(train_info["train_accuracy"])

        if isinstance(scheduler_net, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_net.step(train_info["loss"])

        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]

        # Evaluate the fine-tuned only model.
        eval_info = eval_cbcnet(net, testloader, epoch, device, log)
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
            optimizer_classifier_only.param_groups[0]["lr"],
        )

    # Remove the optimizer related to classification head.
    del optimizer_classifier_only
    del scheduler_classifier_only

    # Second Classifier Training Phase, end-to-end network training.
    # Te-initialize optimizers and schedulers for second training phase.
    # Redefining the MarginLoss criteria, in case you want to try different margins for fine-tuning and end-to-end training stages.
    criterion = MarginLoss(similarity=True, margin=0.025).to(device)

    model_config.lr = model_config.lr_net

    (
        optimizer_net,
        optimizer_classifier,
        _,
        params_to_freeze,
        params_to_train,
        params_backbone,
    ) = get_optimizers_nn(net, model_config, "complete")

    if optimizer_net is not None:
        scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_net, mode="min", patience=10, factor=0.5
        )
    else:
        scheduler_net = None
    if model_config.epochs_net <= 30:
        scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_classifier, mode="max", patience=10, factor=0.5
        )
    else:
        scheduler_classifier = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer_classifier, mode="max", patience=10, factor=0.5
        )

    # Below snippet for making sure that deeper feature extractor layers are kept frozen.
    # Additionally, the below snippet is added to test whether unfreezing certain layers increase or decrease the performance.
    for param in net.module.parameters():
        param.requires_grad = True
    for param in params_backbone:
        param.requires_grad = False
    for param in net.module._classification.parameters():
        param.requires_grad = True

    # Training with MarginLoss, directly using Maxpool output as detection probability from the PIPNet.
    frozen = False
    lrs_net = []
    lrs_classifier = []
    max_acc_score = -999.0
    for epoch in range(1, model_config.epochs_net + 1):
        logging.info("Epoch " + str(epoch) + " frozen: " + str(frozen))

        if scheduler_net is not None:
            logging.info(f"Lr net: {optimizer_net.param_groups[0]['lr']}")

        logging.info(f"Lr classifier: {optimizer_classifier.param_groups[0]['lr']}")

        if (
            epoch == model_config.epochs_net or epoch % 30 == 0
        ) and model_config.epochs_net > 1:
            # Setting small weights to zero threshold.
            with torch.no_grad():
                if model_config.head_type in ("pipnet_head", "rbf_head"):
                    torch.set_printoptions(profile="full")
                    net.module._classification.weight.copy_(
                        torch.clamp(
                            net.module._classification.weight.data - 0.001, min=0.0
                        )
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
                                    net.module._classification.weight.nonzero(
                                        as_tuple=True
                                    )
                                ]
                            ).shape
                        )
                    )
                    if model_config.bias:
                        logging.info(
                            "Classifier bias: " + str(net.module._classification.bias)
                        )
                    torch.set_printoptions(profile="default")

        train_info = train_cbcnet(
            net,
            trainloader,
            optimizer_net,
            optimizer_classifier,
            scheduler_net,
            scheduler_classifier,
            criterion,
            epoch,
            model_config.epochs_net,
            device,
            pretrain=False,
            finetune=True,
            tensor_log_flag=False,
        )

        if isinstance(scheduler_classifier, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_classifier.step(train_info["train_accuracy"])

        if isinstance(scheduler_net, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_net.step(train_info["loss"])

        lrs_net += train_info["lrs_net"]
        lrs_classifier += train_info["lrs_class"]

        # Evaluate model the end-to-end trained model.
        eval_info = eval_cbcnet(net, testloader, epoch, device, log)
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
            optimizer_classifier.param_groups[0]["lr"],
        )

        if model_config.seed == 1 and max_acc_score < eval_info["top1_accuracy"]:
            max_acc_score = eval_info["top1_accuracy"]
            net.eval()
            torch.save(
                {"model_state_dict": net.state_dict()},
                Path(model_config.log_dir)
                / Path("checkpoints")
                / Path("net_end_to_end"),
            )
            net.train()

    # The prototype visualization is needed to be done for both positive and negative reasoning components.
    if model_config.visualize_data:
        # First, visualize the positive reasoning components:
        model_config.dir_for_saving_images = "visualization_positive_prediction_results"
        model_config.reasoning_type = "positive"
        top_ks = visualize_top_k(
            net,
            projectloader,
            len(classes),
            device,
            "visualised_prototypes_positive_top_k",
            model_config,
        )
        visualize_data(
            net,
            projectloader,
            len(classes),
            device,
            "visualised_prototypes_positive",
            model_config,
        )
        testset_img0_path = test_projectloader.dataset.samples[0][0]
        test_path = Path(testset_img0_path).parent.parent
        visualize_predictions(net, test_path, classes, device, model_config)

        # Second, visualize the negative reasoning components:
        model_config.dir_for_saving_images = "visualization_negative_prediction_results"
        model_config.reasoning_type = "negative"
        top_ks = visualize_top_k(
            net,
            projectloader,
            len(classes),
            device,
            "visualised_prototypes_negative_top_k",
            model_config,
        )
        visualize_data(
            net,
            projectloader,
            len(classes),
            device,
            "visualised_prototypes_negative",
            model_config,
        )
        testset_img0_path = test_projectloader.dataset.samples[0][0]
        test_path = Path(testset_img0_path).parent.parent
        visualize_predictions(net, test_path, classes, device, model_config)


if __name__ == "__main__":
    scbc_baseline_trainer()
