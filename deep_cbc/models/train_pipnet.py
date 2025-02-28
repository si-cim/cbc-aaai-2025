import logging

import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from tqdm import tqdm

logger = logging.getLogger()
logger.setLevel(logging.INFO)


def train_pipnet(
    net,
    train_loader,
    optimizer_net,
    optimizer_classifier,
    scheduler_net,
    scheduler_classifier,
    criterion,
    epoch,
    nr_epochs,
    device,
    pretrain=False,
    finetune=False,
    progress_prefix: str = "Train Epoch",
):
    # Make sure the model is in train mode.
    net.train()

    if pretrain:
        # Disable training of classification layer.
        net.module._classification.requires_grad = False
        progress_prefix = "Pretrain Epoch"
    else:
        # Enable training of classification layer (disabled in case of pretraining).
        net.module._classification.requires_grad = True

    # Store info about the procedure.
    train_info = dict()
    total_loss = 0.0
    total_acc = 0.0

    iters = len(train_loader)
    # Show progress on progress bar.
    train_iter = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=progress_prefix + "%s" % epoch,
        mininterval=2.0,
        ncols=0,
    )

    count_param = 0
    for name, param in net.named_parameters():
        if param.requires_grad:
            count_param += 1
    logging.info("Number of parameters that require gradient: " + str(count_param))

    if pretrain:
        align_pf_weight = (
            epoch / nr_epochs
        ) * 1.0  # Keeps on increasing as pre-training progresses.
        unif_weight = 0.5  # Ignored in current implementation.
        t_weight = 5.0  # Stays almost zero as pre-training progresses.
        cl_weight = 0.0
    else:
        align_pf_weight = 5.0
        t_weight = 2.0
        unif_weight = 0.0
        cl_weight = 2.0

    logging.info(
        "Align weight: "
        + str(align_pf_weight)
        + ", U_tanh weight: "
        + str(t_weight)
        + ", Class weight: "
        + str(cl_weight)
    )
    logging.info("Pretrain: " + str(pretrain) + ", Finetune: " + str(finetune))

    lrs_net = []
    lrs_class = []
    # Iterate through the data set to update leaves, prototypes and network.
    for i, (xs1, xs2, ys) in train_iter:

        xs1, xs2, ys = xs1.to(device), xs2.to(device), ys.to(device)

        # Reset the gradients.
        optimizer_classifier.zero_grad(set_to_none=True)
        optimizer_net.zero_grad(set_to_none=True)

        # Perform a forward pass through the network.
        proto_features, pooled, out = net(torch.cat([xs1, xs2]))
        loss, acc = calculate_loss(
            proto_features,
            pooled,
            out,
            ys,
            align_pf_weight,
            t_weight,
            unif_weight,
            cl_weight,
            net.module._classification.normalization_multiplier,
            pretrain,
            finetune,
            criterion,
            train_iter,
            EPS=1e-8,
        )

        # Compute the gradient.
        loss.backward()

        if not pretrain:
            optimizer_classifier.step()
            scheduler_classifier.step(epoch - 1 + (i / iters))
            lrs_class.append(scheduler_classifier.get_last_lr()[0])

        if not finetune:
            optimizer_net.step()
            scheduler_net.step()
            lrs_net.append(scheduler_net.get_last_lr()[0])
        else:
            lrs_net.append(0.0)

        with torch.no_grad():
            total_acc += acc
            total_loss += loss.item()

        if not pretrain:
            with torch.no_grad():
                # Set weights in classification layer < 1e-3 to zero.
                # Note: A manually enforced sparsity constraint for PIPNet model learning to learn sparse representations.
                net.module._classification.weight.copy_(
                    torch.clamp(net.module._classification.weight.data - 1e-3, min=0.0)
                )
                net.module._classification.normalization_multiplier.copy_(
                    torch.clamp(
                        net.module._classification.normalization_multiplier.data,
                        min=1.0,
                    )
                )
                if net.module._classification.bias is not None:
                    net.module._classification.bias.copy_(
                        torch.clamp(net.module._classification.bias.data, min=0.0)
                    )

    train_info["train_accuracy"] = total_acc / float(i + 1)
    train_info["loss"] = total_loss / float(i + 1)
    train_info["lrs_net"] = lrs_net
    train_info["lrs_class"] = lrs_class

    return train_info


def calculate_loss(
    proto_features,
    pooled,
    out,
    ys1,
    align_pf_weight,
    t_weight,
    unif_weight,
    cl_weight,
    net_normalization_multiplier,
    pretrain,
    finetune,
    criterion,
    train_iter,
    EPS=1e-10,
):
    ys = torch.cat([ys1, ys1])
    pooled1, pooled2 = pooled.chunk(2)
    pf1, pf2 = proto_features.chunk(2)
    embv2 = pf2.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)
    embv1 = pf1.flatten(start_dim=2).permute(0, 2, 1).flatten(end_dim=1)

    a_loss_pf = (
        align_loss(embv1, embv2.detach()) + align_loss(embv2, embv1.detach())
    ) / 2.0
    tanh_loss = (
        -(
            torch.log(torch.tanh(torch.sum(pooled1, dim=0)) + EPS).mean()
            + torch.log(torch.tanh(torch.sum(pooled2, dim=0)) + EPS).mean()
        )
        / 2.0
    )

    if not finetune:
        loss = align_pf_weight * a_loss_pf
        loss += t_weight * tanh_loss

    if not pretrain:
        softmax_inputs = torch.log1p(out**net_normalization_multiplier)
        class_loss = criterion(F.log_softmax((softmax_inputs), dim=1), ys)
        if finetune:
            loss = cl_weight * class_loss
        else:
            loss += cl_weight * class_loss
    # The tanh-loss optimizes for uniformity and was sufficient for the experiments.
    # However, if pretraining of the prototypes is not working well for another dataset, try to uniformity loss
    # by uncommenting the below stated lines for training.
    # else:
    #     uni_loss = (uniform_loss(F.normalize(pooled1+EPS,dim=1)) + uniform_loss(F.normalize(pooled2+EPS,dim=1)))/2.
    #     loss += unif_weight * uni_loss

    acc = 0.0
    if not pretrain:
        ys_pred_max = torch.argmax(out, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(ys))

    with torch.no_grad():
        if pretrain:
            pretrain_postfix_str = f"L: {loss.item():.3f}, LA:{a_loss_pf.item():.2f}, LT:{tanh_loss.item():.3f}, num_scores>0.1: {torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}"
            train_iter.set_postfix_str(pretrain_postfix_str, refresh=False)
            logging.info(pretrain_postfix_str)
        else:
            train_postfix_str = f"L: {loss.item():.3f}, LC: {class_loss.item():.3f}, LA: {a_loss_pf.item():.2f}, LT: {tanh_loss.item():.3f}, num_scores>0.1: {torch.count_nonzero(torch.relu(pooled - 0.1), dim=1).float().mean().item():.1f}, Acc.:{acc:.3f}"
            if finetune:
                train_iter.set_postfix_str(train_postfix_str, refresh=False)
                logging.info(train_postfix_str)
            else:
                train_iter.set_postfix_str(train_postfix_str, refresh=False)
                logging.info(train_postfix_str)

    return loss, acc


def uniform_loss(x, t=2):
    # Uniform loss resource: https://www.tongzhouwang.info/hypersphere
    loss = (torch.pdist(x, p=2).pow(2).mul(-t).exp().mean() + 1e-10).log()
    return loss


def align_loss(inputs, targets, EPS=1e-12):
    # Alignment loss resource: https://gitlab.com/mipl/carl/-/blob/main/losses.py
    assert inputs.shape == targets.shape
    assert targets.requires_grad == False

    loss = torch.einsum("nc,nc->n", [inputs, targets])
    loss = -torch.log(loss + EPS).mean()
    return loss
