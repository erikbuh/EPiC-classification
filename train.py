import numpy as np
import time

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import sys
import os
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

# import custom libraries
from dataloader import PointCloudDataset_ZeroPadded
import models
import config
import utils

print("imports done")


def main():
    cfg = config.init()

    if cfg.log_comet:
        experiment = Experiment(
            project_name=cfg.project_prefix,
        )
        experiment.set_name(
            cfg.out_prefix + time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())
        )
        # experiment.log_parameters(cfg.__dict__)

    # Logging
    log_dir = utils.get_new_log_dir(cfg.logdir, prefix=cfg.out_prefix)
    ckpt_mgr = utils.CheckpointManager(log_dir)

    # random seed
    if cfg.seed_all:
        utils.seed_all(seed=42)

    # dataset and loader
    train_dataset = PointCloudDataset_ZeroPadded(cfg.dataset_train)
    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_dataset = PointCloudDataset_ZeroPadded(cfg.dataset_val)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size_val, shuffle=False)
    test_dataset = PointCloudDataset_ZeroPadded(cfg.dataset_test)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size_val, shuffle=False)

    # get model
    # model = models.EPiC_discriminator_mask(cfg).to(cfg.device)
    # model = models.EPiC_discriminator_mask_squash(cfg).to(cfg.device)
    model = models.EPiC_discriminator_mask_squash2(cfg).to(cfg.device)
    cfg.model_parameters = utils.count_parameters(model)  # count model parameters
    print("Model parameters: ", cfg.model_parameters)
    print(model)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    # BCE loss
    criterion = nn.BCEWithLogitsLoss()  # when no sigmoid in last layer

    # Logging
    if cfg.log_comet:
        experiment.log_parameters(cfg.__dict__)

    # for normalisation / standardisation
    if cfg.normalize_points:
        means, stds = train_dataset.get_means_stds()

    ## TRAINING LOOP
    print("start training")
    best_val_loss = float("inf")
    for epoch in range(cfg.epochs):
        ## EPOCH START
        mean_loss = 0.0
        model.train()  # set model to training mode
        # for batch_id, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='Epoch: {}'.format(epoch)):
        for batch_id, data in enumerate(train_loader):
            # get data
            X = data["X"].float().to(cfg.device)
            y = data["y"].float().to(cfg.device)

            # mask
            mask = (
                X[..., 0] != 0.0
            )  # [B,N]    # zero padded values = False,  non padded values = True
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)  # [B,N,1]

            # normalize / standardise after mask is created
            # because mask is used to determine which points are zero padded and standardisation breaks zero paddeding
            if cfg.normalize_points:
                X = utils.normalize_tensor(X, means, stds, sigma=cfg.norm_sigma)

            # forward
            y_hat = model(X, mask)  # [B,1]

            # loss
            loss = criterion(y_hat.flatten(), y)
            mean_loss += loss.item()

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # logging
            if batch_id % cfg.log_interval == 0:
                print(
                    "Epoch: {} | Batch: {} | Loss: {}".format(
                        epoch, batch_id, loss.item()
                    )
                )

                if cfg.log_comet:
                    experiment.log_metric(
                        "loss", loss.item(), step=epoch * len(train_loader) + batch_id
                    )

        ## EPOCH END
        mean_loss /= len(train_loader)
        print("Epoch: {} | Mean Training Loss: {}".format(epoch, mean_loss))
        if cfg.log_comet:
            experiment.log_metric("mean_loss", mean_loss, step=epoch)

        model.eval()  # set model to evaluation mode

        # VALIDATION LOOP
        mean_loss_val = 0.0
        for batch_id, data in enumerate(val_loader):
            # get data
            X = data["X"].float().to(cfg.device)
            y = data["y"].float().to(cfg.device)

            # mask
            mask = X[..., 0] != 0.0
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

            # normalize / standardise
            if cfg.normalize_points:
                X = utils.normalize_tensor(X, means, stds, sigma=cfg.norm_sigma)

            # forward
            y_hat = model(X, mask)  # [B,1]

            # loss
            loss = criterion(y_hat.flatten(), y)
            mean_loss_val += loss.item()

        mean_loss_val /= len(val_loader)
        print("Epoch: {} | Mean Validation Loss: {} \n".format(epoch, mean_loss_val))
        if cfg.log_comet:
            experiment.log_metric("val_mean_loss", mean_loss_val, step=epoch)

        # save checkpoint
        if epoch % cfg.save_interval_epochs == 0:
            opt_states = {
                "optimizer": optimizer.state_dict(),
            }
            ckpt_mgr.save(
                model, cfg, score=mean_loss_val, others=opt_states, step=epoch
            )

        # early stopping
        if mean_loss_val < best_val_loss:
            best_val_loss = mean_loss_val
            best_epoch = epoch
            best_model = model
            best_opt = optimizer
            # save best model
            opt_states = {
                "optimizer": best_opt.state_dict(),
            }
            ckpt_mgr.save(
                best_model, cfg, score=mean_loss_val, others=opt_states, step=epoch
            )
        else:
            if epoch - best_epoch > cfg.early_stopping:
                print("Early stopping at epoch: {}".format(epoch))
                break

    # save final model
    opt_states = {
        "optimizer": optimizer.state_dict(),
    }
    ckpt_mgr.save(model, cfg, score=mean_loss_val, others=opt_states, step=epoch)

    # TEST LOOP with best model
    print("\n\nBest model on test set:")
    mean_loss_test = 0.0
    y_true_list = []
    y_pred_list = []
    for batch_id, data in enumerate(test_loader):
        # get data
        X = data["X"].float().to(cfg.device)
        y = data["y"].float().to(cfg.device)

        # mask
        mask = X[..., 0] != 0.0
        mask = mask.reshape(mask.shape[0], mask.shape[1], 1)

        # normalize / standardise
        if cfg.normalize_points:
            X = utils.normalize_tensor(X, means, stds, sigma=cfg.norm_sigma)

        # forward
        y_hat = best_model(X, mask)

        # loss
        loss = criterion(y_hat.flatten(), y)
        mean_loss_test += loss.item()

        # append to list
        y_true_list.append(y.cpu().detach().numpy())
        y_pred = torch.sigmoid(y_hat.flatten())  # sigmoid since BCEWithLogitsLoss
        y_pred_list.append(y_pred.cpu().detach().numpy())

    y_true = np.concatenate(y_true_list)
    y_pred = np.concatenate(y_pred_list)
    # calc ROC AUC
    roc_auc = roc_auc_score(y_true, y_pred)

    mean_loss_test /= len(test_loader)
    print("Mean Test Loss: {}".format(mean_loss_test))
    print("ROC AUC: {}".format(roc_auc))
    if cfg.log_comet:
        experiment.log_metric("test_mean_loss", mean_loss_test, step=epoch)
        experiment.log_metric("roc_auc", roc_auc, step=epoch)

    # save roc curve to png file in this log_dir folder
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    plt.plot(fpr, tpr, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right", edgecolor="none")
    plt.savefig(os.path.join(log_dir, "roc_curve.pdf"), bbox_inches="tight")
    plt.savefig(os.path.join(log_dir, "roc_curve.png"), bbox_inches="tight")
    plt.savefig("roc_curve.pdf", bbox_inches="tight")  # save to current dir as well
    # upload to comet
    if cfg.log_comet:
        experiment.log_image(os.path.join(log_dir, "roc_curve.png"))

    # calculate accuracy
    acc = accuracy_score(
        y_true, np.round(y_pred)
    )  # round to 0 or 1 because y_pred is sigmoid
    print("Accuracy: {}".format(acc))
    if cfg.log_comet:
        experiment.log_metric("accuracy", acc, step=epoch)


if __name__ == "__main__":
    main()
