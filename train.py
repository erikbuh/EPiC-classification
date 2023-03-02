import numpy as np
import time

from comet_ml import Experiment

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
# torch.autograd.set_detect_anomaly(True)

import sys

# import custom libraries
import models as models
# from dataset.dataloader import PointCloudDataset, PointCloudDataset_noPad, PlotCloudDataset
# import energyflow_torch as efT
# from utils import utils
# from utils import evaluation
# from utils import projected_utils as pj_utils
# from utils import plot_overview as plots
import config as cfg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    params = cfg.init()





if __name__ == "__main__":
    main()