import argparse

def parse_args():
    parser = argparse.ArgumentParser()


    # training parameters
    parser.add_argument('--epochs', '-e', default=2, help='number of epochs to train', type=int)
    parser.add_argument('--batch_size', '-b', default=256, help='batch size', type=int)
    parser.add_argument('--batch_size_val', default=256, type=int, help='validation set batch size')
    parser.add_argument('--lr', '-lr', default=1e-4, help='learning rate', type=float)
    parser.add_argument('--device', default='cuda', type=str, help='device to use for training')
    parser.add_argument('--early_stopping', default=50, type=int, help='number of epochs to wait for early stopping')
    parser.add_argument('--seed_all', default=False, type=bool, help='set seed for all libraries')

    # I/O parameters
    parser.add_argument('--logdir', '-sf', default='/beegfs/desy/user/buhmae/7_EPiC-classification/trainings/', help='folder to save trainings in', type=str)
    parser.add_argument('--dataset_train', default='/beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_train.npz', type=str, help='npz file of training dataset')
    parser.add_argument('--dataset_val', default='/beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_val.npz', type=str, help='npz file of validation dataset')
    parser.add_argument('--dataset_test', default='/beegfs/desy/user/buhmae/7_EPiC-classification/dataset/mixed/top_jetnet30_test.npz', type=str, help='npz file of test dataset')

    # dataset specific parameters
    parser.add_argument('--feats', default=3, type=int, help='number of features, for jets =3 (pt,rapidity,phi), for calorimeters = 4 (x,y,z,E(MeV)')
    parser.add_argument('--normalize_points', default=True, type=bool, help='standardisation of points enabled, default: True')
    parser.add_argument('--norm_sigma', default=5, type=int, help='standardisation with sigma X (with of normal distibution, default: 5')

    # model arguemnts
    parser.add_argument('--model_name', default='EPiC_Claassifier_Masked', type=str, help='model name, i.e. EPiC_Claassifier_Masked')
    parser.add_argument('--epic_layers', '-el', default=3, help='number of epic layers', type=int)
    parser.add_argument('--latent', '-l', default=10, help='number of global latent variables', type=int)   
    parser.add_argument('--hid_d', default=128, type=int, help='hidden dimensionality of model layers, default from EPiC-GAN paper: 128')
    parser.add_argument('--sum_scale', default=1e-2, type=int, help='scale factor sum aggregation in epic layer, sould be order 1/N, default: 1e-2')
    parser.add_argument('--dropout_value', default=0., type=float, help='dropout value for all dropout layers, default: 0.')

    # logging parameters
    parser.add_argument('--log_comet', default=False, type=bool, help='enable comet logging')
    parser.add_argument('--reason', default='first test', type=str, help='explain reason for running this run')
    parser.add_argument('--project_prefix', type=str, default='epic-classification', help='for project naming on W$B or comet.ml')
    parser.add_argument('--out_prefix', type=str, default='test_', help='for run naming on W$B or comet.ml')
    parser.add_argument('--log_interval', default=250, type=int, help='interval for wandb loggging and printouts')
    parser.add_argument('--save_interval_epochs', default=25, type=int, help='interval for model weights saving')


    params = parser.parse_args()

    return params


def init():
    params = parse_args()
    return params