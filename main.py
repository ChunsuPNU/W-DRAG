import argparse
# from train import *

import os

parser = argparse.ArgumentParser(description="WDRAG")

# parser.add_argument("--data_dir", default="./data/", type=str, dest="data_dir")
parser.add_argument("--date", default="2023", type=str, dest="date")
parser.add_argument("--subset_dir", default="train", type=str, dest="subset_dir")
parser.add_argument("--class_mode", default="normal", type=str, dest="class_mode")
parser.add_argument("--train_continue", default="off", type=str, dest="train_continue")
parser.add_argument("--num_epochs", default=100000, type=int, dest="num_epochs")
parser.add_argument("--aug_dloss_weight", default=0.2, type=float, dest="aug_dloss_weight")
parser.add_argument("--aug_gloss_weight", default=0.2, type=float, dest="aug_gloss_weight")
parser.add_argument("--gpu_num", default="0", type=str, dest="gpu_num")
parser.add_argument("--dim", default=128, type=int, dest="dim")
parser.add_argument("--critic_iters", default=5, type=int, dest="critic_iters")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--policy", default=['rotation'], type=str, nargs='+', dest='policy')
parser.add_argument("--policy_weight", default=[1.0], type=float, nargs='+', dest='policy_weight')
parser.add_argument("--save_memo", default='', type=str, dest='save_memo')

parser.add_argument("--height", default=64, type=int, dest="height")

args = parser.parse_args()


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_num

    train(args)