import glob
import sys
import numpy as np
import pandas as pd
import os
import time
import argparse
import pickle
import dgl
import torch
from math import log
import scipy.sparse as sp

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])


def comp_deg_norm(g):
    in_deg = g.in_degrees(g.nodes()).float()
    in_deg[torch.nonzero(in_deg == 0, as_tuple=False).view(-1)] = 1
    norm = 1.0 / in_deg
    return norm

def get_g_r_by_r_edge(args):
    num_nodes, num_rels = get_total_number(args.dp + args.dn, 'stat.txt')
    with open(args.dp + args.dn+'/dg_dict.txt', 'rb') as f:
        graph_dict = pickle.load(f)

    # file = os.path.join(args.dp+args.dn, 'dg_r_dict.txt')
    # g_r_dict = {}
    r_l = list(range(num_rels))
    for r in r_l:
        print(r,' r done')
        # r_dict = {}
        for t in graph_dict:
            g = graph_dict[t]
            print("Nodes",g.nodes())
            print("Edges",g.edges())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", default="../data/", help="dataset path")
    ap.add_argument("--dn", default="AFG", help="dataset name")

    args = ap.parse_args()
    print(args)

    if not os.path.exists(os.path.join(args.dp, args.dn)):
        print('Check if the dataset exists')
        exit()

    get_g_r_by_r_edge(args)