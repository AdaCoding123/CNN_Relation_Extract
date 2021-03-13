# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 20:27:10 2020

@author: Carrie_Lee
"""

import argparse

def config():
    print('parse arguments.')
    parser = argparse.ArgumentParser(description='CRCNN text classificer')
# learning
    parser.add_argument('-lr', type=float, default=0.025, help='initial learning rate [default: 0.001]')
    parser.add_argument('-epochs', type=int, default=300, help='number of epochs for train [default: 16]')
    parser.add_argument('-batch-size', type=int, default=100, help='batch size for training [default: 256]')
    parser.add_argument('-log-interval',  type=int, default=100,   help='how many steps to wait before logging training status [default: 500]')
    parser.add_argument('-dev-interval', type=int, default=300, help='how many steps to wait before testing [default: 100]')
    parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
    parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
    parser.add_argument('-early-stop', type=int, default=2000, help='iteration numbers to stop without performance increasing')
    parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
    parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
    parser.add_argument('-dropout', type=float, default=0.75, help='the probability for dropout [default: 0.5]')
    parser.add_argument('-max-norm', type=float, default=0, help='l2 constraint of parameters [default: 3.0]')
    parser.add_argument('-embed-dim', type=int, default=300, help='number of embedding dimension [default: 128]')
    parser.add_argument('-kernel-num', type=int, default=500, help='number of each kind of kernel')
    parser.add_argument('-kernel-sizes', type=str, default='2,3,4,5', help='comma-separated kernel size to use for convolution')
    parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
    parser.add_argument('-device', type=int, default=2, help='device to use for iterate data, -1 mean cpu [default: -1]')
# option
    parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
    parser.add_argument('-test', action='store_true', default=False, help='train or test')
    args = parser.parse_args()
    

    
    return args
