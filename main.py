import torch
import pandas as pd
import numpy as np
import csv
import spacy
import os
import re
from torchtext import data, datasets
import argparse
import train as trains
import model
import datetime
from utils import  config

args = config.config()

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()): 
    print("\t{}={}".format(attr.upper(),value))

#统一句子长度为90
args.sent_len = 90
#关系类别为19类
args.class_num = 19

args.pos_dim = 90
args.mPos = 2.5
args.mNeg = 0.5
args.gamma = 0.05
# args.device = torch.device(args.device)
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]

nlp = spacy.load('en_core_web_sm')

#对句子进行分词，建立词表
def tokenizer(text): # create a tokenizer function
    # 返回 a list of <class 'spacy.tokens.token.Token'>
    return [tok.text for tok in nlp.tokenizer(text)]

def emb_tokenizer(l):
    r = [y for x in eval(l) for y in x]
    return r

TEXT = data.Field(sequential=True, tokenize=tokenizer,fix_length=args.sent_len)
LABEL = data.Field(sequential=False, unk_token='OTHER') 
POS_EMB = data.Field(sequential=True,unk_token=0,tokenize=emb_tokenizer,use_vocab=False,pad_token=0,fix_length=2*args.sent_len)

print('loading data...')
train,valid,test = data.TabularDataset.splits(path='./data/SemEval2010_task8_all_data',
                                              train='SemEval2010_task8_training/TRAIN_FILE_SUB.CSV',
                                              validation='SemEval2010_task8_training/VALID_FILE.CSV',
                                              test='SemEval2010_task8_testing_keys/TEST_FILE_FULL.CSV',
                                              format='csv',
                                              skip_header=True,csv_reader_params={'delimiter':'\t'},
                                              fields=[('relation',LABEL),('sentence',TEXT),('pos_embed',POS_EMB)])
#建立词表
TEXT.build_vocab(train,vectors='glove.6B.300d')
LABEL.build_vocab(train)

args.vocab = TEXT.vocab
args.cuda = torch.cuda.is_available()
# args.cuda = False
args.save_dir = os.path.join(args.save_dir,datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
train_iter, val_iter, test_iter = data.Iterator.splits((train,valid,test),
                                  batch_sizes=(args.batch_size,len(valid),len(test)),device=args.device,sort_key=lambda x: len(x.sentence),repeat=False)

print('build model...')
cnn = model.CRCNN(args)
if args.snapshot is not None:
    print('\nLoding model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

if args.test:
    try:
        trains.eval(test_iter,cnn,args)
    except Exception as e:
        print("\n test wrong.")
else:
    trains.train(train_iter,val_iter,cnn,args)