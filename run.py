# -*- encoding: utf-8 -*-
'''
@File    :   run.py
@Time    :   2021/11/29 17:20:54
@Author  :   Jun Shi 
@Version :   1.0
@Contact :   shijun18@mail.ustc.edu.cn
@License :   (C)Copyright 2019-2025, USTC-ACSA
'''

# here put the import lib

import os
import argparse
from trainer import SemanticSeg
import pandas as pd
import random
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from config import INIT_TRAINER, SETUP_TRAINER, CURRENT_FOLD, PATH_LIST, FOLD_NUM, ROI_NAME,TEST_PATH
from config import VERSION, ROI_NAME, DISEASE, MODE
import time


def get_cross_validation_by_sample(path_list, fold_num, current_fold):

    sample_list = list(set([os.path.basename(case).split('_')[0] for case in path_list]))
    print('sample len:',len(sample_list))
    sample_list.sort()        
    _len_ = len(sample_list) // fold_num

    train_id = []
    validation_id = []
    end_index = current_fold * _len_
    start_index = end_index - _len_
    if current_fold == fold_num:
        validation_id.extend(sample_list[start_index:])
        train_id.extend(sample_list[:start_index])
    else:
        validation_id.extend(sample_list[start_index:end_index])
        train_id.extend(sample_list[:start_index])
        train_id.extend(sample_list[end_index:])

    train_path = []
    validation_path = []
    for case in path_list:
        if os.path.basename(case).split('_')[0] in train_id:
            train_path.append(case)
        else:
            validation_path.append(case)

    random.shuffle(train_path)
    random.shuffle(validation_path)
    print("Train set length:", len(train_path),
          "\nVal set length:", len(validation_path))
    return train_path, validation_path


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--mode',
                        default='train-cross',
                        choices=["train", 'train-cross', "inf","test"],
                        help='choose the mode',
                        type=str)
    parser.add_argument('-s', '--save', default='no', choices=['no', 'n', 'yes', 'y'],
                        help='save the forward middle features or not', type=str)
    args = parser.parse_args()

    # Set data path & segnetwork
    if args.mode != 'train-cross':
        segnetwork = SemanticSeg(**INIT_TRAINER)
        print(get_parameter_number(segnetwork.net))
    path_list = PATH_LIST
    # Training
    ###############################################
    if args.mode == 'train-cross':
        for current_fold in range(1, FOLD_NUM + 1):
            print("=== Training Fold ", current_fold, " ===")
            segnetwork = SemanticSeg(**INIT_TRAINER)
            print(get_parameter_number(segnetwork.net))
            train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, current_fold)
            SETUP_TRAINER['train_path'] = train_path
            SETUP_TRAINER['val_path'] = val_path
            SETUP_TRAINER['cur_fold'] = current_fold
            start_time = time.time()
            segnetwork.trainer(**SETUP_TRAINER)

            print('run time:%.4f' % (time.time() - start_time))


    if args.mode == 'train':
        train_path, val_path = get_cross_validation_by_sample(path_list, FOLD_NUM, CURRENT_FOLD)
        SETUP_TRAINER['train_path'] = train_path
        SETUP_TRAINER['val_path'] = val_path
        SETUP_TRAINER['cur_fold'] = CURRENT_FOLD
		
        start_time = time.time()
        segnetwork.trainer(**SETUP_TRAINER)

        print('run time:%.4f' % (time.time() - start_time))
    ###############################################

    # Inference
    ###############################################
    if args.mode == 'test':
        start_time = time.time()
        test_path = TEST_PATH
        print("test set len:",len(test_path))

        save_path = './analysis/result/{}/{}/{}'.format(DISEASE,VERSION,MODE)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_flag = False if args.save == 'no' or args.save == 'n' else True
        cls_result = segnetwork.test(test_path,save_path,mode=MODE,save_flag=save_flag)

        if MODE != 'seg':
            csv_path = os.path.join(save_path,ROI_NAME + '.csv')
            info = {}
            info['id'] = test_path
            info['label'] = cls_result['true']
            info['pred'] = cls_result['pred']
            info['prob'] = cls_result['prob']
            print(classification_report(cls_result['true'], cls_result['pred'], target_names=['without','with'],output_dict=False))
            print(confusion_matrix(cls_result['true'], cls_result['pred']))
            csv_file = pd.DataFrame(info)
            csv_file.to_csv(csv_path, index=False)
        print('run time:%.4f' % (time.time() - start_time))
