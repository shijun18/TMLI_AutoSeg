import os 
import numpy as np
import json
import pandas as pd
from utils import hdf5_reader



def data_check(input_path,annotation_list):
    count = 0
    slice_num = 0
    csv_info = []
    error_list = []
    for item in os.scandir(input_path):
        csv_item = []
        print(item.name)
        csv_item.append(item.name)
        img = hdf5_reader(item.path,'image')
        lab = hdf5_reader(item.path,'label')
        print(img.shape)
        slice_num += img.shape[0]
        csv_item.append(img.shape[0])
        print(np.max(img),np.min(img))
        print(np.unique(lab))
        if len(np.unique(lab)) != len(annotation_list) + 1:
            print('============= Error %d ============='% count)
            count += 1
            error_list.append(item.name)
        else:
            print('====================================')
        csv_info.append(csv_item)
    print(error_list)
    col = ['id','slices_num']
    csv_file = pd.DataFrame(columns=col, data=csv_info)
    csv_file.to_csv('./data_check.csv', index=False)

    print('total slice: %d'%slice_num)

if __name__ == "__main__":

    # json_file = './static_files/TMLI_config_up.json'
    json_file = './static_files/TMLI_config_down.json'

    with open(json_file, 'r') as fp:
        info = json.load(fp)
    data_check(info['npy_path'],info['annotation_list'])