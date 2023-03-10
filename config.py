import os
import json
import glob

from utils import get_path_with_annotation,get_path_with_annotation_ratio
from utils import get_weight_path

__disease__ = ['TMLI','TMLI_UP']
__cnn_net__ = ['unet','unet++','deeplabv3+','sfnet']
__trans_net__ = ['UTNet','TransUNet']
__encoder_name__ = ['resnet18','swin_transformer','swinplusr18']
__mode__ = ['cls','seg','mtl']

json_path = {
    'TMLI':'/staff/shijun/torch_projects/TMLI/converter/static_files/TMLI_config.json',
    'TMLI_UP':'/staff/shijun/torch_projects/TMLI/converter/static_files/TMLI_config_up.json',
    'TMLI_DOWN':'/staff/shijun/torch_projects/TMLI/converter/static_files/TMLI_config_down.json',
}
    
DISEASE = 'TMLI_DOWN' 
MODE = 'seg'
NET_NAME = 'sfnet'
ENCODER_NAME = 'swinplusr18'
VERSION = 'v4.3-roi'


with open(json_path[DISEASE], 'r') as fp:
    info = json.load(fp)

DEVICE = '2'
# True if use internal pre-trained model
# Must be True when pre-training and inference
PRE_TRAINED = False
# True if use external pre-trained model 
EX_PRE_TRAINED = True if 'pretrain' in VERSION else False
# True if use resume model
CKPT_POINT = False
# [1-N]
CURRENT_FOLD = 1
GPU_NUM = len(DEVICE.split(','))
FOLD_NUM = 5

# Arguments for trainer initialization
#--------------------------------- single or multiple
ROI_NUMBER = None# or [1-N]
NUM_CLASSES = info['annotation_num'] + 1  # 2 for binary, more for multiple classes
if ROI_NUMBER is not None:
    NUM_CLASSES = 2
    ROI_NAME = info['annotation_list'][ROI_NUMBER - 1]
else:
    ROI_NAME = 'All'
SCALE = info['scale'][ROI_NAME]
#---------------------------------

#--------------------------------- mode and data path setting
#all
PATH_LIST = glob.glob(os.path.join(info['2d_data']['train_path'],'*.hdf5'))

#zero
# PATH_LIST = get_path_with_annotation(info['2d_data']['train_csv_path'],'path',ROI_NAME)

#half
# PATH_LIST = get_path_with_annotation_ratio(info['2d_data']['train_csv_path'],'path',ROI_NAME,ratio=0.5)
#---------------------------------


#--------------------------------- others
INPUT_SHAPE = (512,512)#(512,512) (256,256)
BATCH_SIZE = 64

CKPT_PATH = './ckpt/{}/{}/{}/{}/fold{}'.format(DISEASE,MODE,VERSION,ROI_NAME,str(CURRENT_FOLD))


WEIGHT_PATH = get_weight_path(CKPT_PATH)
print(WEIGHT_PATH)

INIT_TRAINER = {
    'net_name':NET_NAME,
    'encoder_name':ENCODER_NAME,
    'lr':1e-3, 
    'n_epoch':120,
    'channels':1,
    'num_classes':NUM_CLASSES, 
    'roi_number':ROI_NUMBER,
    'scale':SCALE,
    'input_shape':INPUT_SHAPE,
    'crop':0,
    'batch_size':BATCH_SIZE,
    'num_workers':max(8,2*GPU_NUM),
    'device':DEVICE,
    'pre_trained':PRE_TRAINED,
    'ex_pre_trained':EX_PRE_TRAINED,
    'ckpt_point':CKPT_POINT,
    'weight_path':WEIGHT_PATH,
    'use_moco':None if 'moco' not in VERSION else 'moco',
    'weight_decay': 0.0001,
    'momentum': 0.99,
    'gamma': 0.1,
    'milestones': [30,60,90],
    'T_max':5,
    'mode':MODE,
    'topk':20,
    'use_fp16':True, #False if the machine you used without tensor core
    'aux_deepvision':False if 'sup' not in VERSION else True
 }
#---------------------------------

__seg_loss__ = ['CEPlusDice','DiceLoss','Cross_Entropy']
__cls_loss__ = ['BCEWithLogitsLoss']
__mtl_loss__ = ['BCEPlusDice','BCEPlusTopk']
# Arguments when perform the trainer 
loss_index = 0 if len(VERSION.split('.')) == 2 else eval(VERSION.split('.')[-1].split('-')[0])
if MODE == 'cls':
    LOSS_FUN = __cls_loss__[loss_index]
elif MODE == 'seg' :
    LOSS_FUN = 'CEPlusDice' if ROI_NUMBER is not None else __seg_loss__[loss_index] #'CEPlusDice'
else:
    LOSS_FUN = __mtl_loss__[loss_index]

print('>>>>> loss fun:%s'%LOSS_FUN)

SETUP_TRAINER = {
    'output_dir':'./ckpt/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME),
    'log_dir':'./log/{}/{}/{}/{}'.format(DISEASE,MODE,VERSION,ROI_NAME), 
    'optimizer':'AdamW',
    'loss_fun':LOSS_FUN,
    'class_weight':None, #[1,4]
    'lr_scheduler':'CosineAnnealingWarmRestarts',#'CosineAnnealingWarmRestarts','MultiStepLR',
    'freeze_encoder': False if 'freeze' not in VERSION else True,
    'get_roi': False if 'roi' not in VERSION else True,
    'monitor': 'val_acc' if MODE == 'cls' else 'val_run_dice'
  }
#---------------------------------
TEST_PATH = None