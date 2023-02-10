import os
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from data_utils.data_loader import DataGenerator, To_Tensor, CropResize, Trunc_and_Normalize
from data_utils.transformer import Get_ROI
from torch.cuda.amp import autocast as autocast
from utils import get_weight_path,multi_dice,multi_hd,multi_asd,post_seg,ensemble
from utils import hdf5_reader, save_as_nii
import warnings
from skimage.transform import resize
warnings.filterwarnings('ignore')

def resize_and_pad(pred,true,num_classes,target_shape,bboxs):
    from skimage.transform import resize
    final_pred = []
    final_true = []

    for bbox, pred_item, true_item in zip(bboxs,pred,true):
        h,w = bbox[2]-bbox[0], bbox[3]-bbox[1]
        new_pred = np.zeros(target_shape,dtype=np.float32)
        new_true = np.zeros(target_shape,dtype=np.float32)
        for z in range(1,num_classes):
            roi_pred = resize((pred_item == z).astype(np.float32),(h,w),mode='constant')
            new_pred[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_pred>=0.5] = z
            roi_true = resize((true_item == z).astype(np.float32),(h,w),mode='constant')
            new_true[bbox[0]:bbox[2],bbox[1]:bbox[3]][roi_true>=0.5] = z
        final_pred.append(new_pred)
        final_true.append(new_true)
    
    final_pred = np.stack(final_pred,axis=0)
    final_true = np.stack(final_true,axis=0)
    return final_pred, final_true


def get_net(net_name,encoder_name,channels=1,num_classes=2,input_shape=(512,512),**kwargs):

    if net_name == 'unet':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.Unet(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )
            
    elif net_name == 'unet++':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.UnetPlusPlus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )

    
    elif net_name == 'deeplabv3+':
        if encoder_name is None:
            raise ValueError(
                "encoder name must not be 'None'!"
            )
        else:
            import segmentation_models_pytorch as smp
            net = smp.DeepLabV3Plus(
                encoder_name=encoder_name,
                encoder_weights=None,
                in_channels=channels,
                classes=num_classes,                     
                aux_params={"classes":num_classes-1} 
            )

    elif net_name == 'sfnet':
            from model.sfnet import sfnet
            net = sfnet(net_name,
            encoder_name=encoder_name,
            in_channels=channels,
            classes=num_classes,
            **kwargs)
    

    ## external transformer + U-like net
    elif net_name == 'UTNet':
        from model.trans_model.utnet import UTNet
        net = UTNet(channels, base_chan=32,num_classes=num_classes, reduce_size=8, block_list='1234', num_blocks=[1,1,1,1], num_heads=[4,4,4,4], projection='interp', attn_drop=0.1, proj_drop=0.1, rel_pos=True, aux_loss=False, maxpool=True)
    elif net_name =='TransUNet':
        from model.trans_model.transunet import VisionTransformer as ViT_seg
        from model.trans_model.transunet import CONFIGS as CONFIGS_ViT_seg
        config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
        config_vit.n_classes = num_classes 
        config_vit.n_skip = 3 
        config_vit.patches.grid = (int(input_shape[0]/16), int(input_shape[1]/16))
        net = ViT_seg(config_vit, img_size=input_shape[0], num_classes=num_classes)

    
    return net


def eval_process(test_path,config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.device
    device = torch.device("cuda:0")
    # device = torch.device("cpu")
    # data loader
    test_transformer = transforms.Compose([
                Trunc_and_Normalize(config.scale),
                Get_ROI(pad_flag=False) if config.get_roi else transforms.Lambda(lambda x:x),
                CropResize(dim=config.input_shape,num_class=config.num_classes,crop=config.crop),
                To_Tensor(num_class=config.num_classes)
            ])

    test_dataset = DataGenerator(test_path,
                                roi_number=config.roi_number,
                                num_class=config.num_classes,
                                transform=test_transformer)

    test_loader = DataLoader(test_dataset,
                            batch_size=config.batch_size,
                            shuffle=False,
                            num_workers=4,
                            pin_memory=True)
    
    # get weight
    weight_path = get_weight_path(config.ckpt_path)
    print(weight_path)

    s_time = time.time()
    # get net
    net = get_net(config.net_name,
            config.encoder_name,
            config.channels,
            config.num_classes,
            config.input_shape,
            aux_deepvision=config.aux_deepvision,
            aux_classifier=config.aux_classifier
    )
    checkpoint = torch.load(weight_path,map_location='cpu')
    # print(checkpoint['state_dict'])
    msg=net.load_state_dict(checkpoint['state_dict'],strict=False)
    print(msg)
    print('missing key:',msg[0])
    get_net_time = time.time() - s_time
    print('define net and load weight need time:%.3f'%(get_net_time))

    pred = []
    true = []
    s_time = time.time()
    # net = net.cuda()
    # print(device)
    net = net.to(device)
    net.eval()
    move_time = time.time()- s_time
    print('move net to GPU need time:%.3f'%(move_time))

    extra_time = 0.
    with torch.no_grad():
        for step, sample in enumerate(test_loader):
            data = sample['image']
            target = sample['mask']
            ####
            # data = data.cuda()
            data = data.to(device)
            with autocast(True):
                output = net(data)
                
            if isinstance(output,tuple) or isinstance(output,list):
                seg_output = output[0]
            else:
                seg_output = output
            # seg_output = torch.argmax(torch.softmax(seg_output, dim=1),1).detach().cpu().numpy() 
            seg_output = torch.argmax(seg_output,1).detach().cpu().numpy()                           
            s_time = time.time()
            target = torch.argmax(target,1).detach().cpu().numpy()
            if config.get_roi:
                bboxs = torch.stack(sample['bbox'],dim=0).cpu().numpy().T
                seg_output,target = resize_and_pad(seg_output,target,config.num_classes,config.input_shape,bboxs)
            pred.append(seg_output)
            true.append(target)
            extra_time += time.time() - s_time
    pred = np.concatenate(pred,axis=0).squeeze().astype(np.uint8)
    true = np.concatenate(true,axis=0).squeeze().astype(np.uint8)
    print('extra time:%.3f'%extra_time)
    return pred,true,extra_time+move_time+get_net_time


class Config:
    num_classes_dict = {
        'TMLI_UP':8,
        'TMLI_DOWN':2
    }
    scale_dict = {
        'TMLI_UP':[-200,600],
        'TMLI_DOWN':[-200,1400]
    }

    disease = 'TMLI_UP'
    mode = 'seg'
    input_shape = (512,512) #(256,256)(512,512)(448,448) 
    num_classes = num_classes_dict[disease]
    scale = scale_dict[disease]
    channels = 1
    crop = 0
    roi_number = None
    net_name = 'sfnet'
    encoder_name = 'swinplusr18'
    version = 'v4.3-roi'
    device = "0"
    fold = 1
    batch_size = 32
    get_roi = False if 'roi' not in version else True
    aux_deepvision = False if 'sup' not in version else True
    aux_classifier = mode != 'seg'
    ckpt_path = f'./ckpt/{disease}/{mode}/{version}/All'


if __name__ == '__main__':

    # test data
    data_path_dict = {
        'TMLI_UP':'../TMLI/up_2d_test_data',
        'TMLI_DOWN':'..TMLI/down_2d_test_data',
    }
    
    start = time.time()
    config = Config()
    data_path = data_path_dict[config.disease]

    sample_list = list(set([case.name.split('_')[0] for case in os.scandir(data_path)]))
    print(f'Sample num: {len(sample_list)}')
    sample_list.sort()
    print(sample_list)
    ensemble_result = {}


    for fold in range(1,6):
        print('>>>>>>>>>>>> Fold%d >>>>>>>>>>>>'%fold)
        total_dice = []
        total_hd = []
        total_asd = []
        info_dice = []
        info_hd = []
        info_asd = []
        config.fold = fold
        config.ckpt_path = f'./ckpt/{config.disease}/{config.mode}/{config.version}/All/fold{str(fold)}'

        save_dir = f'./result/raw_data/{config.disease}/{config.version}'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        time_list = []
        for sample in sample_list:
            info_item_dice = []
            info_item_hd = []
            info_item_asd = []
            info_item_dice.append(sample)
            info_item_hd.append(sample)
            info_item_asd.append(sample)

            time_item = [sample]
            print('>>>>>>>>>>>> %s is being processed'%sample)
            test_path = [case.path for case in os.scandir(data_path) if case.name.split('_')[0] == sample]
            test_path.sort(key=lambda x:eval(x.split('_')[-1].split('.')[0]))

            img = np.stack([hdf5_reader(item,'image') for item in test_path],axis=0)
            print(img.shape)
            print(len(test_path))
            sample_start = time.time()

            pred,true,extra_time = eval_process(test_path,config)
            # print(pred.shape, true.shape)

            total_time = time.time() - sample_start 
            actual_time = total_time - extra_time

            time_item.append(actual_time)
            time_list.append(time_item)

            print('total time:%.3f'%total_time)
            print('actual time:%.3f'%actual_time)
            print("actual fps:%.3f"%(len(test_path)/actual_time))
            # print(pred.shape,true.shape)

            category_dice, avg_dice = multi_dice(true,pred,config.num_classes - 1)
            total_dice.append(category_dice)
            print('category dice:',category_dice)
            print('avg dice: %s'% avg_dice)

            category_hd, avg_hd = multi_hd(true,pred,config.num_classes - 1)
            total_hd.append(category_hd)
            print('category hd:',category_hd)
            print('avg hd: %s'% avg_hd)

            category_asd, avg_asd = multi_asd(true,pred,config.num_classes - 1)
            total_asd.append(category_asd)
            print('category asd:',category_asd)
            print('avg asd: %s'% avg_asd)

            info_item_dice.extend(category_dice)
            info_item_hd.extend(category_hd)
            info_item_asd.extend(category_asd)

            info_dice.append(info_item_dice)
            info_hd.append(info_item_hd)
            info_asd.append(info_item_asd)


            if sample not in ensemble_result:
                ensemble_result[sample] = {
                    'true':[true],
                    'pred':[],
                    'img':[img]
                }
            ensemble_result[sample]['pred'].append(pred)

        dice_csv = pd.DataFrame(data=info_dice)
        hd_csv = pd.DataFrame(data=info_hd)
        asd_csv = pd.DataFrame(data=info_asd)
        time_csv = pd.DataFrame(data=time_list)
        
        dice_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_dice.csv'))
        hd_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_hd.csv'))
        asd_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_asd.csv'))
        time_csv.to_csv(os.path.join(save_dir,f'fold{config.fold}_time.csv'))

        total_dice = np.stack(total_dice,axis=0) #sample*classes
        total_category_dice = np.mean(total_dice,axis=0)
        total_avg_dice = np.mean(total_category_dice)

        print('total category dice mean:',total_category_dice)
        print('total category dice std:',np.std(total_dice,axis=0))
        print('total dice mean: %s'% total_avg_dice)


        total_hd = np.stack(total_hd,axis=0) #sample*classes
        total_category_hd = np.mean(total_hd,axis=0)
        total_avg_hd = np.mean(total_category_hd)

        print('total category hd mean:',total_category_hd)
        print('total category hd std:',np.std(total_hd,axis=0))
        print('total hd mean: %s'% total_avg_hd)


        total_asd = np.stack(total_asd,axis=0) #sample*classes
        total_category_asd = np.mean(total_asd,axis=0)
        total_avg_asd = np.mean(total_category_asd)

        print('total category asd mean:',total_category_asd)
        print('total category asd std:',np.std(total_asd,axis=0))
        print('total asd mean: %s'% total_avg_asd)

        print("runtime:%.3f"%(time.time() - start))

    #### for ensemble and post-processing

    ensemble_info_dice = []
    ensemble_info_hd = []
    ensemble_info_asd = []
    post_ensemble_info_dice = []
    post_ensemble_info_hd = []
    post_ensemble_info_asd = []

    for sample in sample_list:
        print('>>>> %s in post processing'%sample)
        ensemble_pred = ensemble(np.stack(ensemble_result[sample]['pred'],axis=0),config.num_classes - 1)
        ensemble_true = ensemble_result[sample]['true'][0]
        img = ensemble_result[sample]['img'][0]

        category_dice, avg_dice = multi_dice(ensemble_true,ensemble_pred,config.num_classes - 1)
        category_hd, avg_hd = multi_hd(ensemble_true,ensemble_pred,config.num_classes - 1)
        category_asd, avg_asd = multi_asd(ensemble_true,ensemble_pred,config.num_classes - 1)

        post_ensemble_pred = post_seg(ensemble_pred,list(range(1,config.num_classes - 1)),keep_max=False)
        post_category_dice, post_avg_dice = multi_dice(ensemble_true,post_ensemble_pred,config.num_classes - 1)
        post_category_hd, post_avg_hd = multi_hd(ensemble_true,post_ensemble_pred,config.num_classes - 1)
        post_category_asd, post_avg_asd = multi_asd(ensemble_true,post_ensemble_pred,config.num_classes - 1)


        ### save result as nii
        gt = ensemble_true
        pred = ensemble_pred
        from utils import save_as_nii
        nii_path = os.path.join(save_dir,'nii')
        if not os.path.exists(nii_path):
            os.makedirs(nii_path)
        img_path = os.path.join(nii_path, sample + '_image.nii.gz')
        lab_path = os.path.join(nii_path, sample + '_label.nii.gz')
        gt_path = os.path.join(nii_path, sample + '_gt.nii.gz')
        

        if img.shape != pred.shape:
            print('resize to the same size!!')
            dim = img.shape
            temp_mask = np.zeros_like(img,dtype=np.float32)
            for z in range(config.num_classes - 1):
                roi = resize((pred == z+1).astype(np.float32),dim,mode='constant')
                temp_mask[roi >= 0.5] = z+1
            pred = temp_mask.astype(np.uint8)

        assert img.shape == pred.shape

        save_as_nii(img.astype(np.int16),img_path)
        save_as_nii(pred,lab_path) 
        save_as_nii(gt.astype(np.uint8),gt_path) 
        ###

        print('ensemble category dice:',category_dice)
        print('ensemble avg dice: %s'% avg_dice)
        print('ensemble category hd:',category_hd)
        print('ensemble avg hd: %s'% avg_hd)
        print('ensemble category asd:',category_asd)
        print('ensemble avg asd: %s'% avg_asd)


        print('post ensemble category dice:',post_category_dice)
        print('post ensemble avg dice: %s'% post_avg_dice)
        print('post ensemble category hd:',post_category_hd)
        print('post ensemble avg hd: %s'% post_avg_hd)
        print('post ensemble category asd:',post_category_asd)
        print('post ensemble avg asd: %s'% post_avg_asd)


        ensemble_item_dice = [sample]
        ensemble_item_hd = [sample]
        ensemble_item_asd = [sample]
        post_ensemble_item_dice = [sample]
        post_ensemble_item_hd = [sample]
        post_ensemble_item_asd = [sample]
        
        ensemble_item_dice.extend(category_dice)
        ensemble_item_hd.extend(category_hd)
        ensemble_item_asd.extend(category_asd)
        post_ensemble_item_dice.extend(post_category_dice)
        post_ensemble_item_hd.extend(post_category_hd)
        post_ensemble_item_asd.extend(post_category_asd)
        

        ensemble_info_dice.append(ensemble_item_dice)
        ensemble_info_hd.append(ensemble_item_hd)
        ensemble_info_asd.append(ensemble_item_asd)
        post_ensemble_info_dice.append(post_ensemble_item_dice)
        post_ensemble_info_hd.append(post_ensemble_item_hd)
        post_ensemble_info_asd.append(post_ensemble_item_asd)
    

    ensemble_dice_csv = pd.DataFrame(data=ensemble_info_dice)
    ensemble_hd_csv = pd.DataFrame(data=ensemble_info_hd)
    ensemble_asd_csv = pd.DataFrame(data=ensemble_info_asd)
    post_ensemble_dice_csv = pd.DataFrame(data=post_ensemble_info_dice)
    post_ensemble_hd_csv = pd.DataFrame(data=post_ensemble_info_hd)
    post_ensemble_asd_csv = pd.DataFrame(data=post_ensemble_info_asd)

    
    ensemble_dice_csv.to_csv(os.path.join(save_dir,f'ensemble_dice.csv'))
    ensemble_hd_csv.to_csv(os.path.join(save_dir,f'ensemble_hd.csv'))
    ensemble_asd_csv.to_csv(os.path.join(save_dir,f'ensemble_asd.csv'))
    post_ensemble_dice_csv.to_csv(os.path.join(save_dir,f'post_ensemble_dice.csv'))
    post_ensemble_hd_csv.to_csv(os.path.join(save_dir,f'post_ensemble_hd.csv'))
    post_ensemble_asd_csv.to_csv(os.path.join(save_dir,f'post_ensemble_asd.csv'))
    #### end