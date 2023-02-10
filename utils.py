import os,glob
import pandas as pd
import h5py
import numpy as np
import torch
import random
from skimage.metrics import hausdorff_distance
from monai.metrics.hausdorff_distance import compute_hausdorff_distance
from skimage import measure
import copy
import SimpleITK as sitk



def cal_score(predict,target):
    overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()

    overlap_measures_filter.Execute(target, predict)
    Jaccard = overlap_measures_filter.GetJaccardCoefficient()
    Dice = overlap_measures_filter.GetDiceCoefficient()
    VolumeSimilarity = overlap_measures_filter.GetVolumeSimilarity()
    FalseNegativeError = overlap_measures_filter.GetFalseNegativeError()
    FalsePositiveError = overlap_measures_filter.GetFalsePositiveError()
    # print(Jaccard,Dice,VolumeSimilarity,FalseNegativeError,FalsePositiveError)
    
    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    
    try:
        hausdorff_distance_filter.Execute(target, predict)
    except RuntimeError:
        result = {
            'Jaccard':Jaccard,
            'Dice':Dice,
            'VolumeSimilarity':VolumeSimilarity,
            'FalseNegativeError':FalseNegativeError,
            'FalsePositiveError':FalsePositiveError,
            'HausdorffDistance':np.nan,
            'HausdorffDistance95':np.nan
        }
        return result
    HausdorffDistance = hausdorff_distance_filter.GetHausdorffDistance()
    # Symmetric surface distance measures
    segmented_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(predict, squaredDistance=False, useImageSpacing=True))
    segmented_surface = sitk.LabelContour(predict)

    # Use the absolute values of the distance map to compute the surface distances (distance map sign, outside or inside 
    # relationship, is irrelevant)
    # label = 1
    reference_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(target, squaredDistance=False))
    reference_surface = sitk.LabelContour(target)

    statistics_image_filter = sitk.StatisticsImageFilter()
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(reference_surface)
    num_reference_surface_pixels = int(statistics_image_filter.GetSum()) 
        
    # Multiply the binary surface segmentations with the distance maps. The resulting distance
    # maps contain non-zero values only on the surface (they can also contain zero on the surface)
    seg2ref_distance_map = reference_distance_map*sitk.Cast(segmented_surface, sitk.sitkFloat32)
    ref2seg_distance_map = segmented_distance_map*sitk.Cast(reference_surface, sitk.sitkFloat32)
        
    # Get the number of pixels in the reference surface by counting all pixels that are 1.
    statistics_image_filter.Execute(segmented_surface)
    num_segmented_surface_pixels = int(statistics_image_filter.GetSum())
    
    # Get all non-zero distances and then add zero distances if required.
    seg2ref_distance_map_arr = sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances = list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0]) 
    seg2ref_distances = seg2ref_distances + \
                        list(np.zeros(num_segmented_surface_pixels - len(seg2ref_distances)))
    ref2seg_distance_map_arr = sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances = list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0]) 
    ref2seg_distances = ref2seg_distances + \
                        list(np.zeros(num_reference_surface_pixels - len(ref2seg_distances)))
        
    all_surface_distances = seg2ref_distances + ref2seg_distances

    # The maximum of the symmetric surface distances is the Hausdorff distance between the surfaces. In 
    # general, it is not equal to the Hausdorff distance between all voxel/pixel points of the two 
    # segmentations, though in our case it is. More on this below.
    # mean_surface_distance = np.mean(all_surface_distances)
    # median_surface_distance = np.median(all_surface_distances)
    # std_surface_distance = np.std(all_surface_distances)
    # max_surface_distance = np.max(all_surface_distances)
    HausdorffDistance95 = np.percentile(all_surface_distances,95)
    # print(hd_95)
    # print(HausdorffDistance)
    result = {
        'Jaccard':Jaccard,
        'Dice':Dice,
        'VolumeSimilarity':VolumeSimilarity,
        'FalseNegativeError':FalseNegativeError,
        'FalsePositiveError':FalsePositiveError,
        'HausdorffDistance':HausdorffDistance,
        'HausdorffDistance95':HausdorffDistance95
    }
    return result



# def post_seg(seg_result,post_index=None): 
#     seg_result = copy.deepcopy(seg_result)
#     for i in post_index:
#         tmp_seg_result = (seg_result == i).astype(np.float32)
#         labels = measure.label(tmp_seg_result)
#         area = []
#         for j in range(1,np.amax(labels) + 1):
#             area.append(np.sum(labels == j))
#         if len(area) != 0:
#             tmp_seg_result[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
#         seg_result[seg_result == i] = 0
#         seg_result[tmp_seg_result == 1] = i
#     return seg_result


def post_seg(seg_result,post_index=None,keep_max=True,keep_number=3): 
    seg_result = copy.deepcopy(seg_result)
    for roi in post_index:
        tmp_seg_result = (seg_result == roi).astype(np.float32)
        labels = measure.label(tmp_seg_result)
        area = []
        for j in range(1,np.amax(labels) + 1):
            area.append(np.sum(labels == j))
        if keep_max:
            if len(area) != 0:
                tmp_seg_result[np.logical_and(labels > 0, labels != np.argmax(area) + 1)] = 0
            seg_result[seg_result == roi] = 0
            seg_result[tmp_seg_result == 1] = roi
        else:
            if len(area) != 0:
                area_dict = {}
                for i in range(len(area)):
                    area_dict[i+1] = area[i]
                area_list = sorted(area_dict.items(), key=lambda x:x[1])
                
                for i in range(max(0,len(area_list) - keep_number)):
                    tmp_seg_result[np.logical_and(labels > 0, labels == area_list[i][0])] = 0
            seg_result[seg_result == roi] = 0
            seg_result[tmp_seg_result == 1] = roi

    return seg_result


def ensemble(array,num_classes):
    # print(array.shape)
    _C = array.shape[0]
    result = np.zeros(array.shape[1:],dtype=np.uint8)
    for i in range(num_classes):
        roi = np.sum((array == i+1).astype(np.uint8),axis=0)
        # print(roi.shape)
        result[roi > (_C // 2)] = i+1
    return result

    

def binary_dice(y_true, y_pred):
    smooth = 1e-7
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)

def multi_dice(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    dice_list = []
    for i in range(num_classes):
        dice = cal_score(predict==i+1,target==i+1)['Dice']
        dice_list.append(dice)
    
    dice_list = [round(case, 4) for case in dice_list]
    
    return dice_list, round(np.mean(dice_list),4)

# def multi_dice(y_true,y_pred,num_classes):
#     dice_list = []
#     for i in range(num_classes):
#         true = (y_true == i+1).astype(np.float32)
#         pred = (y_pred == i+1).astype(np.float32)
#         dice = binary_dice(true,pred)
#         dice_list.append(dice)
    
#     dice_list = [round(case, 4) for case in dice_list]
    
#     return dice_list, round(np.mean(dice_list),4)


# def hd_2d(true,pred):
#     hd_list = []
#     for i in range(true.shape[0]):
#         if np.sum(true[i]) != 0 and np.sum(pred[i]) != 0:
#             hd_list.append(hausdorff_distance(true[i],pred[i]))
    
#     return np.mean(hd_list)


def multi_hd(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    hd_list = []
    for i in range(num_classes):
        hd = cal_score(predict==i+1,target==i+1)['HausdorffDistance95']
        hd_list.append(hd)
    
    hd_list = [round(case, 4) for case in hd_list]
    
    return hd_list, round(np.mean(hd_list),4)


# def multi_hd(y_true,y_pred,num_classes):
#     hd_list = []
#     for i in range(num_classes):
#         true = (y_true == i+1).astype(np.float32)
#         pred = (y_pred == i+1).astype(np.float32)
#         hd = hd_2d(true,pred)
#         hd_list.append(hd)
    
#     hd_list = [round(case, 4) for case in hd_list]
    
#     return hd_list, round(np.mean(hd_list),4)



def multi_vs(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    vs_list = []
    for i in range(num_classes):
        vs = cal_score(predict==i+1,target==i+1)['VolumeSimilarity']
        vs_list.append(vs)
    
    vs_list = [round(case, 4) for case in vs_list]
    
    return vs_list, round(np.mean(vs_list),4)



def multi_jc(y_true,y_pred,num_classes):
    predict = copy.deepcopy(y_pred)
    target = copy.deepcopy(y_true)
    predict = sitk.GetImageFromArray(predict)
    target = sitk.GetImageFromArray(target)
    predict = sitk.Cast(predict,sitk.sitkUInt8)
    target = sitk.Cast(target,sitk.sitkUInt8)
    jc_list = []
    for i in range(num_classes):
        jc = cal_score(predict==i+1,target==i+1)['Jaccard']
        jc_list.append(jc)
    
    jc_list = [round(case, 4) for case in jc_list]
    
    return jc_list, round(np.mean(jc_list),4)



def cal_asd(predict,target):

    from monai.metrics.surface_distance import SurfaceDistanceMetric

    asd_cls = SurfaceDistanceMetric(symmetric=True)
    asd = asd_cls._compute_tensor(predict,target)
    
    return asd.numpy()[0][0]


def multi_asd(y_true, y_pred, num_classes):
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    asd_list = []
    for i in range(num_classes):
        roi_pred = (y_pred==i+1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0) # 11HWD
        roi_true = (y_true==i+1).permute(1, 2, 0).contiguous().unsqueeze(0).unsqueeze(0)

        # print(roi_pred.size())
        # print(roi_true.size())

        asd = cal_asd(roi_pred,roi_true)
        asd_list.append(asd)
    
    asd_list = [round(case, 4) for case in asd_list]
    
    return asd_list, round(np.mean(asd_list),4)


def hdf5_reader(data_path, key):
    hdf5_file = h5py.File(data_path, 'r')
    image = np.asarray(hdf5_file[key], dtype=np.float32)
    hdf5_file.close()

    return image


def save_as_nii(data, save_path):
    sitk_data = sitk.GetImageFromArray(data)
    sitk_data.SetSpacing([1.17188, 1.17188, 5])
    sitk.WriteImage(sitk_data, save_path)


def get_path_with_annotation(input_path,path_col,tag_col):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    final_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            final_list.append(path)
    
    return final_list


def get_path_with_annotation_ratio(input_path,path_col,tag_col,ratio=0.5):
    path_list = pd.read_csv(input_path)[path_col].values.tolist()
    tag_list = pd.read_csv(input_path)[tag_col].values.tolist()
    with_list = []
    without_list = []
    for path, tag in zip(path_list,tag_list):
        if tag != 0:
            with_list.append(path)
        else:
            without_list.append(path)
    if int(len(with_list)/ratio) < len(without_list):
        random.shuffle(without_list)
        without_list = without_list[:int(len(with_list)/ratio)]    
    return with_list + without_list


def count_params_and_macs(net,input_shape):
    
    from thop import profile
    input = torch.randn(input_shape)
    input = input.cuda()
    macs, params = profile(net, inputs=(input, ))
    print('%.3f GFLOPs' %(macs/10e9))
    print('%.3f M' % (params/10e6))



def get_weight_path(ckpt_path):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) != 0:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            return os.path.join(ckpt_path,pth_list[-1])
        else:
            return None
    else:
        return None
    
    

def remove_weight_path(ckpt_path,retain=3):

    if os.path.isdir(ckpt_path):
        pth_list = os.listdir(ckpt_path)
        if len(pth_list) >= retain:
            pth_list.sort(key=lambda x:int(x.split('-')[0].split('=')[-1]))
            for pth_item in pth_list[:-retain]:
                os.remove(os.path.join(ckpt_path,pth_item))


def dfs_remove_weight(ckpt_path,retain=3):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_remove_weight(sub_path.path,retain)
        else:
            remove_weight_path(ckpt_path,retain)
            break  

def rename_weight_path(ckpt_path):
    if os.path.isdir(ckpt_path):
        for pth in os.scandir(ckpt_path):
            if ':' in pth.name:
                new_pth = pth.path.replace(':','=')
                print(pth.name,' >>> ',os.path.basename(new_pth))
                os.rename(pth.path,new_pth)
            else:
                break


def dfs_rename_weight(ckpt_path):
    for sub_path in os.scandir(ckpt_path):
        if sub_path.is_dir():
            dfs_rename_weight(sub_path.path)
        else:
            rename_weight_path(ckpt_path)
            break  
