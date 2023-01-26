import os
import pandas as pd
import numpy as np

params_map = {
    'sediao1020': 'hue',
    'sediao1010': 'hue',
    'sediao990': 'hue',
    'sediao980': 'hue',
    's168': 'saturate',
    's148': 'saturate',
    's108': 'saturate',
    's88': 'saturate',
    'q95': 'quality',
    'q55': 'quality',
    'q35': 'quality',
    'q15': 'quality',
    'c10': 'contrast',
    'c20': 'contrast',
    'c30': 'contrast',
    'c40': 'contrast',
    'brightness42': 'brightness',
    'brightness44': 'brightness',
    'brightness46': 'brightness',
    'brightness40': 'brightness',
}
model_map = {
    'resnet18': 'ResNet18',
    'resnet34': 'ResNet34',
    'resnet50': 'ResNet50',
    'resnet101': 'ResNet101',
    'vgg16': 'VGG16',
    'efficientnet_b7': 'EfficientnetB7',
    'convnext_base': 'ConvNext-B',
    'convnext_large': 'ConvNext-L',
    'swin_small_patch4_window7_224': 'Swin-S',
    'swin_base_patch4_window7_224': 'Swin-B',
    'swin_large_patch4_window7_224': 'Swin-L',
    'vit_small_patch16_224': 'ViT-S',
    'vit_base_patch16_224': 'ViT-B',
    'vit_large_patch16_224': 'ViT-L',
}
model_list = model_map.keys()
model_result_list = os.listdir('./predict_result')
model_list_name = []
for model_result in model_result_list:
    model_name = model_result.split('-')[1]
    model_list_name.append(model_name)
model_list_name = list(set(model_list_name))


# Compute the label with the highest predicted frequency
def top1(lst):
    return max(lst, key=lambda v: lst.count(v))


# Map pos/neg to numeric form
map_dict = {
    'pos': 1,
    'neg2': 0
}

parm_list = list(set(params_map.values()))


# Compute model's prediction accuracy
def test_acc(path):
    slice_dict = {}
    acc_dict = {}
    test_dict = np.load(path)
    for (img_path, pred) in test_dict:
        pred = int(pred)
        parm = img_path.split('/')[-3]
        label_str = img_path.split('/')[-4]
        label = map_dict[label_str]
        if parm in slice_dict.keys():
            if pred == label:
                slice_dict[parm]['t'] += 1
            slice_dict[parm]['c'] += 1
        else:
            slice_dict[parm] = {}
            slice_dict[parm]['t'] = 0
            slice_dict[parm]['c'] = 1
            if pred == label:
                slice_dict[parm]['t'] += 1
    for parm, value in slice_dict.items():
        acc_dict[parm] = value['t'] / value['c']
    return acc_dict


# Compute model's prediction consistency
def test_consistency(path):
    slice_dict = {}
    test_dict = np.load(path)
    for (img_path, pred) in test_dict:
        slice_name = img_path.split('/')[-2]
        parm_ = img_path.split('/')[-3]
        # brightness48 represents the default setting of scanner parameter
        if parm_ == 'brightness48':
            continue
        parm = params_map[parm_]

        # img loc is the unique id of the cropped image in same WSI.
        img_loc = img_path.split('/')[-1].split('.')[-2]
        if parm in slice_dict.keys():
            if slice_name in slice_dict[parm].keys():
                if img_loc in slice_dict[parm][slice_name].keys():
                    slice_dict[parm][slice_name][img_loc].append(pred)
                else:
                    slice_dict[parm][slice_name][img_loc] = []
                    slice_dict[parm][slice_name][img_loc].append(pred)
            else:
                slice_dict[parm][slice_name] = {}
                slice_dict[parm][slice_name][img_loc] = []
                slice_dict[parm][slice_name][img_loc].append(pred)
        else:
            slice_dict[parm] = {}
            slice_dict[parm][slice_name] = {}
            slice_dict[parm][slice_name][img_loc] = []
            slice_dict[parm][slice_name][img_loc].append(pred)
    for (img_path, pred) in test_dict:
        slice_name = img_path.split('/')[-2]
        parm = img_path.split('/')[-3]
        img_loc = img_path.split('/')[-1].split('.')[-2]
        # brightness48 represents the default setting of scanner parameter
        # so use it as the default result for each group parameter
        if parm == 'brightness48':
            for parm2 in parm_list:
                slice_dict[parm2][slice_name][img_loc].append(pred)

    sum_ = 0
    cnt = 0
    result_dict = {}
    for parm, dict_slice in slice_dict.items():
        for slice_name, locs in dict_slice.items():
            for loc in locs:
                cnt += 1
                pred_list = dict_slice[slice_name][loc]
                top1_pred = top1(pred_list)
                np_pred_list = np.array(pred_list)
                # if all the prediction of one image is the same, consistency = 1 , otherwise 0
                sum_ += 1.0 * (np_pred_list == top1_pred).sum() // len(pred_list)
        result_dict[parm] = sum_ / cnt
    return result_dict


model_result = {}
for model in model_list_name:
    sum = 0
    cnt = 0
    flag = 0
    final_result = {}
    final_acc_result = {}
    # Due to the existence of crossvalidation, all five folds are calculated
    for fold in ['0', '1', '2', '3', '4']:
        print(model, fold)
        path = os.path.join('../predict_result/', 'total_pred-' + model + '-' + fold + '.npy')
        if os.path.exists(path):
            cnt += 1
            result_consistency = test_consistency(path)
            result_acc = test_acc(path)
            if flag == 0:
                final_result = result_consistency
                final_acc_result = result_acc
            else:
                for parm, value in final_result.items():
                    final_result[parm] += result_consistency[parm]
                for parm, value in result_acc.items():
                    final_acc_result[parm] += result_acc[parm]
            flag = 1
    if cnt == 0:
        continue
    for parm, value in final_result.items():
        final_result[parm] = final_result[parm] / cnt
    for parm, value in final_acc_result.items():
        final_acc_result[parm] = final_acc_result[parm] / cnt
    model_result[model] = {}
    model_result[model]['all_acc'] = final_acc_result
    model_result[model]['consistency'] = final_result

model_map_lists = list(model_map.keys())
# compute model's prediction consistency
model_result_dict = {}
for model_name, model_value in model_result.items():
    model_result_dict[model_name] = {}
    all_acc = model_result[model_name]["all_acc"]
    for param_, acc in all_acc.items():
        # brightness48 represents the default setting of scanner parameter
        if param_ == 'brightness48':
            model_result_dict[model_name]['default'] = round(acc, 3)
            continue
        param = params_map[param_]
        # Calculate the average accuracy under all scanner parameters except default setting
        if param not in model_result_dict[model_name]:
            model_result_dict[model_name][param] = acc / 4
        else:
            model_result_dict[model_name][param] += acc / 4
        model_result_dict[model_name][param] = round(model_result_dict[model_name][param], 3)
    consistency_result = model_result[model_name]['consistency']
    consistency = 0.0
    # calculate average consistency of all parameter groups
    for param, consis_value in consistency_result.items():
        consistency += consis_value / len(consistency_result)
    model_result_dict[model_name]['consistency'] = round(consistency, 3)

# compute Acc-p and rE-P
for model_name, model_value in model_result_dict.items():
    avg = 0
    for param, acc in model_value.items():
        if param in parm_list:
            avg += acc / len(parm_list)
    model_result_dict[model_name]['Acc-P'] = round(avg, 3)
    model_result_dict[model_name]['rE-P'] = round((1 - avg) / (1 - model_result_dict[model_name]['default']), 3)

# aggregate results of all models
result_final = {}
result_final['model_name'] = []
cnt = 0
print(model_map_lists)
while cnt < len(model_map_lists):
    for model_name, model_value in model_result_dict.items():
        print(cnt)
        if cnt == len(model_map_lists):
            break
        if model_name != model_map_lists[cnt]:
            continue
        else:
            print(model_name)
            cnt += 1
        result_final['model_name'].append(model_map[model_name])
        for param, value in model_value.items():
            if param not in result_final:
                result_final[param] = [value]
            else:
                result_final[param].append(value)

df = pd.DataFrame(result_final)
df = df[
    ['model_name', 'default', 'brightness', 'contrast', 'quality', 'saturate', 'hue', 'Acc-P', 'rE-P', 'consistency']]
df = df.set_index('model_name')
# excel format result
df.to_excel('result.xlsx')
# latex format result
print(df.to_latex())
