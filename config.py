# -*- coding: utf-8 -*-
# @Time    : 2019/1/3 17:40
# @Author  : zhoujun

# data config
trainroot = '/data2/dataset/ICD15/train'

output_dir = 'output/psenet_icd2015_resnet152_4gpu_author_crop_adam_MultiStepLR_authorloss12'
data_shape = 860

base_path = r'./data/Tibetan/'
json_path = base_path + 'label/Sub_text.json'
testroot = base_path + 'label/Sub_text.json'

# train config
gpu_id = '0,1'
workers = 4
start_epoch = 0
epochs = 600

train_batch_size = 2

lr = 1e-4
end_lr = 1e-7
lr_gamma = 0.1
lr_decay_step = [200, 400]
weight_decay = 5e-4
warm_up_epoch = 6
warm_up_lr = lr * lr_gamma

display_input_images = False
display_output_images = False
display_interval = 10
show_images_interval = 50

pretrained = True
restart_training = False
checkpoint = ''


use_sub = True
is_output_polygon = True
# net config
backbone = 'resnet50'
Lambda = 0.7
n = 3
m = 0.7
OHEM_ratio = 3
scale = 1
# random seed
seed = 2


def print():
    from pprint import pformat
    tem_d = {}
    for k, v in globals().items():
        if not k.startswith('_') and not callable(v):
            tem_d[k] = v
    return pformat(tem_d)
