# -*- coding: utf-8 -*-
# @Time    : 2018/6/11 15:54
# @Author  : zhoujun
import torch
import shutil
import numpy as np
import config
import os
import cv2
from torch import nn
from tqdm import tqdm
from models import PSENet
from predict import Pytorch_model
from cal_recall.script import cal_recall_precison_f1
from torchvision import transforms
from pse import decode as pse_decode
from utils.utils import load_checkpoint, save_checkpoint, setup_logger, GetPolygon
from utils import draw_bbox

torch.backends.cudnn.benchmark = True

def get_bbox(bbox, i):
    global col_len
    x_l, y_l = 0, 0
    x_u, y_u, x_d, y_d = bbox[i]
    x_u_, y_u_, x_d_, y_d_ = bbox[i - 1]
    if y_u == y_u_ and y_u == 0 and x_u != 0:
        x_l = (x_d_ - x_u) // 2
        x_u += x_l
    elif y_u != y_u_ and x_u == 0 and y_u != 0:
        col_len = y_d_
        y_l = (col_len - y_u) // 2
        y_u += y_l
    elif y_u == y_u_ and x_u != 0 and y_u != 0:
        x_l = (x_d_ - x_u) // 2
        x_u += x_l
        y_l = (col_len - y_u) // 2
        y_u += y_l
    return x_u, y_u, x_d, y_d, x_l, y_l


def eval(model, save_path, test_path, device):
    model.eval()
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    img_path = os.path.join(test_path, 'img')
    gt_path = os.path.join(test_path, 'gt')
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    long_size = 2240
    # 预测所有测试图片
    img_paths = [os.path.join(img_path, x) for x in os.listdir(img_path)]
    for img_path in tqdm(img_paths, desc='test models'):
        img_name = os.path.basename(img_path).split('.')[0]
        save_name = os.path.join(save_path, 'res_' + img_name + '.txt')

        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        # if max(h, w) > long_size:
        scale = long_size / max(h, w)
        img = cv2.resize(img, None, fx=scale, fy=scale)
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = transforms.ToTensor()(img)
        tensor = tensor.unsqueeze_(0)
        tensor = tensor.to(device)
        with torch.no_grad():
            preds = model(tensor)
            preds, boxes_list = pse_decode(preds[0], config.scale)
            scale = (preds.shape[1] * 1.0 / w, preds.shape[0] * 1.0 / h)
            if len(boxes_list):
                boxes_list = boxes_list / scale
        np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # 开始计算 recall precision f1
    result_dict = cal_recall_precison_f1(gt_path, save_path)
    return result_dict['recall'], result_dict['precision'], result_dict['hmean']


def merge_eval(model, save_path, test_path, device, base_path=r'', use_sub=True):
    def get_filename_map(json_file):
        merge_images_ids_to_corner = {image_info["id"]: [] for image_info in json_file["images"]}
        merge_filename_to_info = {image_info["filename"]: image_info for image_info in
                                  json_file["images"]}
        for sub_image_info in json_file["sub_images"]:
            merge_images_ids_to_corner[sub_image_info["id"]].append(sub_image_info["corner"])
        merge_filename_to_corner = {image_info["filename"]: merge_images_ids_to_corner[image_info["id"]] for image_info
                                    in
                                    json_file["images"]}
        return merge_filename_to_corner, merge_filename_to_info

    from utils.metric.metrics import GetScore
    import json

    model.eval()

    metric_cls = GetScore()
    get_polygon = GetPolygon()
    raw_metrics = []
    # torch.cuda.empty_cache()  # speed up evaluating after training finished
    with open(test_path)as f:
        gt_file = json.load(f)

        if os.path.exists(save_path):
            shutil.rmtree(save_path, ignore_errors=True)
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        # 预测所有测试图片

        imgs_name = [x['filename'] for x in gt_file["images"]]
        filename_to_corner, filename_to_info = get_filename_map(gt_file)

        for img_name in tqdm(imgs_name, desc="test models"):
            img_path = os.path.join(base_path + 'img/', img_name)
            assert os.path.exists(img_path), 'file is not exists'
            img = cv2.imread(img_path)
            h, w = img.shape[:2]
            if use_sub:
                merge_pred = torch.zeros([config.n, h, w], requires_grad=False)
                all_corner = filename_to_corner[img_name]
                for index, corner in enumerate(all_corner):
                    sub_im = img[corner[1]:corner[3], corner[0]:corner[2]]
                    #print('corner shape', sub_im.shape)
                    # h, w = sub_im.shape[:2]
                    # scale = long_size / max(h, w)
                    # sub_im = cv2.resize(sub_im, None, fx=scale, fy=scale)
                    # 将图片由(w,h)变为(1,img_channel,h,w)
                    tensor = transforms.ToTensor()(sub_im)
                    tensor = tensor.unsqueeze_(0)
                    tensor = tensor.to(device)
                    with torch.no_grad():
                        preds = model(tensor)
                        if index == 0:
                            merge_pred[:, corner[1]:corner[3], corner[0]:corner[2]] = preds[0]
                        else:
                            x_u, y_u, x_d, y_d, x_l, y_l = get_bbox(all_corner, index)
                            # print('cut_shape', x_l, y_l)


                            # print('merge_shape', merge_pred[:, y_u:y_d, x_u:x_d].shape,
                            #       'preds_old_shape', preds[0].shape,
                            #       'preds_shape', preds[0, :, y_l:, x_l:].shape)
                            merge_pred[:, y_u:y_d, x_u:x_d] = preds[0, :, y_l:, x_l:]
                preds = merge_pred
            else:
                # long_size = 3456
                # scale = long_size / max(h, w)
                # img = cv2.resize(img, None, fx=scale, fy=scale)
                # 将图片由(w,h)变为(1,img_channel,h,w)
                tensor = transforms.ToTensor()(img)
                tensor = tensor.unsqueeze_(0)
                tensor = tensor.to(device)
                with torch.no_grad():
                    preds = model(tensor)
                    preds = preds[0]
            preds = pse_decode(preds)
            m = (preds * 100).astype(np.uint8)
            m = cv2.resize(m, (1280, 720))
            # cv2.imshow("figure", m)
            # cv2.waitKey(0)


            preds[preds > 0] = 1

            pred_polygon = get_polygon(pred_map=preds, dest_shape=[h, w])
            raw_metrics.append(metric_cls(pred_polys=pred_polygon,
                                          ignore_tags=np.zeros(len(filename_to_info[img_name]["polygon"], ),
                                                               dtype=np.bool),
                                          polys=filename_to_info[img_name]["polygon"]))
    metrics = metric_cls.gather_measure(raw_metrics)
    return metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg


def main(model_path, backbone, scale, path, save_path, gpu_id):
    device = torch.device("cuda:" + str(gpu_id))
    logger = setup_logger(os.path.join(config.output_dir, 'test_log'))
    logger.info(config.print())
    if os.path.exists(save_path):
        shutil.rmtree(save_path, ignore_errors=True)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_img_folder = os.path.join(save_path, 'img')
    if not os.path.exists(save_img_folder):
        os.makedirs(save_img_folder)
    save_txt_folder = os.path.join(save_path, 'result')
    if not os.path.exists(save_txt_folder):
        os.makedirs(save_txt_folder)
    img_paths = [os.path.join(path, x) for x in os.listdir(path)]
    net = PSENet(backbone=backbone, pretrained=config.pretrained, result_num=config.n)
    model = Pytorch_model(model_path, net=net, scale=scale, gpu_id=gpu_id)
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    model = nn.DataParallel(model)
    recall, precision, f1 = merge_eval(model=model, save_path=os.path.join(config.output_dir, 'output'),
                                       test_path=config.testroot, device=device, base_path=config.base_path,
                                       use_sub=config.use_sub)
    logger.info('test: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format(recall, precision, f1))

    # total_frame = 0.0
    # total_time = 0.0
    # for img_path in tqdm(img_paths):
    #     img_name = os.path.basename(img_path).split('.')[0]
    #     save_name = os.path.join(save_txt_folder, 'res_' + img_name + '.txt')
    #     _, boxes_list, t = model.predict(img_path)
    #     total_frame += 1
    #     total_time += t
    #     # img = draw_bbox(img_path, boxes_list, color=(0, 0, 255))
    #     # cv2.imwrite(os.path.join(save_img_folder, '{}.jpg'.format(img_name)), img)
    #     np.savetxt(save_name, boxes_list.reshape(-1, 8), delimiter=',', fmt='%d')
    # print('fps:{}'.format(total_frame / total_time))
    return save_txt_folder


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str('2')
    backbone = 'resnet152'
    scale = 4
    model_path = '/home/host/project/Text_location/Tibet/model/PSENet.pytorch-master/output/psenet_icd2015_resnet152_4gpu_author_crop_adam_MultiStepLR_authorloss/Best_18_r0.368421_p0.252654_f10.299748.pth'
    data_path = './data/Tibetan/img'
    gt_path = './data/Tibetan/label'
    save_path = './result/_scale{}'.format(scale)
    gpu_id = 0
    print('backbone:{},scale:{},model_path:{}'.format(backbone,scale,model_path))
    save_path = main(model_path, backbone, scale, data_path, save_path, gpu_id=gpu_id)

    # result = cal_recall_precison_f1(gt_path=gt_path, result_path=save_path)
    # print(result)
    # print(cal_recall_precison_f1('/data2/dataset/ICD151/test/gt', '/data1/zj/tensorflow_PSENet/tmp/'))
