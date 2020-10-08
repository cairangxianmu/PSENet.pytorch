import numpy as np
import cv2
import os


for file in os.listdir('img'):
    image = cv2.imread(os.path.join('img', file))
    h, w, c = image.shape

    mask = cv2.imread(os.path.join('mask', file))

    mask = cv2.resize(mask, (w, h))
    overlay = image.copy()
    alpha = 0.4  # 设置覆盖图片的透明度
    cv2.rectangle(overlay, (0, 0), (image.shape[1], image.shape[0]), (255, 255, 255), -1)  # 设置蓝色为热度图基本色
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)  # 将背景热度图覆盖到原图
    cv2.addWeighted(mask, alpha, image, 1 - alpha, 0, image)  #
    image = cv2.resize(image, (1600, 900))

    cv2.imwrite('pre_img/' + 'pr_' + file, image)
    # cv2.imshow("figure", image)
    # cv2.waitKey(0)
    # cv2.destroyWindow()
