import numpy as np
import configparser
import os, sys, base64, json, cv2

input_file = './land_mask_16.jpg'
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "config.ini")
cf = configparser.ConfigParser()


# 拼接静态wgs84ll坐标的图像
def concat_static_map(maps_dir):
    # shape:(999, 999, 3)
    left_top_img = cv2.imread(maps_dir + 'left_top.jpg')
    left_top_img = cv2.cvtColor(left_top_img, cv2.COLOR_BGR2GRAY)
    right_top_img = cv2.imread(maps_dir + 'right_top.jpg')
    right_top_img = cv2.cvtColor(right_top_img, cv2.COLOR_BGR2GRAY)
    left_down_img = cv2.imread(maps_dir + 'left_down.jpg')
    left_down_img = cv2.cvtColor(left_down_img, cv2.COLOR_BGR2GRAY)
    right_down_img = cv2.imread(maps_dir + 'right_down.jpg')
    right_down_img = cv2.cvtColor(right_down_img, cv2.COLOR_BGR2GRAY)

    # 拼接左右两半部分
    for i, left in enumerate(left_top_img):
        if np.sum((left-left_down_img[0, :])**2) == 0:
            left_con_img = np.concatenate((left_top_img[0:i, :], left_down_img), axis=0)
            right_con_img = np.concatenate((right_top_img[0:i, :], right_down_img), axis=0)
            break
    # 左右拼在一起,图片横坐标为0-999
    for i in range(999):
        if np.sum((left_top_img[:, i]-right_top_img[:, 0])**2) == 0:
            final_img = np.concatenate((left_con_img[:, 0:i], right_con_img), axis=1)
            break

    return final_img


concat_static_map('./static_maps/')
