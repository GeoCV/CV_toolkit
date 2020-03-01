import os
import cv2
import json

input_imgs = '../../scrapy_structure/military/USNI_images/'
files = os.listdir(input_imgs)
input_json = './location.json'
output_json = './labeled_location.json'

with open(input_json, 'r')as c_f:
    s = c_f.readline()
    obj_j = json.loads(s)
    for o in obj_j:
        print(o['time'])
    # for f in files:
    #     template = cv2.imread(input_path, 0)
    #     h, w = template.shape[:2]
    #     font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
    #     for r in de_res:
    #         cv2.circle(img_rgb, (int(r['location'][1]), int(r['location'][0])), 40, (0, 0, 255), 4)
    #         imgzi = cv2.putText(img_rgb, r['words'], (int(r['location'][1] + w), int(r['location'][0] + h)), font, 1.2, (255, 255, 255), 2)
    #         # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
    #     cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
    #     cv2.imshow('img_rgb', img_blue)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
