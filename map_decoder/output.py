import numpy as np
import csv
import os
import cv2
import map_decoder.Decoder as decoder

input_path = '../../scrapy_structure/military/USNI_images/'


# 写入dict数据
def to_csv(data, file='./decode.csv'):
    with open(file, 'w')as f:
        f_csv = csv.DictWriter(f, data[0].keys())
        f_csv.writerows(data)


# 写入一行list数据
def write_line(data, file='./decode.csv'):
    with open(file, 'w')as f:
        f_csv = csv.writer(f)
        f_csv.writerow(data)


csv_file = './decode.csv'
files = os.listdir(input_path)
headers = ['location', 'words', 'box_idx', 'time']
with open(csv_file, 'w')as f:
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    land_mask = []
    for f in files:
        img_path = os.path.join(input_path, f)
        print('----------------processing:' + img_path + '----------------')
        de_res = decoder.Decoder(img_path).decode_res
        for d in de_res:
            d['time'] = f
            f_csv.writerow(d)

        # 叠加出一张陆地mask
        # blue = decoder.Decoder(img_path).get_dark_blue()
        # land_mask.append(blue)

    # input_path1 = './August 28, 2017 10:21 AM'
    # input_path2 = './February 12, 2018 5:15 PM'
    # land_mask.append(decoder.Decoder(input_path1).get_dark_blue())
    # land_mask.append(decoder.Decoder(input_path2).get_dark_blue())
    # land_mask = np.array(land_mask)
    # land_mask = np.average(land_mask, axis=0)
    # land_avg = np.where(land_mask > 16)
    # land_mask.fill(0)
    # land_mask[land_avg] = 255
    #
    # cv2.imwrite('./land_mask_16.jpg', land_mask)
    # cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_rgb', land_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
