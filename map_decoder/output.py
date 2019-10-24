import numpy as np
import csv
import os
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
    for f in files:
        img_path = os.path.join(input_path, f)
        print('----------------processing:' + img_path + '----------------')
        de_res = decoder.Decoder(img_path).decode_res
        for d in de_res:
            d['time'] = f
            f_csv.writerow(d)
