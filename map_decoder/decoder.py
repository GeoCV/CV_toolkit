import cv2
import numpy as np
import sklearn

# 读取图片
input_path = '../../scrapy_structure/military/USNI_images/April 1, 2019 12:07 PM'
search_path = './target.jpg'

def site_point(input_path, search_path):
    img_rgb = cv2.imread(input_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(search_path, 0)
    h, w = template.shape[:2]

    # cv2.matchTemplate标准相关模板匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8

    loc = np.where(res >= threshold)  # 匹配程度大于%80的坐标y,x
    points_list = []
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        right_bottom = (pt[0] + w, pt[1] + h)
        points_list.append((int(pt[0] + w/2), int(pt[1] + h/2)))
        cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
    for point in points_list:
        cv2.circle(img_rgb, point, 1, (0, 0, 255), 4)

    #显示标记好的图片
    # cv2.namedWindow('img_rgb', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
    cv2.imshow('img_rgb', img_rgb)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return points_list


site_point(input_path, search_path)
