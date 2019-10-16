import cv2
import numpy as np
from skimage.measure import compare_ssim

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
    threshold = 0.95

    loc = np.where(res >= threshold)  # 匹配程度大于95%的坐标y,x
    boxes_list = []
    nms_list = []
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        right_bottom = (pt[0] + w, pt[1] + h)
        boxes_list.append((pt[1], pt[0], pt[1] + h, pt[0] + w))
        # cv2.rectangle(img_rgb, pt, right_bottom, (0, 0, 255), 2)
    nms_list.append(boxes_list[0])
    #去除重复和相近的点
    for box in boxes_list:
        flag = 0
        for i, nms in enumerate(nms_list):
            if compute_iou(box, nms) < 0.14 and flag == 0:
                continue
            elif box != nms and flag == 0:
                flag = 1
                img_nms = img_gray[nms[0]:nms[2], nms[1]:nms[3]]
                img_box = img_gray[box[0]:box[2], box[1]:box[3]]
                nms_ssim = compare_ssim(img_nms, template, multichannel=False)
                box_ssim = compare_ssim(img_box, template, multichannel=False)
                if box_ssim > nms_ssim:
                    nms_list[i] = box
            else:
                flag = 1
        if flag != 1:
            nms_list.append(box)
    print(nms_list)
    #显示标记好的图片
    # for nms in nms_list:
    #     cv2.circle(img_rgb, (int(nms[1] + w/2), int(nms[0] + h/2)), 40, (0, 0, 255), 4)
    # cv2.namedWindow('img_rgb', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_rgb', img_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return nms_list


# 求交并比
def compute_iou(rec1, rec2):
    """
    computing IoU
    :param rec1: (y0, x0, y1, x1), which reflects
            (top, left, bottom, right)
    :param rec2: (y0, x0, y1, x1)
    :return: scala value of IoU
    """

    # computing area of each rectangles
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0


def get_ocr(input_path):
    return


site_point(input_path, search_path)
