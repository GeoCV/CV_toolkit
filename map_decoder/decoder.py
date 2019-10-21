import cv2
import numpy as np
import sys
sys.path.append(".")
import ocr_loader
from skimage.measure import compare_ssim

# 读取图片
input_path = '../../scrapy_structure/military/USNI_images/December 5, 2017 9:49 AM'
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
    points_list = []
    #显示标记好的图片
    for nms in nms_list:
        points_list.append((int(nms[0] + h/2), int(nms[1] + w/2)))
        # cv2.circle(img_rgb, (int(nms[1] + w/2), int(nms[0] + h/2)), 40, (0, 0, 255), 4)
    # cv2.namedWindow('img_rgb', cv2.WINDOW_AUTOSIZE)
    # cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
    # cv2.imshow('img_rgb', img_rgb)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return points_list


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


# 取黄色区域的mask输入ocr_loader中进行OCR
def cls_ocr_res(input_path):
    img_rgb = cv2.imread(input_path)
    yellow1 = np.array([32, 128, 200])
    yellow2 = np.array([128, 255, 255])
    mask_img = cv2.inRange(img_rgb, yellow1, yellow2)
    res = ocr_loader.get_ocr_res(cv2_obj=mask_img)
    return res


# 通过循环遍历,初步找出ocr信息和定位的关系
# def find_relation(points_list, ocr_boxes):
#     decode_res = []
#     for point in points_list:
#         p = np.array(point, dtype=float)
#         min_res = (ocr_boxes[0], 50000)
#         for i, box in enumerate(ocr_boxes):
#             # 取边界框各顶点进行计算
#             box_point = [(box['location']['top'], box['location']['left'])]
#             box_point.append((box['location']['top'], box['location']['left']+box['location']['width']))
#             box_point.append((box['location']['top']+box['location']['height'], box['location']['left']))
#             box_point.append((box['location']['top']+box['location']['height'], box['location']['left']+box['location']['width']))
#             box_np = np.array(box_point)
#             # 加大纵坐标的权重有一定效果
#             # q = np.array([[10, 1], [10, 1], [10, 1], [10, 1]])
#             # print(q)
#             # 计算最小欧式距离
#             box_min = np.sqrt(((box_np - p)**2).sum(axis=1)).min()
#             if min_res[1] > box_min:
#                 min_res = (box, box_np, i)
#         decode_res.append({'location': p, 'words': min_res[0]['words'], 'distance': min_res[1], 'box': min_res[2]})
#     return decode_res


# 利用动态规划和矩阵计算,初步找出ocr信息和定位的关系
def find_relation(points_list, ocr_boxes):
    decode_res = []
    points_np = np.array(points_list, dtype=float)
    boxes_np = np.array(
        [([box['location']['top'], box['location']['left']], [box['location']['top'], box['location']['left']+box['location']['width']], [box['location']['top']+box['location']['height'], box['location']['left']], [box['location']['top']+box['location']['height'], box['location']['left']+box['location']['width']]) for box
         in ocr_boxes], dtype=float)
    # 得到距离矩阵
    for box in boxes_np:
        for point in points_np:
            decode_res.append(np.sqrt(((box - point)**2).sum(axis=1)))
    decode_res = np.array(decode_res).reshape([11, 8, 4])
    print(decode_res)
    print(decode_res.shape)


    # for point in points_list:
    #     p = np.array(point, dtype=float)
    #     min_res = (ocr_boxes[0], 50000)
    #     for i, box in enumerate(ocr_boxes):
    #         # 取边界框各顶点进行计算
    #         box_point = [(box['location']['top'], box['location']['left'])]
    #         box_point.append((box['location']['top'], box['location']['left']+box['location']['width']))
    #         box_point.append((box['location']['top']+box['location']['height'], box['location']['left']))
    #         box_point.append((box['location']['top']+box['location']['height'], box['location']['left']+box['location']['width']))
    #         box_np = np.array(box_point)
    #         # 加大纵坐标的权重有一定效果
    #         # q = np.array([[10, 1], [10, 1], [10, 1], [10, 1]])
    #         # print(q)
    #         # 计算最小欧式距离
    #         box_min = np.sqrt(((box_np - p)**2).sum(axis=1)).min()
    #         if min_res[1] > box_min:
    #             min_res = (box, box_np, i)
    #     decode_res.append({'location': p, 'words': min_res[0]['words'], 'distance': min_res[1], 'box': min_res[2]})
    return decode_res

points_list = site_point(input_path, search_path)
de_res = find_relation(points_list, cls_ocr_res(input_path))


img_rgb = cv2.imread(input_path)
template = cv2.imread(search_path, 0)
h, w = template.shape[:2]
font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体

# print(de_res)
# for r in de_res:
#     cv2.circle(img_rgb, (int(r['location'][1]), int(r['location'][0])), 40, (0, 0, 255), 4)
#     imgzi = cv2.putText(img_rgb, r['words'], (int(r['location'][1] + w), int(r['location'][0] + h)), font, 1.2, (255, 255, 255), 2)
    # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
# cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
# cv2.imshow('img_rgb', imgzi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
