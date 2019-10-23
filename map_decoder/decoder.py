import cv2
import numpy as np
from skimage.measure import compare_ssim
from collections import Counter
import map_decoder.ocr_loader as ocr_loader


# 根据匹配算法获取标记点坐标,匹配程度大于95%的坐标y,x
def site_point(input_path, search_path='./target.jpg', threshold=0.95):
    img_rgb = cv2.imread(input_path)
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(search_path, 0)
    h, w = template.shape[:2]
    # cv2.matchTemplate标准相关模板匹配
    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)
    boxes_list = []
    nms_list = []
    for pt in zip(*loc[::-1]):  # *号表示可选参数
        boxes_list.append((pt[1], pt[0], pt[1] + h, pt[0] + w))
    nms_list.append(boxes_list[0])

    # 去除重复和相近的点
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
    for nms in nms_list:
        points_list.append((int(nms[0] + h/2), int(nms[1] + w/2)))
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


# 取黄色区域的mask输入ocr_loader中进行OCR,并去除标记点区域的mask
def cls_ocr_res(input_path, search_path='./target.jpg'):
    img_rgb = cv2.imread(input_path)
    yellow1 = np.array([32, 128, 200])
    yellow2 = np.array([128, 255, 255])
    mask_img = cv2.inRange(img_rgb, yellow1, yellow2)

    # 选取标记点位置,设置mask,避免其被ocr识别
    points_list = site_point(input_path)
    template = cv2.imread(search_path, 0)
    h, w = template.shape[:2]
    for point in points_list:
        mask_img[point[0]-int(h/2): point[0]+int(h/2), point[1]-int(w/2): point[1]+int(w/2)] = 0

    res = ocr_loader.get_ocr_res(cv2_obj=mask_img)
    return res


# 利用动态规划和矩阵计算,初步找出ocr信息和定位的关系,纵坐标权值默认为5:1
def find_relation(points_list, ocr_boxes, q=5):
    distance_tensor = []
    points_np = np.array(points_list, dtype=float)
    boxes_np = np.array(
        [([box['location']['top'], box['location']['left']], [box['location']['top'], box['location']['left']+box['location']['width']], [box['location']['top']+box['location']['height'], box['location']['left']], [box['location']['top']+box['location']['height'], box['location']['left']+box['location']['width']]) for box
         in ocr_boxes], dtype=float)
    # 得到距离矩阵
    for point in points_np:
        for box in boxes_np:
            # 计算距离后加入权重
            distance_tensor.append(np.sqrt(((box - point)**2).sum(axis=1)) + q * np.sqrt((box[:, 0] - point[0])**2))
    distance_tensor = np.array(distance_tensor).reshape([points_np.shape[0], boxes_np.shape[0], 4])
    mini_point_obx = distance_tensor.min(axis=2)
    # 最短距离的索引和值
    p_sort_arg = np.argsort(mini_point_obx)
    b_sort_arg = np.argsort(mini_point_obx, axis=0)
    decode_res1 = p_sort_arg[:, 0]
    for i, p in enumerate(p_sort_arg):
        if p[0] not in decode_res1:
            decode_res1[i] = p[0]
        else:
            if b_sort_arg[0, p[0]] == i:
                decode_res1[i] = p[0]
            else:
                decode_res1[b_sort_arg[0, p[0]]] = p[0]
                decode_res1[i] = p[1]

    decode_res = []
    for i, b in enumerate(decode_res1):
        decode_res.append({'location': points_list[i], 'words': ocr_boxes[b]['words'], 'box_idx': b})

    for i, b in enumerate(b_sort_arg[0]):
        if i not in decode_res1:
            decode_res.append({'location': points_list[b], 'words': ocr_boxes[i]['words'], 'box_idx': i})
    return decode_res


# 局部结果测试
# input_path = './December 5, 2017 9:49 AM'
# search_path = './target.jpg'
# points_list = site_point(input_path)
# de_res = find_relation(points_list, cls_ocr_res(input_path))

# 前台展示结果
# img_rgb = cv2.imread(input_path)
# template = cv2.imread(search_path, 0)
# h, w = template.shape[:2]
# font = cv2.FONT_HERSHEY_SIMPLEX  # 定义字体
# for r in de_res:
#     cv2.circle(img_rgb, (int(r['location'][1]), int(r['location'][0])), 40, (0, 0, 255), 4)
#     imgzi = cv2.putText(img_rgb, r['words'], (int(r['location'][1] + w), int(r['location'][0] + h)), font, 1.2, (255, 255, 255), 2)
#     # 图像，文字内容， 坐标 ，字体，大小，颜色，字体厚度
# cv2.namedWindow('img_rgb', cv2.WINDOW_NORMAL)
# cv2.imshow('img_rgb', imgzi)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
