import numpy as np
import configparser
from osgeo import gdal
from osgeo import osr
from osgeo import ogr
import os, sys, base64, json, cv2

input_file = './land_mask_16.jpg'
# dataset = gdal.Open("./static_maps/gdal_map/Continent.shx")
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


def img_registration(vector_file, img_file, gcps_list=None, gcps_file=None):
    # 为了支持中文路径，请添加下面这句代码
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
    # 为了使属性表字段支持中文，请添加下面这句
    gdal.SetConfigOption("SHAPE_ENCODING", "")
    # 注册所有的驱动
    ogr.RegisterAll()
    # 打开数据
    dataset = ogr.Open(vector_file, 0)
    if dataset is None:
        print("打开文件vector_file失败！")
        return

    if gcps_list is None and gcps_file is not None:
        gcps_list=[]
        with open(gcps_file) as gf:
            for r in gf.readline():
                l = r.split(' ')
                gcps_list.append(gdal.GCP(l[0], l[1], 0, l[2], l[3]))

    sr = osr.SpatialReference()
    sr.SetWellKnownGeogCS('WGS84')
    # 添加控制点
    dataset.SetGCPs(gcps_list, sr.ExportToWkt())
    # 进行校正
    # dst_ds = gdal.Warp(r'xxx_dst.tif', dataset, format='GTiff', tps=True, xRes=0.05, yRes=0.05, dstNodata=65535,
    #                    srcNodata=65535, resampleAlg=gdal.GRIORA_NearestNeighbour， outputType = gdal.GDT_Int32)


img_registration("./static_maps/gdal_map/Continent.shx", input_file, gcps_file='./static_maps/gdal_map/map配准.txt')
