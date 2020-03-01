import configparser
import urllib.request
import urllib.parse
# import urllib3
import os, sys, base64, json, cv2, ssl
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
proDir = os.path.split(os.path.realpath(__file__))[0]
configPath = os.path.join(proDir, "config.ini")
cf = configparser.ConfigParser()
# 读取配置文件，如果写文件的绝对路径，就可以不用os模块
cf.read(configPath)
app_id = cf.get("Baidu-OCR", "AppID")
api_key = cf.get("Baidu-OCR", "APIKey")
secret_key = cf.get("Baidu-OCR", "SecretKey")
ocr_url = cf.get("Baidu-OCR", "ocr_url")
access_token = cf.get("Baidu-OCR", "access_token")


def get_access_token():
    api_key = cf.get("Baidu-OCR", "APIKey")
    secret_key = cf.get("Baidu-OCR", "SecretKey")
    access_token_url = cf.get("Baidu-OCR", "access_token_url")
    host = access_token_url + 'client_id=' + api_key + '&client_secret=' + secret_key
    access_token = ''
    try:
        request = urllib.request.Request(host)
        request.add_header('Content-Type', 'application/json; charset=UTF-8')
        response = urllib.request.urlopen(request)
        content = response.read()
        access_token = json.loads(content)['access_token']
        cf.set("Baidu-OCR", "access_token", access_token)
    except:
        print("Unexpected error:", sys.exc_info()[0])
    finally:
        return access_token


def get_ocr_res(img_path=None, cv2_obj=None, base64_encode=None):
    access_token = cf.get("Baidu-OCR", "access_token")
    url = cf.get("Baidu-OCR", "ocr_url") + access_token
    if img_path is not None:
        # 二进制方式打开图文件
        f = open(r'' + img_path, 'rb')
        # 参数image：图像base64编码
        img = base64.b64encode(f.read())
    elif cv2_obj is not None:
        image = cv2.imencode('.jpg', cv2_obj)[1]
        img = str(base64.b64encode(image))[2:-1]
    elif base64_encode is not None:
        img = base64_encode
    else:
        img = None
    params = urllib.parse.urlencode({"image": img}).encode(encoding='UTF8')
    request = urllib.request.Request(url, params)
    request.add_header('Content-Type', 'application/x-www-form-urlencoded')
    response = urllib.request.urlopen(request)
    content = response.read()
    print(content)
    if 'error_code' in json.loads(content):
        get_access_token()
        print('weberror:' + str(content))
        return get_ocr_res(cv2_obj=None, base64_encode=img)
    else:
        return json.loads(content)['words_result']


# print(get_ocr_res(img_path='./December 5, 2017 9:49 AM'))
