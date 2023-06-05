# -*-coding: utf-8 -*-
"""Run inference with a YOLOv5 model on images, videos, directories, streams

Usage:
    $ python path/to/detect.py --source path/to/img.jpg --weights yolov5s.pt --img 640
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import cv2
import argparse
import numpy as np
from typing import List
from utils.datasets import LoadImages
from pybaseutils import file_utils, image_utils
from engine.inference import yolov5

# resnet部分include
import os
import argparse
import torch
import torch.nn as nn
import numpy as np

import time
import datetime
import os
import math
from datetime import datetime
import cv2
import torch.nn.functional as F
from models.resnet import resnet18,resnet34,resnet50,resnet101
from models.squeezenet import squeezenet1_1,squeezenet1_0
from models.shufflenetv2 import ShuffleNetV2
from models.shufflenet import ShuffleNet
from models.mobilenetv2 import MobileNetV2
from torchvision.models import shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0
from models.rexnetv1 import ReXNetV1

from common_utils import *
import copy
from hand_data_iter.datasets import draw_bd_handpose

import pytesseract
import requests
from utils.AuthV3Util import addAuthParams
from PIL import Image
import uuid
import hashlib
from importlib import reload
from playsound import playsound


reload(sys)

## 手势判断的相关操作
handpose_type = 18      ## 当前正在检测的收拾状态，默认状态下no gesture
fist_time = 0           ## 握拳状态的时间，用来返回初始状态
first_time = 0          ## 手势1的时间
second_time = 0         ## 手势2的时间
third_time = 0          ## 手势3的时间
clik_time = 0           ## 背面并指的时间，表示点击的动作
ltrans_time = 0         ## 大拇指向上表示翻译
origin_time = 0         ## 大拇指向下表示显示原文
error_time = 3          ##计时中允许连续出现的误检
finger_top_x = 0        ## 手指在图像中的坐标
finger_top_y = 0

trans = False
not_trans = False

class Yolov5Detector(yolov5.YOLOv5):
    def __init__(self,
                 weights='yolov5s.pt',  # model.pt path(s)
                 imgsz=640,  # inference size (pixels)
                 conf_thres=0.5,  # confidence threshold
                 iou_thres=0.5,  # NMS IOU threshold
                 max_det=1000,  # maximum detections per image
                 class_name=None,  # filter by class: --class 0, or --class 0 2 3
                 classes=None,  # filter by class: --class 0, or --class 0 2 3
                 agnostic_nms=False,  # class-agnostic NMS
                 augment=False,  # augmented inference
                 half=False,  # use FP16 half-precision inference
                 visualize=False,  # visualize features
                 batch_size=4,
                 device='cuda:0',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                 fix_inputs=False,
                 ):
        super(Yolov5Detector, self).__init__(weights=weights,  # model.pt path(s)
                                             imgsz=imgsz,  # inference size (pixels)
                                             conf_thres=conf_thres,  # confidence threshold
                                             iou_thres=iou_thres,  # NMS IOU threshold
                                             max_det=max_det,  # maximum detections per image
                                             class_name=class_name,  # filter by class: --class 0, or --class 0 2 3
                                             classes=classes,  # filter by class: --class 0, or --class 0 2 3
                                             agnostic_nms=agnostic_nms,  # class-agnostic NMS
                                             augment=augment,  # augmented inference
                                             half=half,  # use FP16 half-precision inference
                                             visualize=visualize,  # visualize features
                                             batch_size=batch_size,
                                             device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                                             fix_inputs=fix_inputs, )
    ## 检测目标
    def detect(self, image: List[np.ndarray] or np.ndarray, vis: bool = False) -> List[List]:
        """
        :param image: 图像或者图像列表,BGR格式
        :param vis: 是否可视化显示检测结果
        :return: 返回检测结果[[List]], each bounding box is in [x1,y1,x2,y2,conf,cls] format.
        """
        if isinstance(image, np.ndarray): image = [image]
        dets = super().inference(image)
        if vis:
            self.draw_result(image, dets)
        return dets

    def detect_image_dir(self, image_dir, out_dir=None, vis=True):
        # Dataloader
        dataset = file_utils.get_files_lists(image_dir)
        # Run inference
        for path in dataset:
            print(path)
            image = cv2.imread(path)  # BGR
            # rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            dets = self.detect(image, vis=False)
            image = self.draw_result([image], dets, vis=vis)[0]
            if out_dir:
                out_file = file_utils.create_dir(out_dir, None, os.path.basename(path))
                print("save result：{}".format(out_file))
                cv2.imwrite(out_file, image)

    def draw_result(self, image: List[np.ndarray] or np.ndarray, dets, thickness=2, fontScale=1.0, delay=0, vis=True):
        """
        :param image: 图像或者图像列表
        :param dets: 是否可视化显示检测结果
        """
        vis_image = []
        for i in range(len(dets)):
            image = self.draw_image(image[i], dets[i], thickness=thickness, fontScale=fontScale, delay=delay, vis=vis)
            vis_image.append(image)
        return vis_image

    def draw_image(self, image, dets, thickness=1, fontScale=1.0, delay=0, vis=True):
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # (xmin,ymin,xmax,ymax,conf, cls)
        # 进度条的位置
        boxes = [[1700,940,1830,960]]
        conf = dets[:, 4:5]
        cls = dets[:, 5]
        labels = [int(c) for c in cls]
        #print(boxes)
        
        image_utils.draw_image_detection_bboxes(image, boxes, conf, labels, class_name=self.names,
                                                thickness=thickness, fontScale=fontScale)
        
        
        # for *box, conf, cls in reversed(dets):
        #     c = int(cls)  # integer class
        #     label = "{}{:.2f}".format(self.names[c], conf)
        #     plot_one_box(box, image, label=label, color=colors(c, True), line_thickness=2)
        if vis: image_utils.cv_show_image("image", image, use_rgb=False, delay=delay)
        return image

# yolo的相关参数的配置
def parse_opt():
    image_dir = 'data/HaGRID-test'  # 测试图片的目录
    weights = "runs/yolov5s_640/weights/best.pt"  # 模型文件
    out_dir = "runs/HaGRID-result"  # 保存检测结果
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=weights, help='model.pt')
    parser.add_argument('--image_dir', type=str, default=image_dir, help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--video_file', type=str, default=None, help='camera id or video file')
    parser.add_argument('--out_dir', type=str, default=out_dir, help='save det result image')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.2, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1, help='maximum detections per image')
    parser.add_argument('--device', default='cuda:0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--class_name', nargs='+', type=list, default=None)
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    opt = parser.parse_args()
    return opt
# resnet的初始化
def resnet_ops():
    parser = argparse.ArgumentParser(description=' Project Hand Pose Inference')
    parser.add_argument('--model_path', type=str, default = './ReXNetV1-size-256-loss-wing_loss102-20211104.pth',
        help = 'model_path') # 模型路径
    parser.add_argument('--model', type=str, default = 'ReXNetV1',
        help = '''model : resnet_34,resnet_50,resnet_101,squeezenet1_0,squeezenet1_1,shufflenetv2,shufflenet,mobilenetv2
            shufflenet_v2_x1_5 ,shufflenet_v2_x1_0 , shufflenet_v2_x2_0,ReXNetV1''') # 模型类型
    parser.add_argument('--num_classes', type=int , default = 42,
        help = 'num_classes') #  手部21关键点， (x,y)*2 = 42
    parser.add_argument('--GPUS', type=str, default = '0',
        help = 'GPUS') # GPU选择
    parser.add_argument('--test_path', type=str, default = './image/',
        help = 'test_path') # 测试图片路径
    parser.add_argument('--img_size', type=tuple , default = (256,256),
        help = 'img_size') # 输入模型图片尺寸
    parser.add_argument('--vis', type=bool , default = True,
        help = 'vis') # 是否可视化图片

    #print('\n/******************* {} ******************/\n'.format(parser.description))
    #--------------------------------------------------------------------------
    ops = parser.parse_args()# 解析添加参数
    #--------------------------------------------------------------------------
    #print('----------------------------------')
    return ops


## 更新各个手势的持续时间
def handpose_judge(detect_T_set,gesture,frame):
    global handpose_type
    global fist_time
    global first_time
    global second_time
    global third_time
    global clik_time
    global ltrans_time
    global origin_time
    global error_time
    global trans
    global not_trans
    
    global finger_top_x
    global finger_top_y
    
    t=time.time() - detect_T_set
    #print(handpose_type)
    if handpose_type == 18:             # no gesture,检测到对应的手势后切换到目标手势
        
        handpose_type = gesture
        fist_time = 0           ## 握拳状态的时间，用来返回初始状态
        first_time = 0          ## 手势1的时间
        second_time = 0         ## 手势2的时间
        third_time = 0          ## 手势3的时间
        clik_time = 0           ## 背面并指的时间，表示点击的动作
        ltrans_time = 0         ## 大拇指向上表示翻译
        origin_time = 0         ## 大拇指向下表示显示原文
        finger_top_x = 0
        finger_top_y = 0
        
    elif gesture == handpose_type:      # 如果待检测的手势新检测到的手势相同，则对对应的手势的计时
        if gesture == 0:
            first_time = first_time+t
            if first_time>1.5:
                first_time=1.5
                cv2.putText(frame, 'get 1 gesture!', (250, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 3)
            cv2.line(frame, (1705, 950), (int(130*first_time/1.6+1700), 950), (255,255,255),10)
        elif gesture == 9 or gesture == 10:
            second_time = second_time+t
            if second_time>1.5:
                second_time=1.5
                cv2.putText(frame, 'get 2 gesture!', (250, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*second_time/1.6+1700), 950), (255,255,255),10)
        elif gesture == 3:
            
            third_time = third_time+t
            if third_time>1.5:
                third_time=1.5
                cv2.putText(frame, 'get 3 gesture!', (250, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*third_time/1.6+1700), 950), (255,255,255),10)
        elif gesture == 2 or gesture == 1:
            # 这里要是有空可以加一个防止手移动的补丁
            clik_time = clik_time+t
            if clik_time>1.5:
                clik_time=1.5
                cv2.putText(frame, 'get clik gesture!', (255, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*clik_time/1.6+1700), 950), (255,255,255),10)
        elif gesture == 6:
            fist_time = fist_time+t
            if fist_time>1.5:
                fist_time=1.5
                cv2.putText(frame, 'get fist gesture!', (255, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*fist_time/1.6+1700), 950), (255,255,255),10)
            
        elif gesture == 11:
            ltrans_time = ltrans_time+t
            if ltrans_time>1.5:
                trans = True
                ltrans_time = 0
                cv2.putText(frame, 'get like gesture!', (255, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*ltrans_time/1.6+1700), 950), (255,255,255),10)
            
        elif gesture == 12:
            origin_time = origin_time+t
            if origin_time>1.5:
                not_trans = True
                origin_time = 0
                cv2.putText(frame, 'get like gesture!', (255, 20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 2)
            cv2.line(frame, (1705, 950), (int(130*origin_time/1.6+1700), 950), (255,255,255),10)
        
        error_time = 3
    else:
        error_time = error_time-1
    if error_time<=0:
        handpose_type=18
        error_time = 3

# 显示图片
def cv_show(winname, image):
    cv2.imshow(winname, image)
# 有些原图片的size不好处理，我们可以封装成一个函数来统一图片的size
# 封装resize功能.
def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None # 缩放后的宽和高
    (h, w) = image.shape[:2]
    # 不做处理
    if width is None and height is None:
        return image
    # 指定了resize的height
    if width is None:
        r = height / float(h) # 缩放比例
        dim = (int(w * r), height)
    # 指定了resize的width
    else:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    return resized
    
# 进行透视变换.
# 透视变换要找到变换矩阵
# 变换矩阵要求原图的4个点坐标和变换之后的4个点的坐标
# 现在已经找到了原图的4个点的坐标。需要知道变换后的4个坐标
# 先对获取到的4个角点按照一定顺序（顺/逆时针）排序
# 排序功能是一个独立功能，可以封装成一个函数

def order_points(pts):
    # 创建全是0的矩阵, 来接收等下找出来的4个角的坐标.
    rect = np.zeros((4, 2), dtype='float32')
    # 列相加
    s = pts.sum(axis=1)
    # 左上的坐标一定是x,y加起来最小的坐标. 右下的坐标一定是x,y加起来最大的坐标.
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # 右上角的x,y相减的差值一定是最小的.
    # 左下角的x,y相减的差值, 一定是最大.
    # diff的作用是后一列减前一列得到的差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect
    
# 把透视变换功能封装成一个函数
def four_point_transform(image, pts):
    # 对输入的4个坐标排序
    rect = order_points(pts)
    # top_left简称tl，左上角
    # top_right简称tr，右上角
    # bottom_right简称br，右下角
    # bottom_left简称bl，左下角
    (tl, tr, br, bl) = rect
    # 空间中两点的距离，并且要取最大的距离确保全部文字都看得到
    widthA = np.sqrt((br[0] - bl[0]) ** 2 + (br[1] - bl[1]) ** 2)
    widthB = np.sqrt((tr[0] - tl[0]) ** 2 + (tr[1] - tl[1]) ** 2)
    max_width = max(int(widthA), int(widthB))
    heightA = np.sqrt((tr[0] - br[0]) ** 2 + (tr[1] - br[1]) ** 2)
    heightB = np.sqrt((tl[0] - bl[0]) ** 2 + (tl[1] - bl[1]) ** 2)
    max_height = max(int(heightA), int(heightB))
    # 构造变换之后的对应坐标位置.
    dst = np.array([
        [0, 0],
        [max_width, 0],
        [max_width, max_height],
        [0, max_height]], dtype='float32')
    # 计算变换矩阵
    M = cv2.getPerspectiveTransform(rect, dst)
    # 透视变换
    warped = cv2.warpPerspective(image, M, (max_width, max_height))
    return warped
    
# 把图像预处理的功能封装成一个函数
def Image_Pretreatment(image):
    # 图片预处理
    # 灰度化处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv_show('gray',gray)
    # 高斯平滑
    Gaussian = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv_show('Gaussian',Gaussian)
    # 边缘检测，寻找边界（为后续查找轮廓做准备）
    edged = cv2.Canny(Gaussian, 70, 200)
    # cv_show('edged',edged)
    # 查找轮廓
    cnts = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    # 将轮廓按照面积降序排序
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    # 绘制所有轮廓
    image_contours = cv2.drawContours(image.copy(), cnts, -1, (0, 0, 255), 1)
#     cv_show('image_contours', image_contours)
    # 遍历轮廓找出最大的轮廓.
    for c in cnts:
        # 计算轮廓周长
        perimeter = cv2.arcLength(c, True)
        # 多边形逼近，得到近似的轮廓
        # 近似完后，只剩下四个顶点的角的坐标
        approx = cv2.approxPolyDP(c, 0.02 * perimeter, True)
        # 最大的轮廓
        if len(approx) == 4:
            # 接收approx
            screen_cnt = approx
            break
    # 画出多边形逼近
    image_screen_cnt = cv2.drawContours(image.copy(), [screen_cnt], -1, (0, 0, 255), 1)
    # cv_show('image_screen_cnt', image_screen_cnt)
    # 进行仿射变换，使图片变正
    warped = four_point_transform(image_copy, screen_cnt.reshape(4, 2) * ratio)
    # cv_show('warped', warped)
    # 二值处理，先转成灰度图
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # 再二值化处理
    ref = cv2.threshold(warped_gray, 180, 255, cv2.THRESH_BINARY)[1]
    # 旋转变正
    # dst = cv2.rotate(ref, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv_show('dst', dst)
    return ref

# 语音部分函数-------------------------------------------------------------------------------------
def encrypt(signStr):
    hash_algorithm = hashlib.md5()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data,url):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(url, data=data, headers=headers)


def read(text):
    YOUDAO_URL = 'https://openapi.youdao.com/ttsapi'
    APP_KEY = '75a7e441679aabef'
    APP_SECRET = 'TeJLYjFm6tkXtTG2vjcmOcvtjiejXPLM'
    q = text
    voiceName = "youxiaoqin"

    data = {}
    data['voiceName'] = voiceName
    salt = str(uuid.uuid1())
    signStr = APP_KEY + q + salt + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign

    response = do_request(data,YOUDAO_URL)
    contentType = response.headers['Content-Type']
    if contentType == "audio/mp3":
        millis = int(round(time.time() * 1000))
        filePath = "read.mp3"
        fo = open(filePath, 'wb')
        fo.write(response.content)
        fo.close()
        playsound('read.mp3')
    else:
        print(response.content)
        
# 翻译部分函数-------------------------------------------------------------------------------------
def translate(text):
    '''
    note: 将下列变量替换为需要请求的参数
    '''
    q = text
    lang_from = 'auto'
    lang_to = 'zh-CHS'
    APP_KEY = '16b2cbc6676c9840'
    # 您的应用密钥
    APP_SECRET = 'NWZHac6O94GkZf28G0B0dFYGsfj4SKHd'
    data = {'q': q, 'from': lang_from, 'to': lang_to}

    addAuthParams(APP_KEY, APP_SECRET, data)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    res = doCall('https://openapi.youdao.com/api', header, data, 'post').json()
    print(res['translation'][0])
    read(res['translation'][0])



def doCall(url, header, params, method):
    if 'get' == method:
        return requests.get(url, params)
    elif 'post' == method:
        return requests.post(url, params, header)


if __name__ == "__main__":
    ## YOLOv5s部分的参数，变量与配置
    opt = parse_opt()
    ## 0代表所选用的是序号为0的摄像头
    opt.video_file = 0
    ## yolo的相关配置
    d = Yolov5Detector(weights=opt.weights,  # model.pt path(s)
                       imgsz=opt.imgsz,  # inference size (pixels)
                       conf_thres=opt.conf_thres,  # confidence threshold
                       iou_thres=opt.iou_thres,  # NMS IOU threshold
                       max_det=opt.max_det,  # maximum detections per image
                       class_name=opt.class_name,  # filter by class: --class 0, or --class 0 2 3
                       classes=opt.classes,  # filter by class: --class 0, or --class 0 2 3
                       agnostic_nms=opt.agnostic_nms,  # class-agnostic NMS
                       augment=opt.augment,  # augmented inference
                       device=opt.device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
                       )
    ## 用来储存获取到的图片
    frame = None
    ## 定义读取图片的操作
    video_cap = image_utils.get_video_capture(opt.video_file,width=1920,height=1080)
    ## 几帧图像进行一次检测
    detect_freq=1
    ## 检测得到的图像是否可见
    vis=True
    width, height, numFrames, fps = image_utils.get_video_info(video_cap)
    count = 0
    
    ## resnet的相关配置---------------------------------------------------------------------------------------------------------
    ops = resnet_ops()
        

    unparsed = vars(ops) # parse_args()方法的返回值为namespace，用vars()内建函数化为字典
    for key in unparsed.keys():
        print('{} : {}'.format(key,unparsed[key]))
    os.environ['CUDA_VISIBLE_DEVICES'] = ops.GPUS   # GPU

    test_path =  ops.test_path # 测试图片文件夹路径

    #---------------------------------------------------------------- 构建模型
    print('use model : %s'%(ops.model))

    model_ = ReXNetV1( width_mult=1.0, depth_mult=1.0, num_classes=ops.num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model_ = model_.to(device)
    model_.eval() # 设置为前向推断模式
    
    # 加载测试模型
    if os.access(ops.model_path,os.F_OK):# checkpoint
        chkpt = torch.load(ops.model_path, map_location=device)
        model_.load_state_dict(chkpt)
        print('load test model : {}'.format(ops.model_path))
        
    start_mode = True       ## 开始菜单的状态以及状态机标志位
    choose_mode_1 = False
    choose_mode_2 = False
    choose_mode_3 = False
    
    point1_x = 0
    point1_y = 0
    point2_x = 0
    point2_y = 0
    point3_x = 0
    point3_y = 0
    point4_x = 0
    point4_y = 0
    point_num = 0
    add_flag = False
    result = None
    trans = False
    not_trans = False
    
    # 检测出手部并辨别手势（yolov5s）-----------------------------------------------------------------------------------------------
    while True:
        detect_T_set = time.time()
        # OCR
        if trans:
            # 计算比例. 限定高度500
            # 此时像素点都缩小了一定的比例，进行放射变换时要还原
            ratio = result.shape[0] / 500.0
            # 拷贝一份
            image_copy = result.copy()
            # 修改尺寸
            result = resize(image_copy, height=500)
            # cv_show('image', image)
            # 返回透视变换的结果
            ref = Image_Pretreatment(result)
            kernel = np.ones((5,5),np.uint8)  
            erosion = cv2.erode(ref,kernel,iterations = 1)
            # 把处理好的图片写入图片文件.
            _ = cv2.imwrite('./scan.jpg', erosion)
            # pytesseract要求的image不是opencv读进来的image, 而是pillow这个包, 即PIL
            text = pytesseract.image_to_string(Image.open('./scan.jpg'), lang='chi_sim+eng', config='--oem 1')
            # 保存到本地
            print(text)
            trans = False
            translate(text)

        elif not_trans:
            # 计算比例. 限定高度500
            # 此时像素点都缩小了一定的比例，进行放射变换时要还原
            ratio = result.shape[0] / 500.0
            # 拷贝一份
            image_copy = result.copy()
            # 修改尺寸
            result = resize(image_copy, height=500)
            # cv_show('image', image)
            # 返回透视变换的结果
            ref = Image_Pretreatment(result)
            kernel = np.ones((5,5),np.uint8)  
            erosion = cv2.erode(ref,kernel,iterations = 1)
            # 把处理好的图片写入图片文件.
            _ = cv2.imwrite('./scan.jpg', erosion)
            # pytesseract要求的image不是opencv读进来的image, 而是pillow这个包, 即PIL
            text = pytesseract.image_to_string(Image.open('./scan.jpg'), lang='chi_sim+eng', config='--oem 1')
            # 保存到本地
            print(text)
            not_trans = False
            read(text)

         
        # s视频配置
        if count % detect_freq == 0:
            # 设置抽帧的位置
            if isinstance(opt.video_file, str): video_cap.set(cv2.CAP_PROP_POS_FRAMES, count)
            isSuccess, frame = video_cap.read()
            img=None
            if not isSuccess:
                break
            # each bounding box is in [x1,y1,x2,y2,conf,cls] format.
            dets = d.detect(frame, vis=False)
            if dets[0].size >0:
                #print(dets[0][0])
                #cv2.rectangle(new_img, (int(dets[0][0][0]), int(dets[0][0][1])), (int(dets[0][0][2]), int(dets[0][0][3])), (0, 255, 0),10)
                img = frame[int(dets[0][0][1]):int(dets[0][0][3]),int(dets[0][0][0]):int(dets[0][0][2])]
                #cv2.imshow('hand',img)
    # 检测出手部并辨别手势（yolov5s）-----------------------------------------------------------------------------------------------
    
    ## resnet关键点回归------------------------------------------------------------------------------------------------------------
                img_width = img.shape[1]
                img_height = img.shape[0]
                # 输入图片预处理
                img_ = cv2.resize(img, (ops.img_size[1],ops.img_size[0]), interpolation = cv2.INTER_CUBIC)
                img_ = img_.astype(np.float32)
                img_ = (img_-128.)/256.

                img_ = img_.transpose(2, 0, 1)
                img_ = torch.from_numpy(img_)
                img_ = img_.unsqueeze_(0)

                if use_cuda:
                    img_ = img_.cuda()  # (bs, 3, h, w)
                pre_ = model_(img_.float()) # 模型推理
                output = pre_.cpu().detach().numpy()
                output = np.squeeze(output)

                pts_hand = {} #构建关键点连线可视化结构
                for i in range(int(output.shape[0]/2)):
                    x = (output[i*2+0]*float(img_width))
                    y = (output[i*2+1]*float(img_height))
                    
                    pts_hand[str(i)] = {}
                    pts_hand[str(i)] = {
                        "x":x,
                        "y":y,
                        }
                draw_bd_handpose(img,pts_hand,0,0) # 绘制关键点连线

                ## 绘制关键点
                finger_top_x = dets[0][0][0]+(output[8*2+0]*float(img_width))
                finger_top_y = dets[0][0][1]+(output[8*2+1]*float(img_height))

                cv2.circle(frame, (int(finger_top_x),int(finger_top_y)), 5, (255,50,60),-1)
                cv2.circle(frame, (int(finger_top_x),int(finger_top_y)), 3, (255,150,180),-1)
                """
                for i in range(int(output.shape[0]/2)):
                    x = (output[i*2+0]*float(img_width))
                    y = (output[i*2+1]*float(img_height))

                    cv2.circle(img, (int(x),int(y)), 3, (255,50,60),-1)
                    cv2.circle(img, (int(x),int(y)), 1, (255,150,180),-1)
                """
                    #print(x)
            if dets[0].size >0:
                pose = dets[0][0][5]
            else:
                pose = 18
            # 各个手势检测与滤波
            handpose_judge(detect_T_set,pose,frame)
            
            
            # 主要功能部分
            if fist_time == 1.5:
                if not start_mode:
                    cv2.destroyAllWindows()
                start_mode = True   ##握拳返回主菜单
                choose_mode_1 = False
                choose_mode_2 = False
                choose_mode_3 = False
                point1_x = 0
                point1_y = 0
                point2_x = 0
                point2_y = 0
                point3_x = 0
                point3_y = 0
                point4_x = 0
                point4_y = 0
                point_num = 0
                trans = False
                not_trans = False
            if start_mode:
                cv2.putText(frame, 'START PAGE', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                if first_time == 1.5:
                    choose_mode_1 = True
                    start_mode = False
                elif second_time == 1.5:
                    choose_mode_2 = True
                    start_mode = False
                elif third_time == 1.5:         #这个模式暂时没用，就是调试的
                    choose_mode_3 = True
                    start_mode = False
            # 两点框选方式
            elif choose_mode_1 == True:
                cv2.putText(frame, 'MODE:1', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                #print(point_num)
                
                if point_num == 0:
                    if clik_time == 1.5:
                        if point1_x == 0:
                            point1_x = int(finger_top_x)
                        if point1_y == 0:
                            point1_y = int(finger_top_y)
                        add_flag = True
                    elif add_flag:
                        point_num = point_num+1
                        add_flag = False
                elif point_num == 1:
                    cv2.circle(frame, (int(point1_x),int(point1_y)), 5, (255,50,60),-1)
                    if finger_top_x !=0:
                        cv2.rectangle(frame, (point1_x,point1_y), (int(finger_top_x), int(finger_top_y)), (0, 0, 255), 2)
                        if clik_time == 1.5:
                            if point2_x == 0:
                                point2_x = int(finger_top_x)
                            if point2_y == 0:
                                point2_y = int(finger_top_y)
                            add_flag = True
                        elif add_flag:
                            point_num = point_num+1
                            add_flag = False
                elif point_num == 2:
                    # cv2.imshow('text',frame[point1_y:point2_y,point1_x:point2_x])
                    cv2.rectangle(frame, (point1_x,point1_y), (int(point2_x), int(point2_y)), (0, 0, 255), 2)
                    result = frame.copy()
                    cv2.rectangle(result, (point1_x,point1_y), (int(point2_x), int(point2_y)), (0, 0, 0), -1)
                    result = frame.copy() - result
                    ## 在此之后进行OCR的操作

            elif choose_mode_2 == True:
                cv2.putText(frame, 'MODE:2', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                if point_num == 0:
                    if clik_time == 1.5:
                        if point1_x == 0:
                            point1_x = int(finger_top_x)
                        if point1_y == 0:
                            point1_y = int(finger_top_y)
                        add_flag = True
                    elif add_flag:
                        point_num = point_num+1
                        add_flag = False
                elif point_num == 1:
                    cv2.circle(frame, (int(point1_x),int(point1_y)), 5, (255,50,60),-1)
                    if finger_top_x !=0:
                        cv2.line(frame, (point1_x,point1_y), (int(finger_top_x), int(finger_top_y)), (0, 0, 255), 2)
                        if clik_time == 1.5:
                            if point2_x == 0:
                                point2_x = int(finger_top_x)
                            if point2_y == 0:
                                point2_y = int(finger_top_y)
                            add_flag = True
                        elif add_flag:
                            point_num = point_num+1
                            add_flag = False
                elif point_num == 2:
                    cv2.circle(frame, (int(point1_x),int(point1_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point2_x),int(point2_y)), 5, (255,50,60),-1)
                    cv2.line(frame, (point1_x,point1_y), (int(point2_x), int(point2_y)), (0, 0, 255), 2)
                    if finger_top_x !=0:
                        
                        cv2.line(frame, (point2_x,point2_y), (int(finger_top_x), int(finger_top_y)), (0, 0, 255), 2)
                        if clik_time == 1.5:
                            if point3_x == 0:
                                point3_x = int(finger_top_x)
                            if point3_y == 0:
                                point3_y = int(finger_top_y)
                            add_flag = True
                        elif add_flag:
                            point_num = point_num+1
                            add_flag = False
                elif point_num == 3:
                    cv2.circle(frame, (int(point1_x),int(point1_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point2_x),int(point2_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point3_x),int(point3_y)), 5, (255,50,60),-1)
                    if finger_top_x !=0:
                        pts = np.array([[point1_x, point1_y], [point2_x, point2_y], [point3_x, point3_y], [int(finger_top_x), int(finger_top_y)]], np.int32)
                        pts = pts.reshape((-1, 1, 2))
                        cv2.polylines(frame, [pts], isClosed=True, color=(0,0,255), thickness=2)
                        if clik_time == 1.5:
                            if point4_x == 0:
                                point4_x = int(finger_top_x)
                            if point4_y == 0:
                                point4_y = int(finger_top_y)
                            add_flag = True
                        elif add_flag:
                            point_num = point_num+1
                            add_flag = False
                elif point_num == 4:
                    cv2.circle(frame, (int(point1_x),int(point1_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point2_x),int(point2_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point3_x),int(point3_y)), 5, (255,50,60),-1)
                    cv2.circle(frame, (int(point3_x),int(point3_y)), 5, (255,50,60),-1)
                    pts = np.array([[point1_x, point1_y], [point2_x, point2_y], [point3_x, point3_y], [point4_x, point4_y]], np.int32)
                    pts = pts.reshape((-1, 1, 2))
                    result = frame.copy()
                    cv2.fillPoly(result, [pts], color=(0,0,0))
                    result = frame.copy() - result
                    # cv2.imshow('text',result)
                    cv2.polylines(frame, [pts], isClosed=True, color=(0,0,255), thickness=2)
                    
                    

            elif choose_mode_3 == True:
                cv2.putText(frame, 'MODE:3', (10, 10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 255), 2)
                
            #print(first_time)
            frame = d.draw_result([frame], dets, thickness=2, fontScale=0.5, delay=20, vis=vis)[0]
        count += 1
    video_cap.release()
