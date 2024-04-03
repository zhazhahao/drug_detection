import json
import torch
import numpy as np
import cv2
import tools.infer.pytorchocr_utility as utility
from tools.infer.predict_video import TextSystem
from ultralytics1.ultralytics import YOLO
import re
import Levenshtein
from tools.infer.predict_video import processs

# video_stream_path = 'rtsp://192.168.123.210/live'  # video_stream_path=0的时候，电脑开启前置摄像头
# user, pwd, ip, channel = "", "", "", 1
# video_stream_path = 0
video_stream_path = 'rtsp://192.168.3.100/live'  # 在这里更接受推流的地址
cv2.CAP_PROP_READ_TIMEOUT_MSEC = 1e3
# tryToConnect = [1, 2, 3]  #
args = utility.parse_args()
text_sys = TextSystem(args)
none_detection = None
model_path = "./models/last.pt"
database_path = "./drug-v3.json"
data_dict = "./dicts.txt"
model = YOLO(model_path)

# 打开药品数据库
with open(database_path, 'r', encoding='utf-8') as file:
    try:
        data = json.load(file)
    except json.JSONDecodeError as e:
        print("Error decoding JSON:", e)
        exit()


def data_list():
    data_list=[]
    with open('dicts.txt', 'r', encoding='utf-8') as file:
        lines = file.readlines()
    for line in lines:
        # 去除行尾的换行符并按空格切分
        items = line.strip().split()
        data_list.append(items)
        # 使用第一列元素作为键，整行作为对应的值
    return data_list


data_list = data_list()


def curr_false(text):

    search_ch_text = re.compile(r'[\u4e00-\u9fff]')
    if search_ch_text.search(text):
        if text in data_list:
            return text
        similarities = [Levenshtein.ratio(text, str2[0]) for str2 in data_list]
        max_similarity = max(similarities)
        max_index = similarities.index(max_similarity)
        most_similar_drug = data_list[max_index][0]
        if max_similarity > 0.2:
            return most_similar_drug


def yolo_and_ocr(frame):
    # ocr
    ocr_dt_boxes, ocr_rec_res = processs(frame, text_sys)
    ocr_current_shelf = ''
    for i in range(len(ocr_dt_boxes)):
        matching_medicines = find_medicine_by_name(data, curr_false(ocr_rec_res[i][0]))
        # print(1111111111111111111)
        if matching_medicines:
            ocr_current_shelf = matching_medicines.get("货架号")  # or other info in the dataset
    # yolo
    results = model(frame)
    for result in results:
        boxes = result.boxes
        # probs = result.probs
        cls, conf, xywh = boxes.cls, boxes.conf, boxes.xywh  # get info needed
        print([converter(cls), converter(xywh)])
        if cls.__len__()==0:
            pass
        else:
            current_drug = get_drug_by_index(int(cls[0]))

            if current_drug.get("货架号") != ocr_current_shelf:
                detect_res_cls = "nomatch"
                return [detect_res_cls, []]
                
            else:
                detect_res_cls = cls
                return [converter(detect_res_cls), converter(xywh)]


def converter(tensor):
    tensor = tensor.cpu()
    numpy_array = tensor.numpy()
    values_list = numpy_array.tolist()
    return values_list


def find_medicine_by_name(medicines, target_name):
    # 查找药品信息
    for med in medicines:
        if med.get('药品名称') == target_name:
            return med


def get_drug_by_index(cls):
    for med in data:
        if med.get('index') == cls:
            return med
    return None

