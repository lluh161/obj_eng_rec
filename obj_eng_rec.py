from ultralytics import YOLO
import cv2
import pyttsx3
import time
import requests
import json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ==========【务必改这里】百度密钥==========
API_KEY = "wUsCYSuXaZLhn2GGk4mbDkGb"
SECRET_KEY = "qSpIrjqwuFSXvllNfxbMRoiwTUeP68VY"
TOKEN_URL = "https://aip.baidubce.com/oauth/2.0/token"
DETECT_URL = "https://aip.baidubce.com/rest/2.0/image-classify/v2/advanced_general"

# 中英对照字典
cn2en_dict = {
    "手机": "cell phone",
    "智能手机": "cell phone",
    "杯子": "cup",
    "水杯": "cup",
    "矿泉水瓶": "bottle",
    "瓶子": "bottle",
    "书本": "book",
    "书": "book",
    "椅子": "chair",
    "桌子": "table",
    "苹果": "apple",
    "鼠标": "mouse",
    "键盘": "keyboard",
    "人": "person",
    "玩偶": "doll",
    "花盆": "potted plant"
}

# 加载YOLO离线模型
model = YOLO("yolov8n.pt")

# 语音初始化
try:
    tts = pyttsx3.init(driverName='nsss')
    tts.setProperty("rate", 150)
    tts.setProperty("volume", 0.9)
except:
    tts = pyttsx3.init()

# 画面参数
W, H = 640, 480
box_w, box_h = 280, 280
roi_x1, roi_y1 = 180, 100
roi_x2, roi_y2 = 460, 380

# 按钮坐标：左下角识别按钮
btn_x1, btn_y1 = 20, H - 60
btn_x2, btn_y2 = 180, H - 20

# 全局变量
display_en = ""    # 识别结果，点击新按钮才更新
last_api_time = 0
api_interval = 1.2
limit_flag = False

# 获取百度token
def get_access_token():
    try:
        data = {"grant_type":"client_credentials","client_id":API_KEY,"client_secret":SECRET_KEY}
        res = requests.post(TOKEN_URL, data=data, timeout=8)
        js = res.json()
        if "access_token" in js:
            print("✅Token获取成功")
            return js["access_token"]
        else:
            print("❌Token失败:", js)
            return ""
    except Exception as e:
        print("❌联网失败", e)
        return ""
token = get_access_token()

# 百度识别
def baidu_detect(frame):
    global last_api_time, limit_flag
    now = time.time()
    if not token or now - last_api_time < api_interval or limit_flag:
        return None
    last_api_time = now
    crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    _, buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY,60])
    post_data = {"image": buf.tobytes()}
    param = {"access_token":token}
    try:
        ret = requests.post(DETECT_URL, data=post_data, params=param, timeout=5).json()
        print("百度返回：", ret)
        if "error_code" in ret:
            if ret["error_code"] == 18:
                limit_flag = True
                print("⚠️百度限流，切换YOLO本地识别")
            return None
        if "result" in ret and len(ret["result"])>0:
            return ret["result"][0]["keyword"]
    except Exception as e:
        print("百度请求异常：",e)
    return None

# YOLO本地识别
def yolo_local_detect(frame):
    crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    res = model(crop, verbose=False, conf=0.3)
    for r in res:
        if len(r.boxes.cls) > 0:
            cls_id = int(r.boxes.cls[0])
            return model.names[cls_id]
    return None

# 综合识别：优先百度，限流走YOLO
def run_recognize(frame):
    cn_name = baidu_detect(frame)
    if cn_name:
        res_en = cn2en_dict.get(cn_name, cn_name)
        return res_en
    en_name = yolo_local_detect(frame)
    if en_name:
        print("本地YOLO识别：", en_name)
        return en_name
    return None

# 朗读单词
def speak_text(word):
    try:
        tts.say(word)
        tts.runAndWait()
    except:
        pass

# PIL文字绘制
def draw_pil_text(bgr_img, text, x, y, color=(0,255,0), size=26):
    rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size)
    except:
        font = ImageFont.load_default()
    draw.text((x,y), text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

# 鼠标点击回调
def mouse_click(event, x, y, flags, param):
    global display_en
    if event == cv2.EVENT_LBUTTONDOWN:
        # 判断点击在按钮区域
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            print("👉点击识别按钮，开始识别...")
            res = run_recognize(frame_cache)
            if res:
                display_en = res
                speak_text(display_en)
                print(f"✅更新单词：{display_en}")

# 初始化摄像头+鼠标绑定
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cv2.namedWindow("Object English")
cv2.setMouseCallback("Object English", mouse_click)
frame_cache = None

print("操作说明：物体放入白色框 → 点击左下角【识别】按钮，单词常驻，下次点按钮才更新 | Q退出")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame,1)
    frame_cache = frame.copy()

    # 绘制中间识别框
    cv2.rectangle(frame, (roi_x1,roi_y1), (roi_x2,roi_y2), (255,255,255), 2)

    # 绘制左下角识别按钮
    cv2.rectangle(frame,(btn_x1,btn_y1),(btn_x2,btn_y2),(0,200,0),-1)
    frame = draw_pil_text(frame, "RECOGNIZE", btn_x1+8, btn_y1+8, color=(0,0,0), size=20)

    # 单词永久显示（除非再次点按钮刷新）
    if display_en != "":
        frame = draw_pil_text(frame, f"Word: {display_en}", roi_x2+10, 30, color=(0,255,0))

    cv2.imshow("Object English", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()