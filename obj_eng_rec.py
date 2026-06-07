from ultralytics import YOLO
import cv2
import pyttsx3
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ======================
# 最强精度模型 YOLOv9-e
# ======================
model = YOLO("yolov9e.pt")  # 最精准版本，识别率大幅提升

# 语音
try:
    tts = pyttsx3.init(driverName='nsss')
    tts.setProperty("rate", 150)
    tts.setProperty("volume", 0.9)
except:
    tts = pyttsx3.init()

# 画面区域
W, H = 640, 480
roi_x1, roi_y1 = 170, 90
roi_x2, roi_y2 = 470, 390

# 按钮
btn_x1, btn_y1 = 20, H - 60
btn_x2, btn_y2 = 180, H - 20

display_en = ""

# 超高精度识别（不过滤、不优先、纯靠AI实力）
def detect_object(frame):
    crop = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # 超高精度推理
    results = model(crop, verbose=False, conf=0.65, imgsz=640)

    for r in results:
        if len(r.boxes) > 0:
            idx = r.boxes.cls[0]
            name = model.names[int(idx)]
            return name
    return None

# 朗读
def speak(word):
    try:
        tts.say(word)
        tts.runAndWait()
    except:
        pass

# 绘图
def draw_text(img, text, x, y, color=(0,255,0), size=26):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", size)
    except:
        font = ImageFont.load_default()
    draw.text((x, y), text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

# 点击
def on_click(event, x, y, flags, param):
    global display_en
    if event == cv2.EVENT_LBUTTONDOWN:
        if btn_x1 <= x <= btn_x2 and btn_y1 <= y <= btn_y2:
            res = detect_object(frame_cache)
            if res:
                display_en = res
                speak(res)

# 摄像头
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cv2.namedWindow("Precision Detect")
cv2.setMouseCallback("Precision Detect", on_click)
frame_cache = None

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    frame_cache = frame.copy()

    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (255, 255, 255), 2)
    cv2.rectangle(frame, (btn_x1, btn_y1), (btn_x2, btn_y2), (0, 210, 0), -1)
    frame = draw_text(frame, "DETECT", btn_x1 + 10, btn_y1 + 5, (0, 0, 0), 22)

    if display_en:
        frame = draw_text(frame, f"Object: {display_en}", 20, 30)

    cv2.imshow("Precision Detect", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()