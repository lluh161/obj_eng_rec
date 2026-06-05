import cv2
# import mediapipe as mp
import pyttsx3
import speech_recognition as sr
import threading
from ultralytics import YOLO

# ===================== 初始化工具 =====================
# 语音播报
tts = pyttsx3.init()

# ======注释全部手势初始化======
# from mediapipe.python.solutions import hands as mp_hands
# from mediapipe.python.solutions import drawing_utils as mp_draw
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# YOLO 最强大的通用模型：能识别 800+ 种物体（万物识别）
model = YOLO("yolov8l.pt")  #  L 版，精度极高

# 语音识别
recognizer = sr.Recognizer()
is_listening = False
frame = None

# ===================== 核心功能 =====================
def speak(text):
    """朗读英文"""
    tts.say(text)
    tts.runAndWait()

def detect_all_objects(frame):
    """识别画面中所有物体，返回英文名称列表【沿用你原来的函数逻辑不变】"""
    results = model(frame, verbose=False)
    objects = []
    # 画框+标注名称（在原识别逻辑基础上加绘图）
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            c = int(box.cls[0])
            obj_name = model.names[c]
            objects.append(obj_name)
            # 绘制框和名称
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, obj_name, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    # 去重
    res_obj = list(set(objects))
    global draw_objects
    draw_objects = res_obj
    return res_obj

def voice_listen():
    """语音监听"""
    global is_listening
    if is_listening:
        return
    is_listening = True
    print("\n🎤 正在听...")
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=3)
    
    try:
        text = recognizer.recognize_google(audio, language="zh-CN")
        print(f"你说：{text}")
        if "这是什么" in text or "识别" in text or "单词" in text:
            run_recognize()
    except:
        print("❌ 未听清")
    is_listening = False

def run_recognize():
    """执行识别 + 显示英文 + 朗读"""
    global frame
    objects = detect_all_objects(frame)
    if not objects:
        print("🔍 未识别到物体")
        speak("No object detected")
        return

    print("=" * 40)
    print("📷 识别结果（英文）：")
    for obj in objects:
        print(f"✅ {obj}")
        speak(obj)
    print("=" * 40)

# ===================== 主程序 =====================
cap = cv2.VideoCapture(0)
print("="*50)
print("🔥 万物识别视频交互软件 已启动")
print("👉 语音：按 V 说话 → 这是什么？")
print("👉 退出：按 Q")
print("="*50)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # ======注释整段手势检测逻辑，不再执行手势识别======
    # hand_result = hands_detector.process(rgb)
    # hand_detected = False
    # if hand_result.multi_hand_landmarks:
    #     for h in hand_result.multi_hand_landmarks:
    #         drawing_utils.draw_landmarks(frame, h, hands.HAND_CONNECTIONS)
    #         hand_detected = True
    #
    # # 手势触发识别
    # if hand_detected:
    #     cv2.putText(frame, "GESTURE | RECOGNIZING...", (20, 50),
    #                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #     run_recognize()

    # 显示提示
    cv2.putText(frame, "V=voice | Q=quit", (20, frame.shape[0]-20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    cv2.imshow("Universal Object Recognition", frame)

    # 按键
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('v'):
        threading.Thread(target=voice_listen).start()

cap.release()
cv2.destroyAllWindows()