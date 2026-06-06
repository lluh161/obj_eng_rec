# README.md
## 视频交互智能物体识别项目
基于 **YOLOv8 + OpenCV + pyttsx3 + SpeechRecognition** 实现摄像头实时物体识别，**语音触发识别、英文语音播报**，无手势依赖版本。

### 一、项目功能
1. **语音触发识别（V键）**
按下键盘 `V` 启用麦克风，说出关键词：`这是什么/识别/单词`，自动识别画面内所有物体。
2. **画面可视化**
识别物体自动在画面画绿色检测框，框上标注物体英文名称，画面左上角汇总所有识别物品。
3. **语音朗读**
识别成功后电脑自动播报对应英文单词；无物体时播报 `No object detected`。
4. **退出程序**
摄像头窗口按 `Q` 关闭程序。

### 二、环境安装
#### 1、创建虚拟环境（可选）
```bash
# Mac/Linux
python3 -m venv venv
source venv/bin/activate
```
#### 2、一键安装全部依赖
```bash
pip install opencv-python pyttsx3 speechrecognition ultralytics pyaudio -i https://pypi.tuna.tsinghua.edu.cn/simple
```
> Mac 安装pyaudio报错：先执行 `brew install portaudio` 再pip安装
> Windows 安装pyaudio报错：`pip install pipwin && pipwin install pyaudio`

### 三、项目目录
```
视频交互智能识别物体/
├─ obj_eng_rec.py      # 主运行代码
├─ yolov8l.pt          # YOLOv8大模型（首次运行自动下载）
└─ README.md           # 项目说明文档
```

### 四、运行项目
```bash
# 终端进入项目目录，激活虚拟环境后运行
python3 obj_eng_rec.py
```
> 首次启动自动在线下载 `yolov8l.pt` 权重文件，等待下载完成自动打开摄像头。
> Mac系统需在【系统设置→隐私与安全性→相机/麦克风】授予VS Code权限。

### 五、快捷键说明
| 按键 | 功能 |
| ---- | ---- |
| V | 开启麦克风语音识别，说话「这是什么」触发物体检测 |
| Q | 关闭摄像头窗口、退出程序 |

### 六、上传git忽略配置（.gitignore）
如需上传代码至github，在项目新建 `.gitignore`：
```
# 虚拟环境
venv/
# 模型权重（大文件不上传）
*.pt
*.pth
# 缓存文件
__pycache__/
*.pyc
# Mac隐藏文件
.DS_Store
# 日志
*.log
```

### 七、项目技术栈
- 目标检测：ultralytics-YOLOv8l
- 摄像头画面：OpenCV
- 语音合成播报：pyttsx3
- 语音收音识别：SpeechRecognition+谷歌语音API