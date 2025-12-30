# 实时人脸检测系统

基于 InsightFace 和 OpenCV 的实时人脸检测项目，集成 **ChromaDB 向量数据库**和 **MiniFASNet 活体检测**，可以通过摄像头实时检测、识别人脸并防止照片/视频欺骗。

## 功能特性

- ✨ **实时检测**：使用摄像头进行实时人脸检测
- 🎯 **高精度**：基于 InsightFace 的 buffalo_sc 模型，检测准确率高
- 🔐 **活体检测**：集成 MiniFASNet 进行静默活体检测，防止照片/视频攻击
- 💾 **向量数据库**：使用 ChromaDB 存储和检索人脸特征
- 👤 **人脸识别**：自动识别已注册人脸并显示姓名
- 📊 **详细信息**：显示人脸边界框、置信度、活体状态等信息
- 🔍 **关键点检测**：标记面部5个关键特征点（眼睛、鼻子、嘴角）
- 💾 **图像保存**：支持保存当前检测结果
- ⚡ **性能优化**：支持 CPU 和 GPU 运行，异步检测，跳帧处理

## 显示效果

- 🟢 **绿色框** - 活体且已识别（显示姓名 + 相似度）
- 🟠 **橙色框** - 活体但未识别（显示 "Unknown"）
- 🔴 **红色框** - 非活体检测（显示 "FAKE!"）

## 安装依赖

### 1. 创建虚拟环境（推荐）

```powershell
# 使用 Python 创建虚拟环境
python -m venv venv

# 激活虚拟环境
.\venv\Scripts\Activate.ps1
```

### 2. 安装依赖包

```powershell
pip install -r requirements.txt
```

### 3. 下载模型

首次运行时，InsightFace 会自动下载所需的模型文件（buffalo_l），请确保网络连接正常。

模型会下载到：`~/.insightface/models/buffalo_l/`

## 使用方法

### 1. 添加人脸到数据库

将人脸照片放入 `face_database` 文件夹：
- 命名格式：`姓名.jpg`（例如：`张三.jpg`, `李四.png`）
- 支持格式：jpg, jpeg, png, bmp
- 每张图片只包含一张人脸效果最佳

### 2. （可选）添加活体检测模型

将预训练的 MiniFASNet 模型文件 `minifasnet.pth` 放入 `models` 文件夹。如果没有模型，系统会使用随机初始化（效果不佳，建议获取预训练模型）。

### 3. 运行程序

```powershell
python face_detector.py
```

### 运行后的操作

- **实时检测**：程序会自动打开摄像头并开始检测人脸
- **按 'q' 键**：退出程序
- **按 's' 键**：保存当前帧到本地

### 显示信息说明

程序会在视频中显示：
- 🟢 **绿色边框**：活体且已识别（显示姓名和相似度）
- 🟠 **橙色边框**：活体但未识别（显示 "Unknown"）
- 🔴 **红色边框**：非活体（显示 "FAKE!"）
- 🔵 **蓝色圆点**：面部关键特征点（眼睛、鼻子、嘴角）
- 📝 **文字信息**：
  - Live/FAKE：活体检测状态和置信度
  - 姓名：识别到的人脸姓名（仅活体）
  - Det：检测置信度
  - Faces：当前帧检测到的人脸数量
  - FPS：帧率
  - DB：人脸库中的人数
  - Liveness：活体检测状态（ON/OFF）

## 自定义配置

你可以在 `face_detector.py` 的 `main()` 函数中修改参数：

```python
detector = RealtimeFaceDetector(
    camera_id=0,              # 摄像头ID，0为默认摄像头
    det_size=(320, 320),      # 检测尺寸，越大越准确但速度越慢
    skip_frames=3,            # 每3帧检测一次（约100ms）
    face_db_path="./face_database",  # 人脸库路径
    enable_liveness=True      # 是否启用活体检测
)
```

### 调整活体检测阈值

在代码中修改 `liveness_threshold`：

```python
self.liveness_threshold = 0.7  # 默认0.7，越高越严格
```

### GPU 加速

如果你有 NVIDIA GPU 并安装了 CUDA：

1. 安装 GPU 版本的依赖：
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

2. 修改代码使用 GPU：
```python
# InsightFace 使用 GPU
self.app = FaceAnalysis(
    name='buffalo_sc',
    providers=['CUDAExecutionProvider']
)

# MiniFASNet 会自动使用 CUDA（如果可用）
```

## 常见问题

### 1. 无法打开摄像头

- 确保摄像头已连接并且没有被其他程序占用
- 尝试修改 `camera_id` 参数（0, 1, 2...）
- 检查摄像头权限设置

### 2. 模型下载失败

- 检查网络连接
- 可以手动下载模型放到 `~/.insightface/models/buffalo_l/` 目录

### 3. 检测速度慢

- 降低摄像头分辨率
- 减小 `det_size` 参数，例如改为 `(320, 320)`
- 使用 GPU 加速

### 4. 导入错误

如果遇到 `ModuleNotFoundError`，请确保：
- 已激活虚拟环境
- 所有依赖都已正确安装：`pip install -r requirements.txt`

## 项目结构

```
face-detector/
├── face_detector.py      # 主程序
├── requirements.txt      # 依赖列表
└── README.md            # 说明文档
```

## 技术栈

- **InsightFace**：人脸检测和分析
- **OpenCV**：图像处理和摄像头调用
- **ONNX Runtime**：模型推理引擎
- **NumPy**：数值计算

## 许可证

MIT License

## 参考资料

- [InsightFace 官方文档](https://github.com/deepinsight/insightface)
- [OpenCV 文档](https://docs.opencv.org/)

---

**注意**：首次运行需要下载模型文件，请耐心等待。
