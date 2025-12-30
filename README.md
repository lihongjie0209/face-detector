# 实时人脸检测系统

基于 InsightFace 和 OpenCV 的实时人脸检测项目，可以通过摄像头实时检测人脸并显示相关信息。

## 功能特性

- ✨ **实时检测**：使用摄像头进行实时人脸检测
- 🎯 **高精度**：基于 InsightFace 的 buffalo_l 模型，检测准确率高
- 📊 **详细信息**：显示人脸边界框、置信度、年龄、性别等信息
- 🔍 **关键点检测**：标记面部5个关键特征点（眼睛、鼻子、嘴角）
- 💾 **图像保存**：支持保存当前检测结果
- ⚡ **性能优化**：支持 CPU 和 GPU 运行

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

### 基本运行

```powershell
python face_detector.py
```

### 运行后的操作

- **实时检测**：程序会自动打开摄像头并开始检测人脸
- **按 'q' 键**：退出程序
- **按 's' 键**：保存当前帧到本地

### 显示信息说明

程序会在视频中显示：
- 🟢 **绿色边框**：检测到的人脸位置
- 🔵 **蓝色圆点**：面部关键特征点（眼睛、鼻子、嘴角）
- 📝 **文字信息**：
  - Confidence：检测置信度（0-1之间）
  - Age：预测年龄
  - Gender：性别（Male/Female）
  - Faces：当前帧检测到的人脸数量
  - FPS：大致帧率

## 自定义配置

你可以在 `face_detector.py` 的 `main()` 函数中修改参数：

```python
detector = RealtimeFaceDetector(
    camera_id=0,        # 摄像头ID，0为默认摄像头，1为第二个摄像头
    det_size=(640, 640) # 检测尺寸，越大越准确但速度越慢
)
```

### GPU 加速

如果你有 NVIDIA GPU 并安装了 CUDA，可以修改代码使用 GPU 加速：

```python
# 在 face_detector.py 的 __init__ 方法中修改
self.app = FaceAnalysis(
    name='buffalo_l',
    providers=['CUDAExecutionProvider']  # 使用 GPU
)
```

需要先安装 GPU 版本的 onnxruntime：

```powershell
pip uninstall onnxruntime
pip install onnxruntime-gpu
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
