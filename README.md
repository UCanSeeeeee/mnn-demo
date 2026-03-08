# MNN Face Detection Demo

基于 [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) 的轻量级人脸检测 Demo，使用 MNN 推理框架，包含 Python 脚本和 iOS App 两种实现。

## 模型信息

| 项目 | 值 |
|------|-----|
| 模型文件 | `model/slim-320.mnn` (~1MB) |
| 原始框架 | Ultra-Light-Fast-Generic-Face-Detector (SSD) |
| 输入尺寸 | 320 x 240 (W x H) |
| 输入格式 | RGB, NCHW, 归一化 `(pixel - 127) / 128` |
| 输出节点 | `scores` (N, num_priors, 2) + `boxes` (N, num_priors, 4) |
| Prior Boxes | 4420 个，4 级 feature map |

### Prior Box 生成参数

```
STRIDES   = [8, 16, 32, 64]
MIN_BOXES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
```

每个 stride 对应一个 feature map 层级，在每个 cell 中心按 `MIN_BOXES` 生成不同大小的 anchor。

### 后处理流程

1. **Box 解码** — SSD 标准方式，`CENTER_VARIANCE=0.1`, `SIZE_VARIANCE=0.2`
2. **置信度过滤** — 阈值 0.3（face class, index=1）
3. **Hard NMS** — `IOU_THRESHOLD=0.3`, `CANDIDATE_SIZE=200`
4. **坐标映射** — 归一化坐标 → 原图像素坐标

## Python Demo

```bash
# 依赖：MNN, opencv-python, numpy
pip install MNN opencv-python numpy

# 运行（默认读取 imgs/ 目录，输出到 results/）
python detect.py

# 自定义参数
python detect.py --model model/slim-320.mnn --imgs imgs --output results --threshold 0.7 --input_size 320,240
```

## iOS Demo

位于 `mnn-demo/` 目录，将 Python 推理逻辑移植到 iOS 端。

### 技术方案

- **MNN 集成**：CocoaPods (`pod 'MNN'`, v1.2.0)
- **推理桥接**：ObjC++ (`.mm`) 调用 MNN C++ API，通过 Bridging Header 暴露给 Swift
- **预处理**：MNN `ImageProcess` API 完成 resize + normalize，零拷贝
- **UI**：纯代码 UIKit，单页面（图片展示 + 相册选择按钮）

### 项目结构

```
mnn-demo/
├── create_project.rb          # xcodeproj gem 生成 Xcode 工程
├── Podfile
└── FaceDetect/
    ├── ViewController.swift   # 主页面：图片展示 + 检测结果绘制
    ├── OverlayView.swift      # 人脸框绘制（绿色矩形 + 置信度）
    ├── Inference/
    │   ├── MNNFaceDetector.h   # ObjC 接口
    │   └── MNNFaceDetector.mm  # C++ 推理核心
    └── Resources/
        └── slim-320.mnn       # 模型文件
```

### 构建步骤

```bash
cd mnn-demo
ruby create_project.rb    # 生成 FaceDetect.xcodeproj
pod install               # 安装 MNN
# 用 Xcode 打开 FaceDetect.xcworkspace，真机运行
```

### 关键代码说明

推理核心在 `MNNFaceDetector.mm`，完整移植了 Python 版的：

- `generatePriors()` — Prior box 生成，与 `detect.py` 中 `generate_priors()` 对应
- `convert_locations_to_boxes` — 内联到检测循环中，SSD box 解码
- `hardNMS()` — Hard NMS，与 Python 版 `hard_nms()` 逻辑一致

预处理使用 MNN 内置的 `ImageProcess`，等效于 Python 版的：
```python
image = cv2.resize(image, (320, 240))
image = (image - 127) / 128.0
```
