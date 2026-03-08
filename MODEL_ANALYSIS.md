# 如何获取 MNN 模型的基础信息

拿到一个 `.mnn` 模型文件后，需要搞清楚以下信息才能正确编写推理代码：

1. 输入/输出 tensor 的名称、shape、数据格式
2. 预处理参数（归一化均值、方差）
3. 后处理逻辑（anchor 参数、NMS 阈值等）

前两项可以通过工具直接获取，第三项必须查阅原始模型的文档或训练代码。

---

## 1. 用 MNN Python API 查看模型结构

安装 MNN Python 包后，几行代码即可提取所有张量信息：

```python
import MNN

interp = MNN.Interpreter("model/slim-320.mnn")
session = interp.createSession()

# 模型版本
print("Version:", interp.getModelVersion())

# 输入
for name, t in interp.getSessionInputAll(session).items():
    print(f"Input  [{name}]: shape={t.getShape()}, dimType={t.getDimensionType()}")

# 输出
for name, t in interp.getSessionOutputAll(session).items():
    print(f"Output [{name}]: shape={t.getShape()}, dimType={t.getDimensionType()}")
```

对 `slim-320.mnn` 的输出结果：

```
Version: <2.0.0
Input  [input]:  shape=(1, 3, 240, 320), dimType=1
Output [boxes]:  shape=(1, 4420, 4),     dimType=1
Output [scores]: shape=(1, 4420, 2),     dimType=1
```

从这里可以直接读出：
- **输入名**：`input`，shape `(1, 3, 240, 320)` → batch=1, channels=3, H=240, W=320
- **dimType=1** → Caffe 格式 (NCHW)
- **输出名**：`scores` 和 `boxes`，4420 个 prior box，scores 有 2 类（背景 + 人脸）

## 2. 关键 API 说明

| 方法 | 用途 |
|------|------|
| `getSessionInputAll(session)` | 返回 `{name: Tensor}` 字典，获取所有输入名和 shape |
| `getSessionOutputAll(session)` | 返回所有输出名和 shape |
| `tensor.getShape()` | 获取张量维度 |
| `tensor.getDimensionType()` | 1=Caffe(NCHW), 2=TF(NHWC) |
| `getModelVersion()` | 模型转换时的 MNN 版本 |

## 3. 预处理参数：从哪里来？

**预处理参数无法从 .mnn 文件中读取**，必须查阅原始模型的资料。获取途径：

### 方法 A：查看原始训练/导出代码

本项目的模型来自 [Ultra-Light-Fast-Generic-Face-Detector-1MB](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB)，其推理代码中写明：

```python
# vision/transforms.py 或推理脚本中
image = (image - 127) / 128  # 归一化到 [-0.992, 1.0]
```

因此 `mean=127, std=128`。

### 方法 B：查看模型仓库的 README / config

很多模型仓库会在文档中注明预处理方式。常见模式：

| 模型类型 | 常见预处理 |
|----------|-----------|
| ImageNet 分类 | mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]（基于 /255 后的值） |
| SSD 系列检测 | mean=[127,127,127], std=128（像素级） |
| YOLO 系列 | 直接 /255，无 mean 偏移 |

### 方法 C：查看 ONNX 中间格式

如果有 `.onnx` 文件，可以用 Netron 可视化查看是否内置了预处理算子：

```bash
pip install netron
netron model.onnx
```

## 4. 后处理参数：从哪里来？

后处理（anchor 生成、box 解码、NMS）的参数也来自原始模型代码。

### 4.1 Anchor / Prior Box 参数

本模型是 SSD 架构，anchor 参数定义在训练配置中：

```python
MIN_BOXES = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
STRIDES   = [8, 16, 32, 64]
```

验证方式：`scores` 输出有 4420 个 prior，可以反推：

```python
from math import ceil
total = 0
for stride, boxes in zip([8,16,32,64], [[10,16,24],[32,48],[64,96],[128,192,256]]):
    total += ceil(320/stride) * ceil(240/stride) * len(boxes)
print(total)  # 4420 ✓
```

如果算出来不等于模型输出的 prior 数量，说明参数有误。

### 4.2 Box 解码参数

SSD 标准解码公式：

```
cx = raw_cx * CENTER_VARIANCE * prior_w + prior_cx
cy = raw_cy * CENTER_VARIANCE * prior_h + prior_cy
w  = exp(raw_w * SIZE_VARIANCE) * prior_w
h  = exp(raw_h * SIZE_VARIANCE) * prior_h
```

本模型：`CENTER_VARIANCE=0.1`, `SIZE_VARIANCE=0.2`，这是 SSD 系列的常用默认值。

### 4.3 NMS 参数

```
IOU_THRESHOLD = 0.3    # IoU 大于此值的重叠框被抑制
PROB_THRESHOLD = 0.3   # 置信度低于此值的直接丢弃
```

这两个参数是推理时可调的，不是模型固有的。降低 `PROB_THRESHOLD` 会检出更多框（召回率高但误检多）。

## 5. 速查流程

拿到一个新的 `.mnn` 模型，按以下顺序获取信息：

```
1. MNN Python API
   └─ getSessionInputAll / getSessionOutputAll
   └─ 得到：输入输出名称、shape、NCHW/NHWC

2. 原始模型仓库
   └─ README / config / 推理脚本
   └─ 得到：预处理参数 (mean, std)

3. 模型架构文档
   └─ SSD → anchor 参数 + variance
   └─ YOLO → anchor sizes + stride
   └─ 分类模型 → 无后处理 / 只需 argmax

4. 验证
   └─ 用 Python 跑一张图，对比原始推理脚本的输出
   └─ 检查 prior 数量是否与输出 shape 匹配
```
