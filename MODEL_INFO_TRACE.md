# slim-320.mnn 模型信息溯源记录

本文档记录了 `slim-320.mnn` 这个模型的所有推理参数是如何从原始项目中获取的。

原始项目路径：`/Users/xswang/Desktop/SHP_Projects/Ultra-Light-Fast-Generic-Face-Detector-1MB`

---

## 第一步：从 MNN Python API 获取输入输出结构

这是唯一可以直接从 `.mnn` 文件本身获取的信息：

```python
import MNN
interp = MNN.Interpreter("model/slim-320.mnn")
session = interp.createSession()

for name, t in interp.getSessionInputAll(session).items():
    print(f"Input  [{name}]: shape={t.getShape()}, dimType={t.getDimensionType()}")
for name, t in interp.getSessionOutputAll(session).items():
    print(f"Output [{name}]: shape={t.getShape()}, dimType={t.getDimensionType()}")
```

得到：

```
Input  [input]:  shape=(1, 3, 240, 320), dimType=1    # NCHW
Output [boxes]:  shape=(1, 4420, 4),     dimType=1
Output [scores]: shape=(1, 4420, 2),     dimType=1
```

**到这里能确定的信息：**
- 输入名 `input`，输出名 `scores` 和 `boxes`
- 输入 shape `(1,3,240,320)` → 3通道，H=240，W=320
- 有 4420 个 prior box，2 类（背景+人脸）

**还不知道的信息：** 预处理参数、anchor 参数、box 解码参数 → 必须查原始项目

---

## 第二步：在原始项目中找到中心配置文件

原始项目的参数集中定义在一个 config 文件中：

**文件：`vision/ssd/config/fd_config.py`**

```python
image_mean_test = image_mean = np.array([127, 127, 127])   # 行5
image_std = 128.0                                            # 行6
iou_threshold = 0.3                                          # 行7
center_variance = 0.1                                        # 行8
size_variance = 0.2                                          # 行9

min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]  # 行11
```

同一个文件中还定义了不同输入尺寸对应的 feature map 大小：

```python
feature_map_w_h_list_dict = {
    320: [[40, 20, 10, 5], [30, 15, 8, 4]],   # 320x240 输入的 feature map
}
```

stride 可以从 `image_size / feature_map_size` 反推：`320/40=8, 320/20=16, 320/10=32, 320/5=64`

**这个文件一次性给出了 6 个关键参数。**

---

## 第三步：在 MNN 推理脚本中验证参数

原始项目提供了 MNN Python 推理脚本，可以交叉验证：

**文件：`MNN/python/ultraface_py_mnn.py`**

```python
# 行33-39
image_mean = np.array([127, 127, 127])
image_std = 128.0
iou_threshold = 0.3
center_variance = 0.1
size_variance = 0.2
min_boxes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]]
strides = [8, 16, 32, 64]    # 这里直接写明了 stride
```

同一文件中的预处理流程（行124-129）：

```python
image = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)   # BGR→RGB
image = cv2.resize(image, tuple(input_size))          # resize 到 320x240
image = (image - image_mean) / image_std              # (x-127)/128
image = image.transpose((2, 0, 1))                    # HWC→CHW
```

**这里额外确认了 strides 和完整的预处理管线。**

---

## 第四步：在 box_utils 中找到解码公式

**文件：`vision/utils/box_utils.py`**

Prior box 生成（行6-29）：

```python
def generate_priors(feature_map_list, shrinkage_list, image_size, min_boxes):
    for index in range(len(feature_map_list[0])):
        scale_w = image_size[0] / shrinkage_list[0][index]
        scale_h = image_size[1] / shrinkage_list[1][index]
        for j in range(feature_map_list[1][index]):
            for i in range(feature_map_list[0][index]):
                x_center = (i + 0.5) / scale_w
                y_center = (j + 0.5) / scale_h
                for min_box in min_boxes[index]:
                    w = min_box / image_size[0]
                    h = min_box / image_size[1]
                    priors.append([x_center, y_center, w, h])
```

SSD box 解码（行32-55）：

```python
def convert_locations_to_boxes(locations, priors, center_variance, size_variance):
    # cx = raw_cx * center_variance * prior_w + prior_cx
    # cy = raw_cy * center_variance * prior_h + prior_cy
    # w  = exp(raw_w * size_variance) * prior_w
    # h  = exp(raw_h * size_variance) * prior_h
    return torch.cat([
        locations[..., :2] * center_variance * priors[..., 2:] + priors[..., :2],
        torch.exp(locations[..., 2:] * size_variance) * priors[..., 2:]
    ], dim=locations.dim() - 1)
```

Hard NMS（行168-198）：

```python
def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    # candidate_size=200 是默认值
```

**这里确认了 box 解码的数学公式和 NMS 的 candidate_size=200。**

---

## 第五步：从 ONNX 导出脚本确认输入输出名

**文件：`convert_to_onnx.py`（行40-43）**

```python
dummy_input = torch.randn(1, 3, 240, 320)
torch.onnx.export(net, dummy_input, model_path,
                  input_names=['input'],
                  output_names=['scores', 'boxes'])
```

**确认了 tensor 名称 `input`、`scores`、`boxes` 是在导出 ONNX 时手工指定的。**

---

## 第六步：验证 prior 数量

用 anchor 参数反算 prior 总数，验证是否与模型输出的 4420 一致：

```python
from math import ceil
strides = [8, 16, 32, 64]
min_boxes = [[10,16,24], [32,48], [64,96], [128,192,256]]
W, H = 320, 240

total = 0
for stride, boxes in zip(strides, min_boxes):
    total += ceil(W/stride) * ceil(H/stride) * len(boxes)
# 40*30*3 + 20*15*2 + 10*8*2 + 5*4*3
# = 3600 + 600 + 160 + 60 = 4420 ✓
```

与模型输出 `(1, 4420, 4)` 吻合，参数正确。

---

## 第七步：模型文件的来源

`.mnn` 文件位于原始项目中：

```
Ultra-Light-Fast-Generic-Face-Detector-1MB/MNN/model/version-slim/slim-320.mnn
```

转换链路（来自 `MNN/README.md`）：

```
PyTorch (.pth) → ONNX (.onnx) → onnx-simplifier → MNNConvert → MNN (.mnn)
```

具体步骤：
1. 修改 `vision/ssd/ssd.py` 注释掉后处理（行35-42），让模型直接输出 raw scores 和 boxes
2. 运行 `convert_to_onnx.py` 生成 `version-slim-320_without_postprocessing.onnx`
3. `python3 -m onnxsim input.onnx simplified.onnx` 简化计算图
4. 使用 MNNConvert 或在线工具 convertmodel.com 转为 `.mnn`

---

## 汇总：每个参数的出处

| 参数 | 值 | 出处文件 | 获取方式 |
|------|-----|---------|---------|
| 输入 shape | (1,3,240,320) | .mnn 文件自身 | MNN Python API |
| 输入名 | `input` | convert_to_onnx.py | ONNX 导出时指定 |
| 输出名 | `scores`, `boxes` | convert_to_onnx.py | ONNX 导出时指定 |
| Prior 数量 | 4420 | .mnn 文件自身 | MNN Python API |
| image_mean | [127,127,127] | fd_config.py 行5 | 训练配置 |
| image_std | 128.0 | fd_config.py 行6 | 训练配置 |
| center_variance | 0.1 | fd_config.py 行8 | SSD 标准默认值 |
| size_variance | 0.2 | fd_config.py 行9 | SSD 标准默认值 |
| min_boxes | [[10,16,24],...] | fd_config.py 行11 | 训练配置 |
| strides | [8,16,32,64] | ultraface_py_mnn.py 行39 | feature map 推算 |
| iou_threshold | 0.3 | fd_config.py 行7 | 推理时可调 |
| candidate_size | 200 | box_utils.py 行168 | NMS 默认参数 |
