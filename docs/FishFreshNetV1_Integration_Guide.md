# FishFreshNetV1 集成指南

FishFreshNetV1 是 Fish-Agent 的专用视觉模型入口，用于对鱼眼图像进行三分类：高度新鲜、新鲜、不新鲜。仓库中提供模型结构、FastAPI 推理服务和工作流客户端；模型权重文件体积较大，默认不纳入源码仓库。

## 代码组成

| 模块 | 文件 | 说明 |
| --- | --- | --- |
| 模型定义 | `src/models/fishfreshnet_model.py` | EfficientNet-B0 特征提取 + 通道/空间注意力 + 三分类头 |
| 推理服务 | `src/api/model_service.py` | 提供预测和 Grad-CAM 接口 |
| 工作流客户端 | `src/tools/fishfreshnet_client.py` | 被新鲜度分类节点和 Grad-CAM 节点调用 |
| 工作流节点 | `src/graphs/nodes/freshness_classification_node.py` | 优先调用模型服务，失败时回退多模态大模型 |

## 模型输入输出

输入图像在模型服务中统一预处理为 `224 x 224` RGB，并使用 ImageNet 均值方差归一化。

输出字段：

| 字段 | 类型 | 说明 |
| --- | --- | --- |
| `freshness_level` | string | `高度新鲜`、`新鲜` 或 `不新鲜` |
| `freshness_label` | int | 0、1、2 |
| `confidence_score` | float | 预测类别概率 |
| `all_probabilities` | object | 三个类别的概率分布 |
| `description` | string | 面向用户的结果说明 |
| `timestamp` | string | 推理时间 |

## 模型权重路径

模型服务启动时按以下顺序查找权重：

1. 环境变量 `FISHFRESHNET_MODEL_PATH`
2. `projects/src/storage/fishfreshnet_v1.pth`
3. `Fish-Agent/fishfreshnet_v1.pth`

推荐在本地设置显式路径：

```bash
# Linux / macOS
export FISHFRESHNET_MODEL_PATH=/path/to/fishfreshnet_v1.pth

# PowerShell
$env:FISHFRESHNET_MODEL_PATH="C:\path\to\fishfreshnet_v1.pth"
```

## 启动模型服务

安装基础依赖和模型依赖：

```bash
pip install -r requirements.txt
pip install -r requirements-model.txt
```

启动服务时可按部署环境指定端口：

```bash
python -m uvicorn src.api.model_service:app --host 0.0.0.0 --port <port>
```

健康检查：

```bash
curl "$FISHFRESHNET_API_URL/"
```

## API

### `POST /predict`

上传本地图片文件并返回分类结果。

```bash
curl -X POST "$FISHFRESHNET_API_URL/predict" \
  -F "file=@fish_eye.jpg"
```

### `POST /predict_url`

传入远程图片 URL。

```bash
curl -X POST "$FISHFRESHNET_API_URL/predict_url" \
  -H "Content-Type: application/json" \
  -d "{\"image_url\":\"https://example.com/fish_eye.jpg\"}"
```

### `POST /predict_with_gradcam`

上传本地图片，返回预测结果和 Grad-CAM 热力图 base64。

### `POST /gradcam_url`

传入远程图片 URL，返回预测结果和 Grad-CAM 热力图 base64。

## 工作流接入方式

`freshness_classification_node` 会先读取 `FISHFRESHNET_API_URL`。如果模型服务健康检查失败、接口超时或结果中包含错误，节点会调用 OpenAI-compatible 多模态大模型作为回退路径。

```bash
# Linux / macOS
export FISHFRESHNET_API_URL=http://localhost:8000
export MODEL_SERVICE_URL=http://localhost:8000

# PowerShell
$env:FISHFRESHNET_API_URL="http://localhost:8000"
$env:MODEL_SERVICE_URL="http://localhost:8000"
```

## Grad-CAM

`src/api/model_service.py` 会读取模型最后的注意力特征图，计算目标类别的梯度权重，生成热力图并叠加到原图。工作流中的 `enhanced_gradcam_node` 会优先调用 `/gradcam_url`；如果服务不可用，则根据区域检测结果生成降级关注区域图，确保报告仍有视觉依据。

## 注意事项

- 模型权重来源需要可信；`load_model()` 优先使用 PyTorch 的 `weights_only=True` 加载方式。
- 本仓库不内置训练数据说明，实际性能应以你本地权重和验证集为准。
- 远程 URL 接口会下载图片到临时文件，服务结束后会清理临时文件。
- Windows 本地安装器通常内置推理核心，源码仓库只保留可复现的打包配置和启动器代码。
