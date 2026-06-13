# Fish-Agent 工作流技术说明

本文档说明 Fish-Agent 当前开源版本的工作流结构、节点职责、分支策略和本地部署方式。它与 `README.md`、`AGENTS.md` 和源码中的实际节点保持一致。

## 总览

Fish-Agent 是一个面向水产品新鲜度评估的 LangGraph 工作流。输入是一张鱼眼图像或包含鱼眼的鱼类图片，输出包括新鲜度等级、置信度、可解释热力图、结构化报告、处理建议和反馈记录。

核心设计：

- 先判断图片是否有效，再进入新鲜度分析。
- 优先使用 FishFreshNetV1 本地模型服务。
- 模型服务不可用时回退 OpenAI-compatible 多模态大模型。
- 根据置信度自动选择高、中、低三条路径。
- 所有用户可见结果统一汇总到报告、建议和可视化节点。

## 工作流图

```text
image_upload
  -> enhanced_quality_check
      -> [非鱼类 / 无鱼眼 / 质量不合格] quality_unqualified -> END
      -> [合格] image_preprocess
  -> fish_region_detection
  -> freshness_classification
      -> [confidence >= 0.8] enhanced_gradcam
      -> [0.5 <= confidence < 0.8] multi_region_fusion -> enhanced_gradcam
      -> [confidence < 0.5] backup_model_review
            -> [复核后高/中置信度] enhanced_gradcam
            -> [复核后低置信度] low_confidence_prompt -> END
  -> structured_analysis
  -> temporal_analysis
  -> report_generation
  -> recommendation_generation
  -> result_visualization
  -> feedback_loop
  -> END
```

## 节点职责

| 节点 | 文件 | 职责 |
| --- | --- | --- |
| `image_upload` | `src/graphs/nodes/image_upload_node.py` | 接收图像并记录上传时间、用户 ID |
| `enhanced_quality_check` | `src/graphs/nodes/enhanced_quality_check_node.py` | 判断鱼类、鱼眼和图像质量 |
| `quality_unqualified` | `src/graphs/nodes/quality_unqualified_node.py` | 输出不合格原因和重拍建议 |
| `image_preprocess` | `src/graphs/nodes/image_preprocess_node.py` | 按场景和质量等级执行轻量图像预处理 |
| `fish_region_detection` | `src/graphs/nodes/fish_region_detection_node.py` | 识别鱼眼、鳃部和体表区域 |
| `freshness_classification` | `src/graphs/nodes/freshness_classification_node.py` | 调用 FishFreshNetV1 或多模态回退模型 |
| `multi_region_fusion` | `src/graphs/nodes/multi_region_fusion_node.py` | 中置信度时融合多区域判断 |
| `backup_model_review` | `src/graphs/nodes/backup_model_review_node.py` | 低置信度时进行大模型复核 |
| `low_confidence_prompt` | `src/graphs/nodes/low_confidence_prompt_node.py` | 提示重拍或人工确认 |
| `enhanced_gradcam` | `src/graphs/nodes/enhanced_gradcam_node.py` | 生成 Grad-CAM 或降级关注区域图 |
| `structured_analysis` | `src/graphs/nodes/structured_analysis_node.py` | 生成鱼眼视觉特征分析 |
| `temporal_analysis` | `src/graphs/nodes/temporal_analysis_node.py` | 预测新鲜度趋势和剩余保质时间 |
| `report_generation` | `src/graphs/nodes/report_generation_node.py` | 生成结构化评估报告 |
| `recommendation_generation` | `src/graphs/nodes/recommendation_generation_node.py` | 生成储存、食用、处理和安全建议 |
| `result_visualization` | `src/graphs/nodes/result_visualization_node.py` | 汇总最终展示数据 |
| `feedback_loop` | `src/graphs/nodes/feedback_loop_node.py` | 生成记录 ID 和样本池入口 |

## 状态字段

全局状态定义位于 `src/graphs/state.py`。重要字段包括：

| 字段 | 说明 |
| --- | --- |
| `uploaded_image` / `processed_image` | 原始和预处理图像 |
| `is_fish_image` / `contains_fish_eye` / `is_qualified` | 质量检测结果 |
| `fish_eye_regions` / `gill_regions` / `body_regions` | 区域检测结果 |
| `freshness_level` / `confidence_score` / `confidence_level` | 新鲜度分类结果 |
| `heatmap_image` / `heatmap_interpretation` | 可解释视觉结果 |
| `freshness_report` / `recommendations` | 报告和建议 |
| `record_id` / `sample_pool_id` | 反馈闭环标识 |

## 分支策略

质量分支：

- 非鱼类图片：终止并提示重新上传。
- 无鱼眼：终止并提示调整拍摄角度。
- 质量不合格：终止并给出拍摄建议。
- 合格：进入预处理和后续分析。

置信度分支：

- `confidence_score >= 0.8`：直接进入 Grad-CAM 和报告路径。
- `0.5 <= confidence_score < 0.8`：进入多区域融合，再进入报告路径。
- `confidence_score < 0.5`：进入备用复核；复核仍低置信度时提示重拍。

## LLM 配置

节点提示词和模型参数位于 `config/*.json`。运行时需要配置 OpenAI-compatible 接口：

```bash
export OPENAI_API_KEY=your_api_key
export OPENAI_BASE_URL=https://your-openai-compatible-endpoint/v1
```

`src/tools/llm_client.py` 会把本地图片路径转换为 data URL，因此工作流支持本地图片和远程图片两种输入。

## 本地模型服务

模型服务位于 `src/api/model_service.py`，提供：

- `GET /`
- `POST /predict`
- `POST /predict_url`
- `POST /predict_with_gradcam`
- `POST /gradcam_url`

工作流客户端位于 `src/tools/fishfreshnet_client.py`。本地路径会走文件上传接口，远程 URL 会走 JSON 接口。

## Windows 本地程序

`installer/launcher_app.py` 是 V3.4.0 Windows 启动器源码。它面向本地用户提供：

- 图片选择、预览和 AI 关注区域叠加。
- 数值居中的固定比例置信度仪表盘、概率分布和结构化结果卡片。
- 无占位符图标的选项卡、舒展行距和历史记录缩略图。
- 批量分析和 CSV 汇总导出。
- PDF/JSON 导出，PDF 图像按宽高比缩放。

## 验证建议

```bash
python -m compileall src installer
python src/main.py -m http -p 5000
python -m uvicorn src.api.model_service:app --host 0.0.0.0 --port <port>
```

如果当前环境尚未安装 `langgraph`、`torch` 等依赖，编译可以通过，但导入或运行服务需要先安装对应依赖。
