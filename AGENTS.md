# Fish-Agent 节点索引

本文档面向开发者，说明 Fish-Agent 当前源码中的工作流节点、职责、配置和分支关系。

## 项目定位

- 名称：Fish-Agent 水产品新鲜度智能评估系统
- 输入：鱼眼图像或包含鱼眼的鱼类图片
- 输出：新鲜度等级、置信度、Grad-CAM/关注区域解释、结构化报告、处理建议、反馈记录
- 编排：LangGraph
- 节点数量：16 个实际节点，14 个逻辑步骤

## 节点清单

| 序号 | 节点 | 文件 | 类型 | 职责 |
| --- | --- | --- | --- | --- |
| 1 | `image_upload` | `src/graphs/nodes/image_upload_node.py` | task | 接收图像并记录上传时间 |
| 2 | `enhanced_quality_check` | `src/graphs/nodes/enhanced_quality_check_node.py` | agent | 判断是否为鱼类、是否包含鱼眼、图像质量是否合格 |
| 3 | `quality_unqualified` | `src/graphs/nodes/quality_unqualified_node.py` | task | 输出不合格原因和重拍建议 |
| 4 | `image_preprocess` | `src/graphs/nodes/image_preprocess_node.py` | task | 根据场景和质量等级执行轻量预处理 |
| 5 | `fish_region_detection` | `src/graphs/nodes/fish_region_detection_node.py` | agent | 识别鱼眼、鳃部、体表区域 |
| 6 | `freshness_classification` | `src/graphs/nodes/freshness_classification_node.py` | agent | 优先调用 FishFreshNetV1，失败时回退多模态大模型 |
| 7 | `multi_region_fusion` | `src/graphs/nodes/multi_region_fusion_node.py` | agent | 中置信度路径，融合鱼眼、鳃部、体表信息 |
| 8 | `backup_model_review` | `src/graphs/nodes/backup_model_review_node.py` | agent | 低置信度路径，大模型复核仲裁 |
| 9 | `low_confidence_prompt` | `src/graphs/nodes/low_confidence_prompt_node.py` | task | 二次低置信度时提示重拍或人工确认 |
| 10 | `enhanced_gradcam` | `src/graphs/nodes/enhanced_gradcam_node.py` | agent | 调用真实 Grad-CAM 服务，失败时生成降级关注区域解释 |
| 11 | `structured_analysis` | `src/graphs/nodes/structured_analysis_node.py` | agent | 输出鱼眼外观、清澈度、颜色、纹理和质量指标 |
| 12 | `temporal_analysis` | `src/graphs/nodes/temporal_analysis_node.py` | agent | 预测新鲜度趋势和剩余保质时间 |
| 13 | `report_generation` | `src/graphs/nodes/report_generation_node.py` | agent | 生成结构化报告和知识图谱匹配结果 |
| 14 | `recommendation_generation` | `src/graphs/nodes/recommendation_generation_node.py` | agent | 生成储存、食用、处理、安全建议 |
| 15 | `result_visualization` | `src/graphs/nodes/result_visualization_node.py` | task | 汇总最终展示数据 |
| 16 | `feedback_loop` | `src/graphs/nodes/feedback_loop_node.py` | task | 生成记录 ID，并为中低置信度样本预留样本池入口 |

## 分支规则

质量检测：

```text
enhanced_quality_check
  -> 合格: image_preprocess
  -> 非鱼类图片 / 无鱼眼 / 质量不合格: quality_unqualified -> END
```

置信度：

```text
freshness_classification
  -> confidence_score >= 0.8: enhanced_gradcam
  -> 0.5 <= confidence_score < 0.8: multi_region_fusion -> enhanced_gradcam
  -> confidence_score < 0.5: backup_model_review
```

复核：

```text
backup_model_review
  -> 高/中置信度: enhanced_gradcam
  -> 低置信度: low_confidence_prompt -> END
```

统一报告路径：

```text
enhanced_gradcam
  -> structured_analysis
  -> temporal_analysis
  -> report_generation
  -> recommendation_generation
  -> result_visualization
  -> feedback_loop
  -> END
```

## 配置文件

| 配置 | 节点 |
| --- | --- |
| `config/enhanced_quality_check_llm_cfg.json` | `enhanced_quality_check` |
| `config/fish_region_detection_llm_cfg.json` | `fish_region_detection` |
| `config/freshness_classification_llm_cfg.json` | `freshness_classification` 回退路径 |
| `config/multi_region_fusion_llm_cfg.json` | `multi_region_fusion` |
| `config/backup_model_review_llm_cfg.json` | `backup_model_review` |
| `config/enhanced_gradcam_llm_cfg.json` | `enhanced_gradcam` |
| `config/structured_analysis_llm_cfg.json` | `structured_analysis` |
| `config/temporal_analysis_llm_cfg.json` | `temporal_analysis` |
| `config/report_generation_llm_cfg.json` | `report_generation` |
| `config/recommendation_generation_llm_cfg.json` | `recommendation_generation` |

## 运行入口

- 主图：`src/graphs/graph.py` 中的 `main_graph`
- HTTP/CLI：`src/main.py`
- 模型服务：`src/api/model_service.py`
- 本地安装器：`installer/launcher_app.py`

## 设计要点

- 节点使用 Pydantic 输入输出模型，字段在 `src/graphs/state.py` 中集中定义。
- `confidence_score` 是工作流分支的唯一置信度依据。
- `multi_region_fusion` 和 `backup_model_review` 会回写 `freshness_level` 与 `confidence_score`，后续节点始终读取最新结果。
- 大模型调用通过 `src/tools/llm_client.py` 的 OpenAI-compatible 客户端完成。
- FishFreshNetV1 通过 HTTP 模型服务接入，客户端位于 `src/tools/fishfreshnet_client.py`。
