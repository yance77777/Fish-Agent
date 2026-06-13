# Fish-Agent 水产品新鲜度智能评估系统

Fish-Agent 是一个面向水产品食品安全场景的 AI 智能体工作流。系统以鱼眼图像为核心输入，结合 FishFreshNetV1 视觉模型、多模态大模型、LangGraph 编排和 Grad-CAM 可解释分析，输出新鲜度等级、置信度、视觉依据、结构化报告和处理建议。

项目已整理为标准 Python 源码仓库，不依赖特定智能体平台；Windows 本地程序的启动器源码与打包配置也保留在仓库中，便于复现和二次开发。

## 核心能力

- 判断上传图片是否为鱼类、是否包含清晰鱼眼、图像质量是否合格。
- 使用 FishFreshNetV1 对鱼眼图像进行三分类：高度新鲜 / 新鲜 / 不新鲜。
- 当专用模型服务不可用时，回退到 OpenAI-compatible 多模态大模型。
- 根据置信度自动选择高、中、低三条分析路径。
- 生成 Grad-CAM 热力图或降级关注区域标注，并配套自然语言解释。
- 输出结构化报告、保质期趋势预测、储存/食用/处理/安全建议。
- 记录反馈入口，为后续样本池扩充和模型迭代保留接口。

## 工作流

```text
image_upload
  -> enhanced_quality_check
      -> [不合格 / 非鱼类 / 无鱼眼] quality_unqualified -> END
      -> [合格] image_preprocess
  -> fish_region_detection
  -> freshness_classification
      -> [confidence >= 0.8] enhanced_gradcam
      -> [0.5 <= confidence < 0.8] multi_region_fusion -> enhanced_gradcam
      -> [confidence < 0.5] backup_model_review
            -> [二次高/中置信度] enhanced_gradcam
            -> [二次低置信度] low_confidence_prompt -> END
  -> structured_analysis
  -> temporal_analysis
  -> report_generation
  -> recommendation_generation
  -> result_visualization
  -> feedback_loop
  -> END
```

## 目录结构

```text
Fish-Agent/
├── config/                     # 多模态/文本大模型提示词与模型参数
├── docs/                       # 模型集成与工作流说明
├── installer/                  # Windows 本地程序源码与打包配置
├── scripts/                    # 辅助脚本
├── src/
│   ├── api/model_service.py    # FishFreshNetV1 推理服务
│   ├── graphs/                 # LangGraph 工作流、状态和节点
│   ├── models/                 # FishFreshNetV1 模型结构
│   ├── tools/                  # LLM 客户端与模型服务客户端
│   └── utils/                  # 文件对象和文档解析工具
├── AGENTS.md                   # 节点索引
├── LICENSE
├── pyproject.toml
├── requirements.txt
└── requirements-model.txt
```

## 快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/yance77777/Fish-Agent.git
cd Fish-Agent

# 2. 安装依赖
pip install -r requirements.txt
# 如需本地运行模型服务，额外安装：
pip install -r requirements-model.txt

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env。OPENAI_API_KEY 和 OPENAI_BASE_URL 为工作流必填；
# FISHFRESHNET_API_URL / MODEL_SERVICE_URL 为专用模型服务可选配置。

# 4. 启动工作流服务
python src/main.py -m http -p <port>

# 5. （可选）启动 FishFreshNetV1 模型服务
python -m uvicorn src.api.model_service:app --host <host> --port <port>
```

## 配置说明

基础工作流依赖写在 `requirements.txt`，模型服务相关依赖写在 `requirements-model.txt`。如需完全可复现部署，可参考 `requirements-lock.example.txt` 在目标环境生成锁定文件。多模态大模型调用通过 OpenAI-compatible 接口完成，可通过环境变量配置：

| 变量 | 说明 |
| --- | --- |
| `OPENAI_API_KEY` | 大模型接口密钥 |
| `OPENAI_BASE_URL` / `LLM_BASE_URL` | OpenAI-compatible 服务地址 |
| `FISH_AGENT_WORKSPACE` | 项目工作区路径 |
| `FISHFRESHNET_API_URL` | FishFreshNetV1 推理服务地址 |
| `MODEL_SERVICE_URL` | Grad-CAM 服务地址 |
| `FISHFRESHNET_MODEL_PATH` | FishFreshNetV1 权重文件路径 |

工作流必须配置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`。如果未配置 FishFreshNetV1 模型服务地址，系统会跳过专用模型服务并回退到 OpenAI-compatible 多模态模型。模型权重文件体积较大，不随源码仓库提交；请根据实际部署环境自行放置并配置路径。

## Windows 本地程序

`installer/launcher_app.py` 是 V3.4.0 Windows 启动器源码，主要能力包括：

- 图像预览与 AI 关注区域叠加滑块。
- 结果卡片、置信度仪表盘、概率分布、处理建议和历史记录。
- 无占位符图标的强对比度选项卡、滚动边界控制和历史记录缩略图。
- 固定比例置信度仪表盘，数值整体居中于半弧中央，概率分布图例和列表行距更舒展。
- PDF/JSON 导出，PDF 中图像保持原始宽高比。
- 批量分析工作台与 CSV 汇总导出。

打包配置位于：

```text
installer/Fish-Agent.spec
installer/Fish-Agent-installer.iss
```

## 文档

- `AGENTS.md`：工作流节点索引和分支规则。
- `docs/FishFreshNetV1_Integration_Guide.md`：模型服务与 Grad-CAM 集成说明。
- `docs/Workflow_Optimization_Complete_Report.md`：工作流技术说明。

## 许可证

MIT License
