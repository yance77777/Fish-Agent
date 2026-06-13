"""
Fish-Agent 主图编排（16节点，14个逻辑步骤）
实现置信度驱动的自适应分支、多区域融合和反馈闭环。

节点结构（14个逻辑步骤，16个实际节点）：
- 步骤① 图片上传 (image_upload)
- 步骤② 增强版图像质量检测 (enhanced_quality_check)
- 步骤③ 质量不合格处理 (quality_unqualified)
- 步骤④ 图像预处理 (image_preprocess)
- 步骤⑤ 鱼体区域识别 (fish_region_detection)
- 步骤⑥ 新鲜度分类 (freshness_classification)
- 步骤⑦a 高置信度路径（无额外节点，直接进入⑧）
- 步骤⑦b 中置信度路径：多区域融合 (multi_region_fusion)
- 步骤⑦c 低置信度路径：备用模型复核 (backup_model_review)
- 步骤⑦d 低置信度路径：提示重拍 (low_confidence_prompt)
- 步骤⑧ Grad-CAM可视化 (enhanced_gradcam)
- 步骤⑨ 结构化分析 (structured_analysis)
- 步骤⑩ 时序分析 (temporal_analysis)
- 步骤⑪ 报告生成 (report_generation)
- 步骤⑫ 建议生成 (recommendation_generation)
- 步骤⑬ 结果可视化 (result_visualization)
- 步骤⑭ 反馈闭环 (feedback_loop)
"""
import math

from langgraph.graph import StateGraph, END
from graphs.state import (
    GlobalState,
    GraphInput,
    GraphOutput,
    QualityDecisionInput,
    ConfidenceDecisionInput
)

# 导入所有节点函数
from graphs.nodes.image_upload_node import image_upload_node
from graphs.nodes.enhanced_quality_check_node import enhanced_quality_check_node
from graphs.nodes.quality_unqualified_node import quality_unqualified_node
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.fish_region_detection_node import fish_region_detection_node
from graphs.nodes.freshness_classification_node import freshness_classification_node
from graphs.nodes.multi_region_fusion_node import multi_region_fusion_node
from graphs.nodes.backup_model_review_node import backup_model_review_node
from graphs.nodes.low_confidence_prompt_node import low_confidence_prompt_node
from graphs.nodes.enhanced_gradcam_node import enhanced_gradcam_node
from graphs.nodes.structured_analysis_node import structured_analysis_node
from graphs.nodes.temporal_analysis_node import temporal_analysis_node
from graphs.nodes.report_generation_node import report_generation_node
from graphs.nodes.recommendation_generation_node import recommendation_generation_node
from graphs.nodes.result_visualization_node import result_visualization_node
from graphs.nodes.feedback_loop_node import feedback_loop_node


# ==================== 条件分支函数 ====================

def quality_decision(state: QualityDecisionInput) -> str:
    """
    title: 质量检测决策（增强版）
    desc: 根据图像质量检测结果决定后续流程，包括鱼类识别、鱼眼检测、质量评估
    """
    # 三重检查：1.是否为鱼类 2.是否包含鱼眼 3.质量是否合格
    if not state.is_fish_image:
        return "非鱼类图片"
    elif not state.contains_fish_eye:
        return "无鱼眼"
    elif not state.is_qualified:
        return "质量不合格"
    else:
        return "合格"


def confidence_decision(state: ConfidenceDecisionInput) -> str:
    """
    title: 置信度分支决策
    desc: 根据置信度分数自适应选择后续处理路径
    """
    if state.confidence_score is None or math.isnan(float(state.confidence_score)):
        return "低置信度"
    if state.confidence_score >= 0.8:
        return "高置信度"
    elif state.confidence_score >= 0.5:
        return "中置信度"
    else:
        return "低置信度"


# ==================== 主图编排 ====================

# 创建状态图
builder = StateGraph(
    GlobalState,
    input_schema=GraphInput,
    output_schema=GraphOutput
)

# ==================== 添加节点 ====================

# 节点① 图像采集与上传
builder.add_node("image_upload", image_upload_node)

# 节点② 增强版图像质量检测（判断鱼类、鱼眼、质量）
builder.add_node(
    "enhanced_quality_check",
    enhanced_quality_check_node,
    metadata={"type": "agent", "llm_cfg": "config/enhanced_quality_check_llm_cfg.json"}
)

# 节点③ 质量不合格提示（新增：非鱼类/无鱼眼/质量差）
builder.add_node("quality_unqualified", quality_unqualified_node)

# 节点④ 图像预处理（场景自适应）
builder.add_node("image_preprocess", image_preprocess_node)

# 节点⑤ 鱼体区域识别（多区域：鱼眼/鳃部/体表）
builder.add_node(
    "fish_region_detection",
    fish_region_detection_node,
    metadata={"type": "agent", "llm_cfg": "config/fish_region_detection_llm_cfg.json"}
)

# 节点⑥ 新鲜度分类（主模型）
builder.add_node(
    "freshness_classification",
    freshness_classification_node,
    metadata={"type": "agent", "llm_cfg": "config/freshness_classification_llm_cfg.json"}
)

# ==================== 置信度驱动的自适应分支节点 ====================

# 节点⑦a 高置信度路径：无额外节点，直接进入Grad-CAM

# 节点⑦b 中置信度路径：多区域融合分析
builder.add_node(
    "multi_region_fusion",
    multi_region_fusion_node,
    metadata={"type": "agent", "llm_cfg": "config/multi_region_fusion_llm_cfg.json"}
)

# 节点⑦c 低置信度路径：备用模型复核
builder.add_node(
    "backup_model_review",
    backup_model_review_node,
    metadata={"type": "agent", "llm_cfg": "config/backup_model_review_llm_cfg.json"}
)

# 节点⑦d 低置信度路径：提示重拍
builder.add_node(
    "low_confidence_prompt",
    low_confidence_prompt_node
)

# ==================== 后续节点（统一路径） ====================

# 节点⑧ Grad-CAM可视化生成（强化版）
builder.add_node(
    "enhanced_gradcam",
    enhanced_gradcam_node,
    metadata={"type": "agent", "llm_cfg": "config/enhanced_gradcam_llm_cfg.json"}
)

# 节点⑨ 大模型结构化专业分析
builder.add_node(
    "structured_analysis",
    structured_analysis_node,
    metadata={"type": "agent", "llm_cfg": "config/structured_analysis_llm_cfg.json"}
)

# 节点⑩ 时序新鲜度分析（历史趋势/保质期预测）
builder.add_node(
    "temporal_analysis",
    temporal_analysis_node,
    metadata={"type": "agent", "llm_cfg": "config/temporal_analysis_llm_cfg.json"}
)

# 节点⑪ 报告生成（结构化）
builder.add_node(
    "report_generation",
    report_generation_node,
    metadata={"type": "agent", "llm_cfg": "config/report_generation_llm_cfg.json"}
)

# 节点⑫ 建议生成
builder.add_node(
    "recommendation_generation",
    recommendation_generation_node,
    metadata={"type": "agent", "llm_cfg": "config/recommendation_generation_llm_cfg.json"}
)

# 节点⑬ 结果可视化展示
builder.add_node("result_visualization", result_visualization_node)

# 节点⑭ 用户反馈收集与闭环
builder.add_node("feedback_loop", feedback_loop_node)


# ==================== 设置边 ====================

# 设置入口点
builder.set_entry_point("image_upload")

# 节点① → 节点②
builder.add_edge("image_upload", "enhanced_quality_check")

# 节点② 质量检测条件分支（四条路径）
builder.add_conditional_edges(
    source="enhanced_quality_check",
    path=quality_decision,
    path_map={
        "合格": "image_preprocess",
        "非鱼类图片": "quality_unqualified",
        "无鱼眼": "quality_unqualified",
        "质量不合格": "quality_unqualified"
    }
)

# 质量不合格节点直接结束
builder.add_edge("quality_unqualified", END)

# 节点④ → 节点⑤
builder.add_edge("image_preprocess", "fish_region_detection")

# 节点⑤ → 节点⑥
builder.add_edge("fish_region_detection", "freshness_classification")

# 节点⑥ 置信度条件分支（三条路径）
builder.add_conditional_edges(
    source="freshness_classification",
    path=confidence_decision,
    path_map={
        "高置信度": "enhanced_gradcam",
        "中置信度": "multi_region_fusion",
        "低置信度": "backup_model_review"
    }
)

# 中置信度路径：多区域融合后进入Grad-CAM
builder.add_edge("multi_region_fusion", "enhanced_gradcam")

# 低置信度路径：备用模型复核后再决策
# 复核后根据结果决定：进入Grad-CAM 或 提示重拍
builder.add_conditional_edges(
    source="backup_model_review",
    path=confidence_decision,
    path_map={
        "高置信度": "enhanced_gradcam",
        "中置信度": "enhanced_gradcam",
        "低置信度": "low_confidence_prompt"
    }
)

# 低置信度提示重拍后结束
builder.add_edge("low_confidence_prompt", END)

# ==================== 后续统一路径 ====================

# 节点⑧ → 节点⑨
builder.add_edge("enhanced_gradcam", "structured_analysis")

# 节点⑨ → 节点⑩
builder.add_edge("structured_analysis", "temporal_analysis")

# 节点⑩ → 节点⑪
builder.add_edge("temporal_analysis", "report_generation")

# 节点⑪ → 节点⑫
builder.add_edge("report_generation", "recommendation_generation")

# 节点⑫ → 节点⑬
builder.add_edge("recommendation_generation", "result_visualization")

# 节点⑬ → 节点⑭
builder.add_edge("result_visualization", "feedback_loop")

# 节点⑭ → END
builder.add_edge("feedback_loop", END)


# ==================== 编译图 ====================

main_graph = builder.compile()
