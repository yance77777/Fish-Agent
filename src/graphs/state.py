"""
Fish-Agent 状态定义文件
支持置信度驱动的自适应分支、多区域融合和反馈闭环。
"""
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from utils.file.file import File


# ==================== 全局状态定义 ====================

class GlobalState(BaseModel):
    """全局状态定义。"""
    # ==================== 原始输入 ====================
    original_image: Optional[File] = Field(default=None, description="用户上传的原始图片")
    user_id: Optional[str] = Field(default=None, description="用户ID（用于历史数据查询）")
    
    # ==================== 图像质量检测（增强版） ====================
    uploaded_image: Optional[File] = Field(default=None, description="上传后的图片")
    is_fish_image: bool = Field(default=False, description="是否为鱼类图片")
    fish_type: Optional[str] = Field(default=None, description="鱼种识别结果")
    contains_fish_eye: bool = Field(default=False, description="是否包含鱼眼")
    is_qualified: bool = Field(default=False, description="图像质量是否合格")
    quality_level: str = Field(default="", description="质量等级：优秀/良好/合格/不合格")
    quality_issues: List[str] = Field(default=[], description="质量问题列表")
    quality_score: float = Field(default=0.0, description="质量评分（0-100）")
    
    # ==================== 图像预处理 ====================
    processed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    scene_type: str = Field(default="", description="场景类型：菜市场/超市/家庭厨房/其他")
    preprocessing_log: Dict[str, Any] = Field(default={}, description="预处理日志")
    
    # ==================== 鱼体区域识别（多区域融合） ====================
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="检测到的鱼眼区域")
    gill_regions: List[Dict[str, Any]] = Field(default=[], description="检测到的鳃部区域")
    body_regions: List[Dict[str, Any]] = Field(default=[], description="检测到的体表区域")
    region_detection_confidence: float = Field(default=0.0, description="区域检测置信度")
    
    # ==================== 新鲜度分类（主模型） ====================
    freshness_level: str = Field(default="", description="新鲜度等级：高度新鲜/新鲜/不新鲜")
    confidence_score: float = Field(default=0.0, description="主模型置信度分数")
    confidence_level: str = Field(default="", description="置信度水平：高/中/低")
    classification_details: Dict[str, Any] = Field(default={}, description="分类详细信息")
    
    # ==================== 低置信度处理 ====================
    need_secondary_analysis: bool = Field(default=False, description="是否需要二次分析")
    secondary_analysis_type: str = Field(default="", description="二次分析类型：多区域融合/备用模型/人工确认")
    
    # ==================== 多区域融合分析 ====================
    fish_eye_score: float = Field(default=0.0, description="鱼眼区域评分")
    gill_score: float = Field(default=0.0, description="鳃部区域评分")
    body_score: float = Field(default=0.0, description="体表区域评分")
    fusion_score: float = Field(default=0.0, description="融合评分")
    fusion_method: str = Field(default="", description="融合方法描述")
    
    # ==================== 备用模型复核 ====================
    backup_model_used: bool = Field(default=False, description="是否使用备用模型")
    backup_model_result: Dict[str, Any] = Field(default={}, description="备用模型结果")
    final_confidence_after_review: float = Field(default=0.0, description="复核后的最终置信度")
    
    # ==================== Grad-CAM 可视化（强化版） ====================
    heatmap_image: Optional[File] = Field(default=None, description="Grad-CAM热力图")
    heatmap_interpretation: str = Field(default="", description="热力图自然语言解释")
    key_attention_regions: List[str] = Field(default=[], description="关键关注区域描述")
    visualization_url: str = Field(default="", description="可视化结果URL")
    
    # ==================== 结构化专业分析 ====================
    structured_analysis: Dict[str, Any] = Field(default={}, description="结构化分析结果")
    eye_appearance: str = Field(default="", description="鱼眼外观描述")
    clarity_assessment: str = Field(default="", description="清澈度评估")
    color_evaluation: str = Field(default="", description="颜色评估")
    texture_analysis: str = Field(default="", description="纹理分析")
    quality_indicators: Dict[str, Any] = Field(default={}, description="质量指标：积极信号/警告信号")
    
    # ==================== 时序新鲜度分析 ====================
    has_history: bool = Field(default=False, description="是否有历史数据")
    history_records: List[Dict[str, Any]] = Field(default=[], description="历史检测记录")
    freshness_trend: str = Field(default="", description="新鲜度变化趋势")
    predicted_remaining_hours: Optional[int] = Field(default=None, description="预计剩余保质时间（小时）")
    trend_analysis: Dict[str, Any] = Field(default={}, description="趋势分析详情")
    
    # ==================== 结构化报告 ====================
    freshness_report: Dict[str, Any] = Field(default={}, description="新鲜度评估报告（结构化）")
    recommendations: Dict[str, Any] = Field(default={}, description="处理建议（结构化）")
    knowledge_graph_match: Dict[str, Any] = Field(default={}, description="知识图谱匹配结果")
    
    # ==================== 用户反馈闭环 ====================
    allow_user_feedback: bool = Field(default=True, description="是否允许用户反馈")
    user_feedback: Optional[str] = Field(default=None, description="用户反馈内容")
    feedback_type: str = Field(default="", description="反馈类型：正确/错误-偏高/错误-偏低")
    user_corrected_level: Optional[str] = Field(default=None, description="用户修正的新鲜度等级")
    sample_pool_id: Optional[str] = Field(default=None, description="样本池ID（用于再训练）")
    
    # ==================== 最终结果 ====================
    final_result: Dict[str, Any] = Field(default={}, description="最终综合结果")
    record_id: str = Field(default="", description="记录ID")
    processing_time: float = Field(default=0.0, description="处理时间（秒）")


# ==================== 图的输入输出定义 ====================

class GraphInput(BaseModel):
    """工作流输入。"""
    image: File = Field(..., description="用户上传的鱼眼图片")
    user_id: Optional[str] = Field(default=None, description="用户ID（可选，用于历史数据查询）")
    scene_type: Optional[str] = Field(default=None, description="场景类型（可选）")


class GraphOutput(BaseModel):
    """工作流输出。"""
    # 核心结果
    freshness_level: str = Field(default="", description="新鲜度等级")
    confidence_score: float = Field(default=0.0, description="置信度分数")
    confidence_level: str = Field(default="", description="置信度水平：高/中/低")
    
    # 可视化
    heatmap_image: Optional[File] = Field(default=None, description="Grad-CAM热力图")
    heatmap_interpretation: str = Field(default="", description="热力图自然语言解释")
    
    # 结构化报告
    freshness_report: Dict[str, Any] = Field(default={}, description="新鲜度评估报告")
    recommendations: Dict[str, Any] = Field(default={}, description="处理建议")
    
    # 时序分析
    freshness_trend: str = Field(default="", description="新鲜度变化趋势")
    predicted_remaining_hours: Optional[int] = Field(default=None, description="预计剩余保质时间")
    
    # 反馈闭环
    record_id: str = Field(default="", description="记录ID")
    sample_pool_id: Optional[str] = Field(default=None, description="样本池ID")
    
    # 错误处理
    quality_level: str = Field(default="", description="质量等级")
    quality_issues: List[str] = Field(default=[], description="质量问题")
    error_message: str = Field(default="", description="错误信息")


# ==================== 节点输入输出定义 ====================

# ==================== 节点① 图像采集与上传 ====================
class ImageUploadInput(BaseModel):
    """图像采集与上传节点的输入"""
    image: File = Field(..., description="用户上传的图片")
    user_id: Optional[str] = Field(default=None, description="用户ID")


class ImageUploadOutput(BaseModel):
    """图像采集与上传节点的输出"""
    uploaded_image: File = Field(..., description="上传后的图片")
    upload_time: str = Field(..., description="上传时间")
    user_id: Optional[str] = Field(default=None, description="用户ID")


# ==================== 节点② 增强版图像质量检测 ====================
class EnhancedQualityCheckInput(BaseModel):
    """增强版图像质量检测节点的输入"""
    uploaded_image: File = Field(..., description="待检测的图片")


class EnhancedQualityCheckOutput(BaseModel):
    """增强版图像质量检测节点的输出"""
    is_fish_image: bool = Field(..., description="是否为鱼类图片")
    fish_type: Optional[str] = Field(default=None, description="鱼种识别")
    contains_fish_eye: bool = Field(..., description="是否包含鱼眼")
    is_qualified: bool = Field(..., description="质量是否合格")
    quality_level: str = Field(..., description="质量等级：优秀/良好/合格/不合格")
    quality_score: float = Field(..., description="质量评分（0-100）")
    quality_issues: List[str] = Field(default=[], description="质量问题列表")
    scene_type: str = Field(default="", description="场景类型")


# ==================== 节点③ 图像预处理（场景自适应） ====================
class ImagePreprocessInput(BaseModel):
    """图像预处理节点的输入"""
    uploaded_image: File = Field(..., description="原始图片")
    quality_level: str = Field(..., description="质量等级")
    scene_type: str = Field(..., description="场景类型")


class ImagePreprocessOutput(BaseModel):
    """图像预处理节点的输出"""
    processed_image: File = Field(..., description="预处理后的图片")
    preprocessing_log: Dict[str, Any] = Field(default={}, description="预处理日志")


# ==================== 节点④ 鱼体区域识别（多区域） ====================
class FishRegionDetectionInput(BaseModel):
    """鱼体区域识别节点的输入"""
    processed_image: File = Field(..., description="预处理后的图片")


class FishRegionDetectionOutput(BaseModel):
    """鱼体区域识别节点的输出"""
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域")
    gill_regions: List[Dict[str, Any]] = Field(default=[], description="鳃部区域")
    body_regions: List[Dict[str, Any]] = Field(default=[], description="体表区域")
    region_detection_confidence: float = Field(default=0.0, description="区域检测置信度")


# ==================== 节点⑤ 新鲜度分类（主模型） ====================
class FreshnessClassificationInput(BaseModel):
    """新鲜度分类节点的输入"""
    processed_image: File = Field(..., description="预处理后的图片")
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域")


class FreshnessClassificationOutput(BaseModel):
    """新鲜度分类节点的输出"""
    freshness_level: str = Field(..., description="新鲜度等级")
    confidence_score: float = Field(..., description="置信度分数")
    confidence_level: str = Field(..., description="置信度水平：高(>=0.8)/中(0.5-0.8)/低(<0.5)")
    classification_details: Dict[str, Any] = Field(default={}, description="分类详细信息")


# ==================== 节点⑦a 高置信度路径：直接生成结果 ====================
# 使用 FreshnessClassificationOutput 作为输入，无额外节点


# ==================== 节点⑦b 中置信度路径：多区域融合分析 ====================
class MultiRegionFusionInput(BaseModel):
    """多区域融合分析节点的输入。"""
    processed_image: File = Field(..., description="预处理后的图片")
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域")
    gill_regions: List[Dict[str, Any]] = Field(default=[], description="鳃部区域")
    body_regions: List[Dict[str, Any]] = Field(default=[], description="体表区域")
    freshness_level: Optional[str] = Field(default="", description="新鲜度等级（来自上游）")
    confidence_score: Optional[float] = Field(default=0.0, description="置信度分数（来自上游）")


class MultiRegionFusionOutput(BaseModel):
    """多区域融合分析节点的输出。"""
    fish_eye_score: float = Field(..., description="鱼眼评分")
    gill_score: float = Field(..., description="鳃部评分")
    body_score: float = Field(..., description="体表评分")
    fusion_score: float = Field(..., description="融合评分")
    freshness_level: str = Field(..., description="融合后新鲜度等级（覆盖全局状态）")
    confidence_score: float = Field(..., description="融合后置信度分数（覆盖全局状态）")
    confidence_level: str = Field(..., description="置信度水平：高(>=0.8)/中(0.5-0.8)/低(<0.5)")
    fusion_method: str = Field(..., description="融合方法描述")
    fusion_details: Dict[str, Any] = Field(default={}, description="融合详细信息")


# ==================== 节点⑦c 低置信度路径：备用模型复核 ====================
class BackupModelReviewInput(BaseModel):
    """备用模型复核节点的输入。"""
    processed_image: File = Field(..., description="预处理后的图片")
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域")
    freshness_level: Optional[str] = Field(default="", description="新鲜度等级（来自上游）")
    confidence_score: Optional[float] = Field(default=0.0, description="置信度分数（来自上游）")


class BackupModelReviewOutput(BaseModel):
    """备用模型复核节点的输出。"""
    backup_freshness_level: str = Field(..., description="备用模型新鲜度等级")
    backup_confidence: float = Field(..., description="备用模型置信度")
    freshness_level: str = Field(..., description="最终新鲜度等级（覆盖全局状态）")
    confidence_score: float = Field(..., description="最终置信度分数（覆盖全局状态）")
    confidence_level: str = Field(..., description="置信度水平：高(>=0.8)/中(0.5-0.8)/低(<0.5)")
    review_method: str = Field(..., description="复核方法描述")
    review_details: Dict[str, Any] = Field(default={}, description="复核详细信息")
    uncertainty_reason: str = Field(default="", description="低置信度原因说明")


# ==================== 节点⑧ 低置信度提示重拍 ====================
class LowConfidencePromptInput(BaseModel):
    """低置信度提示重拍节点的输入"""
    confidence_score: float = Field(..., description="置信度分数")
    quality_issues: List[str] = Field(default=[], description="质量问题")


class LowConfidencePromptOutput(BaseModel):
    """低置信度提示重拍节点的输出"""
    prompt_message: str = Field(..., description="提示信息")
    suggested_actions: List[str] = Field(default=[], description="建议操作")
    final_result: Dict[str, Any] = Field(default={}, description="最终结果（与GraphOutput对齐）")


# ==================== 节点⑨ Grad-CAM可视化生成（强化版） ====================
class EnhancedGradCAMInput(BaseModel):
    """强化版Grad-CAM节点的输入"""
    processed_image: File = Field(..., description="原始图片")
    freshness_level: str = Field(..., description="新鲜度等级")
    confidence_score: float = Field(default=0.0, description="置信度分数")
    classification_details: Dict[str, Any] = Field(default={}, description="分类详细信息")
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域（用于降级方案）")
    gill_regions: List[Dict[str, Any]] = Field(default=[], description="鳃部区域（用于降级方案）")


class EnhancedGradCAMOutput(BaseModel):
    """强化版Grad-CAM节点的输出"""
    heatmap_image: Optional[File] = Field(default=None, description="热力图")
    heatmap_interpretation: str = Field(..., description="热力图自然语言解释")
    key_attention_regions: List[str] = Field(default=[], description="关键关注区域")


# ==================== 节点⑩ 大模型结构化专业分析 ====================
class StructuredAnalysisInput(BaseModel):
    """结构化专业分析节点的输入"""
    processed_image: Optional[File] = Field(default=None, description="预处理后的图片")
    freshness_level: str = Field(..., description="新鲜度等级")
    confidence_score: float = Field(..., description="置信度分数")
    confidence_level: str = Field(..., description="置信度水平")
    heatmap_interpretation: str = Field(..., description="热力图解释")
    fish_eye_regions: List[Dict[str, Any]] = Field(default=[], description="鱼眼区域")
    gill_regions: List[Dict[str, Any]] = Field(default=[], description="鳃部区域")
    body_regions: List[Dict[str, Any]] = Field(default=[], description="体表区域")
    fish_type: Optional[str] = Field(default=None, description="鱼种")


class StructuredAnalysisOutput(BaseModel):
    """结构化专业分析节点的输出"""
    structured_analysis: Dict[str, Any] = Field(..., description="结构化分析结果（包含eye_appearance、clarity_assessment、color_evaluation、texture_analysis、quality_indicators）")


# ==================== 节点⑪ 时序新鲜度分析 ====================
class TemporalAnalysisInput(BaseModel):
    """时序新鲜度分析节点的输入
    
    【关键修改】：使用全局状态字段名 freshness_level 和 confidence_score，
    确保能读取到多区域融合或备用模型复核后的最新结果。
    """
    user_id: Optional[str] = Field(default=None, description="用户ID")
    freshness_level: str = Field(default="", description="当前新鲜度等级")
    confidence_score: float = Field(default=0.0, description="当前置信度分数")
    confidence_level: str = Field(default="", description="置信度水平")


class TemporalAnalysisOutput(BaseModel):
    """时序新鲜度分析节点的输出"""
    has_history: bool = Field(..., description="是否有历史数据")
    history_records: List[Dict[str, Any]] = Field(default=[], description="历史记录")
    freshness_trend: str = Field(default="", description="新鲜度变化趋势")
    predicted_remaining_hours: Optional[int] = Field(default=None, description="预计剩余保质时间")
    trend_analysis: Dict[str, Any] = Field(default={}, description="趋势分析详情")


# ==================== 节点⑫ 报告生成（结构化） ====================
class ReportGenerationInput(BaseModel):
    """报告生成节点的输入"""
    freshness_level: str = Field(..., description="新鲜度等级")
    confidence_score: float = Field(..., description="置信度分数")
    confidence_level: str = Field(..., description="置信度水平")
    structured_analysis: Dict[str, Any] = Field(..., description="结构化分析")
    heatmap_interpretation: str = Field(..., description="热力图解释")
    temporal_analysis: Dict[str, Any] = Field(default={}, description="时序分析（字典格式）")
    # 时序分析字段（从 TemporalAnalysisOutput 获取）
    freshness_trend: Optional[str] = Field(default="无历史数据", description="新鲜度趋势")
    predicted_remaining_hours: Optional[int] = Field(default=0, description="预计保质时间")
    trend_analysis: Optional[Dict[str, Any]] = Field(default={}, description="趋势分析详情")
    fish_type: Optional[str] = Field(default=None, description="鱼种")


class ReportGenerationOutput(BaseModel):
    """报告生成节点的输出"""
    freshness_report: Dict[str, Any] = Field(..., description="新鲜度评估报告")
    knowledge_graph_match: Dict[str, Any] = Field(default={}, description="知识图谱匹配")


# ==================== 节点⑬ 建议生成 ====================
class RecommendationGenerationInput(BaseModel):
    """建议生成节点的输入"""
    freshness_level: str = Field(..., description="新鲜度等级")
    freshness_report: Dict[str, Any] = Field(..., description="评估报告")
    knowledge_graph_match: Dict[str, Any] = Field(default={}, description="知识图谱匹配")
    predicted_remaining_hours: Optional[int] = Field(default=None, description="预计保质时间")
    fish_type: Optional[str] = Field(default=None, description="鱼种")


class RecommendationGenerationOutput(BaseModel):
    """建议生成节点的输出"""
    recommendations: Dict[str, Any] = Field(..., description="处理建议")


# ==================== 节点⑭ 结果可视化展示 ====================
class ResultVisualizationInput(BaseModel):
    """结果可视化展示节点的输入"""
    processed_image: Optional[File] = Field(default=None, description="原始图片")
    heatmap_image: Optional[File] = Field(default=None, description="热力图")
    freshness_level: str = Field(..., description="新鲜度等级")
    confidence_score: float = Field(..., description="置信度分数")
    freshness_report: Dict[str, Any] = Field(default={}, description="评估报告")
    recommendations: Dict[str, Any] = Field(default={}, description="处理建议")


class ResultVisualizationOutput(BaseModel):
    """结果可视化展示节点的输出"""
    final_result: Dict[str, Any] = Field(..., description="最终完整结果")
    visualization_data: Dict[str, Any] = Field(..., description="可视化数据")


# ==================== 节点⑮ 用户反馈收集与闭环 ====================
class FeedbackLoopInput(BaseModel):
    """用户反馈收集节点的输入"""
    record_id: Optional[str] = Field(default=None, description="记录ID（如缺失则自动生成）")
    freshness_level: str = Field(..., description="检测的新鲜度等级")
    confidence_score: float = Field(..., description="置信度分数")
    user_id: Optional[str] = Field(default=None, description="用户ID")


class FeedbackLoopOutput(BaseModel):
    """用户反馈收集节点的输出"""
    record_id: str = Field(..., description="记录ID")
    sample_pool_id: Optional[str] = Field(default=None, description="样本池ID")
    feedback_prompt: str = Field(..., description="反馈提示语")


# ==================== 节点⑯ 质量不合格提示 ====================
class QualityUnqualifiedInput(BaseModel):
    """质量不合格提示节点的输入"""
    quality_issues: List[str] = Field(..., description="质量问题列表")
    is_fish_image: bool = Field(..., description="是否为鱼类")
    contains_fish_eye: bool = Field(..., description="是否包含鱼眼")


class QualityUnqualifiedOutput(BaseModel):
    """质量不合格提示节点的输出"""
    error_message: str = Field(..., description="错误信息")
    suggested_actions: List[str] = Field(default=[], description="建议操作")
    final_result: Dict[str, Any] = Field(default={}, description="最终结果（与GraphOutput对齐）")


# ==================== 条件分支函数的输入定义 ====================

class QualityDecisionInput(BaseModel):
    """质量检测决策的输入"""
    is_fish_image: bool = Field(..., description="是否为鱼类")
    contains_fish_eye: bool = Field(..., description="是否包含鱼眼")
    is_qualified: bool = Field(..., description="质量是否合格")


class ConfidenceDecisionInput(BaseModel):
    """置信度分支决策的输入"""
    confidence_score: float = Field(..., description="置信度分数")
    quality_level: str = Field(..., description="质量等级")
