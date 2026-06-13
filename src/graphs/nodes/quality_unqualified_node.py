"""
节点⑯ 质量不合格提示（增强版）
- 非鱼类图片
- 无鱼眼
- 质量不合格（模糊、过曝、低光照等）
"""
import uuid
from typing import Any, List
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import QualityUnqualifiedInput, QualityUnqualifiedOutput


def quality_unqualified_node(
    state: QualityUnqualifiedInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> QualityUnqualifiedOutput:
    """
    title: 质量不合格提示
    desc: 根据不合格类型生成针对性的提示信息和改进建议
    """
    # 初始化结果
    error_message: str = ""
    suggested_actions: List[str] = []
    
    # 根据不合格原因生成针对性提示
    if not state.is_fish_image:
        # 非鱼类图片
        error_message = "检测到上传的图片不是鱼类图片。本系统专门用于评估鱼类新鲜度，请上传包含鱼类的图片。"
        suggested_actions = [
            "请确保拍摄对象为鱼类",
            "如需评估其他水产品，请使用相应的评估系统"
        ]
        
    elif not state.contains_fish_eye:
        # 无鱼眼
        error_message = "检测到图片中没有清晰的鱼眼区域。新鲜度评估主要基于鱼眼特征，请确保鱼眼清晰可见。"
        suggested_actions = [
            "调整拍摄角度，确保鱼眼区域清晰可见",
            "避免鱼眼被遮挡或处于画面边缘",
            "如鱼眼已被移除，建议选择其他评估方法"
        ]
        
    else:
        # 质量不合格
        error_message = "图像质量不符合分析要求，存在以下问题："
        
        # 根据具体质量问题生成建议
        for issue in state.quality_issues:
            error_message += f"\n- {issue}"
            
            if "模糊" in issue or "清晰度" in issue:
                suggested_actions.extend([
                    "请使用更高分辨率的相机拍摄",
                    "拍摄时保持相机稳定，避免抖动",
                    "确保拍摄距离适中，不要太远或太近"
                ])
            elif "过曝" in issue or "亮" in issue:
                suggested_actions.extend([
                    "避免在强光下拍摄",
                    "调整曝光设置或使用HDR模式",
                    "尝试在光线柔和的环境拍摄"
                ])
            elif "低光照" in issue or "暗" in issue:
                suggested_actions.extend([
                    "在光线充足的环境下拍摄",
                    "使用辅助光源照亮拍摄对象",
                    "避免在阴影或室内昏暗处拍摄"
                ])
            elif "反光" in issue or "反射" in issue:
                suggested_actions.extend([
                    "调整拍摄角度，避开反光区域",
                    "使用偏振镜减少反光",
                    "避免在有强烈光源反射的位置拍摄"
                ])
        
        # 通用建议
        if not suggested_actions:
            suggested_actions = [
                "请重新拍摄，确保图像质量符合要求",
                "检查拍摄环境和设备设置",
                "如有疑问，请参考拍摄指南"
            ]
    
    final_result: dict = {
        "status": "quality_unqualified",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "is_fish_image": state.is_fish_image,
        "contains_fish_eye": state.contains_fish_eye,
        "quality_issues": state.quality_issues,
        "error_message": error_message,
        "suggested_actions": suggested_actions,
        "run_id": str(uuid.uuid4())
    }

    return QualityUnqualifiedOutput(
        error_message=error_message,
        suggested_actions=suggested_actions,
        final_result=final_result
    )


