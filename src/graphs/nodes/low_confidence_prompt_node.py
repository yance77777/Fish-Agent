"""
节点⑦d 低置信度路径：提示重拍
"""
import uuid
from typing import Any, List
from datetime import datetime
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import LowConfidencePromptInput, LowConfidencePromptOutput


def low_confidence_prompt_node(
    state: LowConfidencePromptInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> LowConfidencePromptOutput:
    """
    title: 低置信度提示重拍
    desc: 置信度过低（<0.5）且备用模型复核仍低时，提示用户重新拍摄或人工确认
    """
    # 生成提示信息
    prompt_message: str = ""
    suggested_actions: List[str] = []

    # 根据置信度和质量问题生成针对性提示
    if state.confidence_score < 0.3:
        prompt_message = "图片识别置信度极低，无法给出可靠判断。"
        suggested_actions = [
            "请确保拍摄对象为鱼类",
            "请确保图片中包含清晰的鱼眼",
            "请在光线充足的环境下拍摄",
            "请避免模糊、过曝或反光"
        ]
    elif state.confidence_score < 0.5:
        prompt_message = "图片识别置信度较低，建议重新拍摄以获得更准确的结果。"
        suggested_actions = [
            "调整拍摄角度，确保鱼眼清晰可见",
            "改善光线条件",
            "减少背景杂乱干扰"
        ]

    # 如果有质量问题，添加相应建议
    if state.quality_issues:
        for issue in state.quality_issues:
            if "模糊" in issue:
                suggested_actions.append("请使用更高分辨率的相机或稳定拍摄")
            elif "过曝" in issue:
                suggested_actions.append("请避免强光直射，调整曝光设置")
            elif "低光照" in issue:
                suggested_actions.append("请在光线充足的环境下拍摄")

    final_result: dict = {
        "status": "low_confidence",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "confidence_score": state.confidence_score,
        "quality_issues": state.quality_issues,
        "prompt_message": prompt_message,
        "suggested_actions": suggested_actions,
        "run_id": str(uuid.uuid4())
    }

    return LowConfidencePromptOutput(
        prompt_message=prompt_message,
        suggested_actions=suggested_actions,
        final_result=final_result
    )


