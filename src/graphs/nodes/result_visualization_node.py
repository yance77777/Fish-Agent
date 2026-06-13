"""
节点⑬：结果可视化展示
整合热力图、报告和建议，生成最终可视化结果
"""
import json
from datetime import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import ResultVisualizationInput, ResultVisualizationOutput


def result_visualization_node(
    state: ResultVisualizationInput, 
    config: RunnableConfig, 
    runtime: Runtime[Any]
) -> ResultVisualizationOutput:
    """
    title: 结果可视化展示
    desc: 整合热力图、新鲜度评估报告和处理建议，生成最终可视化结果
    """
    # 安全处理可选字段 heatmap_image
    heatmap_url: str = ""
    if state.heatmap_image is not None:
        heatmap_url = state.heatmap_image.url

    final_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "freshness_level": state.freshness_level,
        "confidence_score": state.confidence_score,
        "heatmap_url": heatmap_url,
        "report": state.freshness_report,
        "recommendations": state.recommendations
    }
    
    # 构建可视化数据
    visualization_data = {
        "heatmap_image_url": heatmap_url,
        "summary": state.freshness_report.get("summary", {}),
        "key_recommendations": {
            "storage": state.recommendations.get("storage", {}),
            "consumption": state.recommendations.get("consumption", {}),
            "safety_warnings": state.recommendations.get("safety_warnings", {})
        }
    }
    
    return ResultVisualizationOutput(
        final_result=final_result,
        visualization_data=visualization_data
    )


