"""
节点⑩ 时序新鲜度分析
使用大模型结合当前检测结果推理新鲜度趋势。

"""
import json
from typing import Dict, Any, List, Optional
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import TemporalAnalysisInput, TemporalAnalysisOutput
from graphs.utils import config_path


def temporal_analysis_node(
    state: TemporalAnalysisInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> TemporalAnalysisOutput:
    """
    title: 时序新鲜度分析
    desc: 分析新鲜度变化趋势，预测剩余保质时间（使用大模型推理）
    integrations: 大语言模型
    """
    ctx = runtime.context
    
    # 读取配置文件
    cfg_path: str = config.get("metadata", {}).get("llm_cfg", "")
    llm_config: Dict[str, Any] = {}
    
    if cfg_path:
        full_cfg_path = config_path(cfg_path)
        try:
            with open(full_cfg_path, "r", encoding="utf-8") as f:
                llm_config = json.load(f)
        except Exception:
            llm_config = {}
    
    # 初始化结果
    has_history: bool = False
    history_records: List[Dict[str, Any]] = []
    freshness_trend: str = "无历史数据"
    predicted_remaining_hours: Optional[int] = None
    trend_analysis: Dict[str, Any] = {}
    
    # 使用全局状态字段名获取当前检测信息
    current_freshness: str = state.freshness_level or "新鲜"
    current_confidence: float = state.confidence_score or 0.0
    current_confidence_level: str = state.confidence_level or "中"
    
    # 构建用户提示词（使用大模型推理趋势）
    user_prompt_template: str = llm_config.get("up", """
请根据当前水产品新鲜度检测结果，预测新鲜度变化趋势和剩余保质时间。

当前检测结果：
- 新鲜度等级：{{current_freshness}}
- 置信度分数：{{current_confidence}}
- 置信度水平：{{current_confidence_level}}

请分析并输出以下信息（JSON格式）：
{
  "freshness_trend": "稳定|缓慢下降|快速下降",
  "predicted_remaining_hours": 预计剩余保质小时数（整数）,
  "trend_analysis": {
    "trend_description": "趋势描述",
    "risk_level": "低|中|高",
    "recommendation": "建议操作",
    "factors": ["影响因素1", "影响因素2"]
  }
}

注意：
- 高度新鲜（置信度>0.9）：预计可保存24小时以上
- 新鲜（置信度0.6-0.9）：预计可保存12-24小时
- 不新鲜（置信度<0.6）：预计可保存不足12小时，建议尽快处理
""")
    
    # 使用jinja2模板渲染提示词
    up_tpl: Template = Template(user_prompt_template)
    user_prompt: str = up_tpl.render({
        "current_freshness": current_freshness,
        "current_confidence": current_confidence,
        "current_confidence_level": current_confidence_level
    })
    
    # 系统提示词
    system_prompt: str = llm_config.get("sp", """你是水产品新鲜度预测专家。
根据当前新鲜度状态和环境因素，科学预测新鲜度变化趋势和剩余保质时间。
请给出专业、谨慎的预测结果。

预测规则：
- 高度新鲜：新鲜度稳定，预计24-48小时保质期
- 新鲜：新鲜度缓慢下降，预计12-24小时保质期
- 不新鲜：新鲜度快速下降，预计6-12小时保质期，建议尽快处理""")
    
    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.5)
    max_tokens: int = model_config.get("max_completion_tokens", 1000)
    
    try:
        # 使用LLMClient调用大模型
        client: LLMClient = LLMClient(ctx=ctx)
        
        # 构建消息列表
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        # 调用大模型
        response = client.invoke(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_completion_tokens=max_tokens
        )
        
        # 解析LLM响应
        response_content: Any = response.content
        
        # 转换为字符串
        response_str: str
        if isinstance(response_content, str):
            response_str = response_content
        elif isinstance(response_content, list):
            text_parts: List[str] = []
            for item in response_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            response_str = " ".join(text_parts)
        else:
            response_str = str(response_content)
        
        # 尝试解析JSON结果
        json_start: int = response_str.find("{")
        json_end: int = response_str.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str: str = response_str[json_start:json_end]
            result: Dict[str, Any] = json.loads(json_str)
            
            # 提取趋势分析结果
            freshness_trend = str(result.get("freshness_trend", "稳定"))
            predicted_remaining_hours = int(result.get("predicted_remaining_hours", 24))
            trend_analysis = result.get("trend_analysis", {})
            
    except json.JSONDecodeError:
        # JSON解析失败，使用基于置信度的默认预测
        if current_freshness == "高度新鲜" and current_confidence >= 0.9:
            freshness_trend = "稳定"
            predicted_remaining_hours = 24
            trend_analysis = {
                "trend_description": "新鲜度保持稳定",
                "risk_level": "低",
                "recommendation": "可在冷藏条件下保存24小时"
            }
        elif current_freshness == "新鲜" or (current_freshness == "高度新鲜" and current_confidence >= 0.6):
            freshness_trend = "缓慢下降"
            predicted_remaining_hours = 12
            trend_analysis = {
                "trend_description": "新鲜度缓慢下降",
                "risk_level": "中",
                "recommendation": "建议当天食用"
            }
        else:
            freshness_trend = "快速下降"
            predicted_remaining_hours = 6
            trend_analysis = {
                "trend_description": "新鲜度快速下降",
                "risk_level": "高",
                "recommendation": "建议尽快烹饪食用"
            }
            
    except Exception:
        # LLM调用失败，使用默认预测
        if current_confidence >= 0.8:
            freshness_trend = "稳定"
            predicted_remaining_hours = 24
        elif current_confidence >= 0.5:
            freshness_trend = "缓慢下降"
            predicted_remaining_hours = 12
        else:
            freshness_trend = "快速下降"
            predicted_remaining_hours = 6
            
        trend_analysis = {
            "trend_description": f"基于置信度{current_confidence:.2f}的预测",
            "risk_level": "中",
            "recommendation": "建议谨慎参考"
        }
    
    return TemporalAnalysisOutput(
        has_history=has_history,
        history_records=history_records,
        freshness_trend=freshness_trend,
        predicted_remaining_hours=predicted_remaining_hours,
        trend_analysis=trend_analysis
    )


