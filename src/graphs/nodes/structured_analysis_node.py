"""
节点⑨ 大模型结构化专业分析
使用大模型生成鱼眼视觉特征的结构化分析。
"""
import json
from typing import Dict, Any, List, Optional
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import StructuredAnalysisInput, StructuredAnalysisOutput
from graphs.utils import config_path


def structured_analysis_node(
    state: StructuredAnalysisInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> StructuredAnalysisOutput:
    """
    title: 结构化专业分析
    desc: 大模型输出结构化的专业分析，包括鱼眼外观、清澈度、颜色、纹理、质量指标等
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
    
    # 初始化结构化分析结果
    structured_analysis: Dict[str, Any] = {}
    
    # 构建结构化专业分析提示词
    system_prompt: str = llm_config.get("sp", """你是水产品新鲜度评估专家，擅长鱼眼视觉特征分析。
请对鱼眼图片进行结构化专业分析，输出以下内容：

1. 鱼眼外观描述（eye_appearance）
   - 角膜透明度
   - 边界清晰度
   - 瞳孔颜色
   - 虹膜状态

2. 清澈度评估（clarity_assessment）
   - 晶状体清澈程度
   - 是否有浑浊、发白、雾状现象

3. 颜色评估（color_evaluation）
   - 瞳孔颜色深浅
   - 虲膜色彩鲜艳度
   - 整体色调协调性

4. 纹理分析（texture_analysis）
   - 表面光泽度
   - 水润程度
   - 鲜活状态

5. 质量指标（quality_indicators）
   - 积极信号（positive_signs）
   - 警告信号（warning_signs）

输出JSON格式：
{
  "eye_appearance": "详细描述",
  "clarity_assessment": "清澈度评估",
  "color_evaluation": "颜色评估",
  "texture_analysis": "纹理分析",
  "quality_indicators": {
    "positive_signs": ["信号1", "信号2"],
    "warning_signs": ["警告1"]
  }
}""")
    
    # 构建用户提示词（使用Jinja2渲染模板）
    up_template: str = llm_config.get("up", """请根据以下检测结果进行结构化专业分析：

新鲜度等级：{{freshness_level}}
置信度水平：{{confidence_level}}
置信度分数：{{confidence_score}}

热力图解释：{{heatmap_interpretation}}

请输出JSON格式的结构化分析结果。""")

    # 使用Jinja2渲染模板
    image_url: str = state.processed_image.url if state.processed_image else ""
    up_tpl: Template = Template(up_template)
    user_prompt: str = up_tpl.render(
        freshness_level=state.freshness_level,
        confidence_level=state.confidence_level,
        confidence_score=f"{state.confidence_score:.2f}",
        heatmap_interpretation=state.heatmap_interpretation,
        image_url=image_url
    )
    
    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.3)
    max_tokens: int = model_config.get("max_completion_tokens", 1500)
    timeout: float = float(model_config.get("timeout", 60))
    
    try:
        # 使用LLMClient调用大模型
        client: LLMClient = LLMClient(ctx=ctx)
        
        # 构建消息列表。有图片时使用多模态输入，避免仅依赖上游文本描述。
        messages: List[Any]
        if image_url:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ])
            ]
        else:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        
        # 调用大模型
        response = client.invoke(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout
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
            structured_analysis = json.loads(json_str)
            
    except json.JSONDecodeError:
        # JSON解析失败，使用基于新鲜度等级的默认分析
        if state.freshness_level == "高度新鲜":
            structured_analysis = {
                "eye_appearance": "鱼眼角膜透明度极高，边界清晰",
                "clarity_assessment": "晶状体完全清澈",
                "color_evaluation": "瞳孔深黑色，虹膜色彩鲜亮",
                "texture_analysis": "表面水润光泽",
                "quality_indicators": {
                    "positive_signs": ["透明度高", "瞳孔深黑"],
                    "warning_signs": []
                }
            }
        elif state.freshness_level == "新鲜":
            structured_analysis = {
                "eye_appearance": "鱼眼角膜透明度良好",
                "clarity_assessment": "晶状体基本清澈",
                "color_evaluation": "瞳孔黑色，虹膜色彩正常",
                "texture_analysis": "表面光泽度适中",
                "quality_indicators": {
                    "positive_signs": ["透明度良好"],
                    "warning_signs": ["轻微浑浊"]
                }
            }
        else:
            structured_analysis = {
                "eye_appearance": "鱼眼角膜透明度较低",
                "clarity_assessment": "晶状体浑浊明显",
                "color_evaluation": "瞳孔灰白色",
                "texture_analysis": "表面光泽度低",
                "quality_indicators": {
                    "positive_signs": [],
                    "warning_signs": ["透明度低", "瞳孔灰白"]
                }
            }
            
    except Exception:
        # LLM调用失败，使用默认分析
        structured_analysis = {
            "eye_appearance": "基于新鲜度等级的分析",
            "clarity_assessment": "清澈度评估",
            "color_evaluation": "颜色评估",
            "texture_analysis": "纹理分析",
            "quality_indicators": {
                "positive_signs": [],
                "warning_signs": ["建议谨慎参考"]
            }
        }
    
    return StructuredAnalysisOutput(
        structured_analysis=structured_analysis
    )
