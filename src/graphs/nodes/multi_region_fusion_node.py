"""
节点⑦b 中置信度路径：多区域融合分析
使用真实的大模型API调用

"""
import json
from typing import Dict, Any, List, Optional
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import MultiRegionFusionInput, MultiRegionFusionOutput
from graphs.utils import get_confidence_level, config_path


def multi_region_fusion_node(
    state: MultiRegionFusionInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> MultiRegionFusionOutput:
    """
    title: 多区域融合分析
    desc: 融合鱼眼、鳃部、体表等多个区域的新鲜度判断，提升复杂场景下的鲁棒性
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
    
    initial_freshness: str = state.freshness_level or "新鲜"
    initial_confidence_val: float = state.confidence_score or 0.0
    
    # 获取图片URL
    image_url: str = state.processed_image.url if state.processed_image else ""
    
    # 初始化评分变量（默认值）
    fish_eye_score: float = initial_confidence_val * 0.95
    gill_score: float = initial_confidence_val * 0.85
    body_score: float = initial_confidence_val * 0.90
    fusion_score: float = fish_eye_score * 0.5 + gill_score * 0.3 + body_score * 0.2
    freshness_level: str = initial_freshness
    confidence_score: float = fusion_score
    confidence_level: str = get_confidence_level(confidence_score)
    fusion_method: str = "默认加权融合"
    fusion_details: Dict[str, Any] = {}
    
    # 构建用户提示词
    user_prompt_template: str = llm_config.get("up", """
请根据以下信息进行多区域融合新鲜度分析：

初始检测结果：
- 新鲜度等级：{{initial_freshness}}
- 置信度：{{initial_confidence}}

区域检测结果：
- 鱼眼区域数量：{{fish_eye_count}}
- 鳃部区域数量：{{gill_count}}
- 体表区域数量：{{body_count}}

图片URL: {{image_url}}

请分析各区域特征并给出融合评分结果。输出JSON格式：
{
  "fish_eye_score": 0.0-1.0,
  "gill_score": 0.0-1.0,
  "body_score": 0.0-1.0,
  "freshness_level": "高度新鲜|新鲜|不新鲜",
  "confidence_score": 0.0-1.0,
  "confidence_level": "高|中|低",
  "fusion_method": "融合方法描述",
  "fusion_details": {
    "eye_analysis": "鱼眼区域分析",
    "gill_analysis": "鳃部区域分析",
    "body_analysis": "体表区域分析"
  }
}
""")
    
    # 使用jinja2模板渲染提示词
    up_tpl: Template = Template(user_prompt_template)
    user_prompt: str = up_tpl.render({
        "initial_freshness": initial_freshness,
        "initial_confidence": initial_confidence_val,
        "fish_eye_count": len(state.fish_eye_regions),
        "gill_count": len(state.gill_regions),
        "body_count": len(state.body_regions),
        "image_url": image_url
    })
    
    # 系统提示词
    system_prompt: str = llm_config.get("sp", """你是水产品新鲜度评估专家，擅长多区域融合分析。
基于鱼眼、鳃部、体表等多个区域特征，综合判断水产品新鲜度。
请给出专业、客观的融合评分结果。

融合权重建议：
- 鱼眼区域：50%（核心判断依据）
- 鳃部区域：30%（重要辅助依据）
- 体表区域：20%（辅助依据）""")
    
    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.3)
    max_tokens: int = model_config.get("max_completion_tokens", 1000)
    
    # 构建消息列表（如果有图片，使用多模态）
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
    
    try:
        # 使用LLMClient调用大模型
        client: LLMClient = LLMClient(ctx=ctx)
        
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
            
            # 提取评分结果
            fish_eye_score = float(result.get("fish_eye_score", fish_eye_score))
            gill_score = float(result.get("gill_score", gill_score))
            body_score = float(result.get("body_score", body_score))
            fusion_score = fish_eye_score * 0.5 + gill_score * 0.3 + body_score * 0.2
            
            # 使用全局状态字段名
            freshness_level = str(result.get("freshness_level", result.get("fusion_freshness_level", freshness_level)))
            confidence_score = float(result.get("confidence_score", result.get("fusion_confidence", fusion_score)))
            confidence_level = str(result.get("confidence_level", get_confidence_level(confidence_score)))
            fusion_method = str(result.get("fusion_method", "大模型融合分析"))
            fusion_details = result.get("fusion_details", {})
            
    except json.JSONDecodeError:
        # JSON解析失败，使用默认融合逻辑
        fusion_method = "JSON解析失败，使用默认融合"
        confidence_level = get_confidence_level(confidence_score)
        
    except Exception:
        # LLM调用失败，使用默认融合逻辑
        fusion_method = "降级加权融合"
        confidence_level = get_confidence_level(confidence_score)
    
    # 根据融合评分确定新鲜度等级（如果LLM没有给出明确结果）
    if fusion_score >= 0.8:
        if freshness_level not in ["高度新鲜", "新鲜", "不新鲜"]:
            freshness_level = "高度新鲜"
    elif fusion_score >= 0.5:
        if freshness_level not in ["高度新鲜", "新鲜", "不新鲜"]:
            freshness_level = "新鲜"
    else:
        if freshness_level not in ["高度新鲜", "新鲜", "不新鲜"]:
            freshness_level = "不新鲜"
    
    # 确保置信度水平正确
    confidence_level = get_confidence_level(confidence_score)
    
    return MultiRegionFusionOutput(
        fish_eye_score=fish_eye_score,
        gill_score=gill_score,
        body_score=body_score,
        fusion_score=fusion_score,
        freshness_level=freshness_level,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        fusion_method=fusion_method,
        fusion_details=fusion_details
    )


