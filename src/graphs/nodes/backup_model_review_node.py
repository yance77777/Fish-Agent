"""
节点⑦c 低置信度路径：备用模型复核
使用大模型对低置信度结果进行二次复核。

"""
import json
from typing import Dict, Any, Optional, List
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import BackupModelReviewInput, BackupModelReviewOutput
from graphs.utils import get_confidence_level, config_path


def backup_model_review_node(
    state: BackupModelReviewInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> BackupModelReviewOutput:
    """
    title: 备用模型复核（大模型仲裁）
    desc: 低置信度时调用大模型进行复核仲裁，提供更可靠的判断
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

    # 获取初始值（处理Optional类型）
    initial_freshness: str = state.freshness_level or "新鲜"
    initial_confidence_val: float = state.confidence_score if state.confidence_score is not None else 0.0

    # 初始化结果
    backup_freshness_level: str = initial_freshness
    backup_confidence: float = 0.0
    freshness_level: str = initial_freshness
    confidence_score: float = initial_confidence_val
    confidence_level: str = get_confidence_level(confidence_score)
    review_method: str = "大模型复核仲裁"
    review_details: Dict[str, Any] = {}
    uncertainty_reason: str = ""

    # 获取图片URL
    image_url: str = state.processed_image.url if state.processed_image else ""

    # 系统提示词
    system_prompt: str = llm_config.get("sp", """你是水产品新鲜度评估复核专家。
你的任务是对低置信度的检测结果进行复核仲裁。

请根据以下信息进行专业分析：
1. 图片中的鱼眼特征（透明度、清澈度、颜色）
2. 原始检测结果的可信度
3. 可能的干扰因素（如反光、模糊、角度不当）
4. 是否需要建议用户重新拍摄

输出JSON格式的复核结果：
{
  "backup_freshness_level": "高度新鲜|新鲜|不新鲜",
  "backup_confidence": 0.3-0.8,
  "freshness_level": "高度新鲜|新鲜|不新鲜",
  "confidence_score": 0.3-0.8,
  "confidence_level": "高|中|低",
  "review_method": "复核方法描述",
  "review_details": {
    "key_factors": ["关键因素1", "关键因素2"],
    "analysis_summary": "分析摘要"
  },
  "uncertainty_reason": "低置信度原因说明",
  "recommendation": "建议操作：继续检测 或 建议重拍"
}""")

    # 用户提示词
    user_prompt_template: str = llm_config.get("up", """请对以下低置信度检测结果进行复核：

原始检测结果：
- 新鲜度等级：{{freshness_level}}
- 置信度分数：{{confidence_score}}

图片URL: {{image_url}}

这是一个置信度较低（<0.5）的结果，请仔细分析后给出更可靠的判断。
如果图片质量有明显问题（模糊、过曝、角度不当），请建议用户重拍。""")
    user_prompt: str = Template(user_prompt_template).render(
        freshness_level=initial_freshness,
        confidence_score=f"{initial_confidence_val:.2f}",
        image_url=image_url,
        fish_eye_regions=state.fish_eye_regions
    )

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

    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.2)
    max_tokens: int = model_config.get("max_completion_tokens", 800)
    timeout: float = float(model_config.get("timeout", 60))

    try:
        # 使用LLMClient进行复核
        client: LLMClient = LLMClient(ctx=ctx)

        # 调用大模型进行复核
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
            result: Dict[str, Any] = json.loads(json_str)

            # 提取复核结果
            backup_freshness_level = str(result.get("backup_freshness_level", initial_freshness))
            backup_confidence = float(result.get("backup_confidence", max(initial_confidence_val, 0.3)))

            # 使用全局状态字段名
            freshness_level = str(result.get("freshness_level", result.get("final_freshness_level", backup_freshness_level)))
            confidence_score = float(result.get("confidence_score", result.get("final_confidence", backup_confidence)))
            confidence_level = str(result.get("confidence_level", get_confidence_level(confidence_score)))

            review_method = str(result.get("review_method", "大模型复核仲裁"))
            review_details = result.get("review_details", {})
            uncertainty_reason = str(result.get("uncertainty_reason", ""))

    except json.JSONDecodeError:
        # JSON解析失败，保持原值不变
        freshness_level = initial_freshness
        confidence_score = initial_confidence_val
        confidence_level = get_confidence_level(confidence_score)
        review_method = "复核结果解析失败，保持原结果（置信度不变）"
        uncertainty_reason = "LLM响应JSON解析失败"

    except Exception as e:
        # LLM调用失败，保持原结果
        freshness_level = initial_freshness
        confidence_score = initial_confidence_val
        confidence_level = get_confidence_level(confidence_score)
        review_method = f"大模型复核失败: {str(e)}"
        uncertainty_reason = "大模型调用异常"

    # 确保置信度水平正确
    confidence_level = get_confidence_level(confidence_score)

    return BackupModelReviewOutput(
        backup_freshness_level=backup_freshness_level,
        backup_confidence=backup_confidence,
        freshness_level=freshness_level,
        confidence_score=confidence_score,
        confidence_level=confidence_level,
        review_method=review_method,
        review_details=review_details,
        uncertainty_reason=uncertainty_reason
    )

