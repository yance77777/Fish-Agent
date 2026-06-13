"""
节点⑥：FishFreshNetV1 新鲜度分类
1. 输出置信度水平（高/中/低）
2. 高置信度：> 0.8
3. 中置信度：0.5 - 0.8
4. 低置信度：< 0.5
"""
import json
import logging
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import FreshnessClassificationInput, FreshnessClassificationOutput
from graphs.utils import get_confidence_level, config_path
from tools.fishfreshnet_client import get_client

logger = logging.getLogger(__name__)


def freshness_classification_node(
    state: FreshnessClassificationInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> FreshnessClassificationOutput:
    """
    title: 新鲜度分类
    desc: 使用FishFreshNetV1模型对鱼眼图像进行新鲜度分类，输出置信度水平（高/中/低）
    integrations: FishFreshNetV1模型、多模态大模型
    """
    ctx = runtime.context

    # 尝试使用真实的FishFreshNetV1模型
    model_result: FreshnessClassificationOutput | None = try_use_fishfreshnet_model(state)

    if model_result:
        logger.info("✅ 使用FishFreshNetV1模型预测成功")
        return model_result

    # 如果模型不可用，回退到多模态大模型
    logger.warning("⚠️ FishFreshNetV1模型不可用，回退到多模态大模型")
    return fallback_to_multimodal_llm(state, ctx, config)


def try_use_fishfreshnet_model(state: FreshnessClassificationInput) -> FreshnessClassificationOutput | None:
    """
    尝试使用FishFreshNetV1模型进行预测
    """
    try:
        # 获取模型客户端
        client: Any = get_client()

        # 检查服务是否可用
        if not client.health_check():
            logger.warning("FishFreshNetV1 API服务不可用")
            return None

        # 获取图像URL
        image_url: str | None = state.processed_image.url if state.processed_image else None

        if not image_url:
            logger.error("未找到有效的图像URL")
            return None

        # 调用模型API
        result: Dict[str, Any] = client.predict_from_url(image_url)

        # 检查是否有错误
        if "error" in result:
            logger.error(f"模型预测失败: {result['error']}")
            return None

        # 计算置信度水平
        confidence_score: float = result["confidence_score"]
        confidence_level: str = get_confidence_level(confidence_score)

        # 构建输出
        return FreshnessClassificationOutput(
            freshness_level=result["freshness_level"],
            confidence_score=confidence_score,
            confidence_level=confidence_level,
            classification_details={
                "model": "FishFreshNetV1",
                "all_probabilities": result["all_probabilities"],
                "description": result["description"],
                "prediction_time": result.get("timestamp", ""),
                "confidence_level_description": f"置信度{confidence_level}（{'≥0.8' if confidence_level == '高' else '≥0.5' if confidence_level == '中' else '<0.5'}）"
            }
        )

    except Exception as e:
        logger.error(f"FishFreshNetV1模型调用失败: {e}")
        return None


def fallback_to_multimodal_llm(
    state: FreshnessClassificationInput,
    ctx: Any,
    config: RunnableConfig
) -> FreshnessClassificationOutput:
    """
    回退到多模态大模型进行预测，从配置文件读取模型参数
    """
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

    # 系统提示词（从配置文件读取）
    system_prompt: str = llm_config.get("sp", """你是水产品新鲜度评估专家，负责根据鱼眼图像判断新鲜度。
请根据以下标准进行评估：

1. 高度新鲜：
   - 鱼眼清澈明亮，角膜透明
   - 瞳孔清晰，颜色正常
   - 鱼眼饱满，无凹陷
   - 表面有光泽，无浑浊

2. 新鲜：
   - 鱼眼基本清澈，角膜略有浑浊
   - 瞳孔可见，颜色略暗
   - 鱼眼轻微凹陷
   - 表面光泽减弱

3. 不新鲜：
   - 鱼眼浑浊，角膜不透明
   - 瞳孔模糊或消失
   - 鱼眼严重凹陷或塌陷
   - 表面无光泽，有明显变质迹象

输出格式：
{
  "freshness_level": "高度新鲜" | "新鲜" | "不新鲜",
  "confidence_score": 0.0-1.0,
  "classification_details": {
    "eye_clarity": "清澈度评估",
    "pupil_condition": "瞳孔状态",
    "surface_quality": "表面质量",
    "overall_appearance": "整体外观"
  }
}

注意：只返回JSON，不要有其他文字。""")

    # 默认值使用有效枚举值"不新鲜"，避免下游节点出现意外行为
    default_level: str = "不新鲜"
    default_score: float = 0.4
    default_level_str: str = "低"
    default_details: Dict[str, Any] = {
        "eye_clarity": "无法确定",
        "pupil_condition": "无法确定",
        "surface_quality": "无法确定",
        "overall_appearance": "无法确定",
        "confidence_level_description": "置信度低（<0.5）",
        "note": "LLM分析失败，使用保守默认值"
    }

    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.2)
    max_tokens: int = model_config.get("max_completion_tokens", 500)
    timeout: float = float(model_config.get("timeout", 60))

    try:
        # 初始化LLM客户端
        client: LLMClient = LLMClient(ctx=ctx)

        # 获取图像URL
        image_url: str = state.processed_image.url if state.processed_image else ""

        if not image_url:
            logger.warning("图像URL为空，返回默认值")
            return FreshnessClassificationOutput(
                freshness_level=default_level,
                confidence_score=default_score,
                confidence_level=default_level_str,
                classification_details=default_details
            )

        # 构建消息
        messages: list = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {
                    "type": "text",
                    "text": "请根据鱼眼图像评估新鲜度等级。"
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url}
                }
            ])
        ]

        # 调用模型
        response = client.invoke(
            messages=messages,
            model=model_id,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=timeout
        )

        # 处理响应内容
        result_text: str = ""
        if isinstance(response.content, str):
            result_text = response.content.strip()
        elif isinstance(response.content, list):
            text_parts: list = []
            for item in response.content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text_parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    text_parts.append(item)
            result_text = " ".join(text_parts).strip()
        else:
            result_text = str(response.content)

        # 尝试解析JSON
        json_start: int = result_text.find("{")
        json_end: int = result_text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str: str = result_text[json_start:json_end]
            result: Dict[str, Any] = json.loads(json_str)

            freshness_level: str = result.get("freshness_level", default_level)
            confidence_score: float = result.get("confidence_score", default_score)
            classification_details: Dict[str, Any] = result.get("classification_details", {})

            # 计算置信度水平
            confidence_level: str = get_confidence_level(confidence_score)

            # 添加置信度水平描述
            classification_details["confidence_level_description"] = f"置信度{confidence_level}（{'≥0.8' if confidence_level == '高' else '≥0.5' if confidence_level == '中' else '<0.5'}）"
            classification_details["note"] = "多模态大模型分析结果"

            return FreshnessClassificationOutput(
                freshness_level=freshness_level,
                confidence_score=confidence_score,
                confidence_level=confidence_level,
                classification_details=classification_details
            )
        else:
            logger.warning("JSON解析失败，返回默认值")
            return FreshnessClassificationOutput(
                freshness_level=default_level,
                confidence_score=default_score,
                confidence_level=default_level_str,
                classification_details=default_details
            )

    except Exception as e:
        logger.error(f"多模态大模型调用失败: {e}")
        return FreshnessClassificationOutput(
            freshness_level=default_level,
            confidence_score=default_score,
            confidence_level=default_level_str,
            classification_details=default_details
        )

