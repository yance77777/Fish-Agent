"""
节点⑤ 鱼体区域识别（多区域）
使用多模态大模型识别鱼眼、鳃部、体表等区域。
"""
import json
from typing import Dict, Any, List
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import FishRegionDetectionInput, FishRegionDetectionOutput
from graphs.utils import config_path


def _sanitize_regions(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    valid_regions: List[Dict[str, Any]] = []
    for region in value:
        if isinstance(region, dict):
            valid_regions.append({
                "bbox": region.get("bbox", [0.1, 0.1, 0.9, 0.9]),
                "confidence": float(region.get("confidence", 0.5)),
                "label": str(region.get("label", "unknown")),
                "description": str(region.get("description", ""))
            })
    return valid_regions


def fish_region_detection_node(
    state: FishRegionDetectionInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> FishRegionDetectionOutput:
    """
    title: 鱼体区域识别
    desc: 使用多模态大模型识别鱼眼、鳃部、体表等多个区域，为多区域融合分析提供真实数据
    integrations: 多模态大模型
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

    # 初始化检测结果
    fish_eye_regions: List[Dict[str, Any]] = []
    gill_regions: List[Dict[str, Any]] = []
    body_regions: List[Dict[str, Any]] = []
    region_detection_confidence: float = 0.0
    response_str: str = ""

    # 获取图片URL
    image_url: str = state.processed_image.url if state.processed_image else ""

    if not image_url:
        return FishRegionDetectionOutput(
            fish_eye_regions=[],
            gill_regions=[],
            body_regions=[],
            region_detection_confidence=0.0
        )

    # 系统提示词（从配置文件读取，若不存在则使用默认值）
    system_prompt: str = llm_config.get("sp", """你是鱼体区域检测专家，负责从鱼类图片中识别关键区域。

请识别以下区域并提供位置信息：
1. 鱼眼区域（fish_eye）：角膜、瞳孔、虹膜等特征区域
2. 鳃部区域（gill）：鳃盖、鳃丝等区域
3. 体表区域（body）：鱼鳞、皮肤、整体形态

对于每个检测到的区域，请输出：
- bbox: 大致位置 [x1, y1, x2, y2]，使用相对坐标（0-100表示百分比位置）
- confidence: 检测置信度（0.0-1.0）
- label: 区域标签
- description: 区域描述

输出JSON格式：
{
  "fish_eye_regions": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.95, "label": "fish_eye", "description": "清晰的鱼眼区域"}
  ],
  "gill_regions": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.85, "label": "gill", "description": "鳃盖区域"}
  ],
  "body_regions": [
    {"bbox": [x1, y1, x2, y2], "confidence": 0.90, "label": "body", "description": "鱼体表区域"}
  ],
  "region_detection_confidence": 整体检测置信度,
  "detection_summary": "检测总结说明"
}

注意：
- 如果某个区域在图片中不清晰或不可见，confidence应较低
- bbox使用相对坐标，便于后续处理
- 必须输出合法JSON格式""")

    user_prompt: str = f"""请分析这张鱼类图片，识别鱼眼、鳃部和体表区域：

图片URL: {image_url}

请检测以下区域并提供详细的位置和置信度信息：
1. 鱼眼区域（最重要，用于新鲜度评估）
2. 鳃部区域（次要，辅助新鲜度判断）
3. 体表区域（辅助整体形态分析）

请以JSON格式输出检测结果。"""

    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.3)
    max_tokens: int = model_config.get("max_completion_tokens", 1000)
    timeout: float = float(model_config.get("timeout", 60))

    try:
        # 使用LLMClient进行多模态调用
        client: LLMClient = LLMClient(ctx=ctx)

        # 构建多模态消息（图片 + 文本）
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])
        ]

        # 调用多模态大模型
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

            # 提取检测结果
            fish_eye_regions = _sanitize_regions(result.get("fish_eye_regions", []))
            gill_regions = _sanitize_regions(result.get("gill_regions", []))
            body_regions = _sanitize_regions(result.get("body_regions", []))
            region_detection_confidence = float(result.get("region_detection_confidence", 0.0))

    except json.JSONDecodeError:
        # JSON解析失败，基于文本判断是否有关键区域
        fish_eye_regions = []
        gill_regions = []
        body_regions = []
        region_detection_confidence = 0.0

        if "鱼眼" in response_str or "eye" in response_str.lower():
            fish_eye_regions = [{"bbox": [30, 30, 70, 70], "confidence": 0.6, "label": "fish_eye", "description": "基于文本判断"}]
        if "鳃" in response_str or "gill" in response_str.lower():
            gill_regions = [{"bbox": [20, 40, 60, 80], "confidence": 0.5, "label": "gill", "description": "基于文本判断"}]
        if "体表" in response_str or "body" in response_str.lower() or "鱼" in response_str:
            body_regions = [{"bbox": [0, 0, 100, 100], "confidence": 0.7, "label": "body", "description": "整体鱼体区域"}]

    except Exception as e:
        # LLM调用失败，返回空结果
        fish_eye_regions = []
        gill_regions = []
        body_regions = []
        region_detection_confidence = 0.0

    return FishRegionDetectionOutput(
        fish_eye_regions=fish_eye_regions,
        gill_regions=gill_regions,
        body_regions=body_regions,
        region_detection_confidence=region_detection_confidence
    )
