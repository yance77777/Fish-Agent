"""
节点⑧ Grad-CAM可视化生成（强化版）

1. 优先调用模型服务的Grad-CAM API
2. 如果模型服务不可用，生成基于区域检测的关注区域标注图
3. 使用大模型生成自然语言解释
4. 从配置文件读取模型参数
"""
import os
import json
import base64
import tempfile
import uuid
import requests
from typing import Dict, Any, List, Optional
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import EnhancedGradCAMInput, EnhancedGradCAMOutput
from graphs.utils import generate_attention_overlay, config_path
from utils.file.file import File


def enhanced_gradcam_node(
    state: EnhancedGradCAMInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> EnhancedGradCAMOutput:
    """
    title: Grad-CAM可视化生成（强化版）
    desc: 生成Grad-CAM热力图（真实或降级方案），并提供自然语言解释和关键关注区域描述
    integrations: 多模态大模型 + Grad-CAM模型服务
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
    heatmap_image: Optional[File] = None
    heatmap_interpretation: str = ""
    key_attention_regions: List[str] = []

    # 获取图片URL
    image_url: str = state.processed_image.url if state.processed_image else ""

    if not image_url:
        return EnhancedGradCAMOutput(
            heatmap_image=None,
            heatmap_interpretation="无法获取图片URL，无法生成热力图解释",
            key_attention_regions=[]
        )

    # 尝试调用真实Grad-CAM模型服务
    gradcam_url: Optional[str] = None
    is_real_gradcam: bool = False

    try:
        # 模型服务地址（从环境变量获取或使用默认值）
        model_service_url: str = os.getenv("MODEL_SERVICE_URL", "http://localhost:8000")

        # 调用正确的Grad-CAM生成接口
        gradcam_endpoint: str = f"{model_service_url}/gradcam_url"

        response = requests.post(
            gradcam_endpoint,
            json={"image_url": image_url},
            timeout=30
        )

        if response.status_code == 200:
            gradcam_result: Dict[str, Any] = response.json()
            heatmap_base64: Optional[str] = gradcam_result.get("heatmap_base64")
            if heatmap_base64:
                is_real_gradcam = True
                output_path = os.path.join(tempfile.gettempdir(), f"fish_gradcam_{uuid.uuid4().hex[:8]}.jpg")
                with open(output_path, "wb") as f:
                    f.write(base64.b64decode(heatmap_base64))
                gradcam_url = output_path

    except Exception:
        is_real_gradcam = False

    # 如果真实Grad-CAM不可用，尝试基于区域检测生成标注图
    if not is_real_gradcam and state.fish_eye_regions:
        attention_url = generate_attention_overlay(
            image_url,
            state.fish_eye_regions,
            state.gill_regions if state.gill_regions else []
        )
        if attention_url:
            gradcam_url = attention_url

    # 设置最终热力图
    if gradcam_url:
        heatmap_image = File(url=gradcam_url, file_type="image")
    else:
        heatmap_image = state.processed_image

    # 构建热力图类型说明
    heatmap_type_note: str = ""
    if is_real_gradcam:
        heatmap_type_note = "（真实Grad-CAM热力图，显示模型实际关注区域）"
    else:
        heatmap_type_note = "（示意热力图，模型服务暂不可用时的降级方案）"

    # 系统提示词（从配置文件读取，若不存在则使用默认值）
    system_prompt: str = llm_config.get("sp", """你是Grad-CAM可视化解释专家，负责分析模型关注的区域并给出自然语言解释。

Grad-CAM热力图显示模型在判断新鲜度时重点关注哪些区域：
- 红色/高亮区域：模型高度关注的区域
- 蓝色/暗淡区域：模型较少关注的区域

请根据图片中的鱼眼特征和新鲜度检测结果，分析模型可能关注的区域并解释原因。

关注区域判断标准：
1. 鱼眼角膜透明度区域：模型判断清澈度的关键区域
2. 瞳孔黑色区域：模型判断瞳孔状态的区域
3. 虹膜边缘区域：模型判断边缘清晰度的区域
4. 鱼眼整体形状：模型判断饱满程度的区域

输出JSON格式：
{
  "heatmap_interpretation": "自然语言解释，说明模型关注哪些区域以及为什么",
  "key_attention_regions": ["区域1描述", "区域2描述", "区域3描述"],
  "attention_analysis": {
    "primary_region": "主要关注区域",
    "secondary_regions": ["次要关注区域1", "次要关注区域2"],
    "explanation": "解释为什么这些区域重要"
  },
  "freshness_correlation": "解释关注区域与新鲜度判断的关联"
}""")

    # 用户提示词
    classification_details_str: str = json.dumps(state.classification_details, ensure_ascii=False) if state.classification_details else "{}"
    user_prompt: str = f"""请分析这张鱼眼图片的Grad-CAM热力图区域：

新鲜度检测结果：{state.freshness_level}
置信度：{state.confidence_score}
分类详细信息：{classification_details_str}

图片URL: {image_url}
热力图类型：{heatmap_type_note}

请分析：
1. 模型在判断新鲜度时重点关注哪些区域？
2. 为什么这些区域是判断新鲜度的关键？
3. 热力图分布说明了什么？
4. 关注区域与新鲜度等级的关联是什么？

请以JSON格式输出解释结果。"""

    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.4)
    max_tokens: int = model_config.get("max_completion_tokens", 1000)

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

            # 提取解释结果
            heatmap_interpretation = str(result.get("heatmap_interpretation", ""))
            key_attention_regions = result.get("key_attention_regions", [])

            # 添加热力图类型说明
            if heatmap_type_note and not is_real_gradcam:
                heatmap_interpretation = f"[示意热力图]{heatmap_interpretation}"

            # 确保key_attention_regions是列表
            if not isinstance(key_attention_regions, list):
                key_attention_regions = []

    except json.JSONDecodeError:
        # JSON解析失败，使用原始文本作为解释
        heatmap_interpretation = f"[示意热力图]{response_str}"
        key_attention_regions = []

    except Exception as e:
        # LLM调用失败，返回基础解释
        heatmap_interpretation = f"热力图生成失败，无法提供区域解释。错误信息：{str(e)}"
        key_attention_regions = []

    return EnhancedGradCAMOutput(
        heatmap_image=heatmap_image,
        heatmap_interpretation=heatmap_interpretation,
        key_attention_regions=key_attention_regions
    )



