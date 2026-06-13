"""
节点② 增强版图像质量检测
1. 判断是否为鱼类图片
2. 判断是否包含鱼眼
3. 图像质量检测（模糊、过曝、低光照等）

"""
import json
from typing import Dict, Any, List
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import EnhancedQualityCheckInput, EnhancedQualityCheckOutput
from graphs.utils import config_path
from utils.file.file import File


def enhanced_quality_check_node(
    state: EnhancedQualityCheckInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> EnhancedQualityCheckOutput:
    """
    title: 增强版图像质量检测
    desc: 检测是否为鱼类图片、是否包含鱼眼、图像质量等级（优秀/良好/合格/不合格）
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
    
    # 初始化检测结果（保守默认值）
    is_fish_image: bool = False
    fish_type: str = ""
    contains_fish_eye: bool = False
    is_qualified: bool = False
    quality_level: str = "不合格"
    quality_score: float = 0.0
    quality_issues: List[str] = []
    scene_type: str = ""
    
    # 获取图片URL
    image_url: str = state.uploaded_image.url if state.uploaded_image else ""
    
    if not image_url:
        quality_issues = ["无法获取图片URL"]
        return EnhancedQualityCheckOutput(
            is_fish_image=False,
            fish_type="",
            contains_fish_eye=False,
            is_qualified=False,
            quality_level="不合格",
            quality_score=0.0,
            quality_issues=quality_issues,
            scene_type=""
        )
    
    # 系统提示词（质量检测）
    system_prompt: str = llm_config.get("sp", """你是图像质量检测专家，负责判断上传图片是否适合用于鱼类新鲜度评估。

请仔细分析图片并判断以下内容：
1. 是否为鱼类图片（鱼的整体或局部）
2. 是否包含清晰可见的鱼眼区域
3. 图像质量评估（清晰度、亮度、对比度、角度）
4. 场景类型判断（菜市场、超市、家庭厨房、其他）

判断标准：
- 鱼类图片：必须能看到鱼的特征（鳞片、鱼眼、鱼身等）
- 鱼眼可见：鱼眼区域清晰可见，不是侧面或被遮挡
- 质量优秀：清晰度高，光线适中，角度良好
- 质量良好：基本清晰，轻微模糊或光线问题
- 质量合格：可识别但有一定质量问题
- 质量不合格：模糊严重、过曝/欠曝、无法识别

输出JSON格式：
{
  "is_fish_image": true/false,
  "fish_type": "鱼种（如鲤鱼、鲫鱼等，不确定则填'未知鱼种'）",
  "contains_fish_eye": true/false,
  "quality_level": "优秀/良好/合格/不合格",
  "quality_score": 0-100的评分,
  "quality_issues": ["问题1", "问题2"],
  "scene_type": "菜市场/超市/家庭厨房/其他"
}""")
    
    # 用户提示词
    user_prompt_template: str = llm_config.get("up", """请分析这张图片，判断是否适合用于鱼类新鲜度评估：

图片URL: {{image_url}}

请详细分析：
1. 图片中的内容是否为鱼类？
2. 是否包含清晰可见的鱼眼？
3. 图片质量如何（清晰度、光线、角度）？
4. 拍摄场景是什么？

请以JSON格式输出检测结果。""")
    
    # 使用jinja2模板渲染提示词
    up_tpl: Template = Template(user_prompt_template)
    user_prompt: str = up_tpl.render({"image_url": image_url})
    
    try:
        # 使用LLMClient进行真实多模态调用
        client: LLMClient = LLMClient(ctx=ctx)
        
        # 构建多模态消息（图片 + 文本）
        messages: List[Any] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=[
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}}
            ])
        ]
        
        # 获取模型配置
        model_config: Dict[str, Any] = llm_config.get("config", {})
        model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
        temperature: float = model_config.get("temperature", 0.1)
        max_tokens: int = model_config.get("max_completion_tokens", 800)
        
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
            
            # 布尔值解析函数（正确处理字符串 "false"/"否" 等）
            def parse_bool(value: Any) -> bool:
                """安全解析布尔值，处理字符串类型"""
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    # 处理各种字符串表示
                    lower_value = value.strip().lower()
                    if lower_value in ("true", "是", "yes", "1", "有", "包含"):
                        return True
                    elif lower_value in ("false", "否", "no", "0", "无", "不", "不包含"):
                        return False
                    # 无法判断时默认为 False
                    return False
                if isinstance(value, (int, float)):
                    return bool(value)
                return False
            
            # 提取检测结果（使用安全布尔解析）
            is_fish_image = parse_bool(result.get("is_fish_image", False))
            fish_type = str(result.get("fish_type", ""))
            contains_fish_eye = parse_bool(result.get("contains_fish_eye", False))
            quality_level = str(result.get("quality_level", "不合格"))
            quality_score = float(result.get("quality_score", 0.0))
            quality_issues = result.get("quality_issues", [])
            scene_type = str(result.get("scene_type", ""))
            
            # 确保quality_issues是列表
            if not isinstance(quality_issues, list):
                quality_issues = []
            
            # 判断是否合格（三重检查）
            is_qualified = is_fish_image and contains_fish_eye and quality_level in ["优秀", "良好", "合格"]
            
            if not is_qualified:
                if not is_fish_image:
                    quality_issues.append("图片内容不是鱼类")
                if not contains_fish_eye:
                    quality_issues.append("图片中未找到清晰的鱼眼区域")
                if quality_level == "不合格":
                    quality_issues.append("图片质量不合格")
    
    except json.JSONDecodeError:
        # JSON解析失败，尝试从文本中提取关键信息
        is_fish_image = "是鱼类" in response_str or "鱼类图片" in response_str
        contains_fish_eye = "有鱼眼" in response_str or "包含鱼眼" in response_str
        is_qualified = is_fish_image and contains_fish_eye
        quality_issues = ["LLM响应JSON解析失败，基于文本判断"]
        
    except Exception as e:
        # LLM调用失败，返回不合格结果
        is_fish_image = False
        contains_fish_eye = False
        is_qualified = False
        quality_level = "不合格"
        quality_score = 0.0
        quality_issues = [f"多模态大模型调用失败: {str(e)}"]
    
    return EnhancedQualityCheckOutput(
        is_fish_image=is_fish_image,
        fish_type=fish_type,
        contains_fish_eye=contains_fish_eye,
        is_qualified=is_qualified,
        quality_level=quality_level,
        quality_score=quality_score,
        quality_issues=quality_issues,
        scene_type=scene_type
    )


