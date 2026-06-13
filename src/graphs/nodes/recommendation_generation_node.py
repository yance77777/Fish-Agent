"""
节点⑫ 建议生成（知识图谱辅助）
1. 使用大模型生成差异化专业建议
2. 结合知识图谱和时序分析结果
"""
import json
from typing import Dict, Any, List, Optional
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import RecommendationGenerationInput, RecommendationGenerationOutput
from graphs.utils import config_path


def recommendation_generation_node(
    state: RecommendationGenerationInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> RecommendationGenerationOutput:
    """
    title: 建议生成（知识图谱辅助）
    desc: 大模型基于知识图谱和时序分析生成存储、食用、处理建议
    integrations: 大语言模型
    """
    ctx = runtime.context
    
    # 从配置文件读取提示词
    cfg_path: str = config.get("metadata", {}).get("llm_cfg", "")
    llm_config: Dict[str, Any] = {}
    
    if cfg_path:
        full_path = config_path(cfg_path)
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                llm_config = json.load(f)
        except Exception:
            llm_config = {}
    
    # 初始化建议
    recommendations: Dict[str, Any] = {}
    
    # 获取基本信息
    kg: Dict[str, Any] = state.knowledge_graph_match
    fish_type: str = state.fish_type or kg.get("fish_type", "未知鱼种")
    
    # 从报告中获取置信度
    confidence_score: float = 0.0
    if state.freshness_report and "summary" in state.freshness_report:
        confidence_str: str = state.freshness_report.get("summary", {}).get("confidence", "0%")
        try:
            confidence_score = float(confidence_str.replace("%", "")) / 100.0
        except (ValueError, AttributeError):
            confidence_score = 0.85  # 默认置信度
    
    # 构建系统提示词
    system_prompt: str = llm_config.get("sp", """你是水产品处理与食品安全专家，拥有丰富的烹饪知识。
根据鱼眼新鲜度评估结果和鱼种信息，生成专业、实用、差异化的建议。

请输出以下结构化建议（JSON格式）：

{
  "storage": {
    "temperature": "存储温度建议",
    "method": "存储方法详细说明",
    "duration": "保质时间（考虑新鲜度等级）",
    "tips": ["实用存储技巧1", "技巧2"]
  },
  "consumption": {
    "timeline": "最佳食用时间",
    "cooking_methods": ["推荐烹饪方式1", "方式2", "方式3"],
    "pairing_suggestions": ["搭配建议1", "建议2"],
    "flavor_tips": ["烹饪技巧1", "技巧2"]
  },
  "handling": {
    "preparation_steps": ["处理步骤1", "步骤2", "步骤3"],
    "cleaning_tips": ["清洗技巧1", "技巧2"],
    "cutting_advice": "切割方式建议",
    "deodorizing": ["去腥技巧1", "技巧2"]
  },
  "safety_warnings": {
    "risk_level": "高/中/低",
    "warnings": ["安全警告1", "警告2"],
    "signs_to_watch": ["需要关注的异常迹象1", "迹象2"]
  },
  "best_practices": ["最佳实践建议1", "建议2", "建议3"]
}

注意：
- 根据新鲜度等级调整建议内容
- 高度新鲜：推荐刺身、清蒸等简单烹饪
- 新鲜：推荐当天食用，避免长时间储存
- 不新鲜：建议充分加热，谨慎食用
- 根据鱼种调整烹饪推荐（如海鱼适合清蒸，淡水鱼适合红烧）""")
    
    # 构建用户提示词（使用Jinja2渲染模板）
    predicted_hours: str = ""
    if state.predicted_remaining_hours:
        predicted_hours = f"预计剩余保质时间：{state.predicted_remaining_hours}小时"
    
    up_template: str = llm_config.get("up", """请根据以下信息生成专业建议：

鱼种：{{fish_type}}
新鲜度等级：{{freshness_level}}
置信度：{{confidence_score}}
{{predicted_hours}}

知识图谱匹配：
- 典型新鲜特征：{{typical_fresh_signs}}
- 变质预警信号：{{warning_signs}}
- 推荐烹饪方式：{{recommended_cooking}}
- 风险等级：{{risk_level}}

请输出JSON格式的完整建议。""")
    
    # 使用Jinja2渲染模板
    up_tpl: Template = Template(up_template)
    user_prompt: str = up_tpl.render(
        fish_type=fish_type,
        freshness_level=state.freshness_level,
        confidence_score=f"{confidence_score:.2f}",
        predicted_hours=predicted_hours,
        typical_fresh_signs=kg.get('typical_fresh_signs', []),
        warning_signs=kg.get('warning_signs', []),
        recommended_cooking=kg.get('recommended_cooking', []),
        risk_level=kg.get('risk_level', '中'),
        freshness_report=json.dumps(state.freshness_report, ensure_ascii=False, indent=2)
    )
    
    # 获取模型配置
    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.5)
    max_tokens: int = model_config.get("max_completion_tokens", 2000)
    timeout: float = float(model_config.get("timeout", 60))
    
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
            recommendations = json.loads(json_str)
            
    except json.JSONDecodeError:
        # JSON解析失败，使用默认建议
        recommendations = _get_default_recommendations(
            state.freshness_level, 
            fish_type, 
            state.predicted_remaining_hours
        )
            
    except Exception:
        # LLM调用失败，使用默认建议
        recommendations = _get_default_recommendations(
            state.freshness_level, 
            fish_type, 
            state.predicted_remaining_hours
        )
    
    return RecommendationGenerationOutput(
        recommendations=recommendations
    )


def _get_default_recommendations(
    freshness_level: str, 
    fish_type: str, 
    predicted_hours: Optional[float]
) -> Dict[str, Any]:
    """生成默认建议（备用方案）"""
    # 存储建议
    storage: Dict[str, Any] = {}
    consumption: Dict[str, Any] = {}
    handling: Dict[str, Any] = {}
    safety: Dict[str, Any] = {}
    best: List[str] = []
    
    if freshness_level == "高度新鲜":
        storage = {
            "temperature": "短期冷藏0-4℃，长期冷冻-18℃以下",
            "method": "保鲜膜紧密包裹，避免接触空气",
            "duration": f"冷藏3天，冷冻3-6个月" + (f"，预计剩余{predicted_hours}小时" if predicted_hours else ""),
            "tips": ["冷藏置于冰箱内层低温区", "冷冻前涂抹食用油减少水分流失", "解冻时冷藏缓慢解冻"]
        }
        consumption = {
            "timeline": "最佳食用期3天内",
            "cooking_methods": ["刺身", "清蒸", "清炖"],
            "pairing_suggestions": ["日式山葵酱油", "姜丝葱段蒸鱼豉油", "豆腐菌菇汤底"],
            "flavor_tips": ["少用重口味调料", "刺身置于冰盘", "清蒸火候适中"]
        }
        handling = {
            "preparation_steps": ["流动水冲洗", "逆向刮鳞", "剪断鳃根取出鱼鳃"],
            "cleaning_tips": ["刮除鱼腹黑膜", "吸干表面水分", "工具及时清洗"],
            "cutting_advice": "刺身顺着纹理切片，厚度2-3毫米",
            "deodorizing": ["鲜度高几乎无腥味", "仅去除鱼鳃黑膜", "料酒姜片腌制10分钟"]
        }
        safety = {
            "risk_level": "低",
            "warnings": ["新鲜度良好可正常处理食用", "储存时做好分类标记"],
            "signs_to_watch": ["鱼眼浑浊发白", "表面失去光泽", "腥臭味或酸败气", "肉质按压不回弹"]
        }
        best = ["优先选择刺身清蒸等简单烹饪", "储存做好分类标记记录时间", "处理前洗净双手生熟分开"]
        
    elif freshness_level == "新鲜":
        storage = {
            "temperature": "冷藏0-4℃",
            "method": "保鲜膜包裹避免接触空气",
            "duration": f"建议当天食用，冷藏不超过2天" + (f"，预计剩余{predicted_hours}小时" if predicted_hours else ""),
            "tips": ["尽快食用避免长时间储存", "与其他食材分开", "如需冷冻尽快处理"]
        }
        consumption = {
            "timeline": "建议当天食用",
            "cooking_methods": ["清蒸", "红烧", "煎制"],
            "pairing_suggestions": ["蒸鱼豉油", "酱油糖红烧", "柠檬汁煎制"],
            "flavor_tips": ["适度调味", "避免过度烹饪", "搭配蔬菜烹饪"]
        }
        handling = {
            "preparation_steps": ["流动水冲洗", "逆向刮鳞", "剪断鳃根取出鱼鳃"],
            "cleaning_tips": ["刮除鱼腹黑膜", "吸干表面水分", "工具及时清洗"],
            "cutting_advice": "分割成适合烹饪的小块",
            "deodorizing": ["去除异味来源", "料酒腌制去腥", "烹饪加入姜片葱段"]
        }
        safety = {
            "risk_level": "中",
            "warnings": ["建议当天食用避免长时间储存", "冷冻超过6个月口感下降"],
            "signs_to_watch": ["鱼眼浑浊发白", "表面失去光泽", "腥臭味或酸败气", "肉质按压不回弹"]
        }
        best = ["充分加热烹饪确保食品安全", "不建议生食或半熟食用", "如有异常立即停止食用"]
        
    else:  # 不新鲜
        storage = {
            "temperature": "不建议长期储存",
            "method": "如需储存充分加热后冷藏",
            "duration": "建议立即处理不建议储存",
            "tips": ["新鲜度较低尽快烹饪食用", "充分加热降低风险", "不建议冷冻储存"]
        }
        consumption = {
            "timeline": "建议充分加热后立即食用",
            "cooking_methods": ["红烧", "油炸", "炖汤"],
            "pairing_suggestions": ["充分加热至全熟", "姜片去腥", "炖汤时间延长"],
            "flavor_tips": ["必须充分加热", "不建议生食或半熟", "烹饪时间适当延长"]
        }
        handling = {
            "preparation_steps": ["流动水冲洗", "逆向刮鳞", "剪断鳃根取出鱼鳃"],
            "cleaning_tips": ["彻底刮除黑膜", "充分清洗", "工具彻底消毒"],
            "cutting_advice": "分割小块便于充分加热",
            "deodorizing": ["处理去除异味来源", "料酒腌制去腥", "烹饪加入姜片葱段"]
        }
        safety = {
            "risk_level": "高",
            "warnings": ["新鲜度较低请谨慎食用", "必须充分加热烹饪", "如有异味立即停止食用"],
            "signs_to_watch": ["鱼眼浑浊发白", "表面失去光泽", "腥臭味或酸败气", "肉质按压不回弹"]
        }
        best = ["充分加热烹饪确保食品安全", "不建议生食或半熟食用", "如有异常立即停止食用"]
    
    return {
        "storage": storage,
        "consumption": consumption,
        "handling": handling,
        "safety_warnings": safety,
        "best_practices": best
    }

