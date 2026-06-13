"""
节点⑪ 报告生成（结构化+知识图谱）
1. 输出结构化的专业报告
2. 知识图谱匹配（鱼种 -> 关键指标 -> 风险等级 -> 推荐处理方式）
使用大模型动态生成知识图谱匹配结果
"""
import json
from typing import Dict, Any, Optional, List
from jinja2 import Template
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.runtime import Runtime
from tools.llm_client import LLMClient
from graphs.state import ReportGenerationInput, ReportGenerationOutput
from graphs.utils import config_path


def report_generation_node(
    state: ReportGenerationInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> ReportGenerationOutput:
    """
    title: 报告生成（结构化+知识图谱）
    desc: 生成结构化新鲜度评估报告，使用大模型匹配知识图谱提供专业建议
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
    
    # 初始化结果
    freshness_report: Dict[str, Any] = {}
    knowledge_graph_match: Dict[str, Any] = {}
    
    # 根据新鲜度等级和置信度生成结构化报告
    confidence_percent: str = f"{state.confidence_score * 100:.1f}%"
    
    # 构建报告摘要
    overall_assessment: str
    if state.confidence_level == "高":
        overall_assessment = "评估结果高度可信，鱼类新鲜度状态明确"
    elif state.confidence_level == "中":
        overall_assessment = "评估结果可信度中等，建议结合实际情况判断"
    else:
        overall_assessment = "评估结果可信度较低，建议谨慎参考或重新检测"
    
    # 构建结构化报告
    freshness_report = {
        "summary": {
            "grade": state.freshness_level,
            "confidence": confidence_percent,
            "confidence_level": state.confidence_level,
            "overall_assessment": overall_assessment,
            "fish_type": state.fish_type or "未知鱼种"
        },
        "detailed_analysis": {
            "eye_appearance": state.structured_analysis.get("eye_appearance", "鱼眼外观正常"),
            "clarity_assessment": state.structured_analysis.get("clarity_assessment", "清澈度良好"),
            "color_evaluation": state.structured_analysis.get("color_evaluation", "颜色正常"),
            "texture_analysis": state.structured_analysis.get("texture_analysis", "质地正常")
        },
        "quality_indicators": state.structured_analysis.get("quality_indicators", {
            "positive_signs": [],
            "warning_signs": []
        }),
        "heatmap_interpretation": state.heatmap_interpretation,
        "temporal_analysis": {
            "freshness_trend": state.freshness_trend if state.freshness_trend else "无历史数据",
            "predicted_remaining_hours": state.predicted_remaining_hours if state.predicted_remaining_hours else 0,
            "trend_analysis": state.trend_analysis if state.trend_analysis else {}
        },
        "confidence_explanation": f"本次评估置信度为{confidence_percent}，属于{state.confidence_level}置信度区间。",
        "disclaimer": "本报告基于鱼眼视觉特征分析生成，仅供参考。"
    }

    model_config: Dict[str, Any] = llm_config.get("config", {})
    model_id: str = model_config.get("model", "doubao-seed-1-8-251228")
    temperature: float = model_config.get("temperature", 0.4)
    max_tokens: int = model_config.get("max_completion_tokens", 800)
    timeout: float = float(model_config.get("timeout", 60))

    report_system_prompt: str = llm_config.get("sp", "")
    report_user_template: str = llm_config.get("up", "")
    if report_system_prompt and report_user_template:
        try:
            report_user_prompt = Template(report_user_template).render(
                structured_analysis=json.dumps(state.structured_analysis, ensure_ascii=False),
                freshness_level=state.freshness_level,
                confidence_score=f"{state.confidence_score:.2f}",
                freshness_trend=state.freshness_trend or "无历史数据",
                predicted_remaining_hours=state.predicted_remaining_hours or 0,
                heatmap_interpretation=state.heatmap_interpretation,
                fish_type=state.fish_type or "未知鱼种"
            )
            client = LLMClient(ctx=ctx)
            response = client.invoke(
                messages=[
                    SystemMessage(content=report_system_prompt),
                    HumanMessage(content=report_user_prompt)
                ],
                model=model_id,
                temperature=temperature,
                max_completion_tokens=max_tokens,
                timeout=timeout
            )
            response_str = response.content if isinstance(response.content, str) else str(response.content)
            json_start = response_str.find("{")
            json_end = response_str.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                parsed_report = json.loads(response_str[json_start:json_end])
                if isinstance(parsed_report, dict):
                    freshness_report = parsed_report.get("freshness_report", parsed_report)
        except Exception:
            # 配置化报告生成失败时，保留上面的确定性报告。
            pass
    
    # 使用大模型动态生成知识图谱匹配结果
    fish_type: str = state.fish_type or "未知鱼种"
    
    # 构建知识图谱查询提示词
    kg_system_prompt: str = """你是水产品知识图谱专家。
根据鱼种和新鲜度状态，从专业知识库中匹配相应的处理建议。

知识图谱结构：
鱼种 -> 关键新鲜度指标 -> 风险等级 -> 推荐处理方式 -> 推荐烹饪方式

请输出JSON格式的知识图谱匹配结果：
{
  "fish_type": "鱼种名称",
  "key_indicators": ["关键指标1", "关键指标2", "关键指标3"],
  "risk_level": "低|中|高",
  "recommended_handling": "推荐处理方式",
  "recommended_cooking": ["推荐烹饪方式1", "推荐烹饪方式2"],
  "expert_tips": "专家建议",
  "storage_conditions": "储存条件建议",
  "shelf_life": "保质期建议"
}"""
    
    kg_user_prompt: str = f"""请为以下情况匹配知识图谱信息：

鱼种：{fish_type}
新鲜度等级：{state.freshness_level}
置信度：{confidence_percent}

请根据专业知识给出对应的处理建议和烹饪推荐。"""
    
    try:
        # 使用LLMClient进行知识图谱匹配
        client: LLMClient = LLMClient(ctx=ctx)
        
        # 构建消息列表
        messages: List[Any] = [
            SystemMessage(content=kg_system_prompt),
            HumanMessage(content=kg_user_prompt)
        ]
        
        # 调用大模型进行知识图谱匹配
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
            knowledge_graph_match = json.loads(json_str)
            
    except json.JSONDecodeError:
        # JSON解析失败，使用基于新鲜度等级的默认匹配
        if state.freshness_level == "高度新鲜":
            knowledge_graph_match = {
                "fish_type": fish_type,
                "key_indicators": ["鱼眼透明度高", "瞳孔深黑", "角膜清澈"],
                "risk_level": "低",
                "recommended_handling": "24小时内冷藏食用",
                "recommended_cooking": ["刺身", "清蒸"],
                "expert_tips": "新鲜度极佳"
            }
        elif state.freshness_level == "新鲜":
            knowledge_graph_match = {
                "fish_type": fish_type,
                "key_indicators": ["鱼眼透明度良好", "瞳孔颜色正常"],
                "risk_level": "中",
                "recommended_handling": "当天食用",
                "recommended_cooking": ["清蒸", "红烧"],
                "expert_tips": "新鲜度良好"
            }
        else:
            knowledge_graph_match = {
                "fish_type": fish_type,
                "key_indicators": ["鱼眼浑浊", "瞳孔灰白"],
                "risk_level": "高",
                "recommended_handling": "谨慎食用",
                "recommended_cooking": ["红烧", "油炸"],
                "expert_tips": "新鲜度较低"
            }
            
    except Exception:
        # LLM调用失败，使用默认匹配
        knowledge_graph_match = {
            "fish_type": fish_type,
            "key_indicators": ["基于新鲜度等级的判断"],
            "risk_level": "中",
            "recommended_handling": "建议根据实际情况处理",
            "recommended_cooking": ["常规烹饪"],
            "expert_tips": "请结合实际情况判断"
        }
    
    return ReportGenerationOutput(
        freshness_report=freshness_report,
        knowledge_graph_match=knowledge_graph_match
    )
