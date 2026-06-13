"""
节点⑭ 用户反馈收集与闭环
"""
import time
import uuid
from typing import Any, Optional
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import FeedbackLoopInput, FeedbackLoopOutput


def feedback_loop_node(
    state: FeedbackLoopInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> FeedbackLoopOutput:
    """
    title: 用户反馈收集与闭环
    desc: 收集用户反馈，形成再训练样本池，持续优化模型
    integrations: 数据库存储
    """
    # 初始化结果
    record_id: str = state.record_id
    sample_pool_id: Optional[str] = None
    feedback_prompt: str = ""
    
    # 如果没有记录ID，生成一个
    if not record_id:
        record_id = f"fish_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # 根据置信度决定是否加入样本池
    # 低置信度样本优先加入样本池，用于模型优化
    if state.confidence_score < 0.8:
        sample_pool_id = f"pool_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    
    # 生成反馈提示语
    if state.confidence_score >= 0.9:
        feedback_prompt = "检测置信度较高。结果如有偏差，可点击反馈。"
    elif state.confidence_score >= 0.7:
        feedback_prompt = "检测置信度良好。请确认结果是否与实际相符。"
    elif state.confidence_score >= 0.5:
        feedback_prompt = "检测置信度中等。建议提交反馈以帮助优化模型。"
    else:
        feedback_prompt = "检测置信度较低。提交实际判断有助于改进模型准确性。"
    
    # 实际实现应：
    # 1. 将检测结果存入数据库
    # 2. 将低置信度样本加入样本池（用于再训练）
    # 3. 提供反馈入口给用户
    
    return FeedbackLoopOutput(
        record_id=record_id,
        sample_pool_id=sample_pool_id,
        feedback_prompt=feedback_prompt
    )


