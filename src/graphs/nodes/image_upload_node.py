"""
节点②：图像采集与上传
接收用户上传的图片，记录上传时间
"""
from datetime import datetime
from typing import Any
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import ImageUploadInput, ImageUploadOutput


def image_upload_node(
    state: ImageUploadInput, 
    config: RunnableConfig, 
    runtime: Runtime[Any]
) -> ImageUploadOutput:
    """
    title: 图像采集与上传
    desc: 接收用户上传的鱼眼图片，记录上传时间并传递给下游节点
    """
    # 记录上传时间
    upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 返回上传结果
    return ImageUploadOutput(
        uploaded_image=state.image,
        upload_time=upload_time,
        user_id=state.user_id,
    )


