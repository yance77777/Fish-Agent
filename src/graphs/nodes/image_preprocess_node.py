"""
节点④：图像预处理（场景自适应）
- 菜市场：处理杂乱背景和强反光
- 超市：处理冷光源、包装膜反射
- 家庭厨房：处理低光照、拍摄角度不稳定

注意：此节点记录预处理策略，实际图像处理由后续节点执行
"""
import os
import io
import uuid
import requests
import tempfile
from datetime import datetime
from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime
from graphs.state import ImagePreprocessInput, ImagePreprocessOutput
from utils.file.file import File


def image_preprocess_node(
    state: ImagePreprocessInput,
    config: RunnableConfig,
    runtime: Runtime[Any]
) -> ImagePreprocessOutput:
    """
    title: 图像预处理（场景自适应）
    desc: 根据场景类型和质量等级自适应调整预处理策略，提高后续分析的准确性
    integrations: 无（预处理策略记录）
    """
    # 记录预处理日志
    preprocess_log: Dict[str, Any] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "scene_type": state.scene_type,
        "quality_level": state.quality_level,
        "operations": [],
        "status": "completed"
    }

    # 根据场景类型调整预处理策略
    operations: list = []

    # 基础预处理（所有场景通用）
    operations.extend(["normalize", "resize"])

    # 场景自适应预处理
    if state.scene_type == "菜市场":
        operations.extend([
            "background_reduction",
            "glare_reduction",
            "contrast_enhancement"
        ])
        preprocess_log["scene_strategy"] = "处理杂乱背景和强反光"

    elif state.scene_type == "超市":
        operations.extend([
            "color_correction",
            "reflection_removal",
            "white_balance"
        ])
        preprocess_log["scene_strategy"] = "处理冷光源和包装膜反射"

    elif state.scene_type == "家庭厨房":
        operations.extend([
            "low_light_enhancement",
            "angle_correction",
            "noise_reduction"
        ])
        preprocess_log["scene_strategy"] = "处理低光照和拍摄角度不稳定"

    else:
        operations.extend([
            "contrast_enhancement",
            "brightness_adjustment"
        ])
        preprocess_log["scene_strategy"] = "标准预处理"

    # 根据质量等级调整预处理强度
    if state.quality_level == "不合格":
        operations.extend(["aggressive_enhancement", "detail_preservation"])
        preprocess_log["quality_strategy"] = "强化预处理以改善质量"
    elif state.quality_level == "合格":
        operations.extend(["moderate_enhancement"])
        preprocess_log["quality_strategy"] = "适度预处理"
    else:
        operations.extend(["light_enhancement"])
        preprocess_log["quality_strategy"] = "轻微预处理以保持原始质量"

    preprocess_log["operations"] = operations

    # 执行实际的图像预处理
    image_url: str = state.uploaded_image.url

    # 默认使用原图作为处理后的图片（异常时或无需处理时）
    processed_file: File = state.uploaded_image

    try:
        if image_url.startswith(("http://", "https://")):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_url).convert("RGB")

        # 根据质量等级执行预处理
        if state.quality_level == "不合格":
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.3)
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(1.5)
            image = image.filter(ImageFilter.MedianFilter(size=3))
        elif state.quality_level == "合格":
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(1.1)
        else:
            pass

        # 标准化尺寸
        image = image.resize((224, 224), Image.Resampling.LANCZOS)

        tmp_dir = tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        local_path = f"{tmp_dir}/fish_processed_{uuid.uuid4().hex[:8]}.jpg"
        image.save(local_path, format="JPEG", quality=95)

        processed_file = File(url=local_path, file_type="image")
        preprocess_log["actual_processing"] = f"已保存预处理后图片到: {local_path}"
        preprocess_log["processed_size"] = image.size

    except Exception as e:
        preprocess_log["preprocessing_error"] = f"预处理执行失败: {str(e)}"
        preprocess_log["actual_processing"] = "预处理失败，使用原图"

    return ImagePreprocessOutput(
        processed_image=processed_file,
        preprocessing_log=preprocess_log
    )



