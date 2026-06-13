"""
工作流公共工具函数

此模块包含工作流中多个节点共用的工具函数，避免代码重复。
"""
import os
import io
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from PIL import Image, ImageDraw, ImageFont
from utils.file.file import File


def project_root() -> Path:
    env_path = os.getenv("FISH_AGENT_WORKSPACE")
    if env_path:
        return Path(env_path).resolve()
    return Path(__file__).resolve().parents[2]


def config_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return project_root() / path


def parse_bbox(bbox: Any, image_size: tuple[int, int] | None = None) -> tuple[int, int, int, int]:
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        values = [float(item) for item in bbox[:4]]
    elif isinstance(bbox, dict):
        values = [
            float(bbox.get("x1", 0)),
            float(bbox.get("y1", 0)),
            float(bbox.get("x2", 0)),
            float(bbox.get("y2", 0)),
        ]
    else:
        return 0, 0, 0, 0

    if image_size and max(values) <= 100:
        width, height = image_size
        values = [
            values[0] / 100 * width,
            values[1] / 100 * height,
            values[2] / 100 * width,
            values[3] / 100 * height,
        ]
    if image_size:
        width, height = image_size
        values = [
            max(0, min(values[0], width)),
            max(0, min(values[1], height)),
            max(0, min(values[2], width)),
            max(0, min(values[3], height)),
        ]
    return int(values[0]), int(values[1]), int(values[2]), int(values[3])


def get_confidence_level(confidence_score: float) -> str:
    """
    根据置信度分数计算置信度水平
    
    置信度水平："高"(>=0.8) / "中"(0.5-0.8) / "低"(<0.5)
    """
    if confidence_score >= 0.8:
        return "高"
    elif confidence_score >= 0.5:
        return "中"
    else:
        return "低"


def safe_bool_parse(value: Any) -> bool:
    """
    安全解析布尔值，正确处理字符串 "false"/"否" 等情况
    
    Handles booleans returned as strings by LLMs.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower_value = value.lower().strip()
        false_values = ["false", "否", "no", "n", "0", "不", "无", "非"]
        if lower_value in false_values:
            return False
        true_values = ["true", "是", "yes", "y", "1", "有", "含"]
        if lower_value in true_values:
            return True
        return False
    if isinstance(value, (int, float)):
        return value > 0
    return False


def generate_attention_overlay(
    image_url: str,
    fish_eye_regions: List[Dict[str, Any]],
    gill_regions: List[Dict[str, Any]]
) -> Optional[str]:
    """
    基于区域检测结果生成标注图（Grad-CAM降级方案）
    使用纯 PIL 实现，无需 OpenCV/NumPy 重型依赖
    
    Returns a local image path, or None if generation fails.
    """
    try:
        if image_url.startswith(("http://", "https://")):
            response = requests.get(image_url, timeout=10)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content)).convert("RGB")
        else:
            image = Image.open(image_url).convert("RGB")
        draw = ImageDraw.Draw(image)

        try:
            import platform
            if platform.system() == "Windows":
                font_path = "C:/Windows/Fonts/arial.ttf"
            else:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font = ImageFont.truetype(font_path, 12)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for region in fish_eye_regions:
            x1, y1, x2, y2 = parse_bbox(region.get("bbox", {}), image.size)
            confidence = region.get("confidence", 0.5)

            intensity = int(255 * confidence)
            color = (intensity, 0, 0)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 15)), f"Eye:{confidence:.2f}", fill=color, font=font)

        for region in gill_regions:
            x1, y1, x2, y2 = parse_bbox(region.get("bbox", {}), image.size)
            confidence = region.get("confidence", 0.5)

            intensity = int(255 * confidence)
            color = (0, 0, intensity)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            draw.text((x1, max(0, y1 - 15)), f"Gill:{confidence:.2f}", fill=color, font=font)

        output_path = str(Path(os.getenv("TMPDIR") or os.getenv("TEMP") or "/tmp") / f"attention_overlay_{uuid.uuid4().hex[:8]}.jpg")
        image.save(output_path, quality=95)

        return output_path

    except Exception:
        return None
