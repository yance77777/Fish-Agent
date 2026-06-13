"""
Fish-Agent V3.4.0 local desktop application.

This installer-facing app keeps FishFreshNetV1 inference fully local and adds
guardrails for image validation, OpenMP startup stability, Grad-CAM explainable
visualization, structured reports, and PDF / JSON export.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# These flags must be set before importing torch / torchvision.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageStat, ImageTk
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


APP_VERSION = "3.4.0"
APP_TITLE = f"Fish-Agent V{APP_VERSION} 水产品新鲜度智能评估系统"
GITHUB_URL = "https://github.com/yance77777/Fish-Agent"
SUPPORT_URL = f"{GITHUB_URL}/issues"

SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_FILE_TYPES = [
    ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]
MAX_IMAGE_SIZE_MB = 25
MIN_IMAGE_SIZE_KB = 1

LABELS = ["Highly Fresh", "Fresh", "Not Fresh"]
DISPLAY_LABELS = {
    "Highly Fresh": "高度新鲜",
    "Fresh": "新鲜",
    "Not Fresh": "不新鲜",
}
LABEL_COLORS = {
    "Highly Fresh": "#0F9F6E",
    "Fresh": "#2368D8",
    "Not Fresh": "#C2410C",
}
LABEL_DESCRIPTIONS = {
    "Highly Fresh": "鱼眼清澈明亮，角膜透明，瞳孔边界相对清晰，整体品质处于理想状态。",
    "Fresh": "鱼眼仍具可辨识结构，但清澈度或光泽可能略有下降，建议尽快完成储存或烹饪。",
    "Not Fresh": "鱼眼浑浊、光泽弱或结构模糊的风险较高，应谨慎处理并优先关注食品安全。",
}

COLORS = {
    "bg": "#F4F7FA",
    "panel": "#FFFFFF",
    "line": "#D8E0EA",
    "text": "#17202A",
    "muted": "#5B677A",
    "primary": "#1565C0",
    "primary_dark": "#0B4EA2",
    "accent": "#0F766E",
    "warning": "#B45309",
    "danger": "#B91C1C",
    "canvas": "#E9EEF5",
}


def configure_logging() -> logging.Logger:
    log_dir = Path(tempfile.gettempdir()) / "Fish-Agent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Fish-Agent")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(
            log_dir / f"fish_agent_v22_{datetime.now():%Y%m%d}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


logger = configure_logging()


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


def temp_path(filename: str) -> Path:
    root = Path(tempfile.gettempdir()) / "Fish-Agent"
    root.mkdir(parents=True, exist_ok=True)
    return root / filename


def find_chinese_font(preferred_size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    names = [
        "msyhbd.ttc" if bold else "msyh.ttc",
        "simhei.ttf",
        "simsun.ttc",
        "Deng.ttf",
        "arial.ttf",
    ]
    for name in names:
        path = windir / "Fonts" / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), preferred_size)
            except Exception:
                continue
    return ImageFont.load_default()


@dataclass
class QualitySummary:
    width: int
    height: int
    brightness: float
    contrast: float
    issues: list[str]

    @property
    def status(self) -> str:
        return "合格" if not self.issues else "需注意"


def validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在。"
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_IMAGE_FORMATS:
        formats = ", ".join(sorted(SUPPORTED_IMAGE_FORMATS))
        return False, f"暂不支持该格式，请选择 {formats}。"
    size = path.stat().st_size
    if size > MAX_IMAGE_SIZE_MB * 1024 * 1024:
        return False, f"图片大于 {MAX_IMAGE_SIZE_MB}MB，请压缩后再试。"
    if size < MIN_IMAGE_SIZE_KB * 1024:
        return False, "图片文件过小，可能不是有效图像。"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False, "图片无法读取或文件已损坏。"
    return True, ""


class ChannelAttention(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=[2, 3], keepdim=True)
        y = torch.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = torch.sigmoid(self.conv(y))
        return x * y


class FishFreshNetV1(nn.Module):
    def __init__(self, num_classes: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b0(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feature_dim = 1280
        self.att = nn.ModuleDict(
            {
                "ca": ChannelAttention(feature_dim),
                "sa": SpatialAttention(),
            }
        )
        self.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(feature_dim, num_classes))
        self.feature_maps: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        features = self.att["ca"](features)
        features = self.att["sa"](features)
        if torch.is_grad_enabled():
            features.retain_grad()
        self.feature_maps = features
        pooled = self.avgpool(features)
        pooled = torch.flatten(pooled, 1)
        return self.classifier(pooled)

    def get_feature_maps(self) -> torch.Tensor | None:
        return self.feature_maps


def load_model(model_path: Path, device: str) -> FishFreshNetV1:
    model = FishFreshNetV1(num_classes=3, pretrained=False)
    try:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=True)
    except Exception:
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
    elif isinstance(checkpoint, nn.Module):
        model = checkpoint
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device)
    return model


class LocalFishAnalyzer:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path or resource_path("src", "storage", "fishfreshnet_v1.pth")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.model: FishFreshNetV1 | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def ensure_model(self) -> FishFreshNetV1:
        if self.model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(f"未找到模型权重：{self.model_path}")
            logger.info("Loading FishFreshNetV1 from %s", self.model_path)
            self.model = load_model(self.model_path, self.device)
        return self.model

    def analyze(self, image_path: Path) -> dict[str, Any]:
        valid, error = validate_image(image_path)
        if not valid:
            raise ValueError(error)

        start = time.perf_counter()
        model = self.ensure_model()
        image = Image.open(image_path).convert("RGB")
        quality = self.inspect_quality(image)
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            outputs = model(input_tensor)
            probabilities_tensor = torch.softmax(outputs, dim=1)[0]
            label_index = int(torch.argmax(probabilities_tensor).item())
            target_score = outputs[0, label_index]
            target_score.backward()

        probabilities = {
            LABELS[index]: round(float(probabilities_tensor[index].detach().cpu().item()), 4)
            for index in range(len(LABELS))
        }
        label = LABELS[label_index]
        confidence = float(probabilities_tensor[label_index].detach().cpu().item())
        gradcam_path = self.generate_gradcam(image, model, label, confidence)
        report = self.build_report(label, confidence, probabilities, quality)
        recommendations = self.build_recommendations(label)

        return {
            "app_version": APP_VERSION,
            "model_name": "FishFreshNetV1",
            "model_source": "EfficientNet-B0 + CBAM-style attention",
            "freshness_label": label_index,
            "freshness_level": label,
            "freshness_level_display": DISPLAY_LABELS[label],
            "confidence_score": round(confidence, 4),
            "confidence_name": self.confidence_name(confidence),
            "probabilities": probabilities,
            "quality_summary": asdict(quality),
            "quality_status": quality.status,
            "freshness_report": report,
            "recommendations": recommendations,
            "gradcam_path": str(gradcam_path) if gradcam_path else "",
            "image_path": str(image_path),
            "inference_time_seconds": round(time.perf_counter() - start, 3),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def inspect_quality(self, image: Image.Image) -> QualitySummary:
        width, height = image.size
        grayscale = ImageOps.grayscale(image)
        stat = ImageStat.Stat(grayscale)
        brightness = float(stat.mean[0])
        contrast = float(stat.stddev[0])
        issues: list[str] = []
        if width < 224 or height < 224:
            issues.append("分辨率偏低，建议使用更清晰的鱼眼图片。")
        if brightness < 35:
            issues.append("画面偏暗，可能影响鱼眼透明度判断。")
        elif brightness > 230:
            issues.append("画面过曝，强反光可能干扰模型关注区域。")
        if contrast < 18:
            issues.append("对比度偏低，鱼眼边界可能不够清楚。")
        return QualitySummary(width, height, round(brightness, 2), round(contrast, 2), issues)

    def generate_gradcam(
        self,
        image: Image.Image,
        model: FishFreshNetV1,
        label: str,
        confidence: float,
    ) -> Path | None:
        feature_maps = model.get_feature_maps()
        if feature_maps is None or feature_maps.grad is None:
            return None

        gradients = feature_maps.grad.detach()
        activations = feature_maps.detach()
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations).sum(dim=1, keepdim=False))[0]
        cam = cam.detach().cpu().numpy()
        cam_min = float(cam.min())
        cam_max = float(cam.max())
        if cam_max - cam_min < 1e-8:
            return None
        cam = (cam - cam_min) / (cam_max - cam_min)

        heatmap = Image.fromarray(np.uint8(cam * 255)).resize(image.size, Image.Resampling.BICUBIC)
        heatmap_np = np.array(heatmap).astype(np.float32) / 255.0
        red = np.clip(2.2 * heatmap_np, 0, 1)
        green = np.clip(1.7 * (1 - np.abs(heatmap_np - 0.5) * 2), 0, 1)
        blue = np.clip(1.7 * (1 - heatmap_np), 0, 1)
        color_map = np.stack([red, green, blue], axis=2)

        base = np.array(image).astype(np.float32) / 255.0
        overlay = np.clip(base * 0.58 + color_map * 0.42, 0, 1)
        overlay_image = Image.fromarray(np.uint8(overlay * 255))
        draw = ImageDraw.Draw(overlay_image)
        font = find_chinese_font(18, bold=True)
        text = f"{DISPLAY_LABELS[label]}  {confidence:.1%}"
        draw.rounded_rectangle((14, 14, 14 + 260, 52), radius=10, fill=(15, 23, 42, 210))
        draw.text((28, 21), text, fill=(255, 255, 255), font=font)

        out = temp_path(f"gradcam_{uuid.uuid4().hex}.png")
        overlay_image.save(out)
        return out

    def confidence_name(self, confidence: float) -> str:
        if confidence >= 0.85:
            return "高置信度"
        if confidence >= 0.65:
            return "中置信度"
        return "低置信度"

    def build_report(
        self,
        label: str,
        confidence: float,
        probabilities: dict[str, float],
        quality: QualitySummary,
    ) -> dict[str, Any]:
        level = DISPLAY_LABELS[label]
        confidence_name = self.confidence_name(confidence)
        probability_text = "，".join(
            f"{DISPLAY_LABELS[name]} {value:.1%}" for name, value in probabilities.items()
        )
        quality_note = "图像质量合格。" if not quality.issues else "；".join(quality.issues)
        details = {
            "Highly Fresh": {
                "eye_appearance": "鱼眼整体更接近清澈、饱满、透明的高度新鲜特征。",
                "risk": "食品安全风险较低，但仍建议保持低温并尽快食用。",
                "positive": ["角膜透明度较好", "瞳孔和眼球结构更清晰", "模型输出集中度较高"],
                "warning": [],
            },
            "Fresh": {
                "eye_appearance": "鱼眼仍保留新鲜特征，但透明度、光泽或边界清晰度可能出现轻微下降。",
                "risk": "适合正常烹饪处理，建议缩短常温暴露时间。",
                "positive": ["鱼眼关键结构仍可辨认", "整体仍处于可接受新鲜区间"],
                "warning": ["建议尽快冷藏或烹饪"],
            },
            "Not Fresh": {
                "eye_appearance": "鱼眼可能呈现浑浊、暗淡或结构模糊等不新鲜信号。",
                "risk": "存在较高食品安全风险，不建议作为生食或低温短时烹饪食材。",
                "positive": [],
                "warning": ["谨慎食用", "检查气味、鳃色、体表黏液等更多证据", "必要时丢弃"],
            },
        }[label]
        return {
            "summary": {
                "grade": level,
                "confidence": confidence_name,
                "overall_assessment": f"系统判断样本为“{level}”，置信度 {confidence:.1%}。{LABEL_DESCRIPTIONS[label]}",
            },
            "detailed_analysis": {
                "eye_appearance": details["eye_appearance"],
                "model_probability": probability_text,
                "image_quality": quality_note,
                "risk_assessment": details["risk"],
            },
            "quality_indicators": {
                "positive_signs": details["positive"],
                "warning_signs": details["warning"] + quality.issues,
            },
            "confidence_explanation": (
                "本次评估由本地 FishFreshNetV1 完成。置信度越高，说明模型对当前等级的概率集中度越高；"
                "若图片反光、模糊或鱼眼不完整，应结合人工观察进行复核。"
            ),
            "disclaimer": "本系统用于辅助判断水产品新鲜度，不替代食品安全检测或专业质检结论。",
        }

    def build_recommendations(self, label: str) -> dict[str, Any]:
        templates = {
            "Highly Fresh": {
                "storage": {
                    "temperature": "0-4°C 冷藏或冰鲜保存",
                    "duration": "建议 24 小时内食用以保持最佳口感",
                    "tips": ["避免常温长时间放置", "尽量密封并与熟食分开"],
                },
                "consumption": {
                    "timeline": "当天食用最佳",
                    "cooking_methods": ["清蒸", "煎烤", "刺身级处理需另行确认正规来源"],
                },
                "handling": {
                    "preparation_steps": ["流水冲洗表面", "去鳞、去鳃、去内脏", "低温环境下处理"],
                    "deodorizing": "可使用姜片、葱段或少量料酒辅助去腥。",
                },
                "safety_warnings": {
                    "risk_level": "低",
                    "warnings": ["仍需确认来源、运输和储存条件可靠"],
                },
            },
            "Fresh": {
                "storage": {
                    "temperature": "0-4°C 冷藏",
                    "duration": "建议 12-24 小时内食用",
                    "tips": ["减少开盖暴露", "不要反复解冻"],
                },
                "consumption": {
                    "timeline": "尽快烹饪",
                    "cooking_methods": ["红烧", "炖煮", "香煎", "清蒸"],
                },
                "handling": {
                    "preparation_steps": ["检查鱼鳃、气味和腹部弹性", "彻底加热后食用"],
                    "deodorizing": "可提前用姜葱和少量盐进行短时处理。",
                },
                "safety_warnings": {
                    "risk_level": "中",
                    "warnings": ["不建议长时间常温放置", "老人、儿童和孕妇食用需更谨慎"],
                },
            },
            "Not Fresh": {
                "storage": {
                    "temperature": "不建议继续储存",
                    "duration": "如伴随异味、黏液异常或组织松散，应直接丢弃",
                    "tips": ["不要与其他食材混放", "处理后清洁刀具和案板"],
                },
                "consumption": {
                    "timeline": "不建议食用",
                    "cooking_methods": ["不建议通过重口味调料掩盖变质风险"],
                },
                "handling": {
                    "preparation_steps": ["复核气味、鳃色和体表状态", "无法确认安全时不要食用"],
                    "deodorizing": "不建议依赖去腥手段处理疑似变质鱼类。",
                },
                "safety_warnings": {
                    "risk_level": "高",
                    "warnings": ["可能存在腐败或微生物风险", "切勿生食或半生食"],
                },
            },
        }
        result = templates[label].copy()
        result["best_practices"] = ["全程低温", "生熟分开", "充分加热", "结合气味、鳃色和触感复核"]
        return result


def format_report(result: dict[str, Any]) -> str:
    report = result.get("freshness_report", {})
    summary = report.get("summary", {})
    detail = report.get("detailed_analysis", {})
    indicators = report.get("quality_indicators", {})
    probabilities = result.get("probabilities", {})
    lines = [
        "Fish-Agent 水产品新鲜度评估报告",
        "",
        f"评估时间：{result.get('created_at', '')}",
        f"模型：{result.get('model_name', 'FishFreshNetV1')}",
        f"新鲜度等级：{summary.get('grade', result.get('freshness_level_display', ''))}",
        f"置信度：{result.get('confidence_score', 0):.1%}（{result.get('confidence_name', '')}）",
        f"图像质量：{result.get('quality_status', '')}",
        "",
        "综合结论",
        summary.get("overall_assessment", ""),
        "",
        "详细分析",
    ]
    for key in ["eye_appearance", "model_probability", "image_quality", "risk_assessment"]:
        value = detail.get(key, "")
        if value:
            lines.append(f"- {value}")
    lines.extend(["", "概率分布"])
    for name, value in probabilities.items():
        lines.append(f"- {DISPLAY_LABELS.get(name, name)}：{value:.1%}")
    positives = indicators.get("positive_signs", [])
    warnings = indicators.get("warning_signs", [])
    if positives:
        lines.extend(["", "积极指标"])
        lines.extend(f"- {item}" for item in positives)
    if warnings:
        lines.extend(["", "注意事项"])
        lines.extend(f"- {item}" for item in warnings)
    lines.extend(["", report.get("confidence_explanation", ""), report.get("disclaimer", "")])
    return "\n".join(lines).strip()


def format_recommendations(result: dict[str, Any]) -> str:
    rec = result.get("recommendations", {})
    lines = ["Fish-Agent 处理建议", ""]
    storage = rec.get("storage", {})
    lines.extend(
        [
            "储存建议",
            f"- 温度：{storage.get('temperature', '')}",
            f"- 时长：{storage.get('duration', '')}",
        ]
    )
    for tip in storage.get("tips", []):
        lines.append(f"- {tip}")
    consumption = rec.get("consumption", {})
    lines.extend(
        [
            "",
            "食用建议",
            f"- 时限：{consumption.get('timeline', '')}",
            f"- 烹饪方式：{'、'.join(consumption.get('cooking_methods', []))}",
        ]
    )
    handling = rec.get("handling", {})
    lines.extend(["", "处理建议"])
    for step in handling.get("preparation_steps", []):
        lines.append(f"- {step}")
    if handling.get("deodorizing"):
        lines.append(f"- {handling.get('deodorizing')}")
    safety = rec.get("safety_warnings", {})
    lines.extend(["", "安全提醒", f"- 风险等级：{safety.get('risk_level', '')}"])
    for warning in safety.get("warnings", []):
        lines.append(f"- {warning}")
    practices = rec.get("best_practices", [])
    if practices:
        lines.extend(["", "最佳实践", f"- {'、'.join(practices)}"])
    return "\n".join(lines).strip()


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for char in paragraph:
            trial = current + char
            bbox = draw.textbbox((0, 0), trial, font=font)
            if bbox[2] - bbox[0] <= width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = char
        lines.append(current)
    return lines


def export_pdf_report(result: dict[str, Any], output_path: Path) -> Path:
    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(page)
    title_font = find_chinese_font(42, bold=True)
    h2_font = find_chinese_font(28, bold=True)
    body_font = find_chinese_font(22)
    small_font = find_chinese_font(18)

    y = 64
    draw.text((64, y), f"Fish-Agent V{APP_VERSION} 新鲜度评估报告", font=title_font, fill="#102033")
    y += 62
    draw.text((64, y), f"生成时间：{result.get('created_at', '')}", font=small_font, fill="#5B677A")
    y += 58

    label = result.get("freshness_level", "Fresh")
    color = LABEL_COLORS.get(label, "#2368D8")
    draw.rounded_rectangle((64, y, width - 64, y + 142), radius=20, fill="#FFFFFF", outline="#D8E0EA", width=2)
    draw.text((96, y + 30), result.get("freshness_level_display", ""), font=h2_font, fill=color)
    draw.text(
        (96, y + 82),
        f"置信度 {result.get('confidence_score', 0):.1%} · {result.get('confidence_name', '')} · 图像质量 {result.get('quality_status', '')}",
        font=body_font,
        fill="#17202A",
    )
    y += 182

    report_text = format_report(result)
    rec_text = format_recommendations(result)
    for title, text in [("评估报告", report_text), ("处理建议", rec_text)]:
        draw.text((64, y), title, font=h2_font, fill="#102033")
        y += 42
        for line in wrap_text(draw, text, body_font, width - 128):
            if y > height - 110:
                break
            draw.text((64, y), line, font=body_font, fill="#17202A")
            y += 32
        y += 28
        if y > height - 360:
            break

    image_boxes = []
    if result.get("image_path"):
        image_boxes.append(("原始图像", result["image_path"]))
    if result.get("gradcam_path"):
        image_boxes.append(("Grad-CAM 热力图", result["gradcam_path"]))

    if image_boxes and y < height - 330:
        box_w, box_h = 520, 300
        x = 64
        for title, path_text in image_boxes[:2]:
            path = Path(path_text)
            if not path.exists():
                continue
            draw.text((x, y), title, font=small_font, fill="#5B677A")
            with Image.open(path).convert("RGB") as image:
                image.thumbnail((box_w, box_h), Image.Resampling.LANCZOS)
                px = x + (box_w - image.width) // 2
                py = y + 32 + (box_h - image.height) // 2
                draw.rounded_rectangle((x, y + 32, x + box_w, y + 32 + box_h), radius=18, fill="#FFFFFF", outline="#D8E0EA")
                page.paste(image, (px, py))
            x += box_w + 56

    draw.text((64, height - 54), GITHUB_URL, font=small_font, fill="#5B677A")
    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, "PDF", resolution=150.0)
    logger.info("PDF report saved: %s", output_path)
    return output_path


class FishAgentDesktop(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(980, 660)
        self.configure(bg=COLORS["bg"])

        self.analyzer = LocalFishAnalyzer()
        self.selected_image: Path | None = None
        self.preview_image: Image.Image | None = None
        self.gradcam_image: Image.Image | None = None
        self.tk_preview: ImageTk.PhotoImage | None = None
        self.result: dict[str, Any] | None = None
        self.show_gradcam = tk.BooleanVar(value=False)

        self.status_var = tk.StringVar(value="就绪")
        self.level_var = tk.StringVar(value="--")
        self.confidence_var = tk.StringVar(value="--")
        self.probability_var = tk.StringVar(value="")

        self.configure_style()
        self.build_ui()
        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"], relief="flat")
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 20, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Microsoft YaHei UI", 10))
        style.configure("Primary.TButton", font=("Microsoft YaHei UI", 10, "bold"), padding=(16, 8))
        style.configure("TButton", font=("Microsoft YaHei UI", 10), padding=(12, 7))
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 10), padding=(14, 8))

    def build_ui(self) -> None:
        root = ttk.Frame(self, padding=22)
        root.pack(fill=tk.BOTH, expand=True)

        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 16))
        ttk.Label(header, text="Fish-Agent", style="Title.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            header,
            text=f"水产品新鲜度智能评估系统  ·  V{APP_VERSION}",
            style="Subtitle.TLabel",
        ).pack(side=tk.LEFT, padx=(16, 0), pady=(8, 0))

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=5)
        body.columnconfigure(1, weight=6)
        body.rowconfigure(0, weight=1)

        left = self.panel(body)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = self.panel(body)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        left.rowconfigure(1, weight=1)
        right.rowconfigure(2, weight=1)

        ttk.Label(left, text="图像", style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(left, bg=COLORS["canvas"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=12)
        self.canvas.bind("<Configure>", lambda _event: self.redraw_preview())

        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.grid(row=2, column=0, sticky="ew")
        ttk.Button(controls, text="选择图片", command=self.select_image).pack(side=tk.LEFT)
        self.analyze_button = ttk.Button(controls, text="开始分析", command=self.start_analysis, style="Primary.TButton")
        self.analyze_button.pack(side=tk.LEFT, padx=8)
        ttk.Button(controls, text="清空", command=self.clear_image).pack(side=tk.LEFT)
        ttk.Checkbutton(
            controls,
            text="Grad-CAM",
            variable=self.show_gradcam,
            command=self.redraw_preview,
        ).pack(side=tk.RIGHT)

        ttk.Label(right, text="结果", style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        summary = ttk.Frame(right, style="Panel.TFrame")
        summary.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        self.level_label = ttk.Label(summary, textvariable=self.level_var, style="Panel.TLabel", font=("Microsoft YaHei UI", 22, "bold"))
        self.level_label.grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.confidence_var, style="Panel.TLabel", font=("Microsoft YaHei UI", 13)).grid(row=0, column=1, sticky="e")
        ttk.Label(summary, textvariable=self.probability_var, style="Panel.TLabel", foreground=COLORS["muted"]).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self.report_text = self.text_tab("评估报告")
        self.recommend_text = self.text_tab("处理建议")
        self.tech_text = self.text_tab("技术信息")

        export_row = ttk.Frame(right, style="Panel.TFrame")
        export_row.grid(row=3, column=0, sticky="ew", pady=(12, 0))
        ttk.Button(export_row, text="保存 PDF", command=self.export_pdf).pack(side=tk.LEFT)
        ttk.Button(export_row, text="保存 JSON", command=self.export_json).pack(side=tk.LEFT, padx=8)
        ttk.Label(export_row, textvariable=self.status_var, style="Panel.TLabel", foreground=COLORS["muted"]).pack(side=tk.RIGHT)

        self.draw_empty_canvas()

    def panel(self, parent: ttk.Frame) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=18)
        frame.configure(style="Panel.TFrame")
        return frame

    def text_tab(self, title: str) -> tk.Text:
        frame = ttk.Frame(self.notebook, style="Panel.TFrame", padding=8)
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)
        text = tk.Text(
            frame,
            wrap=tk.WORD,
            borderwidth=0,
            bg="#FFFFFF",
            fg=COLORS["text"],
            insertbackground=COLORS["primary"],
            font=("Microsoft YaHei UI", 10),
            padx=8,
            pady=8,
        )
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        text.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.notebook.add(frame, text=title)
        return text

    def draw_empty_canvas(self) -> None:
        self.canvas.delete("all")
        width = max(self.canvas.winfo_width(), 420)
        height = max(self.canvas.winfo_height(), 420)
        self.canvas.create_text(
            width // 2,
            height // 2,
            text="Fish-Agent",
            fill=COLORS["muted"],
            font=("Microsoft YaHei UI", 20, "bold"),
        )

    def select_image(self) -> None:
        path_text = filedialog.askopenfilename(title="选择鱼眼图片", filetypes=IMAGE_FILE_TYPES)
        if path_text:
            self.load_image(Path(path_text))

    def load_image(self, path: Path) -> None:
        valid, error = validate_image(path)
        if not valid:
            messagebox.showwarning("图片不可用", error)
            return
        try:
            self.preview_image = Image.open(path).convert("RGB")
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return
        self.selected_image = path
        self.gradcam_image = None
        self.result = None
        self.show_gradcam.set(False)
        self.level_var.set("--")
        self.confidence_var.set("--")
        self.probability_var.set("")
        self.set_text(self.report_text, "")
        self.set_text(self.recommend_text, "")
        self.set_text(self.tech_text, "")
        self.status_var.set(path.name)
        self.redraw_preview()

    def redraw_preview(self) -> None:
        image = self.gradcam_image if self.show_gradcam.get() and self.gradcam_image else self.preview_image
        if image is None:
            self.draw_empty_canvas()
            return
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        preview = image.copy()
        preview.thumbnail((width - 24, height - 24), Image.Resampling.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(width // 2, height // 2, image=self.tk_preview)

    def clear_image(self) -> None:
        self.selected_image = None
        self.preview_image = None
        self.gradcam_image = None
        self.result = None
        self.show_gradcam.set(False)
        self.level_var.set("--")
        self.confidence_var.set("--")
        self.probability_var.set("")
        self.status_var.set("就绪")
        self.set_text(self.report_text, "")
        self.set_text(self.recommend_text, "")
        self.set_text(self.tech_text, "")
        self.draw_empty_canvas()

    def start_analysis(self) -> None:
        if self.selected_image is None:
            messagebox.showinfo("未选择图片", "请先选择一张鱼眼图片。")
            return
        self.analyze_button.configure(state=tk.DISABLED)
        self.status_var.set("正在分析...")
        self.level_var.set("分析中")
        thread = threading.Thread(target=self.analysis_worker, args=(self.selected_image,), daemon=True)
        thread.start()

    def analysis_worker(self, path: Path) -> None:
        try:
            result = self.analyzer.analyze(path)
            self.after(0, lambda: self.analysis_finished(result))
        except Exception as exc:
            details = traceback.format_exc()
            logger.error("Analysis failed: %s\n%s", exc, details)
            self.after(0, lambda: self.analysis_failed(str(exc), details))

    def analysis_finished(self, result: dict[str, Any]) -> None:
        self.result = result
        label = result.get("freshness_level", "Fresh")
        display = result.get("freshness_level_display", DISPLAY_LABELS.get(label, label))
        self.level_var.set(display)
        self.level_label.configure(foreground=LABEL_COLORS.get(label, COLORS["primary"]))
        self.confidence_var.set(f"{result.get('confidence_score', 0):.1%} · {result.get('confidence_name', '')}")
        probabilities = result.get("probabilities", {})
        self.probability_var.set(
            "  ".join(f"{DISPLAY_LABELS.get(name, name)} {value:.1%}" for name, value in probabilities.items())
        )
        self.set_text(self.report_text, format_report(result))
        self.set_text(self.recommend_text, format_recommendations(result))
        self.set_text(self.tech_text, json.dumps(result, ensure_ascii=False, indent=2))
        gradcam_path = result.get("gradcam_path")
        if gradcam_path and Path(gradcam_path).exists():
            self.gradcam_image = Image.open(gradcam_path).convert("RGB")
            self.show_gradcam.set(True)
        self.redraw_preview()
        self.status_var.set(f"完成 · {result.get('inference_time_seconds', 0)}s")
        self.analyze_button.configure(state=tk.NORMAL)

    def analysis_failed(self, message: str, details: str) -> None:
        self.level_var.set("--")
        self.confidence_var.set("--")
        self.status_var.set("分析失败")
        self.set_text(self.tech_text, details)
        self.analyze_button.configure(state=tk.NORMAL)
        messagebox.showerror("分析失败", message)

    def set_text(self, widget: tk.Text, value: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", value)
        widget.configure(state=tk.DISABLED)

    def export_pdf(self) -> None:
        if not self.result:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        default_name = f"fish_agent_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        path_text = filedialog.asksaveasfilename(
            title="保存 PDF 报告",
            defaultextension=".pdf",
            initialfile=default_name,
            filetypes=[("PDF 文件", "*.pdf")],
        )
        if not path_text:
            return
        try:
            saved = export_pdf_report(self.result, Path(path_text))
            messagebox.showinfo("保存成功", f"PDF 报告已保存：\n{saved}")
        except Exception as exc:
            logger.error("PDF export failed: %s\n%s", exc, traceback.format_exc())
            messagebox.showerror("保存失败", str(exc))

    def export_json(self) -> None:
        if not self.result:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        default_name = f"fish_agent_result_{datetime.now():%Y%m%d_%H%M%S}.json"
        path_text = filedialog.asksaveasfilename(
            title="保存 JSON 结果",
            defaultextension=".json",
            initialfile=default_name,
            filetypes=[("JSON 文件", "*.json")],
        )
        if not path_text:
            return
        path = Path(path_text)
        path.write_text(json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("保存成功", f"JSON 结果已保存：\n{path}")

    def on_close(self) -> None:
        self.destroy()


def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=f"Fish-Agent V{APP_VERSION} local analyzer")
    parser.add_argument("image", nargs="?", help="Path to fish-eye image")
    parser.add_argument("--output", "-o", help="Write result JSON to file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)
    if not args.image:
        parser.print_help()
        return 1
    analyzer = LocalFishAnalyzer()
    result = analyzer.analyze(Path(args.image))
    json_text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).write_text(json_text, encoding="utf-8")
    else:
        print(json_text)
    return 0


def main() -> None:
    if len(sys.argv) > 1:
        try:
            raise SystemExit(run_cli(sys.argv[1:]))
        except Exception as exc:
            logger.error("CLI failed: %s\n%s", exc, traceback.format_exc())
            raise
    app = FishAgentDesktop()
    app.mainloop()


if __name__ == "__main__":
    main()
