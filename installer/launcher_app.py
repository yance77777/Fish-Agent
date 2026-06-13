"""
Fish-Agent V3.4.0 installer launcher.

Fish-Agent V3.4.0 Windows desktop application with local FishFreshNetV1
inference, Grad-CAM visualization, PDF/JSON export, and batch analysis.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import subprocess
import sys
import tempfile
import threading
import traceback
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any

import tkinter as tk
from tkinter import filedialog, font as tkfont, messagebox, ttk

from PIL import Image, ImageDraw, ImageFont, ImageOps, ImageTk


APP_VERSION = "3.4.0"
APP_TITLE = f"Fish-Agent V{APP_VERSION} 水产品新鲜度智能评估系统"
GITHUB_URL = "https://github.com/yance77777/Fish-Agent"
APP_SUBTITLE = "水产品新鲜度智能评估系统"
SUPPORTED_IMAGE_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
IMAGE_FILE_TYPES = [
    ("图像文件", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"),
    ("所有文件", "*.*"),
]
LEVEL_COLORS = {
    "高度新鲜": "#0F9F6E",
    "新鲜": "#2368D8",
    "不新鲜": "#C2410C",
}
COLORS = {
    "bg": "#F3F8FC",
    "panel": "#FFFFFF",
    "panel_alt": "#F7FBFF",
    "line": "#C9D8E8",
    "line_soft": "#E3ECF5",
    "text": "#102033",
    "muted": "#5A6B80",
    "primary": "#0B5CAD",
    "primary_dark": "#084A8C",
    "primary_soft": "#E7F2FF",
    "ocean": "#0F766E",
    "canvas": "#EAF3FB",
    "danger": "#C2410C",
}

def setup_logger() -> logging.Logger:
    log_dir = Path(tempfile.gettempdir()) / "Fish-Agent" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("Fish-Agent-V3.4.0")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(log_dir / f"launcher_{datetime.now():%Y%m%d}.log", encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def set_dpi_awareness() -> None:
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def resource_path(*parts: str) -> Path:
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base.joinpath(*parts)


def core_exe_path() -> Path:
    candidates = [
        resource_path("core", "Fish-Agent-Core.exe"),
        resource_path("core", "Fish-Agent.exe"),
        Path(__file__).resolve().parent / "core" / "Fish-Agent-Core.exe",
        Path(__file__).resolve().parent / "core" / "Fish-Agent.exe",
        Path(__file__).resolve().parent.parent / ".installer_innounp" / "{app}" / "Fish-Agent.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def validate_image(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "文件不存在。"
    if path.suffix.lower() not in SUPPORTED_IMAGE_FORMATS:
        return False, "暂不支持该图片格式，请选择 JPG、PNG、BMP、WEBP 或 TIFF 文件。"
    if path.stat().st_size < 1024:
        return False, "图片文件过小，可能不是有效图像。"
    try:
        with Image.open(path) as image:
            image.verify()
    except Exception:
        return False, "图片无法读取或文件已损坏。"
    return True, ""


def run_core_analysis(image_path: Path) -> dict[str, Any]:
    core = core_exe_path()
    if not core.exists():
        raise FileNotFoundError(f"未找到内置分析核心：{core}")

    output = Path(tempfile.gettempdir()) / "Fish-Agent" / f"core_result_{datetime.now():%Y%m%d_%H%M%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")

    command = [str(core), "--analyze", str(image_path.resolve()), "--output", str(output), "--pretty"]
    logger.info("Running core: %s", command)
    completed = subprocess.run(
        command,
        cwd=str(core.parent),
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=300,
        check=False,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "分析核心运行失败。\n"
            f"退出码：{completed.returncode}\n"
            f"输出：{completed.stdout}\n"
            f"错误：{completed.stderr}"
        )
    if not output.exists():
        raise RuntimeError("分析核心未生成结果文件，请检查日志。")

    raw = json.loads(output.read_text(encoding="utf-8"))
    raw["app_version"] = APP_VERSION
    raw["core_executable"] = str(core)
    raw["result_json_path"] = str(output)
    raw["created_at"] = raw.get("timestamp") or datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return raw


def get_level(result: dict[str, Any]) -> str:
    return str(result.get("freshness_level") or result.get("freshness_report", {}).get("summary", {}).get("grade") or "--")


def get_confidence(result: dict[str, Any]) -> float:
    try:
        return float(result.get("confidence_score", 0))
    except Exception:
        return 0.0


def probability_items(result: dict[str, Any]) -> dict[str, float]:
    values = result.get("all_probabilities") or result.get("probabilities") or {}
    return {str(key): float(value) for key, value in values.items()}


def format_report(result: dict[str, Any]) -> str:
    report = result.get("freshness_report", {})
    summary = report.get("summary", {})
    detail = report.get("detailed_analysis", {})
    indicators = report.get("quality_indicators", {})
    lines = [
        "Fish-Agent 水产品新鲜度评估报告",
        "",
        f"应用版本：V{APP_VERSION}",
        f"评估模型：{result.get('analysis_method_label', 'FishFreshNetV1')}",
        f"评估时间：{result.get('created_at', '')}",
        f"新鲜度等级：{get_level(result)}",
        f"置信度：{get_confidence(result):.1%}",
        "",
        "综合结论",
        str(summary.get("overall_assessment") or result.get("description") or ""),
        "",
        "详细分析",
    ]
    for value in detail.values():
        if value:
            lines.append(f"- {value}")

    probs = probability_items(result)
    if probs:
        lines.extend(["", "概率分布"])
        lines.extend(f"- {name}：{value:.1%}" for name, value in probs.items())

    positives = indicators.get("positive_signs", [])
    warnings = indicators.get("warning_signs", [])
    if positives:
        lines.extend(["", "积极指标"])
        lines.extend(f"- {item}" for item in positives)
    if warnings:
        lines.extend(["", "注意事项"])
        lines.extend(f"- {item}" for item in warnings)

    quality = result.get("quality_check", {})
    if quality:
        details = quality.get("details", {})
        lines.extend(
            [
                "",
                "图像质量",
                f"- 是否合格：{'是' if quality.get('is_qualified') else '需注意'}",
                f"- 分辨率：{details.get('width', '--')} x {details.get('height', '--')}",
                f"- 亮度：{details.get('brightness', '--')}",
                f"- 对比度：{details.get('contrast', '--')}",
            ]
        )
        for issue in quality.get("issues", []):
            lines.append(f"- {issue}")

    if report.get("confidence_explanation"):
        lines.extend(["", str(report.get("confidence_explanation"))])
    if report.get("disclaimer"):
        lines.extend(["", str(report.get("disclaimer"))])
    return "\n".join(lines).strip()


def format_recommendations(result: dict[str, Any]) -> str:
    rec = result.get("recommendations", {})
    lines = ["Fish-Agent 处理建议", ""]
    storage = rec.get("storage", {})
    if storage:
        lines.extend(
            [
                "储存建议",
                f"- 温度：{storage.get('temperature', '')}",
                f"- 方式：{storage.get('method', '')}",
                f"- 时长：{storage.get('duration', '')}",
            ]
        )
        lines.extend(f"- {item}" for item in storage.get("tips", []))
    consumption = rec.get("consumption", {})
    if consumption:
        lines.extend(["", "食用建议"])
        if consumption.get("timeline"):
            lines.append(f"- 时限：{consumption.get('timeline')}")
        if consumption.get("cooking_methods"):
            lines.append(f"- 烹饪方式：{'、'.join(consumption.get('cooking_methods', []))}")
        if consumption.get("flavor_tips"):
            lines.extend(f"- {item}" for item in consumption.get("flavor_tips", []))
    handling = rec.get("handling", {})
    if handling:
        lines.extend(["", "处理建议"])
        lines.extend(f"- {item}" for item in handling.get("preparation_steps", []))
        if handling.get("cutting_advice"):
            lines.append(f"- {handling.get('cutting_advice')}")
        if handling.get("deodorizing"):
            lines.append(f"- {handling.get('deodorizing')}")
    safety = rec.get("safety_warnings", {})
    if safety:
        lines.extend(["", "安全提醒", f"- 风险等级：{safety.get('risk_level', '')}"])
        lines.extend(f"- {item}" for item in safety.get("warnings", []))
    if rec.get("best_practices"):
        lines.extend(["", "最佳实践", f"- {'、'.join(rec.get('best_practices', []))}"])
    return "\n".join(lines).strip()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    windir = Path(os.environ.get("WINDIR", r"C:\Windows"))
    names = ["msyhbd.ttc" if bold else "msyh.ttc", "simhei.ttf", "simsun.ttc", "arial.ttf"]
    for name in names:
        path = windir / "Fonts" / name
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except Exception:
                pass
    return ImageFont.load_default()


def wrap_pdf_text(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.ImageFont, max_width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        current = ""
        for char in paragraph:
            trial = current + char
            bbox = draw.textbbox((0, 0), trial, font=fnt)
            if bbox[2] - bbox[0] <= max_width:
                current = trial
            else:
                if current:
                    lines.append(current)
                current = char
        lines.append(current)
    return lines


def draw_bullet_lines(
    draw: ImageDraw.ImageDraw,
    lines: list[str],
    x: int,
    y: int,
    max_y: int,
    fnt: ImageFont.ImageFont,
    color: str = "#17202A",
) -> int:
    for raw in lines:
        if y > max_y:
            break
        text = str(raw).strip()
        if not text:
            y += 14
            continue
        bullet = text.startswith("- ") or text.startswith("• ")
        if bullet:
            text = text[2:].strip()
            draw.ellipse((x, y + 11, x + 10, y + 21), fill=COLORS["primary"])
            text_x = x + 24
            max_width = 1060
        else:
            text_x = x
            max_width = 1090
        for wrapped in wrap_pdf_text(draw, text, fnt, max_width):
            if y > max_y:
                break
            draw.text((text_x, y), wrapped, font=fnt, fill=color)
            y += 34
        y += 4 if bullet else 8
    return y


def export_pdf_report(result: dict[str, Any], output_path: Path) -> Path:
    width, height = 1240, 1754
    page = Image.new("RGB", (width, height), "#F7FAFC")
    draw = ImageDraw.Draw(page)
    title_font = font(42, True)
    h2_font = font(28, True)
    body_font = font(22)
    small_font = font(18)

    level = get_level(result)
    color = LEVEL_COLORS.get(level, COLORS["primary"])
    confidence = get_confidence(result)

    def logo(x: int, y: int) -> None:
        draw.ellipse((x, y, x + 50, y + 50), fill=COLORS["primary"])
        draw.ellipse((x + 9, y + 9, x + 41, y + 41), fill=COLORS["ocean"])
        draw.polygon([(x + 12, y + 25), (x + 24, y + 13), (x + 42, y + 17), (x + 47, y + 25), (x + 42, y + 33), (x + 24, y + 37)], fill="#FFFFFF")
        draw.polygon([(x + 12, y + 25), (x + 4, y + 17), (x + 5, y + 33)], fill="#D9ECFF")
        draw.ellipse((x + 29, y + 19, x + 38, y + 28), fill=COLORS["primary"])
        draw.ellipse((x + 32, y + 21, x + 35, y + 24), fill="#FFFFFF")

    def paste_fit(path_text: str | None, box: tuple[int, int, int, int]) -> None:
        draw.rounded_rectangle(box, radius=20, fill="#FFFFFF", outline="#D8E0EA", width=2)
        if not path_text or not Path(path_text).exists():
            draw.text((box[0] + 24, box[1] + 135), "图片不可用", font=body_font, fill=COLORS["muted"])
            return
        with Image.open(path_text).convert("RGB") as img:
            box_w = box[2] - box[0]
            box_h = box[3] - box[1]
            fit_w = box_w - 36
            fit_h = box_h - 36
            scale = min(fit_w / img.width, fit_h / img.height)
            size = (max(1, int(img.width * scale)), max(1, int(img.height * scale)))
            fitted = img.resize(size, Image.Resampling.LANCZOS)
            page.paste(fitted, (box[0] + (box_w - fitted.width) // 2, box[1] + (box_h - fitted.height) // 2))

    logo(64, 54)
    draw.text((128, 55), "Fish-Agent", font=title_font, fill="#102033")
    draw.text((128, 108), f"水产品新鲜度智能评估报告 · V{APP_VERSION}", font=small_font, fill=COLORS["muted"])
    draw.text((width - 440, 70), f"生成时间：{result.get('created_at', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}", font=small_font, fill=COLORS["muted"])

    y = 170
    draw.rounded_rectangle((64, y, width - 64, y + 126), radius=22, fill="#FFFFFF", outline="#D8E0EA", width=2)
    draw.text((96, y + 26), level, font=h2_font, fill=color)
    draw.text((96, y + 76), f"置信度 {confidence:.1%} · 模型 FishFreshNetV1", font=body_font, fill="#17202A")
    draw.rounded_rectangle((width - 342, y + 34, width - 96, y + 92), radius=18, fill=COLORS["primary_soft"], outline=None)
    draw.text((width - 314, y + 48), "AI 图像智能评估", font=body_font, fill=COLORS["primary"])

    y = 336
    draw.text((64, y), "图像证据", font=h2_font, fill="#102033")
    y += 44
    draw.text((78, y), "原始鱼眼图", font=small_font, fill=COLORS["muted"])
    draw.text((666, y), "AI 关注区域热力图", font=small_font, fill=COLORS["muted"])
    y += 28
    paste_fit(result.get("image_path"), (64, y, 594, y + 318))
    paste_fit(result.get("gradcam_image_path"), (646, y, width - 64, y + 318))
    y += 354

    probs = probability_items(result)
    if probs:
        draw.text((64, y), "概率分布", font=h2_font, fill="#102033")
        y += 48
        bar_x, bar_y, bar_w, bar_h = 64, y, width - 128, 34
        total = sum(max(value, 0.0) for value in probs.values()) or 1.0
        colors = ["#0F9F6E", "#2368D8", "#C2410C", "#64748B"]
        x = bar_x
        for index, (name, value) in enumerate(probs.items()):
            segment = bar_w * max(value, 0.0) / total
            draw.rectangle((x, bar_y, x + segment, bar_y + bar_h), fill=colors[index % len(colors)])
            x += segment
        y += 54
        legend_x = 64
        for index, (name, value) in enumerate(probs.items()):
            draw.rectangle((legend_x, y + 7, legend_x + 18, y + 25), fill=colors[index % len(colors)])
            draw.text((legend_x + 28, y), f"{name} {value:.1%}", font=small_font, fill=COLORS["muted"])
            legend_x += 250
        y += 52

    report = result.get("freshness_report", {})
    summary = report.get("summary", {})
    draw.text((64, y), "综合结论", font=h2_font, fill="#102033")
    y += 42
    for line in wrap_pdf_text(draw, summary.get("overall_assessment") or result.get("description") or "", body_font, width - 128)[:5]:
        draw.text((64, y), line, font=body_font, fill="#17202A")
        y += 32

    y += 18
    draw.text((64, y), "处理建议摘要", font=h2_font, fill="#102033")
    y += 42
    y = draw_bullet_lines(draw, format_recommendations(result).splitlines()[:20], 64, y, height - 120, body_font)

    draw.line((64, height - 82, width - 64, height - 82), fill="#D8E0EA", width=2)
    draw.text((64, height - 56), GITHUB_URL, font=small_font, fill="#5B677A")
    draw.text((width - 320, height - 56), f"Fish-Agent V{APP_VERSION}", font=small_font, fill="#5B677A")
    output_path = output_path.with_suffix(".pdf")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(output_path, "PDF", resolution=150.0)
    return output_path


class FishAgentApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1180x760")
        self.minsize(980, 660)
        self.configure(bg=COLORS["bg"])

        self.selected_image: Path | None = None
        self.preview_image: Image.Image | None = None
        self.gradcam_image: Image.Image | None = None
        self.tk_preview: ImageTk.PhotoImage | None = None
        self.result: dict[str, Any] | None = None
        self.show_gradcam = tk.BooleanVar(value=False)
        self.heatmap_opacity = tk.DoubleVar(value=55.0)
        self.opacity_value_var = tk.StringVar(value="55%")
        self.status_var = tk.StringVar(value="就绪")
        self.level_var = tk.StringVar(value="分析结果将在此处显示")
        self.confidence_var = tk.StringVar(value="--")
        self.probability_var = tk.StringVar(value="")
        self.image_meta_var = tk.StringVar(value="支持 JPG、PNG、BMP、WEBP、TIFF 格式")
        self.focus_hint_var = tk.StringVar(value="红色区域表示 AI 模型关注的关键特征")
        self.score_value = 0.0
        self.history: list[dict[str, Any]] = []
        self.history_thumbnails: list[ImageTk.PhotoImage] = []

        self.configure_style()
        self.build_ui()
        self.bind_shortcuts()

    def configure_style(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background=COLORS["bg"])
        style.configure("Panel.TFrame", background=COLORS["panel"])
        style.configure("TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Panel.TLabel", background=COLORS["panel"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 10))
        style.configure("Muted.Panel.TLabel", background=COLORS["panel"], foreground=COLORS["muted"], font=("Microsoft YaHei UI", 9))
        style.configure("Title.TLabel", background=COLORS["bg"], foreground=COLORS["text"], font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("Subtitle.TLabel", background=COLORS["bg"], foreground=COLORS["muted"], font=("Microsoft YaHei UI", 11))
        style.configure("TNotebook", background=COLORS["panel"], borderwidth=0)
        style.configure("TNotebook.Tab", font=("Microsoft YaHei UI", 10, "bold"), padding=(18, 9), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", COLORS["primary_soft"])], foreground=[("selected", COLORS["primary"])])

    def make_button(
        self,
        parent: tk.Misc,
        text: str,
        command: Any,
        *,
        primary: bool = False,
        compact: bool = False,
    ) -> tk.Button:
        bg = COLORS["primary"] if primary else COLORS["primary_soft"]
        fg = "#FFFFFF" if primary else COLORS["primary"]
        hover_bg = COLORS["primary_dark"] if primary else "#D9ECFF"
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg,
            fg=fg,
            activebackground=hover_bg,
            activeforeground=fg,
            relief=tk.FLAT,
            bd=0,
            cursor="hand2",
            padx=18 if not compact else 12,
            pady=9 if not compact else 6,
            font=("Microsoft YaHei UI", 10, "bold" if primary else "normal"),
            highlightthickness=0,
        )
        button._normal_bg = bg
        button._normal_fg = fg
        button._hover_bg = hover_bg
        button._hover_fg = fg
        button.bind("<Enter>", lambda _event: button.configure(bg=button._hover_bg, fg=button._hover_fg))
        button.bind("<Leave>", lambda _event: button.configure(bg=button._normal_bg, fg=button._normal_fg))
        return button

    def set_button_state_style(self, button: tk.Button, *, active: bool = False, ghost: bool = False) -> None:
        if active:
            normal_bg, normal_fg = COLORS["primary"], "#FFFFFF"
            hover_bg, hover_fg = COLORS["primary_dark"], "#FFFFFF"
        elif ghost:
            normal_bg, normal_fg = "#FFFFFF", COLORS["primary"]
            hover_bg, hover_fg = COLORS["primary_soft"], COLORS["primary_dark"]
        else:
            normal_bg, normal_fg = COLORS["primary_soft"], COLORS["primary"]
            hover_bg, hover_fg = "#D9ECFF", COLORS["primary_dark"]
        button._normal_bg = normal_bg
        button._normal_fg = normal_fg
        button._hover_bg = hover_bg
        button._hover_fg = hover_fg
        button.configure(bg=normal_bg, fg=normal_fg)

    def build_opacity_slider(self, parent: tk.Misc) -> tk.Canvas:
        slider = tk.Canvas(parent, width=164, height=28, bg=COLORS["panel"], highlightthickness=0, cursor="hand2")
        slider.bind("<Configure>", lambda event: self.draw_opacity_slider(event.widget))
        slider.bind("<Button-1>", self.set_opacity_from_event)
        slider.bind("<B1-Motion>", self.set_opacity_from_event)
        return slider

    def set_opacity_from_event(self, event: tk.Event) -> None:
        slider = event.widget
        width = max(slider.winfo_width(), 164)
        left, right = 12, width - 12
        ratio = min(1.0, max(0.0, (event.x - left) / max(1, right - left)))
        self.heatmap_opacity.set(int(round(ratio * 100)))
        self.update_opacity_value()
        self.draw_opacity_slider(slider)
        self.redraw_preview()

    def update_opacity_value(self) -> None:
        self.opacity_value_var.set(f"{int(round(self.heatmap_opacity.get()))}%")

    def draw_opacity_slider(self, slider: tk.Canvas | None = None) -> None:
        if slider is None:
            slider = getattr(self, "opacity_slider", None)
        if slider is None or not slider.winfo_exists():
            return
        slider.delete("all")
        width = max(slider.winfo_width(), 164)
        height = max(slider.winfo_height(), 28)
        left, right = 12, width - 12
        y = height // 2
        value = min(100, max(0, int(self.heatmap_opacity.get())))
        self.update_opacity_value()
        knob_x = left + (right - left) * value / 100
        slider.create_line(left, y, right, y, fill=COLORS["line"], width=6, capstyle=tk.ROUND)
        slider.create_line(left, y, knob_x, y, fill=COLORS["primary"], width=6, capstyle=tk.ROUND)
        slider.create_oval(knob_x - 8, y - 8, knob_x + 8, y + 8, fill="#FFFFFF", outline=COLORS["primary"], width=2)

    def bind_shortcuts(self) -> None:
        self.bind("<Control-o>", lambda _event: self.select_image())
        self.bind("<Control-O>", lambda _event: self.select_image())
        self.bind("<Control-Return>", lambda _event: self.start_analysis())
        self.bind("<F5>", lambda _event: self.start_analysis())

    def build_ui(self) -> None:
        root = ttk.Frame(self, padding=22)
        root.pack(fill=tk.BOTH, expand=True)
        header = ttk.Frame(root)
        header.pack(fill=tk.X, pady=(0, 16))
        self.logo = tk.Canvas(header, width=48, height=48, bg=COLORS["bg"], highlightthickness=0)
        self.logo.pack(side=tk.LEFT, padx=(0, 12))
        self.draw_logo()
        title_box = ttk.Frame(header)
        title_box.pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Label(title_box, text="Fish-Agent", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_box, text=APP_SUBTITLE, style="Subtitle.TLabel").pack(anchor="w", pady=(2, 0))
        self.make_button(header, "在线帮助", lambda: webbrowser.open(GITHUB_URL), compact=True).pack(side=tk.RIGHT, pady=(7, 0))

        body = ttk.Frame(root)
        body.pack(fill=tk.BOTH, expand=True)
        body.columnconfigure(0, weight=4)
        body.columnconfigure(1, weight=7)
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        right = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        left.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)

        ttk.Label(left, text="图像输入", style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.canvas = tk.Canvas(left, bg=COLORS["canvas"], highlightthickness=1, highlightbackground=COLORS["line"])
        self.canvas.grid(row=1, column=0, sticky="nsew", pady=12)
        self.canvas.bind("<Configure>", lambda _event: self.redraw_preview())
        self.canvas.bind("<Button-1>", self.canvas_click)

        ttk.Label(left, textvariable=self.image_meta_var, style="Muted.Panel.TLabel").grid(row=2, column=0, sticky="w", pady=(0, 8))

        controls = ttk.Frame(left, style="Panel.TFrame")
        controls.grid(row=3, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        action_row = ttk.Frame(controls, style="Panel.TFrame")
        action_row.grid(row=0, column=0, sticky="ew")
        self.make_button(action_row, "选择图片  Ctrl+O", self.select_image).pack(side=tk.LEFT)
        self.analyze_button = self.make_button(action_row, "开始分析  Ctrl+Enter", self.start_analysis, primary=True)
        self.analyze_button.pack(side=tk.LEFT, padx=8)
        self.make_button(action_row, "清空", self.clear_image, compact=True).pack(side=tk.LEFT)
        self.batch_button = self.make_button(action_row, "▦ 批量分析", self.batch_analysis, compact=True)
        self.batch_button.configure(highlightthickness=1, highlightbackground=COLORS["line"], padx=14)
        self.set_button_state_style(self.batch_button, ghost=True)
        self.batch_button.pack(side=tk.LEFT, padx=(8, 0))
        opacity_box = ttk.Frame(controls, style="Panel.TFrame")
        opacity_box.grid(row=1, column=0, sticky="ew", pady=(12, 0))
        ttk.Label(opacity_box, text="AI 关注区域", style="Muted.Panel.TLabel").pack(side=tk.LEFT, padx=(0, 8))
        self.opacity_slider = self.build_opacity_slider(opacity_box)
        self.opacity_slider.pack(side=tk.LEFT)
        ttk.Label(opacity_box, textvariable=self.opacity_value_var, style="Muted.Panel.TLabel", font=("Microsoft YaHei UI", 9, "bold")).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Label(left, textvariable=self.focus_hint_var, style="Muted.Panel.TLabel").grid(row=4, column=0, sticky="w", pady=(8, 0))
        ttk.Label(left, text="模型：FishFreshNetV1", style="Muted.Panel.TLabel").grid(row=5, column=0, sticky="w", pady=(4, 0))

        ttk.Label(right, text="结果", style="Panel.TLabel", font=("Microsoft YaHei UI", 12, "bold")).grid(row=0, column=0, sticky="w")
        summary = ttk.Frame(right, style="Panel.TFrame")
        summary.grid(row=1, column=0, sticky="ew", pady=(12, 12))
        summary.columnconfigure(0, weight=1)
        summary.columnconfigure(1, weight=1)
        self.level_label = ttk.Label(summary, textvariable=self.level_var, style="Panel.TLabel", font=("Microsoft YaHei UI", 23, "bold"))
        self.level_label.grid(row=0, column=0, sticky="w")
        ttk.Label(summary, textvariable=self.confidence_var, style="Panel.TLabel", font=("Microsoft YaHei UI", 15, "bold")).grid(row=0, column=1, sticky="e")
        self.score_canvas = tk.Canvas(summary, height=18, bg=COLORS["panel"], highlightthickness=0)
        self.score_canvas.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 4))
        self.score_canvas.bind("<Configure>", lambda _event: self.draw_score())
        ttk.Label(summary, textvariable=self.probability_var, style="Panel.TLabel", foreground=COLORS["muted"]).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        self.active_view = "report"
        self.tab_bar = ttk.Frame(right, style="Panel.TFrame")
        self.tab_bar.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.report_tab = self.make_button(self.tab_bar, "评估报告", lambda: self.switch_view("report"), primary=True, compact=True)
        self.report_tab.pack(side=tk.LEFT)
        self.recommend_tab = self.make_button(self.tab_bar, "处理建议", lambda: self.switch_view("recommend"), compact=True)
        self.recommend_tab.pack(side=tk.LEFT, padx=(8, 0))
        self.history_tab = self.make_button(self.tab_bar, "历史记录", lambda: self.switch_view("history"), compact=True)
        self.history_tab.pack(side=tk.LEFT, padx=(8, 0))

        self.card_canvas = tk.Canvas(right, bg=COLORS["panel"], highlightthickness=0)
        self.card_canvas.grid(row=3, column=0, sticky="nsew")
        self.card_scrollbar = ttk.Scrollbar(right, orient=tk.VERTICAL, command=self.card_canvas.yview)
        self.card_scrollbar.grid(row=3, column=1, sticky="ns")
        self.card_canvas.configure(yscrollcommand=self.card_scrollbar.set)
        self.card_frame = ttk.Frame(self.card_canvas, style="Panel.TFrame")
        self.card_window = self.card_canvas.create_window((0, 0), window=self.card_frame, anchor="nw")
        self.card_frame.bind("<Configure>", lambda _event: self.card_canvas.configure(scrollregion=self.card_canvas.bbox("all")))
        self.card_canvas.bind("<Configure>", lambda event: self.card_canvas.itemconfigure(self.card_window, width=event.width))
        self.card_canvas.bind("<Enter>", lambda _event: self.card_canvas.bind_all("<MouseWheel>", self.on_card_mousewheel))
        self.card_canvas.bind("<Leave>", lambda _event: self.card_canvas.unbind_all("<MouseWheel>"))

        export_row = ttk.Frame(right, style="Panel.TFrame")
        export_row.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        self.make_button(export_row, "导出报告", self.show_export_menu, primary=True).pack(side=tk.LEFT)
        self.make_button(export_row, "再次分析", self.start_analysis, compact=True).pack(side=tk.LEFT, padx=8)
        self.loading_bar = ttk.Progressbar(export_row, mode="indeterminate", length=150)
        self.loading_bar.pack(side=tk.LEFT, padx=(8, 0))
        ttk.Label(export_row, textvariable=self.status_var, style="Panel.TLabel", foreground=COLORS["muted"]).pack(side=tk.RIGHT)
        self.render_empty_cards()
        self.draw_empty_canvas()
        self.toast_label = tk.Label(
            self,
            text="",
            bg="#0F9F6E",
            fg="#FFFFFF",
            padx=18,
            pady=10,
            font=("Microsoft YaHei UI", 10, "bold"),
        )

    def draw_logo(self) -> None:
        self.logo.delete("all")
        self.logo.create_oval(3, 3, 45, 45, fill=COLORS["primary"], outline="")
        self.logo.create_oval(10, 10, 38, 38, fill=COLORS["ocean"], outline="")
        self.logo.create_polygon(11, 24, 22, 13, 39, 16, 44, 24, 39, 32, 22, 35, fill="#FFFFFF", outline="")
        self.logo.create_polygon(11, 24, 4, 16, 5, 32, fill="#D9ECFF", outline="")
        self.logo.create_oval(27, 18, 36, 27, fill=COLORS["primary"], outline="")
        self.logo.create_oval(30, 20, 33, 23, fill="#FFFFFF", outline="")

    def show_toast(self, message: str, *, kind: str = "success", duration: int = 2600) -> None:
        bg = "#0F9F6E" if kind == "success" else COLORS["primary"]
        if kind == "warning":
            bg = "#B45309"
        self.toast_label.configure(text=message, bg=bg)
        self.toast_label.place(relx=0.5, rely=0.035, anchor="n")
        self.toast_label.lift()
        self.after(duration, self.toast_label.place_forget)

    def canvas_click(self, _event: tk.Event) -> None:
        if self.preview_image is None:
            self.select_image()

    def draw_score(self) -> None:
        self.score_canvas.delete("all")
        width = max(self.score_canvas.winfo_width(), 240)
        height = 18
        level = self.level_var.get()
        color = LEVEL_COLORS.get(level, COLORS["primary"])
        self.score_canvas.create_rectangle(0, 4, width, 14, fill=COLORS["primary_soft"], outline="")
        if self.score_value > 0:
            self.score_canvas.create_rectangle(0, 4, int(width * self.score_value), 14, fill=color, outline="")

    def on_card_mousewheel(self, event: tk.Event) -> None:
        if self.card_canvas.winfo_exists():
            first, last = self.card_canvas.yview()
            delta = int(-1 * (event.delta / 120))
            if (delta < 0 and first <= 0.001) or (delta > 0 and last >= 0.999):
                return "break"
            self.card_canvas.yview_scroll(delta, "units")
        return "break"

    def switch_view(self, view: str) -> None:
        self.active_view = view
        self.update_tab_buttons()
        if view == "history":
            self.render_history_cards()
        elif not self.result:
            self.render_empty_cards()
        elif view == "recommend":
            self.render_recommendation_cards(self.result)
        else:
            self.render_report_cards(self.result)

    def update_tab_buttons(self) -> None:
        mapping = {
            "report": self.report_tab,
            "recommend": self.recommend_tab,
            "history": self.history_tab,
        }
        for name, button in mapping.items():
            self.set_button_state_style(button, active=name == self.active_view)

    def clear_cards(self) -> None:
        for child in self.card_frame.winfo_children():
            child.destroy()

    def card(self, title: str, accent: str | None = None, *, bg: str = "#FFFFFF") -> tk.Frame:
        outer = tk.Frame(self.card_frame, bg=COLORS["line_soft"], bd=0)
        outer.pack(fill=tk.X, pady=(0, 12))
        inner = tk.Frame(outer, bg=bg, padx=16, pady=14)
        inner.pack(fill=tk.X, padx=1, pady=1)
        title_line = tk.Frame(inner, bg=bg)
        title_line.pack(fill=tk.X, pady=(0, 8))
        if accent:
            tk.Frame(title_line, width=5, height=22, bg=accent).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(
            title_line,
            text=title,
            bg=bg,
            fg=COLORS["text"],
            font=("Microsoft YaHei UI", 12, "bold"),
        ).pack(side=tk.LEFT, anchor="center")
        return inner

    def add_text(self, parent: tk.Frame, text: str, *, muted: bool = False, bold: bool = False) -> tk.Label:
        bg = str(parent.cget("bg"))
        label = tk.Label(
            parent,
            text=text,
            bg=bg,
            fg=COLORS["muted"] if muted else COLORS["text"],
            justify=tk.LEFT,
            anchor="w",
            wraplength=max(520, self.card_canvas.winfo_width() - 72),
            font=("Microsoft YaHei UI", 10, "bold" if bold else "normal"),
        )
        label.pack(fill=tk.X, anchor="w", pady=(3, 5))
        return label

    def add_tag(self, parent: tk.Frame, text: str, color: str = "#E7F2FF", fg: str | None = None) -> None:
        tk.Label(
            parent,
            text=text,
            bg=color,
            fg=fg or COLORS["text"],
            padx=10,
            pady=5,
            font=("Microsoft YaHei UI", 9, "bold"),
        ).pack(side=tk.LEFT, padx=(0, 8), pady=(2, 6))

    def render_empty_cards(self) -> None:
        self.clear_cards()
        card = self.card("等待分析", COLORS["primary"])
        self.add_text(card, "分析结果将在此处显示。请选择一张鱼眼图片，或使用 Ctrl+O 快速打开。", bold=True)
        self.add_text(card, f"V{APP_VERSION} 会在完成分析后展示图像质量标签、置信度仪表盘、概率分布、积极指标和注意事项。", muted=True)
        ghost = self.card("工作流预览", COLORS["ocean"])
        for text in ["1. 上传鱼眼图片", "2. FishFreshNetV1 本地推理", "3. AI 关注区域热力图", "4. 生成报告与处理建议"]:
            self.add_text(ghost, text, muted=True)

    def render_report_cards(self, result: dict[str, Any]) -> None:
        self.clear_cards()
        report = result.get("freshness_report", {})
        summary = report.get("summary", {})
        detail = report.get("detailed_analysis", {})
        indicators = report.get("quality_indicators", {})
        level = get_level(result)
        color = LEVEL_COLORS.get(level, COLORS["primary"])

        conclusion = self.card("综合结论", color)
        self.add_text(conclusion, summary.get("overall_assessment") or result.get("description") or "暂无综合结论。", bold=True)
        self.add_text(conclusion, f"模型：{result.get('analysis_method_label', 'FishFreshNetV1')}    生成时间：{result.get('created_at', '')}", muted=True)

        gauge = self.card("置信度仪表盘", color)
        self.draw_gauge_card(gauge, get_confidence(result), color)

        quality = self.card("图像质量标签", COLORS["primary"])
        tag_row = tk.Frame(quality, bg="#FFFFFF")
        tag_row.pack(fill=tk.X)
        q = result.get("quality_check", {})
        details = q.get("details", {})
        self.add_tag(tag_row, f"分辨率：{details.get('width', '--')} x {details.get('height', '--')}")
        brightness = details.get("brightness", "--")
        contrast = details.get("contrast", "--")
        self.add_tag(tag_row, f"亮度：{brightness}", "#EAF7EF")
        self.add_tag(tag_row, f"对比度：{contrast}", "#FFF7E6")
        for issue in q.get("issues", []):
            self.add_text(quality, f"⚠ {issue}", muted=True)

        probs = probability_items(result)
        if probs:
            probability = self.card("概率分布", COLORS["primary"])
            self.draw_probability_bar(probability, probs)

        detail_card = self.card("详细分析", COLORS["ocean"])
        if detail:
            for value in detail.values():
                if value:
                    self.add_text(detail_card, f"• {value}")
        else:
            self.add_text(detail_card, "暂无详细分析。", muted=True)

        positives = indicators.get("positive_signs", [])
        warnings = indicators.get("warning_signs", [])
        if positives:
            positive_card = self.card("积极指标", "#0F9F6E")
            for item in positives:
                self.add_text(positive_card, f"✓ {item}")
        if warnings:
            warning_card = self.card("注意事项", "#B45309")
            for item in warnings:
                self.add_text(warning_card, f"! {item}", muted=True)

    def draw_gauge_card(self, parent: tk.Frame, confidence: float, color: str) -> None:
        canvas = tk.Canvas(parent, width=244, height=172, bg="#FFFFFF", highlightthickness=0)
        canvas.pack(anchor="center", pady=(10, 4))
        center_x = 122
        bbox = (34, 26, 210, 202)
        canvas.create_arc(*bbox, start=180, extent=-180, style=tk.ARC, width=16, outline=COLORS["primary_soft"])
        canvas.create_arc(*bbox, start=180, extent=-180 * max(0.0, min(1.0, confidence)), style=tk.ARC, width=16, outline=color)
        number = f"{confidence * 100:.0f}"
        number_font = tkfont.Font(family="Microsoft YaHei UI", size=22, weight="bold")
        percent_font = tkfont.Font(family="Microsoft YaHei UI", size=15, weight="bold")
        gap = 2
        total_width = number_font.measure(number) + gap + percent_font.measure("%")
        start_x = center_x - total_width / 2
        value_y = 94
        canvas.create_text(start_x, value_y, text=number, fill=color, font=number_font, anchor="w")
        canvas.create_text(start_x + number_font.measure(number) + gap, value_y + 2, text="%", fill=color, font=percent_font, anchor="w")
        canvas.create_text(center_x, 134, text="模型置信度", fill=COLORS["muted"], font=("Microsoft YaHei UI", 10))

    def draw_probability_bar(self, parent: tk.Frame, probs: dict[str, float]) -> None:
        canvas = tk.Canvas(parent, height=112, bg="#FFFFFF", highlightthickness=0)
        canvas.pack(fill=tk.X, pady=(8, 0))
        total = sum(max(value, 0.0) for value in probs.values()) or 1.0
        width = max(480, self.card_canvas.winfo_width() - 80)
        x = 0
        colors = ["#0F9F6E", "#2368D8", "#C2410C", "#64748B"]
        for index, (name, value) in enumerate(probs.items()):
            segment = width * max(value, 0.0) / total
            canvas.create_rectangle(x, 14, x + segment, 40, fill=colors[index % len(colors)], outline="")
            x += segment
        legend_font = tkfont.Font(family="Microsoft YaHei UI", size=9)
        legend_x = 0
        for index, (name, value) in enumerate(probs.items()):
            label = f"{name} {value:.1%}"
            canvas.create_rectangle(legend_x, 74, legend_x + 13, 87, fill=colors[index % len(colors)], outline="")
            canvas.create_text(legend_x + 24, 80, text=label, anchor="w", fill=COLORS["muted"], font=("Microsoft YaHei UI", 9))
            legend_x += max(172, 13 + 11 + legend_font.measure(label) + 32)

    def render_recommendation_cards(self, result: dict[str, Any]) -> None:
        self.clear_cards()
        rec = result.get("recommendations", {})
        sections = [
            ("储存建议", rec.get("storage", {}), ["temperature", "method", "duration", "tips"]),
            ("食用建议", rec.get("consumption", {}), ["timeline", "cooking_methods", "flavor_tips"]),
            ("处理建议", rec.get("handling", {}), ["preparation_steps", "cutting_advice", "deodorizing"]),
            ("安全提醒", rec.get("safety_warnings", {}), ["risk_level", "warnings"]),
        ]
        for title, data, keys in sections:
            if not data:
                continue
            section = self.card(title, COLORS["ocean"] if title != "安全提醒" else "#B45309")
            for key in keys:
                value = data.get(key)
                if isinstance(value, list):
                    for item in value:
                        self.add_text(section, f"• {item}")
                elif value:
                    self.add_text(section, f"{value}")
        practices = rec.get("best_practices", [])
        if practices:
            best = self.card("最佳实践", "#0F9F6E")
            self.add_text(best, "、".join(practices))

    def render_history_cards(self) -> None:
        self.clear_cards()
        self.history_thumbnails = []
        if not self.history:
            empty = self.card("暂无历史记录", COLORS["primary"])
            self.add_text(empty, "本次打开软件后的分析记录会显示在这里。", muted=True)
            return
        for index, item in enumerate(self.history[:20], start=1):
            level = get_level(item)
            history_bg = COLORS["panel_alt"]
            history_card = self.card(f"#{index} {level}", LEVEL_COLORS.get(level, COLORS["primary"]), bg=history_bg)
            row = tk.Frame(history_card, bg=history_bg)
            row.pack(fill=tk.X)
            thumb_label = tk.Label(row, bg="#EEF4FA", width=50, height=50)
            thumb_label.pack(side=tk.LEFT, padx=(0, 14), anchor="n")
            thumb = self.make_thumbnail(item.get("image_path"), 50)
            if thumb:
                self.history_thumbnails.append(thumb)
                thumb_label.configure(image=thumb)
            else:
                thumb_label.configure(text="图像", fg=COLORS["muted"], font=("Microsoft YaHei UI", 8))
            text_col = tk.Frame(row, bg=history_bg)
            text_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(
                text_col,
                text=f"置信度：{get_confidence(item):.1%}    时间：{item.get('created_at', '')}",
                bg=history_bg,
                fg=COLORS["text"],
                anchor="w",
                justify=tk.LEFT,
                font=("Microsoft YaHei UI", 10, "bold"),
            ).pack(fill=tk.X, anchor="w")
            tk.Label(
                text_col,
                text=str(item.get("image_path", "")),
                bg=history_bg,
                fg=COLORS["muted"],
                anchor="w",
                justify=tk.LEFT,
                wraplength=max(420, self.card_canvas.winfo_width() - 170),
                font=("Microsoft YaHei UI", 9),
            ).pack(fill=tk.X, anchor="w", pady=(2, 0))
            self.make_button(text_col, "载入该结果", lambda result=item: self.load_history_result(result), compact=True).pack(anchor="w", pady=(8, 0))

    def make_thumbnail(self, path_text: str | None, size: int) -> ImageTk.PhotoImage | None:
        if not path_text or not Path(path_text).exists():
            return None
        try:
            image = ImageOps.exif_transpose(Image.open(path_text).convert("RGB"))
            image.thumbnail((size, size), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (size, size), "#EEF4FA")
            canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
            return ImageTk.PhotoImage(canvas)
        except Exception:
            return None

    def load_history_result(self, result: dict[str, Any]) -> None:
        self.analysis_finished(result)

    def draw_empty_canvas(self) -> None:
        self.canvas.delete("all")
        w, h = max(self.canvas.winfo_width(), 420), max(self.canvas.winfo_height(), 420)
        margin = 34
        self.canvas.create_rectangle(
            margin,
            margin,
            w - margin,
            h - margin,
            outline=COLORS["primary"],
            dash=(8, 6),
            width=2,
        )
        self.canvas.create_oval(w // 2 - 34, h // 2 - 92, w // 2 + 34, h // 2 - 24, fill=COLORS["primary_soft"], outline="")
        self.canvas.create_line(w // 2, h // 2 - 76, w // 2, h // 2 - 42, fill=COLORS["primary"], width=4)
        self.canvas.create_line(w // 2 - 14, h // 2 - 58, w // 2, h // 2 - 76, w // 2 + 14, h // 2 - 58, fill=COLORS["primary"], width=4)
        self.canvas.create_text(
            w // 2,
            h // 2 - 4,
            text="点击选择鱼眼图片",
            fill=COLORS["text"],
            font=("Microsoft YaHei UI", 17, "bold"),
        )
        self.canvas.create_text(
            w // 2,
            h // 2 + 34,
            text="支持高清图片，Ctrl+O 快速打开",
            fill=COLORS["muted"],
            font=("Microsoft YaHei UI", 11),
        )
        self.canvas.create_rectangle(w // 2 - 70, h // 2 + 62, w // 2 + 70, h // 2 + 102, fill=COLORS["primary"], outline="")
        self.canvas.create_text(w // 2, h // 2 + 82, text="选择图片", fill="#FFFFFF", font=("Microsoft YaHei UI", 11, "bold"))

    def select_image(self) -> None:
        path_text = filedialog.askopenfilename(title="选择鱼眼图片", filetypes=IMAGE_FILE_TYPES)
        if path_text:
            self.load_image(Path(path_text))

    def load_image(self, path: Path) -> None:
        valid, error = validate_image(path)
        if not valid:
            messagebox.showwarning("图片不可用", error)
            return
        self.selected_image = path
        self.preview_image = ImageOps.exif_transpose(Image.open(path).convert("RGB"))
        self.gradcam_image = None
        self.result = None
        self.show_gradcam.set(False)
        self.heatmap_opacity.set(55.0)
        self.update_opacity_value()
        self.level_var.set("分析结果将在此处显示")
        self.confidence_var.set("--")
        self.probability_var.set("")
        self.score_value = 0.0
        self.status_var.set(path.name)
        size_mb = path.stat().st_size / 1024 / 1024
        width, height = self.preview_image.size
        self.image_meta_var.set(f"{path.name} · {width} x {height} · {size_mb:.2f} MB")
        self.render_empty_cards()
        self.redraw_preview()
        self.draw_score()

    def redraw_preview(self) -> None:
        image = self.preview_image
        if self.preview_image is not None and self.gradcam_image is not None and self.heatmap_opacity.get() > 0:
            alpha = max(0.0, min(1.0, self.heatmap_opacity.get() / 100.0))
            grad = self.gradcam_image.resize(self.preview_image.size, Image.Resampling.LANCZOS)
            image = Image.blend(self.preview_image, grad, alpha)
        if image is None:
            self.draw_empty_canvas()
            return
        w, h = max(self.canvas.winfo_width(), 200), max(self.canvas.winfo_height(), 200)
        preview = image.copy()
        preview.thumbnail((w - 24, h - 24), Image.Resampling.LANCZOS)
        self.tk_preview = ImageTk.PhotoImage(preview)
        self.canvas.delete("all")
        self.canvas.create_image(w // 2, h // 2, image=self.tk_preview)

    def clear_image(self) -> None:
        self.selected_image = None
        self.preview_image = None
        self.gradcam_image = None
        self.result = None
        self.show_gradcam.set(False)
        self.heatmap_opacity.set(55.0)
        self.update_opacity_value()
        self.level_var.set("分析结果将在此处显示")
        self.confidence_var.set("--")
        self.probability_var.set("")
        self.score_value = 0.0
        self.status_var.set("就绪")
        self.image_meta_var.set("支持 JPG、PNG、BMP、WEBP、TIFF 格式")
        self.render_empty_cards()
        self.draw_score()
        self.draw_empty_canvas()

    def start_analysis(self) -> None:
        if self.selected_image is None:
            messagebox.showinfo("未选择图片", "请先选择一张鱼眼图片。")
            return
        self.analyze_button.configure(state=tk.DISABLED)
        self.status_var.set("正在分析，请稍候...")
        self.level_var.set("分析中")
        self.score_value = 0.0
        self.draw_score()
        self.loading_bar.start(12)
        threading.Thread(target=self.analysis_worker, args=(self.selected_image,), daemon=True).start()

    def analysis_worker(self, image_path: Path) -> None:
        try:
            result = run_core_analysis(image_path)
            self.after(0, lambda: self.analysis_finished(result))
        except Exception as exc:
            details = traceback.format_exc()
            logger.error("Analysis failed: %s\n%s", exc, details)
            self.after(0, lambda: self.analysis_failed(str(exc), details))

    def analysis_finished(self, result: dict[str, Any]) -> None:
        self.result = result
        level = get_level(result)
        confidence = get_confidence(result)
        self.level_var.set(level)
        self.level_label.configure(foreground=LEVEL_COLORS.get(level, COLORS["primary"]))
        self.confidence_var.set(f"{confidence:.1%}")
        self.score_value = max(0.0, min(1.0, confidence))
        self.draw_score()
        probs = probability_items(result)
        self.probability_var.set("  ".join(f"{name} {value:.1%}" for name, value in probs.items()))
        self.history.insert(0, result)
        self.autosave_result(result)
        gradcam_path = result.get("gradcam_image_path")
        if gradcam_path and Path(gradcam_path).exists():
            self.gradcam_image = Image.open(gradcam_path).convert("RGB")
            self.show_gradcam.set(True)
            self.heatmap_opacity.set(55.0)
            self.update_opacity_value()
        self.redraw_preview()
        if self.active_view == "recommend":
            self.render_recommendation_cards(result)
        elif self.active_view == "history":
            self.render_history_cards()
        else:
            self.render_report_cards(result)
        self.status_var.set("分析完成")
        self.loading_bar.stop()
        self.analyze_button.configure(state=tk.NORMAL)

    def analysis_failed(self, message: str, details: str) -> None:
        self.level_var.set("分析失败")
        self.confidence_var.set("--")
        self.status_var.set("分析失败")
        logger.error("Analysis details:\n%s", details)
        self.loading_bar.stop()
        self.analyze_button.configure(state=tk.NORMAL)
        messagebox.showerror("分析失败", message)

    def batch_analysis(self) -> None:
        paths = filedialog.askopenfilenames(title="选择多张鱼眼图片", filetypes=IMAGE_FILE_TYPES)
        if not paths:
            return
        self.status_var.set(f"批量分析中：0/{len(paths)}")
        self.loading_bar.start(12)
        threading.Thread(target=self.batch_worker, args=([Path(path) for path in paths],), daemon=True).start()

    def batch_worker(self, paths: list[Path]) -> None:
        results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        for index, path in enumerate(paths, start=1):
            try:
                result = run_core_analysis(path)
                result["app_version"] = APP_VERSION
                results.append(result)
                self.after(0, lambda i=index, total=len(paths): self.status_var.set(f"批量分析中：{i}/{total}"))
            except Exception as exc:
                errors.append({"image": str(path), "error": str(exc)})
                logger.error("Batch item failed: %s\n%s", exc, traceback.format_exc())
        self.after(0, lambda: self.batch_finished(results, errors))

    def batch_finished(self, results: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
        self.loading_bar.stop()
        summary = {
            "app_version": APP_VERSION,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total": len(results) + len(errors),
            "success": len(results),
            "failed": len(errors),
            "results": results,
            "errors": errors,
        }
        path = Path(tempfile.gettempdir()) / "Fish-Agent" / "batch"
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"fish_agent_batch_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if results:
            self.analysis_finished(results[0])
        self.status_var.set(f"批量完成：成功 {len(results)}，失败 {len(errors)}")
        self.show_batch_workspace(results, errors, out)
        self.show_toast(f"批量分析完成：成功 {len(results)}，失败 {len(errors)}")

    def show_batch_workspace(self, results: list[dict[str, Any]], errors: list[dict[str, str]], summary_path: Path) -> None:
        window = tk.Toplevel(self)
        window.title("Fish-Agent 批量分析工作台")
        window.geometry("920x560")
        window.configure(bg=COLORS["bg"])
        header = ttk.Frame(window, padding=16)
        header.pack(fill=tk.X)
        ttk.Label(header, text="批量分析工作台", style="Title.TLabel").pack(side=tk.LEFT)
        self.make_button(header, "导出 Excel 汇总表", lambda: self.export_batch_csv(results, errors), primary=True, compact=True).pack(side=tk.RIGHT)
        ttk.Label(window, text=f"汇总 JSON：{summary_path}", background=COLORS["bg"], foreground=COLORS["muted"]).pack(anchor="w", padx=18)

        table = ttk.Treeview(window, columns=("status", "level", "confidence", "path"), show="headings", height=18)
        table.heading("status", text="状态")
        table.heading("level", text="结果")
        table.heading("confidence", text="置信度")
        table.heading("path", text="图片路径")
        table.column("status", width=90, anchor="center")
        table.column("level", width=110, anchor="center")
        table.column("confidence", width=90, anchor="center")
        table.column("path", width=580)
        table.pack(fill=tk.BOTH, expand=True, padx=18, pady=14)

        for item in results:
            table.insert("", tk.END, values=("已完成", get_level(item), f"{get_confidence(item):.1%}", item.get("image_path", "")))
        for item in errors:
            table.insert("", tk.END, values=("失败", item.get("error", ""), "--", item.get("image", "")))

    def export_batch_csv(self, results: list[dict[str, Any]], errors: list[dict[str, str]]) -> None:
        default_name = f"fish_agent_batch_{datetime.now():%Y%m%d_%H%M%S}.csv"
        path_text = filedialog.asksaveasfilename(title="导出 Excel 汇总表", defaultextension=".csv", initialfile=default_name, filetypes=[("CSV 文件", "*.csv")])
        if not path_text:
            return
        lines = ["status,level,confidence,image_path,error"]
        for item in results:
            lines.append(f"success,{get_level(item)},{get_confidence(item):.4f},\"{item.get('image_path', '')}\",")
        for item in errors:
            lines.append(f"failed,,,\"{item.get('image', '')}\",\"{item.get('error', '')}\"")
        Path(path_text).write_text("\n".join(lines), encoding="utf-8-sig")
        self.show_toast(f"批量汇总表已导出：{Path(path_text).name}")

    def show_export_menu(self) -> None:
        menu = tk.Menu(self, tearoff=False)
        menu.add_command(label="导出为 PDF 报告", command=self.export_pdf)
        menu.add_command(label="导出为 JSON 数据", command=self.export_json)
        try:
            menu.tk_popup(self.winfo_pointerx(), self.winfo_pointery())
        finally:
            menu.grab_release()

    def autosave_result(self, result: dict[str, Any]) -> None:
        try:
            history_dir = Path(tempfile.gettempdir()) / "Fish-Agent" / "history"
            history_dir.mkdir(parents=True, exist_ok=True)
            path = history_dir / f"fish_agent_{datetime.now():%Y%m%d_%H%M%S}.json"
            result["autosave_path"] = str(path)
            path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception as exc:
            logger.warning("Autosave failed: %s", exc)

    def export_pdf(self) -> None:
        if not self.result:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        default_name = f"fish_agent_report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
        path_text = filedialog.asksaveasfilename(title="保存 PDF 报告", defaultextension=".pdf", initialfile=default_name, filetypes=[("PDF 文件", "*.pdf")])
        if not path_text:
            return
        try:
            saved = export_pdf_report(self.result, Path(path_text))
            self.show_toast(f"PDF 报告已导出：{saved.name}")
        except Exception as exc:
            logger.error("PDF export failed: %s\n%s", exc, traceback.format_exc())
            messagebox.showerror("保存失败", str(exc))

    def export_json(self) -> None:
        if not self.result:
            messagebox.showinfo("暂无结果", "请先完成一次分析。")
            return
        default_name = f"fish_agent_result_{datetime.now():%Y%m%d_%H%M%S}.json"
        path_text = filedialog.asksaveasfilename(title="保存 JSON 结果", defaultextension=".json", initialfile=default_name, filetypes=[("JSON 文件", "*.json")])
        if not path_text:
            return
        Path(path_text).write_text(json.dumps(self.result, ensure_ascii=False, indent=2), encoding="utf-8")
        self.show_toast(f"JSON 结果已导出：{Path(path_text).name}")


def run_cli(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=f"Fish-Agent V{APP_VERSION} launcher")
    parser.add_argument("--analyze", metavar="IMAGE", help="Analyze an image with the embedded core")
    parser.add_argument("--output", metavar="JSON", help="Write JSON result to file")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON")
    args = parser.parse_args(argv)
    if not args.analyze:
        parser.print_help()
        return 1
    result = run_core_analysis(Path(args.analyze))
    text = json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None)
    if args.output:
        Path(args.output).resolve().write_text(text, encoding="utf-8")
    else:
        print(text)
    return 0


def main() -> None:
    set_dpi_awareness()
    if len(sys.argv) > 1:
        raise SystemExit(run_cli(sys.argv[1:]))
    app = FishAgentApp()
    app.mainloop()


if __name__ == "__main__":
    main()
