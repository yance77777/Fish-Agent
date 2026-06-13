"""Fish-Agent local HTTP and CLI entry point."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from types import SimpleNamespace
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

from graphs.graph import main_graph
from graphs.state import (
    BackupModelReviewInput,
    EnhancedGradCAMInput,
    EnhancedQualityCheckInput,
    FeedbackLoopInput,
    FishRegionDetectionInput,
    FreshnessClassificationInput,
    GraphInput,
    ImagePreprocessInput,
    ImageUploadInput,
    LowConfidencePromptInput,
    MultiRegionFusionInput,
    QualityUnqualifiedInput,
    RecommendationGenerationInput,
    ReportGenerationInput,
    ResultVisualizationInput,
    StructuredAnalysisInput,
    TemporalAnalysisInput,
)
from graphs.nodes.backup_model_review_node import backup_model_review_node
from graphs.nodes.enhanced_gradcam_node import enhanced_gradcam_node
from graphs.nodes.enhanced_quality_check_node import enhanced_quality_check_node
from graphs.nodes.feedback_loop_node import feedback_loop_node
from graphs.nodes.fish_region_detection_node import fish_region_detection_node
from graphs.nodes.freshness_classification_node import freshness_classification_node
from graphs.nodes.image_preprocess_node import image_preprocess_node
from graphs.nodes.image_upload_node import image_upload_node
from graphs.nodes.low_confidence_prompt_node import low_confidence_prompt_node
from graphs.nodes.multi_region_fusion_node import multi_region_fusion_node
from graphs.nodes.quality_unqualified_node import quality_unqualified_node
from graphs.nodes.recommendation_generation_node import recommendation_generation_node
from graphs.nodes.report_generation_node import report_generation_node
from graphs.nodes.result_visualization_node import result_visualization_node
from graphs.nodes.structured_analysis_node import structured_analysis_node
from graphs.nodes.temporal_analysis_node import temporal_analysis_node
from utils.file.file import File


logging.basicConfig(level=os.getenv("FISH_AGENT_LOG_LEVEL", "INFO"))
logger = logging.getLogger("fish-agent")

app = FastAPI(
    title="Fish-Agent API",
    description="水产品新鲜度智能评估工作流",
    version="3.4.0",
)


NodeFunc = Callable[[Any, RunnableConfig, Any], Any]
NODE_SPECS: dict[str, tuple[NodeFunc, type[BaseModel], dict[str, Any]]] = {
    "image_upload": (image_upload_node, ImageUploadInput, {}),
    "enhanced_quality_check": (
        enhanced_quality_check_node,
        EnhancedQualityCheckInput,
        {"llm_cfg": "config/enhanced_quality_check_llm_cfg.json"},
    ),
    "quality_unqualified": (quality_unqualified_node, QualityUnqualifiedInput, {}),
    "image_preprocess": (image_preprocess_node, ImagePreprocessInput, {}),
    "fish_region_detection": (
        fish_region_detection_node,
        FishRegionDetectionInput,
        {"llm_cfg": "config/fish_region_detection_llm_cfg.json"},
    ),
    "freshness_classification": (
        freshness_classification_node,
        FreshnessClassificationInput,
        {"llm_cfg": "config/freshness_classification_llm_cfg.json"},
    ),
    "multi_region_fusion": (
        multi_region_fusion_node,
        MultiRegionFusionInput,
        {"llm_cfg": "config/multi_region_fusion_llm_cfg.json"},
    ),
    "backup_model_review": (
        backup_model_review_node,
        BackupModelReviewInput,
        {"llm_cfg": "config/backup_model_review_llm_cfg.json"},
    ),
    "low_confidence_prompt": (low_confidence_prompt_node, LowConfidencePromptInput, {}),
    "enhanced_gradcam": (
        enhanced_gradcam_node,
        EnhancedGradCAMInput,
        {"llm_cfg": "config/enhanced_gradcam_llm_cfg.json"},
    ),
    "structured_analysis": (
        structured_analysis_node,
        StructuredAnalysisInput,
        {"llm_cfg": "config/structured_analysis_llm_cfg.json"},
    ),
    "temporal_analysis": (
        temporal_analysis_node,
        TemporalAnalysisInput,
        {"llm_cfg": "config/temporal_analysis_llm_cfg.json"},
    ),
    "report_generation": (
        report_generation_node,
        ReportGenerationInput,
        {"llm_cfg": "config/report_generation_llm_cfg.json"},
    ),
    "recommendation_generation": (
        recommendation_generation_node,
        RecommendationGenerationInput,
        {"llm_cfg": "config/recommendation_generation_llm_cfg.json"},
    ),
    "result_visualization": (result_visualization_node, ResultVisualizationInput, {}),
    "feedback_loop": (feedback_loop_node, FeedbackLoopInput, {}),
}


def normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    image = normalized.get("image")
    if isinstance(image, str):
        normalized["image"] = File(url=image, file_type="image")
    return normalized


def to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    return value


async def run_workflow(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = normalize_payload(payload)
    result = await asyncio.to_thread(main_graph.invoke, normalized)
    return to_jsonable(result)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok", "service": "Fish-Agent"}


@app.post("/run")
async def http_run(request: Request) -> dict[str, Any]:
    try:
        payload = await request.json()
        return await run_workflow(payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Workflow run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/stream_run")
async def http_stream_run(request: Request) -> StreamingResponse:
    try:
        payload = normalize_payload(await request.json())
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc

    async def events():
        try:
            for chunk in main_graph.stream(payload):
                yield f"data: {json.dumps(to_jsonable(chunk), ensure_ascii=False)}\n\n"
            yield 'data: {"status":"done"}\n\n'
        except Exception as exc:
            logger.exception("Streaming run failed")
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@app.post("/node_run/{node_id}")
async def http_node_run(node_id: str, request: Request) -> dict[str, Any]:
    if node_id not in NODE_SPECS:
        raise HTTPException(status_code=404, detail=f"Unknown node: {node_id}")
    try:
        payload = await request.json()
        return run_node(node_id, payload)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
    except Exception as exc:
        logger.exception("Node run failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/graph_parameter")
async def graph_parameter() -> dict[str, Any]:
    return {
        "input_schema": GraphInput.model_json_schema(),
        "nodes": list(NODE_SPECS),
        "code": 0,
        "msg": "",
    }


def run_node(node_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    node_func, input_cls, metadata = NODE_SPECS[node_id]
    state = input_cls.model_validate(payload)
    runtime = SimpleNamespace(context=None)
    result = node_func(state, {"metadata": metadata}, runtime)
    return to_jsonable(result)


def parse_input(input_str: str) -> dict[str, Any]:
    if not input_str:
        return {}
    try:
        parsed = json.loads(input_str)
        if isinstance(parsed, dict):
            return parsed
        return {"image": parsed}
    except json.JSONDecodeError:
        return {"image": input_str}


def start_http_server(port: int) -> None:
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False, workers=1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fish-Agent local runner")
    parser.add_argument("-m", choices=["http", "flow", "node"], default="http", help="Run mode")
    parser.add_argument("-n", default="", help="Node ID for node mode")
    parser.add_argument("-p", type=int, default=int(os.getenv("PORT", "5000")), help="HTTP port")
    parser.add_argument("-i", default="", help="Input JSON or image path/URL")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.m == "http":
        start_http_server(args.p)
        return

    payload = parse_input(args.i)
    if args.m == "flow":
        result = main_graph.invoke(normalize_payload(payload))
    else:
        if not args.n:
            raise SystemExit("node mode requires -n <node_id>")
        result = run_node(args.n, payload)
    print(json.dumps(to_jsonable(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
