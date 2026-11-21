"""Flask HTTP interface for the CineComic-MML pipeline."""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

from src.web_jobs import (
    JOB_ROOT,
    MAX_UPLOAD_BYTES,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    STAGES,
    STAGE_METHODS,
    build_processing_report,
    create_job,
    get_job,
    stop_job,
)


app = Flask(__name__, static_folder=str(PROJECT_ROOT / "web"), static_url_path="/assets")
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES + 2 * 1024 * 1024


def _public_state(state: dict[str, Any]) -> dict[str, Any]:
    """Return the browser-safe subset of job state; never expose paths or tokens."""
    settings = state.get("settings", {})
    return {
        "job_id": state.get("job_id"),
        "filename": state.get("filename"),
        "status": state.get("status"),
        "created_at": state.get("created_at"),
        "video_info": state.get("video_info", {}),
        "settings": {
            "language": settings.get("language"),
            "model_size": settings.get("model_size"),
            "device": settings.get("device"),
            "panels_per_page": settings.get("panels_per_page"),
            "frame_scan": settings.get("frame_scan", "fast"),
        },
        "completed_stages": state.get("completed_stages", []),
        "running_stages": state.get("_running_stages", []),
        "stage_statuses": state.get("stage_statuses", {}),
        "stage_reasons": state.get("stage_reasons", {}),
        "dialogue_detected": state.get("dialogue_detected"),
        "stage_logs": state.get("stage_logs", {}),
        "progress": state.get("progress", 0),
        "current_stage": state.get("current_stage"),
        "failed_stage": state.get("failed_stage"),
        "phase5_progress": state.get("phase5_progress"),
        "error": state.get("error"),
        "warning": state.get("warning"),
        "log_tail": state.get("log_tail", ""),
        "stages": [
            {
                "id": stage,
                "method": STAGE_METHODS[stage],
                "status": state.get("stage_statuses", {}).get(stage, (
                    "completed" if stage in state.get("completed_stages", [])
                    else "failed" if stage == state.get("failed_stage")
                    else "running" if stage in state.get("_running_stages", []) and state.get("status") in {"running", "starting"}
                    else "queued"
                )),
                "reason": state.get("stage_reasons", {}).get(stage),
            }
            for stage in STAGES
        ],
    }


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/healthz")
def health_check():
    return jsonify({"status": "ok"})


@app.get("/api/config")
def config():
    import os

    return jsonify({
        "max_upload_bytes": MAX_UPLOAD_BYTES,
        "max_duration_seconds": 300,
        "hf_token_configured": bool(os.environ.get("HF_TOKEN", "").strip()),
    })


@app.post("/api/jobs")
def create_pipeline_job():
    video = request.files.get("video")
    if video is None:
        return jsonify({"error": "Choose a video file first."}), 400
    payload = video.read(MAX_UPLOAD_BYTES + 1)
    if len(payload) > MAX_UPLOAD_BYTES:
        return jsonify({"error": "Video exceeds the 512 MB upload limit."}), 413
    try:
        state = create_job(
            payload,
            video.filename or "video.mp4",
            {
                "language": request.form.get("language", "en"),
                "model_size": request.form.get("model_size", "small"),
                "device": request.form.get("device", "auto"),
                "panels_per_page": request.form.get("panels_per_page", "6"),
            },
            hf_token=request.form.get("hf_token", ""),
            allow_model_downloads=request.form.get("allow_model_downloads") == "true",
        )
    except (RuntimeError, ValueError, OSError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(_public_state(state)), 201


@app.get("/api/jobs")
def list_pipeline_jobs():
    jobs = []
    for state_path in JOB_ROOT.glob("*/job.json"):
        state = _read_json(state_path)
        if isinstance(state, dict):
            jobs.append(state)
    jobs.sort(key=lambda item: item.get("created_at", ""), reverse=True)
    return jsonify([_public_state(item) for item in jobs[:30]])


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    try:
        return jsonify(_public_state(get_job(job_id)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Job not found."}), 404


@app.post("/api/jobs/<job_id>/stop")
def stop_pipeline_job(job_id: str):
    try:
        return jsonify(_public_state(stop_job(job_id)))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Job not found."}), 404
    except OSError as exc:
        return jsonify({"error": f"Could not stop processing: {exc}"}), 500


@app.get("/api/jobs/<job_id>/pages")
def comic_pages(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        return jsonify({"error": "Invalid job ID."}), 400
    if not (JOB_ROOT / job_id / "job.json").is_file():
        return jsonify({"error": "Job not found."}), 404
    pages = sorted((OUTPUT_ROOT / job_id).glob("page_*.png"))
    return jsonify([page.name for page in pages])


@app.get("/api/jobs/<job_id>/pages/<path:filename>")
def comic_page(job_id: str, filename: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id) or not re.fullmatch(r"page_[A-Za-z0-9_-]+\.png", filename):
        abort(404)
    page_path = OUTPUT_ROOT / job_id / filename
    if not page_path.is_file():
        abort(404)
    # Job output filenames are immutable, so the browser can cache each page
    # and avoid downloading the same full-resolution image on every poll.
    return send_file(page_path, mimetype="image/png", conditional=True, max_age=3600)


@app.get("/api/jobs/<job_id>/comic.pdf")
def comic_pdf(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        abort(404)
    pdf_path = OUTPUT_ROOT / job_id / "comic.pdf"
    if not pdf_path.is_file():
        abort(404)
    return send_file(pdf_path, as_attachment=True, download_name=f"cinecomic-{job_id}.pdf")


@app.get("/api/jobs/<job_id>/pages.zip")
def comic_pages_zip(job_id: str):
    if not re.fullmatch(r"[a-f0-9]{12}", job_id):
        abort(404)
    pages = sorted((OUTPUT_ROOT / job_id).glob("page_*.png"))
    if not pages:
        abort(404)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w", zipfile.ZIP_DEFLATED) as archive:
        for page in pages:
            archive.write(page, arcname=page.name)
    payload.seek(0)
    return send_file(
        payload,
        mimetype="application/zip",
        as_attachment=True,
        download_name=f"cinecomic-{job_id}-pages.zip",
    )


@app.get("/api/jobs/<job_id>/details")
def job_details(job_id: str):
    try:
        get_job(job_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Job not found."}), 404
    output = OUTPUT_ROOT / job_id
    return jsonify({
        "transcript": _read_json(PROJECT_ROOT / "data" / "subtitles" / f"{job_id}.json"),
        "keyframes": _read_json(PROJECT_ROOT / "data" / "keyframes" / f"{job_id}.json"),
        "alignment": _read_json(output / "dialogue_frame_alignment.json"),
        "page": _read_json(output / "page.json"),
    })


@app.get("/api/jobs/<job_id>/report")
def job_report(job_id: str):
    try:
        report = build_processing_report(job_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except FileNotFoundError:
        return jsonify({"error": "Job not found."}), 404
    return app.response_class(
        report,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="cinecomic-{job_id}-report.md"'},
    )


@app.get("/api/project-report")
def project_report():
    path = PROJECT_ROOT / "docs" / "PROJECT_ARCHITECTURE_REPORT.md"
    if not path.is_file():
        abort(404)
    return send_file(path, as_attachment=True, download_name="CineComic-Architecture-Report.md")


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify({"error": "Video exceeds the 512 MB upload limit."}), 413


if __name__ == "__main__":
    import os

    app.run(
        host=os.environ.get("CINECOMIC_HOST", "127.0.0.1"),
        port=int(os.environ.get("CINECOMIC_PORT", "8501")),
        debug=False,
        threaded=True,
    )
