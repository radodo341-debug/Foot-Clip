from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.event_recognition import run_event_model, link_goal_sequences

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data"))).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
CLIP_DIR = DATA_DIR / "clips"
DB_PATH = DATA_DIR / "pitchperfect.sqlite3"
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", "20000000000"))
TRACK_FRAME_STRIDE = max(1, int(os.getenv("TRACK_FRAME_STRIDE", "8")))
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")

for folder in (UPLOAD_DIR, CLIP_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(
    title="PitchPerfect AI Backend",
    version="0.3.0",
    description="Football video upload, tracking, event-candidate review and clip export API.",
)
_allowed_origins = [v.strip() for v in os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if v.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)
_db_lock = threading.Lock()


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _db_lock, db() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY,
            match_name TEXT NOT NULL,
            original_filename TEXT NOT NULL,
            video_path TEXT NOT NULL,
            status TEXT NOT NULL,
            progress INTEGER NOT NULL DEFAULT 0,
            stage TEXT NOT NULL DEFAULT 'uploaded',
            error TEXT,
            created_at TEXT NOT NULL,
            duration_seconds REAL,
            metadata_json TEXT NOT NULL DEFAULT '{}'
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS track_observations (
            job_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            timestamp REAL NOT NULL,
            x1 REAL NOT NULL, y1 REAL NOT NULL, x2 REAL NOT NULL, y2 REAL NOT NULL,
            confidence REAL NOT NULL,
            PRIMARY KEY(job_id, track_id, timestamp)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS player_labels (
            job_id TEXT NOT NULL,
            track_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'player',
            team_name TEXT,
            confirmed_at TEXT NOT NULL,
            PRIMARY KEY(job_id, track_id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS detected_events (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            event_id TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp REAL NOT NULL,
            confidence REAL NOT NULL,
            source TEXT NOT NULL,
            evidence TEXT,
            linked_goal_event_id TEXT,
            reviewed INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            job_id TEXT NOT NULL,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            start_seconds REAL NOT NULL,
            end_seconds REAL NOT NULL,
            file_path TEXT,
            status TEXT NOT NULL DEFAULT 'candidate',
            confidence REAL,
            notes TEXT,
            FOREIGN KEY(job_id) REFERENCES jobs(id)
        )
        """)
        conn.commit()


init_db()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_filename(name: str) -> str:
    # Keep only a basename and a conservative character set.
    name = Path(name or "match.mp4").name
    cleaned = "".join(c for c in name if c.isalnum() or c in "._- ")
    return cleaned[:160] or "match.mp4"


def update_job(job_id: str, **values: Any) -> None:
    if not values:
        return
    allowed = {"status", "progress", "stage", "error", "duration_seconds", "metadata_json"}
    if not set(values).issubset(allowed):
        raise ValueError("Invalid job fields")
    assignments = ", ".join(f"{key}=?" for key in values)
    params = list(values.values()) + [job_id]
    with _db_lock, db() as conn:
        conn.execute(f"UPDATE jobs SET {assignments} WHERE id=?", params)
        conn.commit()


def get_job_row(job_id: str) -> sqlite3.Row:
    with db() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return row


def serialize_job(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "match_name": row["match_name"],
        "original_filename": row["original_filename"],
        "status": row["status"],
        "progress": row["progress"],
        "stage": row["stage"],
        "error": row["error"],
        "created_at": row["created_at"],
        "duration_seconds": row["duration_seconds"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }


def probe_video(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError("Video could not be opened. Check that the file is a valid video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ValueError("Video metadata is invalid or the codec is unsupported.")
    return {
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_seconds": frame_count / fps,
        "width": width,
        "height": height,
    }


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def insert_clip(job_id: str, title: str, category: str, start: float, end: float,
                confidence: float | None = None, notes: str | None = None) -> str:
    clip_id = uuid.uuid4().hex
    with _db_lock, db() as conn:
        conn.execute(
            """INSERT INTO clips
            (id, job_id, title, category, start_seconds, end_seconds, confidence, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (clip_id, job_id, title, category, max(0.0, start), max(start + 0.5, end),
             confidence, notes),
        )
        conn.commit()
    return clip_id


def audio_peak_candidates(video_path: Path, duration: float) -> list[dict[str, Any]]:
    """
    Baseline event proposal only. Audio peaks may correlate with goals/crowd reactions,
    but can also be commentary, whistles, music or unrelated noise. Never treat as labels.
    """
    if not ffmpeg_available():
        return []
    wav_path = video_path.with_suffix(".analysis.wav")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(video_path), "-vn", "-ac", "1",
             "-ar", "8000", "-f", "wav", str(wav_path)],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=1800,
        )
        import wave
        import numpy as np
        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            width = wf.getsampwidth()
            frames = wf.readframes(wf.getnframes())
        if width != 2 or not frames:
            return []
        samples = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
        window = max(1, rate * 2)
        energies = []
        for start in range(0, len(samples), window):
            chunk = samples[start:start + window]
            if len(chunk):
                energies.append((start / rate, float(np.sqrt(np.mean(chunk * chunk)))))
        if len(energies) < 3:
            return []
        vals = np.array([e for _, e in energies])
        threshold = max(float(np.percentile(vals, 90)), float(vals.mean() + vals.std()))
        picks = []
        last_time = -999.0
        for t, energy in energies:
            if energy >= threshold and t - last_time >= 12 and 0 <= t < duration:
                picks.append({
                    "start": max(0, t - 8),
                    "end": min(duration, t + 12),
                    "category": "audio_peak_review",
                    "title": "Possible crowd / commentary reaction — review",
                    "confidence": None,
                    "notes": "Heuristic audio-energy candidate, not a confirmed goal or celebration.",
                })
                last_time = t
        return picks[:200]
    except Exception:
        return []
    finally:
        try:
            wav_path.unlink(missing_ok=True)
        except Exception:
            pass


def run_tracking(job_id: str, video_path: Path, duration: float) -> dict[str, Any]:
    """
    Generic person tracking. IDs are temporary visual tracks, NOT verified player names.
    The generic model can miss small/far-away players and may swap IDs at cuts/occlusions.
    """
    update_job(job_id, status="processing", progress=10, stage="tracking players")
    from ultralytics import YOLO

    model = YOLO(YOLO_MODEL)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError("Cannot open uploaded video for tracking.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_no = 0
    sampled = 0
    track_stats: dict[str, dict[str, Any]] = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_no % TRACK_FRAME_STRIDE == 0:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],  # generic COCO person class
                    verbose=False,
                    conf=0.20,
                )
                sampled += 1
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    ids = boxes.id
                    if ids is not None:
                        ids_list = ids.int().cpu().tolist()
                        xyxy = boxes.xyxy.cpu().tolist()
                        confs = boxes.conf.cpu().tolist()
                        timestamp = frame_no / fps
                        observation_rows = []
                        for track_id, box, conf in zip(ids_list, xyxy, confs):
                            observation_rows.append((job_id, int(track_id), round(timestamp, 3),
                                float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)))
                            key = str(track_id)
                            item = track_stats.setdefault(key, {
                                "track_id": int(track_id),
                                "first_seen_seconds": timestamp,
                                "last_seen_seconds": timestamp,
                                "observations": 0,
                                "max_confidence": 0.0,
                                "sample_boxes": [],
                            })
                            item["last_seen_seconds"] = timestamp
                            item["observations"] += 1
                            item["max_confidence"] = max(item["max_confidence"], float(conf))
                            if len(item["sample_boxes"]) < 12:
                                item["sample_boxes"].append({
                                    "t": round(timestamp, 2),
                                    "xyxy": [round(float(v), 1) for v in box],
                                })
                        if observation_rows:
                            with _db_lock, db() as conn:
                                conn.executemany(
                                    "INSERT OR REPLACE INTO track_observations "
                                    "(job_id, track_id, timestamp, x1, y1, x2, y2, confidence) "
                                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)", observation_rows)
                                conn.commit()
            frame_no += 1
            if frame_no % max(1, int(fps * TRACK_FRAME_STRIDE * 30)) == 0:
                progress = min(75, 10 + int(65 * (frame_no / max(1, fps * duration))))
                update_job(job_id, progress=progress, stage="tracking players")
    finally:
        cap.release()

    tracks = sorted(track_stats.values(), key=lambda x: x["observations"], reverse=True)
    # Persist each track's visible time span as a reviewable candidate, not an event label.
    with _db_lock, db() as conn:
        for track in tracks:
            conn.execute(
                """INSERT INTO clips
                (id, job_id, title, category, start_seconds, end_seconds, status, confidence, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    uuid.uuid4().hex, job_id,
                    f"Visual track #{track['track_id']} — review identity",
                    "player_track",
                    max(0.0, track["first_seen_seconds"] - 2),
                    min(duration, track["last_seen_seconds"] + 2),
                    "candidate", track["max_confidence"],
                    f"{track['observations']} sampled observations. Track ID is not a named-player identity.",
                ),
            )
        conn.commit()
    return {"tracks_detected": len(tracks), "tracks": tracks[:100], "sampled_frames": sampled}


def process_job(job_id: str) -> None:
    row = get_job_row(job_id)
    video_path = Path(row["video_path"])
    try:
        update_job(job_id, status="processing", progress=3, stage="checking video")
        meta = probe_video(video_path)
        update_job(job_id, duration_seconds=meta["duration_seconds"],
                   metadata_json=json.dumps(meta), progress=7, stage="preparing analysis")
        tracking = run_tracking(job_id, video_path, meta["duration_seconds"])
        update_job(job_id, progress=78, stage="recognising football events")
        candidates = audio_peak_candidates(video_path, meta["duration_seconds"])
        predictions, model_status = run_event_model(video_path, meta["duration_seconds"])
        linked_sequences = link_goal_sequences(predictions, meta["duration_seconds"])
        # Persist model predictions and create automatically timestamped review clips.
        with _db_lock, db() as conn:
            for pred in predictions:
                conn.execute("""INSERT INTO detected_events
                    (id, job_id, event_id, label, timestamp, confidence, source, evidence)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                    (uuid.uuid4().hex, job_id, "", pred.label, pred.timestamp,
                     pred.confidence, pred.source, pred.evidence))
            for sequence in linked_sequences:
                goal = sequence["goal"]
                conn.execute("UPDATE detected_events SET event_id=? WHERE job_id=? AND label='goal' AND timestamp=?",
                             (sequence["event_id"], job_id, goal["timestamp"]))
            conn.commit()
        for candidate in candidates:
            insert_clip(job_id, candidate["title"], candidate["category"],
                        candidate["start"], candidate["end"],
                        candidate.get("confidence"), candidate.get("notes"))
        for sequence in linked_sequences:
            goal = sequence["goal"]
            start = sequence["window"]["start_seconds"]
            end = sequence["window"]["end_seconds"]
            linked_labels = sorted({e["label"] for e in sequence["linked_events"]})
            title = f"Goal sequence at {goal['timestamp']:.1f}s"
            note = (f"Auto-detected goal confidence={goal['confidence']:.3f}; linked events="
                    f"{', '.join(linked_labels) if linked_labels else 'none detected'}. "
                    "Review before publishing; temporal association is not proof of causality.")
            insert_clip(job_id, title, "auto_goal_sequence", start, end,
                        goal["confidence"], note)
        final_meta = {**meta, **tracking, "audio_event_candidates": len(candidates),
                      "football_event_model": model_status,
                      "football_event_predictions": len(predictions),
                      "auto_linked_goal_sequences": len(linked_sequences),
                      "analysis_notes": [
                          "Person tracking uses a generic detector; IDs are not player names.",
                          "Audio peaks are review candidates, not confirmed goals or celebrations.",
                          "Automatic semantic events are only produced when a football model adapter is configured.",
                          "Goal-to-reaction links use temporal proximity and must be reviewed before publishing.",
                      ]}
        update_job(job_id, status="completed", progress=100, stage="analysis complete",
                   metadata_json=json.dumps(final_meta))
    except Exception as exc:
        update_job(job_id, status="failed", stage="failed", error=str(exc)[:1000])


class TrackLabelRequest(BaseModel):
    player_name: str = Field(min_length=1, max_length=100)
    role: str = Field(default="player", pattern=r"^(player|manager|official|unknown)$")
    team_name: str | None = Field(default=None, max_length=100)


class GoalLinkRequest(BaseModel):
    goal_seconds: float = Field(ge=0)
    scorer_track_id: int | None = None
    scorer_name: str | None = Field(default=None, max_length=100)
    celebration_start: float | None = Field(default=None, ge=0)
    celebration_end: float | None = Field(default=None, gt=0)
    teammate_reaction_start: float | None = Field(default=None, ge=0)
    teammate_reaction_end: float | None = Field(default=None, gt=0)
    manager_reaction_start: float | None = Field(default=None, ge=0)
    manager_reaction_end: float | None = Field(default=None, gt=0)


class ClipRequest(BaseModel):
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(gt=0)
    title: str = Field(default="Football highlight", min_length=1, max_length=140)
    category: str = Field(default="manual", max_length=50)
    aspect_ratio: str = Field(default="9:16", pattern=r"^(9:16|16:9|1:1)$")
    selected_track_id: int | None = None


@app.get("/health")
def health():
    return {"ok": True, "service": "PitchPerfect AI Backend", "version": "0.3.0"}


@app.post("/api/matches/upload", status_code=202)
async def upload_match(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    match_name: str = Form("Real Madrid vs Barcelona"),
):
    filename = safe_filename(file.filename or "match.mp4")
    suffix = Path(filename).suffix.lower()
    if suffix not in {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}:
        raise HTTPException(status_code=415, detail="Unsupported video extension.")
    job_id = uuid.uuid4().hex
    target = UPLOAD_DIR / f"{job_id}{suffix}"
    total = 0
    try:
        with target.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="Upload exceeds configured size limit.")
                out.write(chunk)
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    try:
        meta = probe_video(target)
    except Exception as exc:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Invalid video: {exc}") from exc

    with _db_lock, db() as conn:
        conn.execute(
            """INSERT INTO jobs
            (id, match_name, original_filename, video_path, status, progress, stage, created_at,
             duration_seconds, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (job_id, match_name[:160], filename, str(target), "queued", 0, "queued",
             utc_now(), meta["duration_seconds"], json.dumps(meta)),
        )
        conn.commit()
    background_tasks.add_task(process_job, job_id)
    return {
        "job_id": job_id,
        "status": "queued",
        "match_name": match_name,
        "duration_seconds": meta["duration_seconds"],
        "message": "Upload accepted. Poll GET /api/jobs/{job_id} for progress.",
    }


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    return serialize_job(get_job_row(job_id))


@app.get("/api/jobs/{job_id}/clips")
def list_clips(job_id: str):
    get_job_row(job_id)
    with db() as conn:
        rows = conn.execute(
            """SELECT id, title, category, start_seconds, end_seconds, file_path,
                      status, confidence, notes
               FROM clips WHERE job_id=? ORDER BY start_seconds ASC""",
            (job_id,),
        ).fetchall()
    return {"job_id": job_id, "clips": [dict(r) for r in rows]}


@app.get("/api/jobs/{job_id}/tracks")
def list_tracks(job_id: str):
    """List tracked visual identities and any user-confirmed labels."""
    get_job_row(job_id)
    with db() as conn:
        rows = conn.execute("""
            SELECT o.track_id, MIN(o.timestamp) AS first_seen_seconds,
                   MAX(o.timestamp) AS last_seen_seconds, COUNT(*) AS observations,
                   MAX(o.confidence) AS max_confidence,
                   l.player_name, l.role, l.team_name
            FROM track_observations o
            LEFT JOIN player_labels l ON l.job_id=o.job_id AND l.track_id=o.track_id
            WHERE o.job_id=?
            GROUP BY o.track_id, l.player_name, l.role, l.team_name
            ORDER BY observations DESC
        """, (job_id,)).fetchall()
    return {"job_id": job_id, "tracks": [dict(r) for r in rows],
            "note": "Track IDs are visual tracks, not automatic proof of player identity. Confirm labels manually."}


@app.post("/api/jobs/{job_id}/tracks/{track_id}/label")
def label_track(job_id: str, track_id: int, request: TrackLabelRequest):
    """User confirms that a visual track corresponds to a named player or manager."""
    get_job_row(job_id)
    with db() as conn:
        exists = conn.execute(
            "SELECT 1 FROM track_observations WHERE job_id=? AND track_id=? LIMIT 1",
            (job_id, track_id)).fetchone()
        if not exists:
            raise HTTPException(status_code=404, detail="Track not found for this match.")
        conn.execute("""INSERT INTO player_labels
            (job_id, track_id, player_name, role, team_name, confirmed_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id, track_id) DO UPDATE SET
            player_name=excluded.player_name, role=excluded.role,
            team_name=excluded.team_name, confirmed_at=excluded.confirmed_at""",
            (job_id, track_id, request.player_name.strip(), request.role,
             request.team_name, utc_now()))
        conn.commit()
    return {"job_id": job_id, "track_id": track_id, "player_name": request.player_name.strip(),
            "role": request.role, "team_name": request.team_name, "confirmed": True}


@app.post("/api/jobs/{job_id}/events/goal-link")
def link_goal_reactions(job_id: str, request: GoalLinkRequest):
    """Create reviewable clips around a goal and optionally link celebration/reaction ranges.

    Goal timestamp and reaction windows are supplied/confirmed by the user in this starter.
    This endpoint does not pretend to automatically recognise goal semantics.
    """
    row = get_job_row(job_id)
    duration = float(row["duration_seconds"] or 0)
    if request.goal_seconds > duration:
        raise HTTPException(status_code=422, detail="Goal timestamp exceeds match duration.")
    event_id = uuid.uuid4().hex[:12]
    scorer = request.scorer_name or "Unconfirmed scorer"
    made = []
    def add_range(title: str, category: str, start: float, end: float, notes: str):
        if start < 0 or end <= start or end > duration:
            raise HTTPException(status_code=422, detail=f"Invalid time range for {category}.")
        clip_id = insert_clip(job_id, title, category, start, end, notes=f"Goal event {event_id}. {notes}")
        made.append({"clip_id": clip_id, "title": title, "category": category,
                     "start_seconds": start, "end_seconds": end})
    add_range(f"Goal: {scorer}", "goal_review", max(0, request.goal_seconds - 12),
              min(duration, request.goal_seconds + 8),
              f"Goal timestamp reviewed at {request.goal_seconds:.2f}s; scorer track={request.scorer_track_id}.")
    optional_ranges = [
        ("Scorer celebration", "celebration", request.celebration_start, request.celebration_end),
        ("Teammates' reaction", "teammate_reaction", request.teammate_reaction_start, request.teammate_reaction_end),
        ("Manager reaction", "manager_reaction", request.manager_reaction_start, request.manager_reaction_end),
    ]
    for title, category, start, end in optional_ranges:
        if start is not None or end is not None:
            if start is None or end is None:
                raise HTTPException(status_code=422, detail=f"Both start and end are required for {category}.")
            add_range(title, category, start, end, "Manually linked to the same goal event; review before export.")
    return {"job_id": job_id, "goal_event_id": event_id, "clips": made,
            "note": "These are reviewable event clips. Semantic detection still needs a validated football-specific model."}


@app.post("/api/jobs/{job_id}/clips")
def create_clip(job_id: str, request: ClipRequest, background_tasks: BackgroundTasks):
    row = get_job_row(job_id)
    duration = row["duration_seconds"]
    if duration is None:
        duration = probe_video(Path(row["video_path"]))["duration_seconds"]
    if request.end_seconds <= request.start_seconds:
        raise HTTPException(status_code=422, detail="end_seconds must be greater than start_seconds.")
    if request.end_seconds > duration:
        raise HTTPException(status_code=422, detail=f"end_seconds exceeds video duration ({duration:.2f}s).")
    if request.end_seconds - request.start_seconds > 180:
        raise HTTPException(status_code=422, detail="Maximum clip duration is 180 seconds.")
    clip_id = insert_clip(job_id, request.title, request.category,
                          request.start_seconds, request.end_seconds,
                          notes=f"User-selected clip. Aspect ratio: {request.aspect_ratio}")
    background_tasks.add_task(render_clip, job_id, clip_id, request)
    return {"clip_id": clip_id, "status": "rendering", "message": "Poll GET /api/clips/{clip_id}."}


def render_clip(job_id: str, clip_id: str, request: ClipRequest) -> None:
    row = get_job_row(job_id)
    source = Path(row["video_path"])
    out = CLIP_DIR / f"{clip_id}.mp4"
    if not ffmpeg_available():
        with _db_lock, db() as conn:
            conn.execute("UPDATE clips SET status='failed', notes=? WHERE id=?",
                         ("FFmpeg is not installed on the server.", clip_id))
            conn.commit()
        return
    width_height = {"9:16": "1080:1920", "16:9": "1920:1080", "1:1": "1080:1080"}[request.aspect_ratio]
    vf = f"scale={width_height}:force_original_aspect_ratio=increase,crop={width_height}"
    # For vertical output, use the selected track's sampled horizontal centre to guide the crop.
    # A piecewise-linear FFmpeg expression follows the player between sampled detections.
    if request.aspect_ratio == "9:16" and request.selected_track_id is not None:
        with db() as conn:
            observations = conn.execute("""SELECT timestamp, x1, x2 FROM track_observations
                WHERE job_id=? AND track_id=? AND timestamp BETWEEN ? AND ? ORDER BY timestamp""",
                (job_id, request.selected_track_id, request.start_seconds, request.end_seconds)).fetchall()
        if observations:
            # Limit expression size while retaining approximately one sample per second.
            chosen = []
            last_t = -999.0
            for ob in observations:
                if ob["timestamp"] - last_t >= 1.0:
                    chosen.append(ob)
                    last_t = ob["timestamp"]
            if len(chosen) > 120:
                step = max(1, len(chosen) // 120)
                chosen = chosen[::step]
            rel = [(max(0.0, float(o["timestamp"]) - request.start_seconds),
                    (float(o["x1"]) + float(o["x2"])) / 2.0) for o in chosen]
            # FFmpeg crop width is ih*9/16. x is clamped to the image bounds.
            # Source observations are in original pixels; normalize against video width.
            source_meta = json.loads(row["metadata_json"] or "{}")
            src_w = max(1.0, float(source_meta.get("width", 1920)))
            pieces = []
            for i, (t, cx) in enumerate(rel):
                cx_norm = min(1.0, max(0.0, cx / src_w))
                x_expr = f"max(0,min(iw-ih*9/16,{cx_norm:.6f}*iw-ih*9/32))"
                if i == 0:
                    pieces.append((t, x_expr))
                else:
                    prev_t, prev_x = rel[i-1]
                    prev_norm = min(1.0, max(0.0, prev_x / src_w))
                    prev_expr = f"max(0,min(iw-ih*9/16,{prev_norm:.6f}*iw-ih*9/32))"
                    pieces.append((t, x_expr))
            # Piecewise-linear crop centre keeps the selected track in frame between detections.
            norm_centres = [min(1.0, max(0.0, cx / src_w)) for _, cx in rel]
            def crop_x(norm_expr: str) -> str:
                return f"max(0,min(iw-ih*9/16,({norm_expr})*iw-ih*9/32))"
            x_dynamic = crop_x(f"{norm_centres[-1]:.6f}")
            for i in range(len(rel)-2, -1, -1):
                t0, t1 = rel[i][0], rel[i+1][0]
                c0, c1 = norm_centres[i], norm_centres[i+1]
                if t1 <= t0:
                    continue
                interpolated = f"{c0:.6f}+({c1-c0:.6f})*(t-{t0:.3f})/{(t1-t0):.3f}"
                x_dynamic = f"if(lt(t,{t1:.3f}),{crop_x(interpolated)},{x_dynamic})"
            vf = f"crop=w='ih*9/16':h=ih:x='{x_dynamic}':y=0,scale=1080:1920"
    command = [
        "ffmpeg", "-y", "-ss", str(request.start_seconds), "-i", str(source),
        "-t", str(request.end_seconds - request.start_seconds),
        "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                       stderr=subprocess.PIPE, timeout=3600)
        with _db_lock, db() as conn:
            conn.execute("UPDATE clips SET file_path=?, status='ready' WHERE id=?",
                         (str(out), clip_id))
            conn.commit()
    except Exception as exc:
        with _db_lock, db() as conn:
            conn.execute("UPDATE clips SET status='failed', notes=? WHERE id=?",
                         (str(exc)[:800], clip_id))
            conn.commit()


@app.get("/api/clips/{clip_id}")
def clip_status(clip_id: str):
    with db() as conn:
        row = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    item = dict(row)
    item.pop("file_path", None)
    item["download_url"] = f"/api/clips/{clip_id}/download" if row["status"] == "ready" else None
    return item


@app.get("/api/clips/{clip_id}/download")
def download_clip(clip_id: str):
    with db() as conn:
        row = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Clip not found")
    if row["status"] != "ready" or not row["file_path"]:
        raise HTTPException(status_code=409, detail=f"Clip is not ready (status={row['status']}).")
    path = Path(row["file_path"])
    if not path.exists():
        raise HTTPException(status_code=410, detail="Rendered clip file is missing.")
    return FileResponse(path, media_type="video/mp4", filename=f"{safe_filename(row['title'])}.mp4")


@app.get("/api/jobs/{job_id}/events")
def list_detected_events(job_id: str):
    """List model predictions and automatically linked goal context for review."""
    get_job_row(job_id)
    with db() as conn:
        rows = conn.execute("""SELECT id, event_id, label, timestamp, confidence, source,
            evidence, reviewed FROM detected_events WHERE job_id=? ORDER BY timestamp""",
            (job_id,)).fetchall()
    return {"job_id": job_id, "events": [dict(r) for r in rows],
            "note": "Predictions require review. A temporal link does not prove an event caused another event."}


@app.get("/api/jobs/{job_id}/source")
def source_metadata(job_id: str):
    row = get_job_row(job_id)
    return {
        "job_id": job_id,
        "match_name": row["match_name"],
        "duration_seconds": row["duration_seconds"],
        "metadata": json.loads(row["metadata_json"] or "{}"),
    }
