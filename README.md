# PitchPerfect AI — football clipping backend (starter)

This is a runnable backend starter for the PitchPerfect AI football clipping app. It provides:

- Upload full-match video files (MP4/MOV/MKV/WebM/AVI) with a configurable size limit.
- Background analysis jobs and progress/status polling.
- Generic person detection + multi-object tracking using Ultralytics YOLO and ByteTrack.
- Persisted track observations and an API for users to confirm a track as a named player/manager.
- Player-aware 9:16 crop guided by the selected track's sampled bounding boxes.
- Goal-event linking endpoint for reviewed goal, scorer celebration, teammate reaction, and manager reaction ranges.
- A Next.js/React uploader example in `frontend-example/PitchPerfectUploader.tsx`.
- Audio-energy peaks as **unverified review candidates** for possible crowd/commentary reactions.
- A clips API and FFmpeg MP4 exports in 9:16, 16:9 or 1:1.
- SQLite persistence for local development.

## What it does NOT claim to solve yet

The default model is a generic person detector, not a football identity model. Users must confirm a track label; labels do not automatically persist across camera cuts. It produces temporary visual track IDs, not verified identities such as Mbappé or Bellingham. Audio peaks are not reliable goal detectors. Celebrations, manager reactions, assists, goal attribution, jersey-number recognition, and player-specific highlight ranking need additional football-specific models, event logic, and human review. Center-crop output is used initially; player-aware vertical reframing is a follow-up integration.

## Requirements

- Python 3.10 or 3.11 recommended.
- FFmpeg installed and available on `PATH`.
- For practical match-length tracking, use a machine with a supported GPU. CPU-only processing can be slow.
- First model use may download model weights from Ultralytics.

## Windows PowerShell setup

```powershell
cd pitchperfect-ai-backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8000 --reload
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again.

## macOS/Linux setup

```bash
cd pitchperfect-ai-backend
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API docs.

## Example: upload Real Madrid vs Barcelona

Use a match video file you have permission to process:

```bash
curl -X POST http://127.0.0.1:8000/api/matches/upload \
  -F "match_name=Real Madrid vs Barcelona" \
  -F "file=@./your-match.mp4"
```

The response contains `job_id`. Poll the job:

```bash
curl http://127.0.0.1:8000/api/jobs/YOUR_JOB_ID
```

List event and track candidates:

```bash
curl http://127.0.0.1:8000/api/jobs/YOUR_JOB_ID/clips
```

Render a selected timestamp range (replace `start_seconds` and `end_seconds`):

```bash
curl -X POST http://127.0.0.1:8000/api/jobs/YOUR_JOB_ID/clips \
  -H "Content-Type: application/json" \
  -d "{\"start_seconds\":120.0,\"end_seconds\":150.0,\"title\":\"Goal celebration review\",\"category\":\"celebration\",\"aspect_ratio\":\"9:16\"}"
```

Poll the returned `clip_id`:

```bash
curl http://127.0.0.1:8000/api/clips/YOUR_CLIP_ID
```

When status is `ready`, download using:

```text
http://127.0.0.1:8000/api/clips/YOUR_CLIP_ID/download
```

## Frontend integration

- `POST /api/matches/upload` — multipart form with `file` and `match_name`.
- `GET /api/jobs/{job_id}` — progress and analysis metadata.
- `GET /api/jobs/{job_id}/clips` — track and event candidates.
- `POST /api/jobs/{job_id}/clips` — render a selected clip.
- `GET /api/clips/{clip_id}` — render status.
- `GET /api/clips/{clip_id}/download` — MP4 download.

For browser uploads, call the upload endpoint with `FormData`; do not manually set the multipart `Content-Type` header because the browser adds its boundary.

## Before production

1. Add authentication, per-user ownership checks, quotas, rate limits, CSRF/CORS policy, and private object storage.
2. Use a durable worker queue (Celery/RQ/Arq) instead of FastAPI in-process background tasks.
3. Store job/clip records in PostgreSQL and use signed object-storage URLs for downloads.
4. Add resumable/multipart uploads for very large matches.
5. Add a football-trained detector and player identity workflow: team/kit classification, jersey-number OCR, roster confirmation, and track re-identification through camera cuts.
6. Train/validate event classifiers for goals, assists, celebrations, player reactions, manager reactions, missed chances and tactical moments.
7. Add a review UI so users confirm event labels and selected-player identity before exporting.
8. Use only footage the uploader has the rights or permission to process; define retention and deletion policies.

## Troubleshooting

- `ffmpeg not found`: install FFmpeg and restart the terminal so it is on `PATH`.
- Model download / torch errors: install a compatible PyTorch build for your CPU/GPU, then retry `pip install -r requirements.txt`.
- Long processing times: increase `TRACK_FRAME_STRIDE` (e.g. 12 or 16) for faster, less detailed sampling, or run on a GPU.
- Track IDs change after camera cuts: expected for the generic baseline; cross-shot player re-identification is not included.


## Phase 2 endpoints

- `GET /api/jobs/{job_id}/tracks` — tracked visual identities and confirmed labels.
- `POST /api/jobs/{job_id}/tracks/{track_id}/label` — confirm a track identity. JSON example:

```json
{"player_name":"Kylian Mbappé","role":"player","team_name":"Real Madrid"}
```

- `POST /api/jobs/{job_id}/events/goal-link` — add a reviewed goal and optionally connect scorer celebration, teammate reaction, and manager reaction time ranges. Example:

```json
{
  "goal_seconds": 1234,
  "scorer_track_id": 8,
  "scorer_name": "Kylian Mbappé",
  "celebration_start": 1238,
  "celebration_end": 1252,
  "teammate_reaction_start": 1241,
  "teammate_reaction_end": 1258,
  "manager_reaction_start": 1236,
  "manager_reaction_end": 1248
}
```

The goal and reaction timestamps are user-reviewed in this phase. The endpoint creates linked clip candidates; it does not claim to infer the semantic labels automatically.


## Phase 4: pretrained SoccerNet goal spotting adapter

Phase 4 adds a concrete adapter for the published OpenSportsLib checkpoint `OpenSportsLab/OSL-loc-snbas-2025-e2e`. It runs localization inference on the uploaded match and converts its `GOAL` events into PitchPerfect timestamps, then the existing backend creates goal-sequence review clips automatically. The checkpoint's documented label set includes GOAL plus ball actions such as pass, shot, header, tackle and free kick; it does **not** contain celebration, emotional-reaction, or manager-response classes. Those categories are intentionally not invented. Published model scores are benchmark results, not a guarantee of performance on every broadcast or upload. See the [model zoo](https://opensportslab.github.io/opensportslib/model-zoo/) and [SoccerNet ball action spotting task](https://www.soccer-net.org/tasks/ball-action-spotting).

### Enable the pretrained goal model (optional)

Use Python 3.12+ in a dedicated environment for the optional model stack. Review OpenSportsLib's AGPL-3.0 and commercial licensing terms before using it in a proprietary SaaS.

```powershell
python -m pip install -r requirements-event-model.txt
# Copy .env.example to .env, then keep these values enabled:
# PITCHPERFECT_EVENT_MODEL_COMMAND=python -m app.soccernet_event_model
# PITCHPERFECT_SOCCERNET_WEIGHTS=OpenSportsLab/OSL-loc-snbas-2025-e2e
uvicorn app.main:app --env-file .env --host 127.0.0.1 --port 8000
```

The adapter first downloads the repository's `config.yaml` and then asks OpenSportsLib to load the published checkpoint by repository ID. The first inference may download model weights and can require substantial RAM/VRAM. This package has not downloaded weights or run inference on a real match here, so it is not represented as end-to-end validated. The model card describes a DALI-backed architecture; if native Windows installation fails, use a supported Linux environment (for example Ubuntu under WSL2 or a Linux GPU machine) and follow OpenSportsLib's current installation requirements. For a real validation result, run the evaluator against manually annotated, held-out matches.

### Adapter smoke test (no weights required)

```powershell
python -m unittest tests.test_soccernet_event_model
```

## Connect the sample frontend

Copy `frontend-example/PitchPerfectUploader.tsx` into a Next.js app (for example `app/components/PitchPerfectUploader.tsx`) and render it from a client page. Set the frontend environment variable:

```env
NEXT_PUBLIC_PITCHPERFECT_API=http://127.0.0.1:8000
```

Set `CORS_ORIGINS` in the backend environment to a comma-separated list of your frontend origins, e.g. `http://localhost:3000,https://your-actual-frontend-domain.example`. Do not use the example domain literally. Restart the backend after changing CORS settings. For production, add authentication and user/job ownership checks before exposing uploads or clip downloads.

## Phase 3: automatic event inference and goal-sequence linking

The backend now has an event-model adapter and automatic linking pipeline. If a configured football event model emits timestamps, the pipeline persists its predictions, creates review clips around detected goals, and links nearby predicted scorer celebrations, emotional reactions, crowd reactions, and manager reactions without asking the user to type timestamps. Results are exposed at `GET /api/jobs/{job_id}/events`; the resulting review clips are included in `GET /api/jobs/{job_id}/clips`.

**Important: this repository does not ship a validated checkpoint for all four semantic categories.** SoccerNet Action Spotting provides established annotations for goals and other match actions, but its listed 17 classes do not directly label emotional reactions, celebrations, or manager responses. Those categories need a separately trained/validated model and suitable labeled examples. The app therefore does not fabricate those labels when no model is configured. It continues to surface audio-energy peaks as review-only candidates.

### Connecting an event model

Set `PITCHPERFECT_EVENT_MODEL_COMMAND` to a local inference adapter command. The backend substitutes `{video}` and `{output}` if used in the command template. If the command template does not include `{video}` and `{output}` placeholders, those arguments are appended automatically. The adapter must write a JSON list to the output path, for example:

```json
[
  {"timestamp": 1324.5, "label": "goal", "confidence": 0.94},
  {"timestamp": 1328.2, "label": "celebration", "confidence": 0.86},
  {"timestamp": 1330.0, "label": "emotional_reaction", "confidence": 0.73},
  {"timestamp": 1335.1, "label": "manager_reaction", "confidence": 0.81}
]
```

Supported labels: `goal`, `celebration`, `emotional_reaction`, `manager_reaction`, `shot_on_target`, `penalty`, `save`, `crowd_reaction`, `unknown`. Timestamps are seconds from the start of the uploaded video. Predictions outside the video duration, unsupported labels, and invalid confidence values are discarded. The linker associates reaction predictions in a configurable context window around each goal; temporal proximity is a hypothesis, not proof that the events are causally connected.

### Validation before production

Use a held-out match set with manually verified timestamps for every target class. Do not split clips from the same match across train and test. Report per-class precision, recall and F1 at fixed temporal tolerances, and manually review false positives/false negatives for broadcast cuts, replays, crowd noise, and off-camera managers. A simple evaluator is included:

```powershell
python scripts_validate_events.py --predictions predictions.json --ground-truth labels.json --tolerance 2.0
```

The prediction and ground-truth files should each be a list of `{"timestamp": seconds, "label": class}` records. This evaluator is a development aid, not a substitute for the official SoccerNet evaluation protocol. See [SoccerNet Action Spotting](https://www.soccer-net.org/tasks/action-spotting) and the [official SoccerNet spotting repository](https://github.com/SoccerNet/sn-spotting) for dataset, benchmark, and access details. The project requires separate model weights and any dataset access approvals.
