# CineComic-MML

Multimodal video-to-comic system with a staged Phase 0–12 processing pipeline.

## Setup

1. Install Python 3.10/3.11+ and FFmpeg
   (`ffmpeg -version` and `ffprobe -version` must work).

2. `python -m venv .venv` and activate it, then:

   `pip install -r requirements.txt`

3. Create the folder skeleton:

   `python setup_project.py`

4. Run:

   `python env_check.py`

   This writes:

   `results/environment_report.json`

## Run a stage or the full pipeline

The runner automatically merges the stage-specific YAML fragments when you use `--stages all`, so a single config file such as `configs/phase2.yaml` is enough for the full pipeline.

```bash
python main.py --config configs/phase1.yaml --video path/to/clip.mp4 --video-id clip_001 --stages ingestion
python main.py --config configs/phase2.yaml --video path/to/clip.mp4 --video-id clip_001 --stages all
```

The CLI runs independent stages concurrently by default: Phases 5 and 6, then
Phases 8 and 9. Set `--max-parallel-stages 1` to run them sequentially. The
current parallel groups each contain two stages. A failed stage is recorded
with its reason and only dependent stages are skipped; independent branches
continue. If ASR finds no dialogue, speaker and emotion stages are skipped and
visual processing continues to compose a visual-only comic with scene captions.
Phase 5 uses the Fast scan profile by default (one face-analysis sample about
every two seconds, with a wider tracking gap); the web app offers a Detailed
profile at one sample per second. On CPU, the fast profile roughly halves the
number of expensive face-detection calls, at the cost of missing some brief
faces or actions.

## Run the web application

Install the requirements, then start the local Flask server:

```bash
python app.py
```

Open `http://127.0.0.1:8501` in a browser. The standalone HTML/CSS/JavaScript
interface is served by Flask; the video pipeline remains in Python. Upload an
MP4, MOV, MKV, AVI, or WEBM file (up to 512 MB), choose the speech model and
page density, then follow stage progress and review the generated comic. The
web interface includes page navigation, zoom, full-screen preview, PDF/PNG
exports, stage details, reports, and stop control. Job inputs and outputs are
stored under `data/uploads/`, `data/jobs/`, and `output/`. While a job runs,
the production timeline shows active stages; selecting a stage displays its
recent live console output, refreshed every two seconds. Failed and skipped
stages show their reasons, and the timeline identifies visual-only runs when
no dialogue is detected.

The app requires FFmpeg/FFprobe and the models/dependencies used by the
existing pipeline. Speaker diarization uses gated pyannote models; create a
Hugging Face access token, accept access to `pyannote/speaker-diarization-3.1`
and `pyannote/segmentation-3.0`, then enter the token in Advanced settings or
set `HF_TOKEN` in the environment. The token is passed to the pipeline process
and is not saved with job state.
Use videos you own or have permission to process.

## Private single-server deployment

The repository includes a Docker deployment for one private CPU server. It
uses Waitress instead of Flask's development server, installs FFmpeg and the
pipeline's ML dependencies, runs as an unprivileged user, and persists uploads,
job state, outputs, results, and downloaded model caches in Docker volumes.
The image is large and first-run model downloads require outbound internet
access. Allow several GB for the image and model cache, and size server memory
and disk for the videos you plan to process.

1. Install Docker Engine and the Docker Compose plugin on the server.
2. Copy the project to the server, then create the environment file:

   ```bash
   cp .env.example .env
   ```

   Put a Hugging Face token in `.env` if diarization is required. Accept the
   model access terms listed above first. Keep `.env` private; it is ignored by
   Git.
3. Build and start the service:

   ```bash
   docker compose up -d --build
   docker compose ps
   docker compose logs -f cinecomic
   ```

4. Open `http://127.0.0.1:8502`. Compose binds to localhost by default so the
   app is not exposed directly to the network. For private remote use, connect
   through an SSH tunnel or an authenticated VPN/reverse proxy. Do not change
   the bind address to a public interface without adding authentication and
   HTTPS at a trusted proxy.

The service intentionally runs one WSGI process because active pipeline child
processes are tracked in memory and only one pipeline job is allowed at a time.
The Docker volumes survive container rebuilds; back them up with Docker stopped
or while the app is idle. This single-server setup does not provide a durable
queue or multi-host job recovery. Public multi-worker deployment needs those
before it is safe to scale out.
