from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from core.downloader import VideoDownloader
from core.analyzer import VideoAnalyzer
from core.clipper import ClipDetector
import uuid, logging, subprocess, os, tempfile, threading
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Increase Flask timeout for long videos
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

downloader = VideoDownloader(UPLOAD_FOLDER)
analyzer   = VideoAnalyzer()
detector   = ClipDetector()
jobs       = {}


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/video-info", methods=["POST"])
def video_info():
    url = request.json.get("url", "").strip()
    if not url:
        return jsonify({"error": "URL required"}), 400
    return jsonify(downloader.get_info(url))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data    = request.json
    url     = data.get("url", "").strip()
    options = data.get("options", {})
    if not url:
        return jsonify({"error": "URL required"}), 400

    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {"status": "downloading", "progress": 0}

    # Run in background thread so request doesn't time out
    def run_job():
        try:
            resolution = options.get("resolution", "1080p")
            result = downloader.download(url, job_id, resolution=resolution)
            if not result["success"]:
                jobs[job_id] = {"status": "error", "error": result["error"]}
                return

            jobs[job_id].update({
                "file_path": result["file_path"],
                "video":     result,
                "status":    "analyzing",
                "progress":  20,
            })

            analysis = analyzer.analyze(result["file_path"], options)

            jobs[job_id].update({
                "status":              "detecting",
                "progress":            80,
                "transcript_segments": (analysis.get("transcription") or {}).get("segments", []),
            })

            clips = detector.detect(
                analysis,
                min_dur  = options.get("min_duration", 15),
                max_dur  = options.get("max_duration", 60),
                keywords = options.get("keywords", []),
            )

            # Build subtopics
            segs      = jobs[job_id]["transcript_segments"]
            subtopics = analysis.get("subtopics", [])

            jobs[job_id].update({
                "status":    "done",
                "progress":  100,
                "clips":     clips,
                "subtopics": subtopics,
                "summary": {
                    "duration":    result.get("duration"),
                    "scenes":      len(analysis.get("scenes", [])),
                    "peaks":       len(analysis.get("peaks", [])),
                    "clips_found": len(clips),
                    "subtopics":   len(subtopics),
                },
            })
            logger.info(f"[{job_id}] Done — {len(clips)} clips found")

        except Exception as e:
            logger.error(f"[{job_id}] Error: {e}")
            jobs[job_id] = {"status": "error", "error": str(e)}

    thread = threading.Thread(target=run_job, daemon=True)
    thread.start()

    return jsonify({"success": True, "job_id": job_id, "status": "processing"})


@app.route("/api/jobs/<job_id>")
def job_status(job_id):
    job = jobs.get(job_id, {})
    status = job.get("status", "not_found")
    resp   = {
        "job_id":   job_id,
        "status":   status,
        "progress": job.get("progress", 0),
        "error":    job.get("error"),
    }
    if status == "done":
        resp.update({
            "success":   True,
            "video":     job.get("video"),
            "clips":     job.get("clips", []),
            "subtopics": job.get("subtopics", []),
            "summary":   job.get("summary", {}),
        })
    return jsonify(resp)


@app.route("/api/clip", methods=["POST"])
def download_clip():
    data      = request.json
    job_id    = data.get("job_id", "")
    start     = float(data.get("start", 0))
    end       = float(data.get("end", 0))
    index     = data.get("clip_index", 0)

    job = jobs.get(job_id)
    if not job or not job.get("file_path"):
        return jsonify({"error": "Job not found — analyse a video first"}), 404

    video_path = job["file_path"]
    if not Path(video_path).exists():
        return jsonify({"error": "Video file not found on disk"}), 404

    if end - start <= 0:
        return jsonify({"error": "Invalid timestamps"}), 400

    try:
        out_dir = UPLOAD_FOLDER / job_id / "clips"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"clip_{index+1}_{int(start)}s_subtitled.mp4"
        segments = job.get("transcript_segments", [])

        # Always burn captions
        _cut_with_captions(video_path, str(out_path), start, end, segments)

        return send_file(str(out_path), mimetype="video/mp4",
                         as_attachment=True, download_name=out_path.name)
    except Exception as e:
        logger.error(f"Clip error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/download-full", methods=["POST"])
def download_full():
    job_id = request.json.get("job_id", "")
    job    = jobs.get(job_id)
    if not job or not job.get("file_path"):
        return jsonify({"error": "Job not found"}), 404

    video_path = job["file_path"]
    if not Path(video_path).exists():
        return jsonify({"error": "Video file not found on disk"}), 404

    title      = (job.get("video") or {}).get("title", "video")
    safe_title = "".join(c for c in title if c.isalnum() or c in " _-")[:60].strip()

    return send_file(video_path, mimetype="video/mp4",
                     as_attachment=True, download_name=f"{safe_title}.mp4")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_srt(segments, start, end):
    lines = []
    idx   = 1
    for seg in segments:
        s = seg["start"] - start
        e = seg["end"]   - start
        if e < 0 or s > (end - start):
            continue
        s = max(0, s)
        e = min(end - start, e)
        lines.append(f"{idx}\n{_srt_ts(s)} --> {_srt_ts(e)}\n{seg['text'].strip()}\n")
        idx += 1
    return "\n".join(lines)


def _srt_ts(s):
    h  = int(s // 3600)
    m  = int((s % 3600) // 60)
    sc = int(s % 60)
    ms = int((s - int(s)) * 1000)
    return f"{h:02}:{m:02}:{sc:02},{ms:03}"


def _cut_with_captions(video_path, out_path, start, end, segments):
    """Cut clip and burn in captions from transcript segments."""
    srt_content = _build_srt(segments, start, end)

    srt_file = tempfile.NamedTemporaryFile(
        suffix=".srt", delete=False, mode="w", encoding="utf-8")
    srt_file.write(srt_content)
    srt_file.close()

    # Escape path for ffmpeg subtitles filter
    srt_escaped = srt_file.name.replace("\\", "/").replace(":", "\\:")

    style = ("FontName=Arial,FontSize=16,Bold=1,"
             "PrimaryColour=&HFFFFFF,OutlineColour=&H000000,"
             "Outline=2,Shadow=1,Alignment=2")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(end - start),
        "-vf", f"subtitles='{srt_escaped}':force_style='{style}'",
        "-c:v", "libx264", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        out_path
    ]

    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    os.unlink(srt_file.name)

    if r.returncode != 0:
        logger.warning(f"Caption burn failed, falling back to plain cut: {r.stderr[-200:]}")
        # Fallback: plain cut without captions
        cmd2 = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(end - start),
            "-c:v", "libx264", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            "-avoid_negative_ts", "make_zero",
            out_path
        ]
        subprocess.run(cmd2, capture_output=True, text=True, timeout=300)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)