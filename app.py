from flask import Flask, request, jsonify
from flask_cors import CORS
from core.downloader import VideoDownloader
from core.analyzer import VideoAnalyzer
from core.clipper import ClipDetector
import uuid, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = Path("./uploads")
UPLOAD_FOLDER.mkdir(exist_ok=True)

downloader = VideoDownloader(UPLOAD_FOLDER)
analyzer   = VideoAnalyzer()
detector   = ClipDetector()
jobs = {}

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

    try:
        result = downloader.download(url, job_id)
        if not result["success"]:
            return jsonify({"error": result["error"]}), 400

        jobs[job_id]["status"] = "analyzing"
        analysis = analyzer.analyze(result["file_path"], options)

        jobs[job_id]["status"] = "detecting"
        clips = detector.detect(
            analysis,
            min_dur  = options.get("min_duration", 15),
            max_dur  = options.get("max_duration", 60),
            keywords = options.get("keywords", []),
        )

        jobs[job_id] = {"status": "done", "progress": 100}
        return jsonify({
            "success": True, "job_id": job_id,
            "video": result, "clips": clips,
            "summary": {
                "duration":    result.get("duration"),
                "scenes":      len(analysis.get("scenes", [])),
                "peaks":       len(analysis.get("peaks", [])),
                "clips_found": len(clips),
            },
        })
    except Exception as e:
        logger.error(f"[{job_id}] Error: {e}")
        jobs[job_id] = {"status": "error", "error": str(e)}
        return jsonify({"error": str(e)}), 500

@app.route("/api/jobs/")
def job_status(job_id):
    return jsonify(jobs.get(job_id, {"error": "Not found"}))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
