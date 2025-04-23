from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import numpy as np
import sys
import os
import tempfile
import subprocess
import json

sys.path.append(os.path.abspath("../"))
import option
from model import Model

app = Flask(__name__)
CORS(app)

args = option.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model()
model_path = '../ckpt/' + args.model_name + 'final.pkl'
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

@app.route('/snap-video', methods=['POST'])
def snap_video():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        video_filename = video_file.filename
        video_path = os.path.join(os.getcwd(), video_filename)
        video_file.save(video_path)

        go_command = ["go", "run", "../video_split.go"]
        go_process = subprocess.Popen(go_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        go_stdout, go_stderr = go_process.communicate()

        if go_process.returncode != 0:
            return jsonify({'error': f"video split code execution failed: {go_stderr.decode()}"}), 500

        chunk_dir = "chunks"
        chunk_files = [f for f in os.listdir(chunk_dir) if f.endswith(".mp4")]
        res = []
        for chunk_file in chunk_files:
            chunk_path = os.path.join(chunk_dir, chunk_file)
            npy_path = run_feature_extractor(chunk_path)
            features = np.load(npy_path)
            input_tensor = torch.tensor(np.array([features]), dtype=torch.float32).to(device)

            with torch.no_grad():
                scores, _ = model(input_tensor)
                scores = torch.nn.Sigmoid()(scores).squeeze()
                result = scores.item()

            os.remove(npy_path)
            res.append(result)
            if result < 0.5:
                os.remove(chunk_path)
        
        go_command = ["go", "run", "../video_merge.go"]
        go_process = subprocess.Popen(go_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        go_stdout, go_stderr = go_process.communicate()

        if go_process.returncode != 0:
            return jsonify({'error': f"video merge code execution failed: {go_stderr.decode()}"}), 500

        return send_file("snapped_video.mp4", mimetype='video/mp4', as_attachment=True, download_name="snapped_video.mp4")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/detect-activity', methods=['POST'])
def detect_activity():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400

        video_file = request.files['video']

        video_filename = video_file.filename
        video_path = os.path.join(os.getcwd(), video_filename)
        video_file.save(video_path)

        result = subprocess.run(
            ['python', '../cctv-recap/cctv_recap.py', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            return jsonify({'error': result.stderr.strip()}), 500

        avi_file = "video_summary.avi"
        mp4_file = "video_summary.mp4"

        if not os.path.exists(avi_file):
            return jsonify({'error': f"{avi_file} not found after script execution."}), 500

        ffmpeg_command = [
            'ffmpeg',
            '-i', avi_file,
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '22',
            mp4_file
        ]

        ffmpeg_result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if ffmpeg_result.returncode != 0:
            return jsonify({'error': f"FFmpeg conversion failed: {ffmpeg_result.stderr}"}), 500

        return send_file(mp4_file, mimetype='video/mp4', as_attachment=True, download_name="video_summary.mp4")

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def run_feature_extractor(video_path):
    """Runs feat_extractor.py and returns the path to the .npy file."""
    command = ["python", "../feat_extractor.py", video_path]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        npy_files = [f for f in os.listdir(".") if f.endswith(".npy")]
        if not npy_files:
            raise Exception("feat_extractor.py did not create a .npy file.")
        return npy_files[0]

    except subprocess.CalledProcessError as e:
        raise Exception(f"Error running feat_extractor.py: {e.stderr}")
    except Exception as e:
        raise Exception(f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)