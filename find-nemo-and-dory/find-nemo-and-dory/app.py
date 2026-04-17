"""
🐠 Web App - Find Marine Friends
======================================
Giao diện web sử dụng Flask để upload ảnh và chọn mô hình nhận dạng.

Cách chạy:
    python app.py
    
Sau đó mở trình duyệt tại: http://localhost:5000

Luồng hoạt động:
    1. Người dùng mở trang web → chọn mô hình + upload ảnh
    2. Flask nhận ảnh → gọi hàm detect từ detect.py
    3. Kết quả (ảnh + bảng thống kê) được trả về trang web
    4. Có thể so sánh cả 3 mô hình cùng lúc
"""

import base64
import io
import time
import tempfile
import os
from pathlib import Path
import json
import threading
import uuid
import cv2
import numpy as np
import torch
try:
    import seaborn  # YOLOv5 (torch.hub) cần seaborn khi load
except ImportError:
    pass
from flask import Flask, render_template, request, jsonify, Response

# ---------------------------------------------------------------------------
# Import các hàm nhận dạng từ detect.py (cùng thư mục)
# ---------------------------------------------------------------------------
from detect import (
    BASE_DIR, DATA_DIR, CLASS_NAMES, COLORS, IMG_SIZE, DEFAULT_CONF,
    load_fasterrcnn, predict_fasterrcnn,
    load_yolov5, predict_yolov5,
    load_yolov8, predict_yolov8,
    draw_results, draw_time_on_image,
    get_training_metrics,
)

# ---------------------------------------------------------------------------
# Khởi tạo Flask app
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),  # Thư mục chứa file HTML
    static_folder=str(BASE_DIR / "static"),        # Thư mục chứa CSS, JS, ảnh tĩnh
)

# Giới hạn kích thước upload: 500 MB (hỗ trợ upload video)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024

# Kích thước inference nhỏ hơn cho video (416 thay vì 640) để tăng tốc
VIDEO_IMGSZ = 416

# Cache video processing progress
_video_progress = {}


def _get_ffmpeg_path():
    """Lấy đường dẫn ffmpeg từ imageio-ffmpeg."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def reencode_to_h264(input_path: str) -> str:
    """
    Re-encode video từ mp4v sang H.264 để browser play được.
    Trả về path file mới (H.264), xóa file gốc.
    """
    import subprocess
    ffmpeg = _get_ffmpeg_path()
    if ffmpeg is None:
        return input_path  # fallback: trả file gốc

    output_path = input_path.replace(".mp4", "_h264.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-i", input_path,
             "-c:v", "libx264", "-preset", "fast", "-crf", "23",
             "-pix_fmt", "yuv420p",
             "-movflags", "+faststart",
             "-an", output_path],
            check=True, capture_output=True, timeout=300,
        )
        os.unlink(input_path)
        return output_path
    except Exception:
        # Nếu re-encode lỗi, trả file gốc
        if os.path.exists(output_path):
            os.unlink(output_path)
        return input_path


# ---------------------------------------------------------------------------
# Đường dẫn trọng số mặc định cho mỗi model
# ---------------------------------------------------------------------------
WEIGHT_MAP = {
    "fasterrcnn": DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth",
    "yolov5":     DATA_DIR / "yolo5_results"   / "weights" / "best.pt",
    "yolov8":     DATA_DIR / "yolo8_results"   / "weights" / "best.pt",
}

# ---------------------------------------------------------------------------
# Cache: Lưu model đã load để không phải load lại mỗi lần request
# Key: (model_name, device) để hỗ trợ chuyển đổi CPU/GPU
# ---------------------------------------------------------------------------
_model_cache = {}

# Device hiện tại (cho phép user chuyển đổi qua API)
_current_device = "cuda:0" if torch.cuda.is_available() else "cpu"


def get_device():
    """Trả về device hiện tại (có thể thay đổi bởi user)."""
    return _current_device


def get_model(model_name: str, device: str = None):
    """
    Lấy model từ cache hoặc load mới nếu chưa có.
    
    Cơ chế cache:
      - Key = (model_name, device) để hỗ trợ chuyển CPU/GPU
      - Lần đầu gọi: load model từ file .pt/.pth → lưu vào _model_cache
      - Lần sau: lấy trực tiếp từ _model_cache (rất nhanh)
    
    Args:
        model_name: 'fasterrcnn', 'yolov5', hoặc 'yolov8'
        device:     'cpu' hoặc 'cuda:0' (None = dùng device hiện tại)
    Returns:
        (model, predict_fn, load_time_ms)
    """
    if device is None:
        device = get_device()
    
    cache_key = (model_name, device)
    
    # Kiểm tra cache
    if cache_key in _model_cache:
        model, predict_fn = _model_cache[cache_key]
        return model, predict_fn, 0.0
    
    # Chưa có trong cache → load model mới
    weights = WEIGHT_MAP[model_name]
    t_start = time.perf_counter()
    
    if model_name == "fasterrcnn":
        model = load_fasterrcnn(str(weights), num_classes=9, device=device)
        predict_fn = lambda img, conf: predict_fasterrcnn(model, img, device, conf)
    elif model_name == "yolov5":
        model = load_yolov5(str(weights), device=device)
        predict_fn = lambda img, conf: predict_yolov5(model, img, conf)
    else:  # yolov8
        model = load_yolov8(str(weights), device=device)
        predict_fn = lambda img, conf: predict_yolov8(model, img, device, conf)
    
    load_time = (time.perf_counter() - t_start) * 1000
    
    _model_cache[cache_key] = (model, predict_fn)
    
    return model, predict_fn, load_time


def image_to_base64(image_bgr: np.ndarray) -> str:
    """
    Chuyển ảnh OpenCV (numpy array) sang chuỗi Base64.
    
    Tại sao dùng Base64?
      - Trả ảnh trực tiếp trong JSON response (không cần lưu file tạm)
      - Hiển thị ảnh trong HTML bằng: <img src="data:image/jpeg;base64,...">
    
    Args:
        image_bgr: Ảnh OpenCV (BGR, numpy array)
    Returns:
        Chuỗi base64 của ảnh JPEG
    """
    # Encode ảnh thành JPEG (nén, nhẹ hơn PNG)
    _, buffer = cv2.imencode(".jpg", image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    # Chuyển bytes → base64 string
    return base64.b64encode(buffer).decode("utf-8")


# ---------------------------------------------------------------------------
# Routes (các trang web)
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """
    Trang chủ - hiển thị giao diện upload ảnh.
    
    Truyền thông tin training metrics sang template để hiển thị
    bảng so sánh độ chính xác các model.
    """
    # Lấy metrics huấn luyện của YOLOv5 và YOLOv8
    metrics = {}
    for name in ["yolov5", "yolov8"]:
        m = get_training_metrics(name)
        if m:
            metrics[name] = m
    
    return render_template("index.html",
                           device=get_device(),
                           cuda_available=torch.cuda.is_available(),
                           gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                           metrics=metrics)


@app.route("/device-info")
def device_info():
    """Trả về thông tin device hiện tại."""
    return jsonify({
        "current_device": get_device(),
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    })


@app.route("/set-device", methods=["POST"])
def set_device():
    """Chuyển đổi device giữa CPU và GPU."""
    global _current_device
    data = request.get_json() or {}
    new_device = data.get("device", "cpu")
    if new_device not in ("cpu", "cuda:0"):
        return jsonify({"error": "Device không hợp lệ. Chọn 'cpu' hoặc 'cuda:0'"}), 400
    if new_device == "cuda:0" and not torch.cuda.is_available():
        return jsonify({"error": "CUDA không khả dụng trên máy này"}), 400
    _current_device = new_device
    return jsonify({"device": _current_device})


@app.route("/detect", methods=["POST"])
def detect():
    """
    API endpoint nhận ảnh + tham số → trả kết quả nhận dạng.
    
    Request (multipart/form-data):
        - file:       Ảnh upload (JPEG/PNG/...)
        - model:      Tên model ('fasterrcnn', 'yolov5', 'yolov8', hoặc 'all')
        - confidence: Ngưỡng confidence (0.0 - 1.0)
    
    Response (JSON):
        - results: list kết quả cho mỗi model, mỗi item chứa:
            - model_name:    Tên model
            - detections:    Danh sách detection [{label, confidence, box}]
            - image_base64:  Ảnh kết quả (base64)
            - inference_ms:  Thời gian inference (ms)
            - load_ms:       Thời gian load model (ms)
            - training_metrics: Metrics huấn luyện (nếu có)
    """
    # --- Kiểm tra file upload ---
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file ảnh"}), 400
    
    try:
        # --- Đọc tham số từ form ---
        model_name = request.form.get("model", "yolov8")      # Model mặc định: yolov8
        conf_thres = float(request.form.get("confidence", DEFAULT_CONF))  # Ngưỡng confidence
        device = get_device()
        
        # --- Đọc ảnh upload vào bộ nhớ ---
        # file.read() → bytes, np.frombuffer() → numpy array, cv2.imdecode() → ảnh BGR
        file_bytes = np.frombuffer(file.read(), np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image_bgr is None:
            return jsonify({"error": "Không đọc được file ảnh. Hãy chọn file JPG/PNG."}), 400
        
        # --- Xác định danh sách model cần chạy ---
        # Nếu chọn "all" → chạy cả 3 model để so sánh
        if model_name == "all":
            model_list = ["fasterrcnn", "yolov5", "yolov8"]
        else:
            model_list = [model_name]
        
        # --- Chạy inference cho từng model ---
        all_results = []
        
        for m_name in model_list:
            # Load model (hoặc lấy từ cache)
            model, predict_fn, load_ms = get_model(m_name)
            
            # Đo thời gian inference
            t_start = time.perf_counter()
            detections = predict_fn(image_bgr, conf_thres)
            infer_ms = (time.perf_counter() - t_start) * 1000
            
            # Vẽ kết quả lên ảnh
            img_result = draw_results(image_bgr, detections)
            img_result = draw_time_on_image(img_result, infer_ms, m_name)
            
            # Chuyển ảnh kết quả sang base64 để gửi qua JSON
            img_b64 = image_to_base64(img_result)
            
            # Lấy training metrics (nếu có)
            training_metrics = get_training_metrics(m_name)
            
            # Gom kết quả
            all_results.append({
                "model_name":    m_name,
                "detections": [
                    {
                        "label":      label,
                        "confidence": round(score, 4),
                        "box":        [x1, y1, x2, y2],
                    }
                    for label, score, (x1, y1, x2, y2) in detections
                ],
                "image_base64":     img_b64,
                "inference_ms":     round(infer_ms, 1),
                "load_ms":          round(load_ms, 1),
                "total_detections": len(detections),
                "training_metrics": training_metrics,
                "device":           device,
            })
        
        return jsonify({"results": all_results})
    
    except Exception as e:
        import traceback
        traceback.print_exc()  # In lỗi chi tiết ra terminal để debug
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


@app.route("/detect-video", methods=["POST"])
def detect_video():
    """
    API endpoint nhận video + tham số → trả video kết quả (base64).
    Xử lý từng frame của video, vẽ bounding box lên và encode lại.
    Hỗ trợ skip frames để tăng tốc.
    """
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file video"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file video"}), 400

    try:
        model_name = request.form.get("model", "yolov8")
        conf_thres = float(request.form.get("confidence", DEFAULT_CONF))
        skip_frames = int(request.form.get("skip_frames", 3))
        if skip_frames < 1:
            skip_frames = 1

        if model_name == "all":
            model_list = ["fasterrcnn", "yolov5", "yolov8"]
        else:
            model_list = [model_name]

        # Tạo task_id để track progress
        task_id = str(uuid.uuid4())

        # Lưu video tạm vào disk
        suffix = Path(file.filename).suffix or ".mp4"
        tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        file.save(tmp_input)
        tmp_input.close()

        all_results = []

        for model_idx, m_name in enumerate(model_list):
            model, predict_fn, load_ms = get_model(m_name)

            # Tạo predict_fn riêng cho video với imgsz nhỏ hơn để tăng tốc
            device = get_device()
            if m_name == "yolov5":
                video_predict_fn = lambda img, conf, _m=model: predict_yolov5(_m, img, conf, imgsz=VIDEO_IMGSZ)
            elif m_name == "yolov8":
                video_predict_fn = lambda img, conf, _m=model: predict_yolov8(_m, img, device, conf, imgsz=VIDEO_IMGSZ)
            else:
                video_predict_fn = predict_fn  # Faster R-CNN không dùng imgsz

            cap = cv2.VideoCapture(tmp_input.name)
            if not cap.isOpened():
                os.unlink(tmp_input.name)
                return jsonify({"error": "Không đọc được video. Hãy chọn file MP4/AVI/MOV."}), 400

            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Tính kích thước resize cho inference (max 480p) để tăng tốc
            max_infer_dim = 480
            if max(w, h) > max_infer_dim:
                infer_scale = max_infer_dim / max(w, h)
                infer_w = int(w * infer_scale)
                infer_h = int(h * infer_scale)
            else:
                infer_scale = None
                infer_w, infer_h = w, h

            # Tạo file video output tạm
            tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tmp_output.close()
            # Output fps giữ nguyên fps gốc, skip chỉ giảm số frame detect
            out_fps = fps / skip_frames
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(tmp_output.name, fourcc, out_fps, (w, h))

            total_detections = 0
            frame_count = 0
            processed_count = 0
            t_start = time.perf_counter()

            # Cập nhật progress
            _video_progress[task_id] = {
                "current_frame": 0,
                "total_frames": total_frames,
                "model_name": m_name,
                "model_index": model_idx + 1,
                "total_models": len(model_list),
                "fps_processing": 0,
                "status": "processing",
            }

            while True:
                ret = cap.grab()  # grab() chỉ đọc, không decode → rất nhanh
                if not ret:
                    break
                frame_count += 1

                # Chỉ decode + detect frame cần xử lý, bỏ qua frame còn lại
                if frame_count % skip_frames == 0 or frame_count == 1:
                    ret2, frame = cap.retrieve()  # Decode frame
                    if not ret2:
                        continue

                    # Resize xuống để inference nhanh hơn
                    if infer_scale is not None:
                        frame_small = cv2.resize(frame, (infer_w, infer_h))
                    else:
                        frame_small = frame

                    detections = video_predict_fn(frame_small, conf_thres)

                    # Scale bounding box về kích thước gốc nếu đã resize
                    if infer_scale is not None:
                        inv = 1.0 / infer_scale
                        detections = [
                            (lbl, sc, (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv)))
                            for lbl, sc, (x1, y1, x2, y2) in detections
                        ]

                    total_detections += len(detections)
                    processed_count += 1

                    frame_out = draw_results(frame, detections)
                    writer.write(frame_out)

                # Cập nhật progress mỗi 10 frames
                if frame_count % 10 == 0 or frame_count == total_frames:
                    elapsed = time.perf_counter() - t_start
                    fps_proc = frame_count / elapsed if elapsed > 0 else 0
                    _video_progress[task_id] = {
                        "current_frame": frame_count,
                        "total_frames": total_frames,
                        "model_name": m_name,
                        "model_index": model_idx + 1,
                        "total_models": len(model_list),
                        "fps_processing": round(fps_proc, 1),
                        "status": "processing",
                    }

            cap.release()
            writer.release()

            infer_ms = (time.perf_counter() - t_start) * 1000

            # Re-encode sang H.264 để browser play được
            final_path = reencode_to_h264(tmp_output.name)

            # Đọc video output và encode base64
            with open(final_path, "rb") as f:
                video_b64 = base64.b64encode(f.read()).decode("utf-8")
            os.unlink(final_path)

            all_results.append({
                "model_name":       m_name,
                "video_base64":     video_b64,
                "inference_ms":     round(infer_ms, 1),
                "load_ms":          round(load_ms, 1),
                "total_frames":     total_frames,
                "processed_frames": processed_count,
                "total_detections": total_detections,
                "fps":              round(fps, 1),
                "skip_frames":      skip_frames,
                "training_metrics": get_training_metrics(m_name),
            })

        os.unlink(tmp_input.name)
        _video_progress[task_id] = {"status": "done"}
        return jsonify({"results": all_results, "task_id": task_id})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý video: {str(e)}"}), 500


@app.route("/video-progress/<task_id>")
def video_progress(task_id):
    """SSE endpoint để stream progress xử lý video realtime."""
    def generate():
        import time as _time
        while True:
            progress = _video_progress.get(task_id, {})
            # Không gửi results qua SSE (quá lớn), chỉ gửi status
            send_data = {k: v for k, v in progress.items() if k != "results"}
            data = json.dumps(send_data)
            yield f"data: {data}\n\n"
            if progress.get("status") in ("done", "error"):
                break
            _time.sleep(0.3)
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/start-video-task", methods=["POST"])
def start_video_task():
    """Nhận video, tạo task_id, bắt đầu xử lý trên thread riêng."""
    if "file" not in request.files:
        return jsonify({"error": "Không tìm thấy file video"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Chưa chọn file video"}), 400

    model_name = request.form.get("model", "yolov8")
    conf_thres = float(request.form.get("confidence", DEFAULT_CONF))
    skip_frames = int(request.form.get("skip_frames", 3))
    if skip_frames < 1:
        skip_frames = 1

    task_id = str(uuid.uuid4())

    # Lưu video tạm
    suffix = Path(file.filename).suffix or ".mp4"
    tmp_input = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    file.save(tmp_input)
    tmp_input.close()

    _video_progress[task_id] = {"status": "uploading", "current_frame": 0, "total_frames": 0}

    def process_video():
        try:
            if model_name == "all":
                model_list = ["fasterrcnn", "yolov5", "yolov8"]
            else:
                model_list = [model_name]

            all_results = []

            for model_idx, m_name in enumerate(model_list):
                model, predict_fn, load_ms = get_model(m_name)

                # Tạo predict_fn riêng cho video với imgsz nhỏ hơn để tăng tốc
                device = get_device()
                if m_name == "yolov5":
                    video_predict_fn = lambda img, conf, _m=model: predict_yolov5(_m, img, conf, imgsz=VIDEO_IMGSZ)
                elif m_name == "yolov8":
                    video_predict_fn = lambda img, conf, _m=model: predict_yolov8(_m, img, device, conf, imgsz=VIDEO_IMGSZ)
                else:
                    video_predict_fn = predict_fn

                cap = cv2.VideoCapture(tmp_input.name)
                if not cap.isOpened():
                    _video_progress[task_id] = {"status": "error", "error": "Không đọc được video"}
                    os.unlink(tmp_input.name)
                    return

                fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Tính kích thước resize cho inference (max 480p) để tăng tốc
                max_infer_dim = 480
                if max(w, h) > max_infer_dim:
                    infer_scale = max_infer_dim / max(w, h)
                    infer_w = int(w * infer_scale)
                    infer_h = int(h * infer_scale)
                else:
                    infer_scale = None
                    infer_w, infer_h = w, h

                tmp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tmp_output.close()
                out_fps = fps / skip_frames
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(tmp_output.name, fourcc, out_fps, (w, h))

                total_detections = 0
                frame_count = 0
                processed_count = 0
                t_start = time.perf_counter()

                _video_progress[task_id] = {
                    "current_frame": 0, "total_frames": total_frames,
                    "model_name": m_name, "model_index": model_idx + 1,
                    "total_models": len(model_list), "fps_processing": 0,
                    "status": "processing",
                }

                while True:
                    ret = cap.grab()
                    if not ret:
                        break
                    frame_count += 1

                    if frame_count % skip_frames == 0 or frame_count == 1:
                        ret2, frame = cap.retrieve()
                        if not ret2:
                            continue

                        # Resize xuống để inference nhanh hơn
                        if infer_scale is not None:
                            frame_small = cv2.resize(frame, (infer_w, infer_h))
                        else:
                            frame_small = frame

                        detections = video_predict_fn(frame_small, conf_thres)

                        # Scale bounding box về kích thước gốc nếu đã resize
                        if infer_scale is not None:
                            inv = 1.0 / infer_scale
                            detections = [
                                (lbl, sc, (int(x1*inv), int(y1*inv), int(x2*inv), int(y2*inv)))
                                for lbl, sc, (x1, y1, x2, y2) in detections
                            ]

                        total_detections += len(detections)
                        processed_count += 1

                        frame_out = draw_results(frame, detections)
                        writer.write(frame_out)

                    if frame_count % 10 == 0 or frame_count == total_frames:
                        elapsed = time.perf_counter() - t_start
                        fps_proc = frame_count / elapsed if elapsed > 0 else 0
                        _video_progress[task_id] = {
                            "current_frame": frame_count, "total_frames": total_frames,
                            "model_name": m_name, "model_index": model_idx + 1,
                            "total_models": len(model_list),
                            "fps_processing": round(fps_proc, 1),
                            "status": "processing",
                        }

                cap.release()
                writer.release()

                infer_ms = (time.perf_counter() - t_start) * 1000

                # Re-encode sang H.264 để browser play được
                final_path = reencode_to_h264(tmp_output.name)

                with open(final_path, "rb") as f:
                    video_b64 = base64.b64encode(f.read()).decode("utf-8")
                os.unlink(final_path)

                all_results.append({
                    "model_name": m_name, "video_base64": video_b64,
                    "inference_ms": round(infer_ms, 1), "load_ms": round(load_ms, 1),
                    "total_frames": total_frames, "processed_frames": processed_count,
                    "total_detections": total_detections, "fps": round(fps, 1),
                    "skip_frames": skip_frames,
                    "training_metrics": get_training_metrics(m_name),
                })

            os.unlink(tmp_input.name)
            _video_progress[task_id] = {"status": "done", "results": all_results}

        except Exception as e:
            import traceback
            traceback.print_exc()
            _video_progress[task_id] = {"status": "error", "error": str(e)}

    thread = threading.Thread(target=process_video, daemon=True)
    thread.start()

    return jsonify({"task_id": task_id})


@app.route("/video-result/<task_id>")
def video_result(task_id):
    """Lấy kết quả video sau khi xử lý xong."""
    progress = _video_progress.get(task_id, {})
    if progress.get("status") == "done":
        results = progress.get("results", [])
        _video_progress.pop(task_id, None)
        return jsonify({"results": results})
    elif progress.get("status") == "error":
        error = progress.get("error", "Unknown error")
        _video_progress.pop(task_id, None)
        return jsonify({"error": error}), 500
    else:
        return jsonify({"status": "processing"}), 202


@app.route("/detect-batch", methods=["POST"])
def detect_batch():
    """
    API endpoint nhận nhiều ảnh (folder) + tham số → trả kết quả cho từng ảnh.
    
    Request (multipart/form-data):
        - files:      Nhiều file ảnh upload cùng lúc
        - model:      Tên model ('fasterrcnn', 'yolov5', 'yolov8', hoặc 'all')
        - confidence: Ngưỡng confidence (0.0 - 1.0)
    
    Response (JSON):
        - images: list kết quả cho mỗi ảnh, mỗi item chứa:
            - filename:  Tên file ảnh
            - results:   list kết quả cho mỗi model (giống /detect)
    """
    files = request.files.getlist("files")
    if not files or len(files) == 0:
        return jsonify({"error": "Không tìm thấy file ảnh"}), 400
    
    try:
        model_name = request.form.get("model", "yolov8")
        conf_thres = float(request.form.get("confidence", DEFAULT_CONF))
        
        if model_name == "all":
            model_list = ["fasterrcnn", "yolov5", "yolov8"]
        else:
            model_list = [model_name]
        
        all_images = []
        
        for file in files:
            if file.filename == "":
                continue
            
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image_bgr is None:
                continue
            
            image_results = []
            
            for m_name in model_list:
                model, predict_fn, load_ms = get_model(m_name)
                
                t_start = time.perf_counter()
                detections = predict_fn(image_bgr, conf_thres)
                infer_ms = (time.perf_counter() - t_start) * 1000
                
                img_result = draw_results(image_bgr, detections)
                img_result = draw_time_on_image(img_result, infer_ms, m_name)
                img_b64 = image_to_base64(img_result)
                training_metrics = get_training_metrics(m_name)
                
                image_results.append({
                    "model_name":    m_name,
                    "detections": [
                        {
                            "label":      label,
                            "confidence": round(score, 4),
                            "box":        [x1, y1, x2, y2],
                        }
                        for label, score, (x1, y1, x2, y2) in detections
                    ],
                    "image_base64":     img_b64,
                    "inference_ms":     round(infer_ms, 1),
                    "load_ms":          round(load_ms, 1),
                    "total_detections": len(detections),
                    "training_metrics": training_metrics,
                })
            
            all_images.append({
                "filename": file.filename,
                "results":  image_results,
            })
        
        if not all_images:
            return jsonify({"error": "Không có ảnh hợp lệ nào trong folder."}), 400
        
        return jsonify({"images": all_images})
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Lỗi xử lý: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Chạy server
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  🐠 Find Marine Friends - Web Interface")
    print("=" * 60)
    print(f"  Device: {get_device()}")
    print(f"  Mở trình duyệt tại: http://localhost:5000")
    print("=" * 60)
    
    # debug=True: tự restart khi thay đổi code (chỉ dùng khi phát triển)
    # host="0.0.0.0": cho phép truy cập từ máy khác trong mạng LAN
    # Nếu chỉ dùng localhost thì đổi thành host="127.0.0.1"
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
