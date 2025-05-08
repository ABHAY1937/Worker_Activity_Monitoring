from ultralytics import YOLO
import cv2
import numpy as np
import time
import os
import logging
from dotenv import load_dotenv
load_dotenv()
 
# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("motion_tracking.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load model
model = YOLO("coco_wholebody.pt")

MOTION_THRESHOLD = 50
SKIP_FRAMES = 3
RESIZE_WIDTH = 320
MIN_CONFIDENCE = 0.5

INPUT_SOURCE = os.environ.get('RTSP_LINK')
print(INPUT_SOURCE)
OUTPUT_PATH = "output_motion_tracking.mp4"

prev_kpts_list = []  

if isinstance(INPUT_SOURCE, str) and (
    INPUT_SOURCE.startswith("rtsp://") or INPUT_SOURCE.startswith("rtmp://")
):
    logger.info("[INFO] Using RTSP stream...")
    cap = cv2.VideoCapture(INPUT_SOURCE, cv2.CAP_FFMPEG)
else:
    logger.info("[INFO] Using local video file...")
    cap = cv2.VideoCapture(INPUT_SOURCE)

if not cap.isOpened():
    logger.error(f"[ERROR] Could not open video source: {INPUT_SOURCE}")
    exit()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = cap.get(cv2.CAP_PROP_FPS) or 30

fps = min(fps, 30)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps / SKIP_FRAMES, (width, height))

frame_count = 0
start_time = time.time()

# Resize scale
scale = RESIZE_WIDTH / width

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        logger.info("[INFO] End of video or failed to read frame.")
        break

    frame_count += 1
    if frame_count % SKIP_FRAMES != 0:
        continue

    resized_frame = cv2.resize(frame, None, fx=scale, fy=scale)

    results = model(resized_frame, verbose=False)
    current_kpts_list = []

    for result in results:
        keypoints = result.keypoints
        boxes = result.boxes

        if keypoints is None or boxes is None:
            logger.info(f"[Frame {frame_count}] No boxes or keypoints detected.")
            prev_kpts_list = []  
            continue

        kpts_xy = keypoints.xy.cpu().numpy()
        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()  

        if len(boxes_xyxy) == 0:
            logger.info(f"[Frame {frame_count}] No bounding boxes detected.")
            prev_kpts_list = []
            continue

        for i, person_kpts in enumerate(kpts_xy):
            if i >= len(boxes_xyxy):
                logger.warning(f"[Frame {frame_count}] Keypoints and boxes mismatch for person {i}.")
                continue

            box = boxes_xyxy[i]
            conf = confs[i]

            if conf < MIN_CONFIDENCE:
                continue

            x1, y1, x2, y2 = map(int, box / scale)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2) 

            person_kpts_scaled = []
            for idx, (x, y) in enumerate(person_kpts):
                x_orig, y_orig = int(x / scale), int(y / scale)
                person_kpts_scaled.append((x_orig, y_orig))
                cv2.circle(frame, (x_orig, y_orig), 5, (0, 255, 0), -1)  
                cv2.putText(frame, str(idx), (x_orig + 5, y_orig),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            current_kpts_list.append(person_kpts_scaled)

    # Motion comparison (for each person separately)
    for idx in range(len(current_kpts_list)):
        if idx >= len(prev_kpts_list):
            logger.info(f"[Frame {frame_count}] Initializing person {idx}")
            prev_kpts_list.append(current_kpts_list[idx])
            continue

        if not prev_kpts_list or idx >= len(prev_kpts_list):
            continue  # Safety fallback

        prev_kpts = prev_kpts_list[idx]
        current_kpts = current_kpts_list[idx]

        motion = sum(np.linalg.norm(np.array(curr) - np.array(prev)) for curr, prev in zip(current_kpts, prev_kpts))
        status = "WORKING" if motion > MOTION_THRESHOLD else "IDLE"

        # Optional: draw on top-left corner of the bounding box
        try:
            x1, y1, _, _ = map(int, boxes_xyxy[idx] / scale)
        except Exception as e:
            logger.error(f"[Frame {frame_count}] Could not get box coordinates: {e}")
            continue

        cv2.putText(frame, f"{status} ({motion:.1f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 255) if status == "WORKING" else (255, 255, 0), 2)

        logger.info(f"[Frame {frame_count}] Person {idx}: {status} | Motion: {motion:.1f}")

    prev_kpts_list = current_kpts_list  
    out.write(frame)

# Cleanup
cap.release()
out.release()
elapsed = time.time() - start_time
logger.info(f"\n✅ Done in {elapsed:.2f} seconds | Avg Processing Rate: {frame_count / elapsed:.2f} frames/sec")
logger.info(f"💾 Output saved to: {OUTPUT_PATH}")