from flask import Blueprint, render_template, request, redirect, url_for, render_template_string, jsonify
import os
import face_recognition
import cv2
import tempfile
import base64
import logging
import shutil 
import psutil
import time
#from app import socketio นำเข้า socketio จากไฟล์หลักของแอป
#from flask_socketio import emit

# กำหนดค่า logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

face_image_bp = Blueprint('face_image_bp', __name__)

ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}

known_face_encodings = [] 

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@face_image_bp.route('/face_image', methods=['GET', 'POST'])
def face_image():
    if request.method == 'POST':
        if 'picture' in request.files:
            file = request.files['picture']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_IMAGE):
                return detect_faces_in_image(file)

        if 'video' in request.files:
            file = request.files['video']
            if file.filename == '':
                return redirect(request.url)
            if file and allowed_file(file.filename, ALLOWED_EXTENSIONS_VIDEO):
                return process_video(file)
    return render_template('face_image.html')

def detect_faces_in_image(file):
    global known_face_encodings
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    known_face_encodings = face_recognition.face_encodings(image)

    if len(known_face_encodings) > 0:
        return jsonify({"face_locations": face_locations})
    else:
        return jsonify({"message": "ไม่เจอใบหน้าในรูปภาพที่คุณอับโหลด"}), 400
def memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

logging.info(f"Memory usage before processing : {memory_usage()} MB")

def format_time(seconds):
    # แปลงเวลาจากวินาทีเป็น ชั่วโมง:นาที:วินาที
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}h {minutes}m {secs:.2f}s"
    elif minutes > 0:
        return f"{minutes}m {secs:.2f}s"
    else:
        return f"{secs:.2f}s"
    
def process_video(file):
    global known_face_encodings
    if not known_face_encodings:
        return jsonify({"message": "No face encodings to compare with."})

    temp_dir = 'D:\\newtemp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=temp_dir) as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name
        logging.info(f"Temporary video file created: {video_path}")

    try:
        video_capture = cv2.VideoCapture(video_path)
        if not video_capture.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            return jsonify({"message": "Failed to open video file."}), 500
        else:
            logging.info("Video file opened successfully.")
    except Exception as e:
        logging.error(f"Error during opening video file: {e}")
        return jsonify({"message": "An error occurred while opening the video file."}), 500

    fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        logging.error("FPS value is zero. Cannot process video.")
        return jsonify({"message": "Invalid FPS value."}), 500

    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    logging.info(f"FPS: {fps}, Total Frames: {total_frames}")

    frame_number = 0
    matches = []
    previous_match_time = -1
    first_match_time = None
    last_match_time = None

    resize_scale = 0.5 # ย่อขนาดภาพลงครึ่งหนึ่ง
    frame_skip = fps
    
    match_dir = tempfile.mkdtemp(dir=temp_dir)
    logging.info(f"Created temporary directory for matched images: {match_dir}")

    try:
        while frame_number < total_frames:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                ret, frame = video_capture.read()
                #logging.info(f"Reading frame number: {frame_number}")
                if not ret:
                    logging.error(f"Failed to read frame {frame_number} from video. Skipping this frame.")
                    frame_number += fps
                    continue

                # คำนวณเวลาจริงจาก cv2.CAP_PROP_POS_MSEC
                current_time_msec = video_capture.get(cv2.CAP_PROP_POS_MSEC)
                time_in_seconds = current_time_msec / 1000.0
                logging.info(f"Processing frame {frame_number}")
                logging.info(f"Calculated time from video: {time_in_seconds:.2f} seconds")


                small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)
                rgb_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for i, face_encoding in enumerate(face_encodings):
                    match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.50)

                    if match[0]:
                        if first_match_time is None:
                            first_match_time = time_in_seconds
                        last_match_time = time_in_seconds
                        if time_in_seconds != previous_match_time:
                            # จัดเก็บข้อมูลการแมทช์ในรูปแบบทูเพิล (ชื่อ, เวลา, เส้นทางภาพ)
                            top, right, bottom, left = [int(v / resize_scale) for v in face_locations[i]]
                            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

                            frame_image_path = os.path.join(match_dir, f"match_{frame_number}.png")
                            success = cv2.imwrite(frame_image_path, frame)
                            if success:
                                matches.append(("Person of Interest", format_time(time_in_seconds), frame_image_path))

                                logging.info(f"Matched face at {time_in_seconds:.4f} seconds")
                                previous_match_time = time_in_seconds


                frame_number += fps 

        logging.info(f"Processed {frame_number}/{total_frames} frames so far.")
        logging.info(f"Memory usage: {memory_usage()} MB")


    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return jsonify({"message": "An internal error occurred."}), 500

    finally:
        video_capture.release()
        cv2.destroyAllWindows()

        if os.path.exists(video_path):
            os.remove(video_path)
            logging.info(f"Temporary video file removed: {video_path}")

        rendered_template = render_template_string(
            '''
            <!doctype html>
            <html>
            <head>
                <title>Matched Faces</title>
                <link rel="shortcut icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">
                <link rel="icon" href="{{ url_for('static', filename='favicon.ico') }}" type="image/x-icon">

                <style>
                    body { font-family: Arial, sans-serif; background-color: #f8f9fa; padding: 20px; }
                    table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
                    th, td { border: 1px solid #dee2e6; padding: 8px; text-align: center; }
                    th { background-color: #007bff; color: white; }
                    img { width: 200px; height: auto; }
                    .container { max-width: 1200px; margin: auto; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Matched Faces</h1>
                    <p>Frame Count: {{ frame_count }}</p>
                    <p>First Match Time: {{ format_time(first_match_time) }}</p>
                    <p>Last Match Time: {{ format_time(first_match_time) }}</p>
                    <h2>Matches</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Name</th>
                                <th>Time (seconds)</th>
                                <th>Face Image</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for name, time, img_path in matches %}
                            <tr>
                                <td>{{ name }}</td>
                                <td>{{ time }}</td>
                                <td><img src="data:image/png;base64,{{ encode_image(img_path) }}" alt="Matched Face"></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </body>
            </html>
            ''', 
            frame_count=frame_number, 
            matches=matches, 
            first_match_time=first_match_time, 
            last_match_time=last_match_time,
            format_time=format_time,
            encode_image=encode_image)
           

        shutil.rmtree(match_dir)
        logging.info(f"Temporary directory for matched images removed: {match_dir}")

        return rendered_template

def encode_image(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string
    else:
        logging.error(f"File not found: {image_path}")
        return None