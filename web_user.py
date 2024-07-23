import face_recognition
from flask import Flask, jsonify, request, redirect, render_template_string
import cv2
import tempfile
import os

ALLOWED_EXTENSIONS_IMAGE = {'png', 'jpg', 'jpeg', 'gif'}
ALLOWED_EXTENSIONS_VIDEO = {'mp4', 'avi', 'mov'}

app = Flask(__name__)  # เปิดใช้แอป Flask

known_face_encodings = []  # ตัวแปรเก็บค่า encodings ของใบหน้าที่ตรวจจับได้จากรูปภาพ

def allowed_file(filename, allowed_extensions):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

@app.route('/', methods=['GET', 'POST'])
def upload_file():
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

    return '''
    <!doctype html>
    <html>
    <head>
       <title>Upload Media</title>
    </head>
    <body>
        <h1>Upload face photo</h1>
       <form method="POST" enctype="multipart/form-data">
           <input type="file" name="picture" accept="image/*">
           <input type="submit" value="Upload Picture">
       </form>

        <h1>Upload a video</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="video" accept="video/*">
            <input type="submit" value="Upload Video">
       </form>
    </body>
    </html>
    '''

def detect_faces_in_image(file):
    global known_face_encodings
    image = face_recognition.load_image_file(file)
    face_locations = face_recognition.face_locations(image)
    known_face_encodings = face_recognition.face_encodings(image)

    if len(known_face_encodings) > 0:
        return jsonify({"face_locations": face_locations})
    else:
        return jsonify({"message": "ไม่เจอใบหน้าในรูปภาพที่คุณอับโหลด"}), 400

def process_video(file):
    global known_face_encodings
    if not known_face_encodings:
        return jsonify({"message": "No face encodings to compare with."})

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        file.save(temp_video.name)
        video_path = temp_video.name

    video_capture = cv2.VideoCapture(video_path)

    # Initialize some variables
    frame_number = 0
    matches = []
    previous_match_time = -1
    first_match_time = None
    last_match_time = None

    frame_skip = 20  # จำนวนเฟรมที่จะข้าม (สามารถเพิ่มได้ตามต้องการ)
    resize_scale = 0.5  # สเกลสำหรับลดขนาดเฟรม

    fps = video_capture.get(cv2.CAP_PROP_FPS)

    # Create a temporary directory to store matched face images
    match_dir = tempfile.mkdtemp()

    while video_capture.isOpened():
        # Grab a single frame of video
        ret, frame = video_capture.read()
        frame_number += 1

        # Skip frames
        if frame_number % frame_skip != 0:
            continue

        # Quit when the input video file ends
        if not ret:
            print("End of video file reached or no frame captured.")
            break

        # Resize the frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=resize_scale, fy=resize_scale)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = small_frame[:, :, ::-1]

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for i, face_encoding in enumerate(face_encodings):
            # See if the face is a match for the known face(s)
            match = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.50)

            name = None
            if match[0]:
                time_in_seconds = frame_number / fps
                name = "Person of Interest"
                if first_match_time is None:
                    first_match_time = time_in_seconds
                last_match_time = time_in_seconds
                if time_in_seconds != previous_match_time:
                    matches.append((name, time_in_seconds))
                    print(f"Found {name} at {time_in_seconds:.4f} seconds")
                    previous_match_time = time_in_seconds

                    # Save the matched face image
                    # Adjust face location based on resize scale
                    top, right, bottom, left = [int(v / resize_scale) for v in face_locations[i]]
                    face_image = frame[top:bottom, left:right]
                    face_image_path = os.path.join(match_dir, f"match_{frame_number}_{i}.png")
                    cv2.imwrite(face_image_path, face_image)

    print(f"First match found at {first_match_time:.4f} seconds")
    print(f"Last match found at {last_match_time:.4f} seconds")

    video_capture.release()
    cv2.destroyAllWindows()

    # Get list of matched face images
    matched_images = os.listdir(match_dir)
    matched_images = [os.path.join(match_dir, img) for img in matched_images]

    return render_template_string('''
    <!doctype html>
    <html>
    <head>
        <title>Matched Faces</title>
    </head>
    <body>
        <h1>Matched Faces</h1>
        <p>Frame Count: {{ frame_count }}</p>
        <p>First Match Time: {{ first_match_time }}</p>
        <p>Last Match Time: {{ last_match_time }}</p>
        <ul>
            {% for name, time in matches %}
            <li>{{ name }} at {{ time }} seconds</li>
            {% endfor %}
        </ul>
        <h2>Matched Face Images</h2>
        <div>
            {% for image_path in matched_images %}
            <img src="data:image/png;base64,{{ image_path }}" alt="Matched Face" width="200"/>
            {% endfor %}
        </div>
    </body>
    </html>
    ''', frame_count=frame_number, matches=matches, first_match_time=first_match_time, last_match_time=last_match_time, matched_images=[encode_image(img) for img in matched_images])

def encode_image(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string

if __name__ == "__main__":
    app.run(debug=True)


