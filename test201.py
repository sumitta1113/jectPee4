import cv2
import face_recognition

"""skip 20 frame/ใช้เวลา20-30 sec per video 1.20 min/บอกละเอียดเรียงตามเวลา/บอกเวลาแรกที่เจอและเวลาสุดท้ายที่เจอะ ในตอนจบการประมวล"""

# Open the input movie file
video_path = "D:\\jectpee4\\hamilton_clip.mp4"
input_movie = cv2.VideoCapture(video_path)
if not input_movie.isOpened():
    print(f"Error: Could not open video file {video_path}")
    exit()

fps = input_movie.get(cv2.CAP_PROP_FPS)
print(f"Frames per second (FPS): {fps}")

# Load the reference images and encode the faces
lmm_image = face_recognition.load_image_file("D:\\jectpee4\\lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("D:\\ฝึกงาน\\facereFromPip\\face_recognition-master\\examples\\alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
frame_number = 0
matches = []
previous_match_time = -1
first_match_time = None
last_match_time = None

frame_skip = 20  # จำนวนเฟรมที่จะข้าม (สามารถเพิ่มได้ตามต้องการ)
resize_scale = 0.5  # สเกลสำหรับลดขนาดเฟรม

while True:  
    # Grab a single frame of video
    ret, frame = input_movie.read()
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

    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            time_in_seconds = frame_number / fps
            name = "Lin-Manuel Miranda"
            if first_match_time is None:
                first_match_time = time_in_seconds
            last_match_time = time_in_seconds
            if time_in_seconds != previous_match_time:
                matches.append(("Person of Interest", time_in_seconds))
                print(f"Found {name} at {time_in_seconds:.4f} seconds")
                previous_match_time = time_in_seconds
        elif match[1]:
            time_in_seconds = frame_number / fps
            name = "Alex Lacamoire"
            if first_match_time is None:
                first_match_time = time_in_seconds
            last_match_time = time_in_seconds
            if time_in_seconds != previous_match_time:
                matches.append(("Person of Interest", time_in_seconds))
                print(f"Found {name} at {time_in_seconds:.4f} seconds")
                previous_match_time = time_in_seconds
print(f"First match found at {first_match_time:.4f} seconds")
print(f"Last match found at {last_match_time:.4f} seconds")
# All done!
input_movie.release()
cv2.destroyAllWindows()

