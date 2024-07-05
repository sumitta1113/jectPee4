import face_recognition
import cv2


# Open the input movie file
input_movie = cv2.VideoCapture("D:\\ฝึกงาน\\facereFromPip\\face_recognition-master\\examples\\hamilton_clip.mp4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

#***********************************************
fps = input_movie.get(cv2.CAP_PROP_FPS)#Frames per second
print(f"Frames per second (FPS): {fps}")
#***********************************************
if not input_movie.isOpened():
    print(f"Error: Could not open video file {input_movie}")
    exit()

#**************************************************

lmm_image = face_recognition.load_image_file("D:\\ฝึกงาน\\facereFromPip\\face_recognition-master\\examples\\lin-manuel-miranda.png")
lmm_face_encoding = face_recognition.face_encodings(lmm_image)[0]

al_image = face_recognition.load_image_file("D:\\ฝึกงาน\\facereFromPip\\face_recognition-master\\examples\\alex-lacamoire.png")
al_face_encoding = face_recognition.face_encodings(al_image)[0]

known_faces = [
    lmm_face_encoding,
    al_face_encoding
]

# Initialize some variables
face_locations = [] #สร้างไว้เก็บ locationsแต่ละรูป
face_encodings = [] #สร้างไว้เก็บ encodingsแต่ละรูป
face_names = []
matches = []
frame_number = 0
previous_match_time = -1


while True:  
    # Grab a single frame of video
    #ถ่ายวิดีโอเฟรมเดียว
    ret, frame = input_movie.read()
    frame_number += 1
     
    # Quit when the input video file ends
    # ออกเมื่อไฟล์วิดีโออินพุตสิ้นสุด
    if not ret:
        print("End of video file reached or no frame captured.")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    # ค้นหาใบหน้าและการเข้ารหัสใบหน้าทั้งหมดในเฟรมปัจจุบันของวิดีโอ
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50) 

        name = None
        if match[0]:
            time_in_seconds = frame_number / fps
            name = "Lin-Manuel Miranda"
            if time_in_seconds != previous_match_time:
                matches.append(("Person of Interest", time_in_seconds))
                print(f"Found {name} at {time_in_seconds:.4f} seconds")
                previous_match_time = time_in_seconds
        elif match[1]:
            time_in_seconds = frame_number / fps
            name = "Alex Lacamoire"
            if time_in_seconds != previous_match_time:
                matches.append(("Person of Interest", time_in_seconds))
                print(f"Found {name} at {time_in_seconds:.4f} seconds")
                previous_match_time = time_in_seconds

        face_names.append(name)


    # Write the resulting image to the output video file
    # เขียนภาพที่ได้ลงในไฟล์วิดีโอเอาท์พุต
    #print("frame {} / {}".format(frame_number, length))
    
# All done!
input_movie.release()
cv2.destroyAllWindows()
# Print all the matches
#for name, time in matches:
    #print(f"Found {name} at {time:.2f} seconds")