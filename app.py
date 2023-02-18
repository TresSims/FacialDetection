import face_recognition
import os
import shutil


log = True
training_file = "./TrainingData/Cal_Sims.jpg"
input_directory = "./InputData/"
output_directory = "./MatchedData/"

known_image = face_recognition.load_image_file(training_file)


cal_encoding = face_recognition.face_encodings(known_image)[0]

files = os.listdir(input_directory)
if log:
    print(files)

for unknown_file in files:
    faces = 1
    unknown_image = face_recognition.load_image_file(input_directory + unknown_file)
    unknown_encodings = face_recognition.face_encodings(unknown_image)
    if log:
        print(f"Analyzing {unknown_file}")
        print(f"\t{len(unknown_encodings)} faces in image")
    match_found = False
    for encoding in unknown_encodings:
        if match_found:
            continue
        if log:
            print(f"\tChecking face {faces}")
        faces += 1
        result = face_recognition.compare_faces([cal_encoding], encoding)
        if result[0]:
            if log:
                print("\tMatch found, copying to matched data")
            shutil.copyfile(input_directory + unknown_file, output_directory + unknown_file)
            match_found = True
