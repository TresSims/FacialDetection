import face_recognition
import os
import shutil

training_file = "./TrainingData/Cal_Sims.jpg"
input_directory = "./InputData/"
output_directory = "./MatchedData/"


known_image = face_recognition.load_image_file(training_file)


cal_encoding = face_recognition.face_encodings(known_image)[0]
print(cal_encoding)
for unknown_file in os.listdir(input_directory):
    faces = 1
    unknown_image = face_recognition.load_image_file(input_directory + unknown_file)
    for encoding in face_recognition.face_encodings(unknown_image):
        print(f"Checking face {faces} in file {unknown_file}")
        faces += 1
        result = face_recognition.compare_faces([cal_encoding], encoding)
        print(result)
        if result[0]:
            print("Copying file")
            shutil.copyfile(input_directory + unknown_file, output_directory + unknown_file)
