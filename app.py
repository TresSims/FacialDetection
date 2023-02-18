import face_recognition
import os
import shutil

training_file = "./TrainingData/Cal_Sims.jpg"
input_directory = "./InputData/"
output_driectory = "./MatchedData/"


known_image = face_recognition.load_image_file(training_file)


cal_encoding = face_recognition.face_encodings(known_image)[0]
for unknown_file in os.listdir(input_directory):
    unknown_image = face_recognition.load_image_file(training_file)
    for encoding in face_recognition.face_encoding(unknown_image)[0]:
        result = face_recognition.compare_faces([cal_encoding], encoding)
        if result:
            shutil.copyfile(unknown_file, output_directory)
