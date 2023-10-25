import cv2
import face_recognition

# Tanınan kişilerin yüzlerinin resimlerini yükleyin ve isimlerini belirtin
known_faces = [
    {
        "name": "umut emre albayrak",
        "image_path": r"C:\Users\albay\staj_proje\proje\umut_foto.jpg"
    }
   
]

# Tanınan yüzlerin kodlanmış özelliklerini ve isimlerini saklayacak listeleri oluşturun
known_face_encodings = []
known_face_names = []

# Tanınan yüzlerin resimlerini okuyarak kodlanmış özelliklerini ve isimlerini listelere ekleyin
for face in known_faces:
    image = face_recognition.load_image_file(face["image_path"])
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)
    known_face_names.append(face["name"])

# Video yakalayıcısını başlatın
video_capture = cv2.VideoCapture(0)

while True:
    # Kameradan bir video çerçevesi yakalayın
    ret, frame = video_capture.read()

    # Yakalanan çerçevedeki yüzleri algılayın
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Algılanan yüzleri döngüye alın
    for face_encoding in face_encodings:
        # Algılanan yüzü tanımak için tanınan yüzlerle karşılaştırın
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        # Eşleşen yüz varsa en yakın eşleşmeyi bulun
        if True in matches:
            matched_indexes = [i for i, match in enumerate(matches) if match]
            distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            closest_index = min(matched_indexes, key=lambda x: distances[x])
            name = known_face_names[closest_index]

        # Yüzü çerçeve üzerine yazdırın
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Çerçeveyi görüntüleyin
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında döngüyü sonlandırın
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kaynakları serbest bırakın
video_capture.release()
cv2.destroyAllWindows()
