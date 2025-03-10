import cv2

# Image path
image_path = "C:\\Users\\pc\\Desktop\\bmi-calculator\\bmi-backend-python\\bmi-backend\\public\\user\\image\\1741583618_67ce750222435.jpg"

# Load image
image = cv2.imread(image_path)

if image is None:
    print("Image not loaded. Check path!")
else:
    print("Image loaded successfully.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load OpenCV Haar cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Check if faces are detected
    if len(faces) == 0:
        print("No human detected")
    else:
        print(f"{len(faces)} human(s) detected")
