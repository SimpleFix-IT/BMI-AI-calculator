import os
import sys
import json
import cv2
import torch
import numpy as np
import traceback
from torchvision import transforms
from models import HeightWeightEstimator  # AI Model Class Import

# ✅ AI Model Load Karna
model = HeightWeightEstimator()
model_path = os.path.join(os.path.dirname(__file__), "model_weights.pth")

try:
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(json.dumps({"error": "Model loading failed", "details": str(e)}))
    sys.exit(1)


# ✅ Image Processing Function
def process_image(image_path):
    if not os.path.exists(image_path):
        return {"error": "Image not found or cannot be read"}

    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Invalid image format or corrupted image"}

    # ✅ Convert BGR to RGB for PyTorch
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 🔹 Image ko resize aur contrast enhance karein
    image = cv2.resize(image, (500, 500))  # Resize for better detection
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=30)  # Increase contrast

    # ✅ Human detection
    human_check = detect_human(image)
    if human_check.get("error"):
        return human_check

    # ✅ AI Model se Height & Weight predict karna
    height, weight = estimate_height_weight(image)

    return {"height": height, "weight": weight}


# ✅ Human Detection Function
def detect_human(image):
    haarcascade_path = cv2.data.haarcascades + "haarcascade_fullbody.xml"
    
    if not os.path.exists(haarcascade_path):
        return {"error": "Haarcascade model not found", "details": haarcascade_path}

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    haar_cascade = cv2.CascadeClassifier(haarcascade_path)
    
    bodies = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
    
    print(f"Detected bodies: {len(bodies)}")  # Debugging log
    
    if len(bodies) == 0:
        cv2.imwrite("debug_no_human.jpg", image)  # Debugging ke liye image save karein
        return {"error": "No human detected. Please upload a clearer image with a visible person."}
    
    return {"success": True}


# ✅ Height & Weight Estimation AI Model
def estimate_height_weight(image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
    ])
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        prediction = model(image_tensor)

    height, weight = prediction[0][0].item(), prediction[0][1].item()
    return height, weight


# ✅ Main Execution with Proper Error Handling
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided. Please upload an image."}))
        sys.exit(1)

    image_path = sys.argv[1]

    try:
        result = process_image(image_path)

        # ✅ Ensure response is always JSON
        if "error" in result:
            print(json.dumps(result))
        else:
            result["height"] = round(result["height"] * 24, 2)  # Adjust height scaling
            result["weight"] = round(result["weight"] * 9, 2)  # Adjust weight scaling
            
            print(json.dumps(result))
    
    except Exception as e:
        error_message = traceback.format_exc()
        print(json.dumps({
            "error": "Python Script Error",
            "details": error_message
        }))
        sys.exit(1)
