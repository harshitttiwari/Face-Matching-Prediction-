import json

# Face Matching Example
from deepface import DeepFace 
result = DeepFace.verify(
    img1_path="s3.jpg",
    img2_path="s5.jpg",
    threshold=0.6,
    enforce_detection=False  # Allow processing even if face/landmarks are not fully detected
)

print(json.dumps(result, indent=4))
