import cv2
import json
import os
import glob
from ultralytics import YOLO
from datetime import datetime

class PCBInspector:
    def __init__(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        self.model = YOLO(model_path)

    def analyze(self, image_path):
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading {image_path}")
            return

        # Inference
        results = self.model.predict(frame, conf=0.25, verbose=False)[0]
        
        report = {
            "file": os.path.basename(image_path),
            "timestamp": datetime.now().isoformat(),
            "defects": []
        }

        h, w, _ = frame.shape
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = self.model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            
            # Severity Logic
            area_ratio = ((x2-x1)*(y2-y1)) / (w*h)
            severity = "Critical" if area_ratio > 0.005 else "Minor"

            report["defects"].append({
                "type": label,
                "confidence": round(conf, 3),
                "severity": severity,
                "bbox": [x1, y1, x2, y2]
            })

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Print JSON
        print(json.dumps(report, indent=2))
        
        # Save Result
        output_name = f"result_{os.path.basename(image_path)}"
        cv2.imwrite(output_name, frame)
        print(f"Saved visual output to: {output_name}")

if __name__ == "__main__":
    # 1. Find the latest trained model automatically
    # It will be in runs/detect/pcb_final_model/weights/best.pt
    try:
        model_path = glob.glob("runs/detect/pcb_final_model*/weights/best.pt")[-1]
        print(f"Using model: {model_path}")
        
        inspector = PCBInspector(model_path)
        
        # 2. Test on a validation image
        test_images = glob.glob("dataset_fixed/val/images/*.jpg")
        if test_images:
            inspector.analyze(test_images[0])
        else:
            print("No test images found.")
            
    except IndexError:
        print("Error: Model file not found. Did the training finish successfully?")