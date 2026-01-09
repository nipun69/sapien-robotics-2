from ultralytics import YOLO

def train_model():
    # Load model
    model = YOLO('yolov8n.pt') 

    # Point to the NEW yaml created by Step 1
    yaml_path = "dataset_fixed/data.yaml"

    print(f"Training on: {yaml_path}")
    
    results = model.train(
        data=yaml_path,
        epochs=50, 
        imgsz=640,
        batch=16,
        name='pcb_final_model'
    )
    
    print(f"Done! Best model: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    train_model()