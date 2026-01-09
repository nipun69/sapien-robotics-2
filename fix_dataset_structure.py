import os
import shutil
import random
import yaml
import glob
import xml.etree.ElementTree as ET

# CONFIGURATION
# Update this to match the path in your screenshot where 'PCB_DATASET' is located
INPUT_ROOT = "dataset/PCB_DATASET" 
OUTPUT_DIR = "dataset_fixed"

# Exact class names from your screenshot folders
CLASSES = [
    "Missing_hole", 
    "Mouse_bite", 
    "Open_circuit", 
    "Short", 
    "Spur", 
    "Spurious_copper"
]

def convert_xml_to_yolo(xml_file, width, height, class_id):
    """Parses XML and returns YOLO format lines."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    yolo_lines = []
    
    for obj in root.findall("object"):
        # If the XML has specific class names, we can verify them, 
        # but here we assume the folder name dictates the class ID.
        bndbox = obj.find("bndbox")
        xmin = float(bndbox.find("xmin").text)
        ymin = float(bndbox.find("ymin").text)
        xmax = float(bndbox.find("xmax").text)
        ymax = float(bndbox.find("ymax").text)
        
        # Calculate normalized coordinates
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        w = (xmax - xmin) / width
        h = (ymax - ymin) / height
        
        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
        
    return yolo_lines

def setup_structure():
    # 1. Create Clean Directory Structure
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'val']:
        os.makedirs(f"{OUTPUT_DIR}/{split}/images", exist_ok=True)
        os.makedirs(f"{OUTPUT_DIR}/{split}/labels", exist_ok=True)

    print(f"Scanning '{INPUT_ROOT}'...")
    
    total_images = 0
    
    # 2. Process Each Class Folder
    for class_id, class_name in enumerate(CLASSES):
        img_dir = os.path.join(INPUT_ROOT, "images", class_name)
        ann_dir = os.path.join(INPUT_ROOT, "Annotations", class_name)
        
        if not os.path.exists(img_dir):
            print(f"Warning: Folder not found {img_dir}")
            continue
            
        # Get all images
        images = glob.glob(os.path.join(img_dir, "*.*"))
        # Filter only image extensions
        images = [x for x in images if x.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
        
        # Split 80% Train / 20% Val
        random.shuffle(images)
        split_idx = int(len(images) * 0.8)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        print(f"Processing '{class_name}': {len(train_imgs)} train, {len(val_imgs)} val")
        
        for file_list, split in [(train_imgs, 'train'), (val_imgs, 'val')]:
            for img_path in file_list:
                filename = os.path.basename(img_path)
                name_no_ext = os.path.splitext(filename)[0]
                
                # Destination Paths
                dst_img_path = os.path.join(OUTPUT_DIR, split, "images", filename)
                dst_lbl_path = os.path.join(OUTPUT_DIR, split, "labels", name_no_ext + ".txt")
                
                # Copy Image
                shutil.copy(img_path, dst_img_path)
                
                # Handle Annotation (Supports both XML and TXT)
                # Check for XML first (Common in this dataset structure)
                xml_source = os.path.join(ann_dir, name_no_ext + ".xml")
                txt_source = os.path.join(ann_dir, name_no_ext + ".txt")
                
                if os.path.exists(xml_source):
                    # Convert XML to YOLO TXT
                    import cv2
                    img = cv2.imread(img_path)
                    h, w = img.shape[:2]
                    lines = convert_xml_to_yolo(xml_source, w, h, class_id)
                    with open(dst_lbl_path, 'w') as f:
                        f.write("\n".join(lines))
                        
                elif os.path.exists(txt_source):
                    # Copy TXT directly
                    shutil.copy(txt_source, dst_lbl_path)
                else:
                    # Create empty label file if missing to prevent errors
                    open(dst_lbl_path, 'w').close()

                total_images += 1

    # 3. Create data.yaml
    yaml_content = {
        'path': os.path.abspath(OUTPUT_DIR),
        'train': 'train/images',
        'val': 'val/images',
        'nc': len(CLASSES),
        'names': CLASSES
    }
    
    with open(f"{OUTPUT_DIR}/data.yaml", 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"\nSUCCESS! Processed {total_images} images.")
    print(f"New dataset created at: {OUTPUT_DIR}")

if __name__ == "__main__":
    setup_structure()