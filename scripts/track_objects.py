import os
import json
import glob
import argparse
import sys
from datetime import datetime

import cv2
import numpy as np
import torch
from PIL import Image

import sam3
from sam3.model_builder import build_sam3_video_predictor

# Setup precision
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def load_frames(video_path):
    """Load frames from a directory or video file."""
    if isinstance(video_path, str) and video_path.endswith(".mp4"):
        cap = cv2.VideoCapture(video_path)
        video_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
    else:
        video_frames = glob.glob(os.path.join(video_path, "*.jpg"))
        video_frames += glob.glob(os.path.join(video_path, "*.jpeg"))
        video_frames += glob.glob(os.path.join(video_path, "*.png"))
        try:
            video_frames.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
        except ValueError:
            video_frames.sort()
    
    print(f"✅ Loaded video: {len(video_frames)} frames")
    return video_frames

def mask_to_rle(binary_mask):
    """Convert binary mask to RLE."""
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    flat_mask = binary_mask.ravel(order='F')
    
    last_val = 0
    count = 0
    for val in flat_mask:
        if val != last_val:
            counts.append(count)
            count = 1
            last_val = val
        else:
            count += 1
    counts.append(count)
    return rle

def mask_to_bbox(binary_mask):
    """Extract bounding box [x, y, width, height]."""
    
    if binary_mask.ndim != 2:
        # Se for 1D ou 0D, retorna um bbox vazio para evitar o AxisError.
        return [0, 0, 0, 0]
        
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    
    # Adicionando um bloco try/except para capturar IndexErrors se np.where retornar vazio
    # (embora o 'if not rows.any() or not cols.any()' já cubra a maioria dos casos)
    try:
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
    except IndexError:
        return [0, 0, 0, 0]
    
    # Retorna o bbox no formato [xmin, ymin, width, height]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]

def propagate_in_video(predictor, session_id):
    """Propagate annotations from the initial frame to the entire video."""
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request=dict(
            type="propagate_in_video",
            session_id=session_id,
        )
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame

def process_class(predictor, video_frames, class_name, class_id, output_dir):
    """Process a single class: prompt, propagate, and save results."""
    print(f"\nProcessing class: {class_name} (ID: {class_id})")
    
    # Start a session
    # Note: For image folders, resource_path should be the folder path if possible, 
    # but the API might expect a list of paths or a video path.
    # The example notebook passes the folder path.
    
    # We need to handle if video_frames is a list of paths or numpy arrays.
    # The predictor expects a path to video or folder.
    
    # Assuming video_frames passed here is just for info, we need the path.
    # But wait, we need to know the path to pass to start_session.
    # Let's pass the path to this function instead of loaded frames if possible,
    # or handle it.
    pass

def save_coco_json(outputs_per_frame, video_frames, class_name, class_id, output_path):
    """Export annotations to COCO JSON format."""
    coco_data = {
        "info": {
            "description": f"SAM 3 Video Annotations - {class_name}",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [
            {
                "id": int(class_id),
                "name": class_name,
                "supercategory": "object",
            }
        ],
    }
    
    # Get dimensions from first frame
    if isinstance(video_frames[0], str):
        first_frame = np.array(Image.open(video_frames[0]))
    else:
        first_frame = video_frames[0]
    
    height, width = first_frame.shape[:2]
    
    annotation_id = 1
    
    # Process each frame
    for frame_idx in sorted(outputs_per_frame.keys()):
        image_id = frame_idx + 1
        
        # Add image info if not already added 
        if isinstance(video_frames[frame_idx], str):
            file_name = os.path.basename(video_frames[frame_idx])
        else:
            file_name = f"frame_{frame_idx:05d}.jpg"
            
        # Check if image is already in coco_data
        img_exists = False
        for img in coco_data["images"]:
            if img["id"] == image_id:
                img_exists = True
                break
        
        if not img_exists:
            coco_data["images"].append({
                "id": image_id,
                "file_name": file_name,
                "height": height,
                "width": width,
                "frame_index": frame_idx,
            })
        
        frame_data = outputs_per_frame[frame_idx]
        
        # frame_data é um dict de obj_id -> result
        
        for sam_obj_id, obj_data in frame_data.items():
            # Tenta extrair a máscara. Se for um dicionário e tiver a chave "mask", usa ela.
            mask = obj_data["mask"] if isinstance(obj_data, dict) and "mask" in obj_data else obj_data
            
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            
            # 🛑 NOVO: Verifica se a máscara é um array NumPy antes de qualquer operação.
            # Isso evita o TypeError se 'mask' for um dict (retorno de erro do SAM 3).
            if not isinstance(mask, np.ndarray) or mask.ndim < 2:
                # print(f"Aviso: Objeto '{class_name}' no quadro {frame_idx} (ID: {sam_obj_id}) retornou um tipo/dimensão inválida. Ignorando.")
                continue
                
            # Agora a operação de comparação funciona, pois 'mask' é um ndarray
            binary_mask = (mask > 0).astype(np.uint8)
            
            # A função mask_to_bbox (que deve estar corrigida no seu código) é chamada aqui
            bbox = mask_to_bbox(binary_mask)
            area = int(np.sum(binary_mask))
            
            if area == 0:
                continue
            
            rle = mask_to_rle(binary_mask)
            
            coco_data["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": int(class_id),
                "segmentation": rle,
                "area": area,
                "bbox": bbox,
                "iscrowd": 0,
                # Store original SAM object ID for reference if needed
                "sam_object_id": sam_obj_id 
            })
            
            annotation_id += 1
            
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"✅ Saved annotations for '{class_name}' to: {output_path}")
    return coco_data

def merge_json_files(json_files, output_path):
    """Merge multiple COCO JSON files into one."""
    if not json_files:
        print("No JSON files to merge.")
        return

    merged_data = {
        "info": {
            "description": "SAM 3 Video Annotations - Merged",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [],
    }
    
    # To avoid duplicate images and categories
    image_map = {} # file_name -> image_info
    category_map = {} # id -> category_info
    
    annotation_id_counter = 1
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        # Merge Categories
        for cat in data["categories"]:
            if cat["id"] not in category_map:
                category_map[cat["id"]] = cat
        
        # Merge Images
        for img in data["images"]:
            if img["file_name"] not in image_map:
                image_map[img["file_name"]] = img
        
        # Merge Annotations
        for ann in data["annotations"]:
            new_ann = ann.copy()
            new_ann["id"] = annotation_id_counter
            annotation_id_counter += 1
            merged_data["annotations"].append(new_ann)
            
    merged_data["categories"] = list(category_map.values())
    merged_data["images"] = list(image_map.values())
    
    # Sort categories by ID
    merged_data["categories"].sort(key=lambda x: x["id"])
    # Sort images by ID
    merged_data["images"].sort(key=lambda x: x["id"])
    
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=2)
        
    print(f"\n✅ Merged JSON saved to: {output_path}")
    print(f"   • Total images: {len(merged_data['images'])}")
    print(f"   • Total annotations: {len(merged_data['annotations'])}")
    print(f"   • Total categories: {len(merged_data['categories'])}")

def main():
    parser = argparse.ArgumentParser(description="SAM 3 Multi-Class Video Tracking")
    parser.add_argument("--input_path", type=str, required=True, help="Path to video file or folder of images")
    parser.add_argument("--output_path", type=str, required=True, help="Directory to save output JSONs")
    args = parser.parse_args()
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    # Load frames to check and get list
    video_frames = load_frames(args.input_path)
    if not video_frames:
        print("No frames found!")
        return

    # Initialize Predictor
    print("Initializing SAM 3 Video Predictor...")
    video_predictor = build_sam3_video_predictor()
    
    # Get list of objects from user
    print("\n📝 Enter the list of objects to track (one per line).")
    print("   Press Enter on an empty line to finish.")
    
    object_classes = []
    while True:
        try:
            line = input("Object name: ").strip()
            if not line:
                break
            object_classes.append(line)
        except EOFError:
            break
            
    if not object_classes:
        print("No objects entered. Exiting.")
        return
        
    print(f"\n🚀 Starting tracking for {len(object_classes)} classes: {object_classes}")
    
    generated_jsons = []
    
    # Process each class
    # We use a 1-based class ID counter
    for i, class_name in enumerate(object_classes):
        class_id = i + 1
        
        # Start Session
        # We need to pass the resource path. If input_path is a folder, pass that.
        # If it's a list of frames, we might need to handle differently, but SAM 3 API usually takes a path.
        
        print(f"\n--- Processing Class {class_id}/{len(object_classes)}: '{class_name}' ---")
        
        try:
            response = video_predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=args.input_path,
                )
            )
            session_id = response["session_id"]
            
            # Add Prompt to Frame 0
            print(f"   • Adding text prompt '{class_name}' to frame 0...")
            video_predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=session_id,
                    frame_index=0,
                    text=class_name,
                )
            )
            
            # Propagate
            print("   • Propagating annotations...")
            outputs = propagate_in_video(video_predictor, session_id)
            
            # Save JSON
            json_filename = f"{class_name.replace(' ', '_')}_id{class_id}.json"
            json_path = os.path.join(args.output_path, json_filename)
            save_coco_json(outputs, video_frames, class_name, class_id, json_path)
            
            generated_jsons.append(json_path)
            
            # Close session to free memory/reset state for next class
            # Assuming start_session creates a new isolated session, but good to be clean.
            # The API doesn't explicitly show close_session in the example, but start_session returns a new ID.
            
        except Exception as e:
            print(f"❌ Error processing class '{class_name}': {e}")
            import traceback
            traceback.print_exc()
            
    # Merge all JSONs
    if generated_jsons:
        merge_path = os.path.join(args.output_path, "final_merged_annotations.json")
        merge_json_files(generated_jsons, merge_path)

if __name__ == "__main__":
    main()
