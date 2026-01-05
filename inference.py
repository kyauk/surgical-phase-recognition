import os
import sys
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from models.model import SurgicalModel
from constants import SEQ_LEN

def run_inference(frames_dir, model_path, output_txt):
     device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

     # Init model

     model = SurgicalModel
     model.load_state_dict(torch.load(model_path, map_location=device))
     model.to(device)
     model.eval()

     # Transform image
     transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
     ])

     # Frame processing
     frame_files = sorted([f for f in os.listdir(frames_dir) if f.lower().endswith('.jpg')])
     predictions = []
     frame_buffer = []

     with torch.no_grad():
        for frame_file in tqdm(frame_files, desc="Processing Video"):
            img = Image.open(os.path.join(frames_dir, frame_file)).convert('RGB')
            frame_tensor = transform(img)
            frame_buffer.append(frame_tensor)

            if len(frame_buffer) > SEQ_LEN:
                frame_buffer.pop(0)
            
            # If sequence is too short, pad it
            if len(frame_buffer < SEQ_LEN):
                padding = frame_bfufer[0] * (SEQ_LEN - len(frame_buffer))
                input_seq = torch.stack(padding + frame_buffer)
                else:
                    input_seq = torch.stack(frame_buffer)

                input_batch = input_seq.unsqueeze(0).to(device)
                output = model(input_batch)
                _, pred = output[0, -1].max(0)
                predictions.append(pred.item())

        # Write predictions into a .txt
        with open(output_txt, 'w') as f:
            f.write("Frame\tPhase\n")
            for i, p in enumerate(predicitons):
                f.write(f"{i}\t{p}\n")