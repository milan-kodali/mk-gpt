import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from datasets import load_dataset

# ------------------------------
# Image & shard parameters
# ------------------------------
shard_size = 250
img_size = 384

# ------------------------------
# Create directory for shards
# ------------------------------
shard_dir = "./.cache/data/finevision-sharegpt4vcoco_"
os.makedirs(shard_dir, exist_ok=True)

# ------------------------------
# Load dataset
# ------------------------------
ds = load_dataset(
  'HuggingFaceM4/FineVision',
  name='sharegpt4v(coco)',
  split='train', streaming=True,
)
ds = iter(ds)

# ------------------------------
# Transform for converting images to tensors with fixed size
# ------------------------------
itot = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# ------------------------------
# Shard collection variables
# ------------------------------
current_shard = []
current_shard_images = []
current_shard_labels = []
shard_index = 0
total_processed = 0
total_valid = 0

# ------------------------------
# Prefix set, to filter for labels
# that are not super "Q&A"-y
# ------------------------------

prefix_set = {
  'This image captures',
  'The image shows',
  'The image depicts',
  'The image features',
  'The image presents',
  'The image showcases',
  'The image portrays',
  'The image illustrates',
  'The image displays',
  'The image highlights',
  'The image captures',
  'In the image,',
  'In this image,',
  'The image is',
  'The image contains',
  'The image displays',
  'The image shows',
  'The image depicts',
  'The image features',
  'The image presents',
  'This image is',
  'This image shows',
  'This image depicts',
  'This image features',
  'This image presents',
  'In the center of the image,',
  'In the given image,',
}

# ------------------------------
# Download and save shards
# ------------------------------

print(f"Starting to process dataset with shard_size={shard_size}")
print(f"Filtering for labels starting with {len(prefix_set)} prefixes")
print(f"Resizing all images to 384x384")

try:
  while True:
    sample = next(ds)
    total_processed += 1
    
    label = sample['texts'][0]['assistant']
    
    # Check if label starts with any prefix
    matching_prefix = None
    for prefix in prefix_set:
      if label.startswith(prefix):
        matching_prefix = prefix
        break
    
    if matching_prefix:
      label = label[len(matching_prefix):].strip()
      label = label.replace('**', '')
      
      image_tensor = itot(sample['images'][0])
      current_shard_images.append(image_tensor)
      current_shard_labels.append(label)
      total_valid += 1
      
      if len(current_shard_images) >= shard_size:
        shard_data = {
          'images': torch.stack(current_shard_images),
          'labels': current_shard_labels
        }
        
        # Save shard
        split = "val" if shard_index == 0 else "train"
        shard_path = f"{shard_dir}/finevision_sharegpt4vcoco_{split}_{shard_index:06d}.pt"
        torch.save(shard_data, shard_path)
        print(f"Saved {shard_path} with {len(current_shard_images)} samples (processed {total_processed}, valid {total_valid})")
        
        # Reset for next shard
        current_shard_images = []
        current_shard_labels = []
        shard_index += 1
    
    # Progress update every 1000 samples
    if total_processed % 1000 == 0:
      print(f"Processed {total_processed} samples, found {total_valid} valid samples")
            
except StopIteration:
  print(f"\nReached end of dataset")
  # Save any remaining samples as final shard
  if len(current_shard_images) > 0:
    shard_data = {
      'images': torch.stack(current_shard_images),
      'labels': current_shard_labels
    }
    shard_path = f"{shard_dir}/shard_{shard_index:04d}.pt"
    torch.save(shard_data, shard_path)
    print(f"Saved final {shard_path} with {len(current_shard_images)} samples")
        
print(f"\nComplete! Processed {total_processed} samples, saved {total_valid} valid samples in {shard_index + 1} shards")
