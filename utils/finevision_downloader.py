import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import os
from datasets import load_dataset
from multiprocessing import Pool

# ------------------------------
# Image & shard parameters
# ------------------------------
shard_size = 250
img_size = 384
n_proc = 8  # Number of parallel workers

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
    transforms.Resize((img_size, img_size)),
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
# Helper function for parallel processing
# ------------------------------

def transform_image(image):
  """Transform image to tensor - used in parallel workers."""
  return itot(image)

# ------------------------------
# Download and save shards
# ------------------------------

if __name__ == "__main__":
  print(f"Starting to process dataset with shard_size={shard_size}")
  print(f"Using {n_proc} parallel processes")
  print(f"Filtering for labels starting with {len(prefix_set)} prefixes")
  print(f"Resizing all images to {img_size}x{img_size}")

  try:
    with Pool(n_proc) as pool:
      pending_images = []
      pending_labels = []
      
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
          
          # Collect images and labels for parallel processing
          pending_images.append(sample['images'][0])
          pending_labels.append(label)
          
          # Process images in parallel when we have enough
          if len(pending_images) >= n_proc:
            image_tensors = pool.map(transform_image, pending_images)
            current_shard_images.extend(image_tensors)
            current_shard_labels.extend(pending_labels)
            total_valid += len(pending_labels)
            
            pending_images = []
            pending_labels = []
          
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
      
      # Process any remaining pending images
      if pending_images:
        image_tensors = pool.map(transform_image, pending_images)
        current_shard_images.extend(image_tensors)
        current_shard_labels.extend(pending_labels)
        total_valid += len(pending_labels)
              
  except StopIteration:
    # Process any remaining pending images
    if pending_images:
      with Pool(n_proc) as pool:
        image_tensors = pool.map(transform_image, pending_images)
        current_shard_images.extend(image_tensors)
        current_shard_labels.extend(pending_labels)
        total_valid += len(pending_labels)

  print(f"\nReached end of dataset")
    
  # Save any remaining samples as final shard
  if len(current_shard_images) > 0:
    shard_data = {
      'images': torch.stack(current_shard_images),
      'labels': current_shard_labels
    }
    split = "train"
    shard_path = f"{shard_dir}/finevision_sharegpt4vcoco_{split}_{shard_index:06d}.pt"
    torch.save(shard_data, shard_path)
    print(f"Saved final {shard_path} with {len(current_shard_images)} samples")
          
  print(f"\nComplete! Processed {total_processed} samples, saved {total_valid} valid samples in {shard_index + 1} shards")
