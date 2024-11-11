import os
import random
import shutil

# Paths to your dataset
dataset_dir = 'drone_dataset_yolo/dataset_txt'  # replace with the path to your dataset
image_folder = dataset_dir  # both images and labels are in the same folder
output_dir = 'dataset_drones'  # this will be the parent directory for the new folder structure

# Split ratio for train/val/test
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure the output directory structure exists
os.makedirs(os.path.join(output_dir, 'train/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'valid/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'train/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'valid/labels'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'test/labels'), exist_ok=True)

# Get list of all image files (.jpg) and label files (.txt)
image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
label_files = [f.replace('.jpg', '.txt') for f in image_files]

# Shuffle the dataset for randomness
combined = list(zip(image_files, label_files))
random.shuffle(combined)
image_files, label_files = zip(*combined)

# Calculate split indices
total_images = len(image_files)
train_size = int(train_ratio * total_images)
val_size = int(val_ratio * total_images)

# Split the data
train_images = image_files[:train_size]
val_images = image_files[train_size:train_size+val_size]
test_images = image_files[train_size+val_size:]

# Helper function to copy images and their labels to the respective directories
def copy_files(image_list, label_list, src_dir, dst_image_dir, dst_label_dir):
    for img, label in zip(image_list, label_list):
        # Copy the image
        shutil.copy(os.path.join(src_dir, img), os.path.join(dst_image_dir, img))
        
        # Copy the corresponding label file
        shutil.copy(os.path.join(src_dir, label), os.path.join(dst_label_dir, label))

# Copy the files into the appropriate directories
copy_files(train_images, label_files[:train_size], image_folder, os.path.join(output_dir, 'train/images'), os.path.join(output_dir, 'train/labels'))
copy_files(val_images, label_files[train_size:train_size+val_size], image_folder, os.path.join(output_dir, 'valid/images'), os.path.join(output_dir, 'valid/labels'))
copy_files(test_images, label_files[train_size+val_size:], image_folder, os.path.join(output_dir, 'test/images'), os.path.join(output_dir, 'test/labels'))

print(f"Dataset split completed: {len(train_images)} for train, {len(val_images)} for val, {len(test_images)} for test.")
