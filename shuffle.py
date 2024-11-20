import os
import random
import shutil
from math import ceil




def train_test_split(src_folder, percentage_valid=10, percentage_test=10):
    """
    Performs train, test, split at a given percentage for test and validation datasets.
    Creates test and valid folders in the parent of data with their own images and labels folders.
    

    Args:
        src_folder (str): Path to the source folder.
        percentage_valid (int): Percentage of overall dataset to be used in validation dataset.
        percentage_test (int): Percentage of overall dataset to be used in test dataset.
    """
    
        
    # Ensure the source and destination folders exist
    if not os.path.exists(src_folder):
        raise ValueError(f"Source folder '{src_folder}' does not exist.")
    
    # Create validation and test directories
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'valid'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'test'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'valid', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder), 'test', 'labels'), exist_ok=True)
    

    # Get a list of all files in the source folder
    files = [f for f in os.listdir(os.path.join(src_folder, 'labels'))]
    total_files = len(files)
    print('Total images in src directory:', total_files)
    files_name_only = [file.rsplit('.', 1)[0] for file in files]
    # Calculate the number of files to move
    num_files_to_move_valid = ceil((percentage_valid/100)*total_files)

    # Randomly select files to move
    files_to_move = random.sample(files_name_only, num_files_to_move_valid)
    
    # Move the files
    for file in files_to_move:
        print(file)
        src_img_path = os.path.join(src_folder, 'images', file + '.jpg')
        dest_img_path = os.path.join(os.path.dirname(src_folder), 'valid', 'images', file + '.jpg')

        src_label_path = os.path.join(src_folder, 'labels', file + '.txt')
        dest_label_path = os.path.join(os.path.dirname(src_folder), 'valid', 'labels', file + '.txt')

        shutil.move(src_img_path, dest_img_path)
        shutil.move(src_label_path, dest_label_path)
    
    
     # Get a list of all files in the source folder
    files = [f for f in os.listdir(os.path.join(src_folder, 'labels'))]
    files_name_only = [file.rsplit('.', 1)[0] for file in files]
    # Calculate the number of files to move
    num_files_to_move_test = ceil((percentage_test/100)*total_files)

    # Randomly select files to move
    files_to_move = random.sample(files_name_only, num_files_to_move_test)
    
    # Move the files
    for file in files_to_move:
        src_img_path = os.path.join(src_folder, 'images', file + '.jpg')
        dest_img_path = os.path.join(os.path.dirname(src_folder), 'test', 'images', file + '.jpg')

        src_label_path = os.path.join(src_folder, 'labels', file + '.txt')
        dest_label_path = os.path.join(os.path.dirname(src_folder), 'test', 'labels', file + '.txt')

        shutil.move(src_img_path, dest_img_path)
        shutil.move(src_label_path, dest_label_path)
    
    print(f'{num_files_to_move_valid} random validation images pulled from training data.')
    print(f'{num_files_to_move_test} random test images pulled from training data.')  
    



