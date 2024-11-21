import os
import random
import shutil
from math import ceil




def train_test_split(src_folder_path, percentage_valid=10, percentage_test=10):
    """
    Performs train, test, split at a given percentage for test and validation datasets.
    Creates test and valid folders in the parent of data with their own images and labels folders.
    

    Args:
        src_folder (str): Path to the source folder.
        percentage_valid (int): Percentage of overall dataset to be used in validation dataset.
        percentage_test (int): Percentage of overall dataset to be used in test dataset.
    
    Returns: 
        None
    """
    
        
    # Ensure the source and destination folders exist
    if not os.path.exists(src_folder_path):
        raise ValueError(f"Source folder '{src_folder_path}' does not exist.")
    
    folders_to_create = ['valid', 'test']
    
    # Create validation and test directories
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'valid'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'test'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'valid', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'test', 'images'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'valid', 'labels'), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(src_folder_path), 'test', 'labels'), exist_ok=True)
    


    
    for folder in folders_to_create:
        
        # Get a list of all files in the source folder
        files = [f for f in os.listdir(os.path.join(src_folder_path, 'labels'))]
        total_files = len(files)
        print('Total images in src directory:', total_files)
        files_name_only = [file.rsplit('.', 1)[0] for file in files]  
        file_move_dict = {'valid': ceil((percentage_valid/100)*total_files), 'test':ceil((percentage_test/100)*total_files) }
        # Randomly select files to move
        files_to_move = random.sample(files_name_only, file_move_dict.get(folder))
       
        
        # Move the files
        for file in files_to_move:
            print(file)
            src_img_path = os.path.join(src_folder_path, 'images', file + '.jpg')
            dest_img_path = os.path.join(os.path.dirname(src_folder_path), folder, 'images', file + '.jpg')

            src_label_path = os.path.join(src_folder_path, 'labels', file + '.txt')
            dest_label_path = os.path.join(os.path.dirname(src_folder_path), folder, 'labels', file + '.txt')

            shutil.move(src_img_path, dest_img_path)
            shutil.move(src_label_path, dest_label_path)
        
        print(f'Moved {file_move_dict.get("valid")} random files to the valid folder.')
        print(f'Moved {file_move_dict.get("test")} random files to the test folder.')
        

    



