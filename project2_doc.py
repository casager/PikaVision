from ultralytics import YOLO
from roboflow import Roboflow
import os
import matplotlib.pyplot as plt
import math
import random
from PIL import Image
from shuffle import train_test_split
import pydoc

"""
This python module serves the purpose of performing basic operation using the YOLO framework. 

Functions: 
    path_to: Provides a convenient way of locating the files in your system.
    download_dataset: Downloads the specified version of the dataset from Roboflow.
    test_model: Makes a prediction using the model and a test image. 
    train_model: Trains the model on the data specified in a .yaml file.
    
"""



def path_to(*p):
    """
    Takes an arbitrary number of strings and converts it to a path
    with the current working directory automatically tacked onto the front of the path.
    
    Parameters:
        *p (str): Single string or list of strings.
    
    Returns:
        (string): Full path to desired directory/file
    """
    return os.path.join(os.getcwd(), *p)


def download_dataset(v: int):
    """
    Function that takes the dataset version and downloads if from our Roboflow Project.
    The dataset will be save under project2-dataset-{v}.
    
    Parameters:
        v (int): the version number of the dataset that you want to download
    Returns: 
        None: This function does not return anything.
        
    """
    rf = Roboflow(api_key="jSHmBYOVLG3O81wzzHz3")
    project = rf.workspace("pikavision").project("project2-dataset")
    version = project.version(v)
    dataset = version.download("yolov11")
    
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
        file_move_dict = {'valid': math.ceil((percentage_valid/100)*total_files), 'test':math.ceil((percentage_test/100)*total_files) }
        # Randomly select files to move
        files_to_move = random.sample(files_name_only, file_move_dict.get(folder))
       
        
        # Move the files
        for file in files_to_move:
            print(file)
            src_img_path = os.path.join(src_folder_path, 'images', file + '.jpg')
            dest_img_path = os.path.join(os.path.dirname(src_folder_path), folder, 'images', file + '.jpg')

            src_label_path = os.path.join(src_folder_path, 'labels', file + '.txt')
            dest_label_path = os.path.join(os.path.dirname(src_folder_path), folder, 'labels', file + '.txt')

            os.shutil.move(src_img_path, dest_img_path)
            os.shutil.move(src_label_path, dest_label_path)
        
        print(f'Moved {file_move_dict.get("valid")} random files to the valid folder.')
        print(f'Moved {file_move_dict.get("test")} random files to the test folder.') 

def test_model(model : YOLO, sample_image : str):
    """
    Takes the model and a sample image and then produces a prediction image.
    The function saves the image to the test_images folder.
    It also shows the predictions using matplotlib.
    
    Parameters:
        model (YOLO): YOLO model that you want to make a prediction with.
        sample_image (str): Sample image to make the prediction on. Can by .png or .jpg
    
    Returns:
        None: This function does not return any value.

    """
    path_to_img = path_to('test_images', sample_image)
    results = model.predict(
        source=path_to_img,
        conf=0.1,
    )

    for r in results:
        name = 'the_detected_' + sample_image
        path_to_predicted_img = path_to('test_images', name)
        r.save(path_to_predicted_img)
    image = Image.open(path_to_predicted_img)
    plt.imshow(image)
    plt.axis('off')
    plt.show()
        

def train_model(model : YOLO, yaml_file : str) -> YOLO:
    """
    Takes the model and a .yaml file and trains the model on the dataset specified in the .yaml file.
    Saves the results of the training to runs/detect
    
    Parameters:
        model (YOLO): The pretrained model that you want want to train ontop of. 
        yaml_file (str): The .yaml file that specifies the data that you want to train on.
        
    Returns: 
        None: This function does not return anything.
    """
    results = model.train(
        data = path_to('project2-dataset-3', yaml_file),
        epochs = 100,
        imgsz = 640,
        save_dir = path_to(),
        device = 'gpu',
        batch=16
        
    )
    
    
# Load custom model
new_model = YOLO(path_to('runs', 'detect', '5_categories_no_aug', 'weights', 'best.pt'))


print('Press Ctrl + C to quit.')

while True: 
    image = input("Enter name of image to make a prediction on.\n")
    try:
        test_model(new_model, image)
    except Exception as e:
        print(f'An error has occured: {e}')
    
