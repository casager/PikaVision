# PikaVision

Requirements:
  pip install ultralytics
  Change path2pika global variable according to where your local machines stores the repo

source for drone dataset https://www.kaggle.com/datasets/dasmehdixtr/drone-dataset-uav



## Latest model:
location: runs/detect/5_categories_no_aug/weights/best.pt

latest model is implemented in **project3.ipynb**

To download dataset through Jupyter you must update the API key. If you cant get it to work just lmk.

This is trained on the dataset found at the link below. 
No augmentation was applied to the dataset before training. Better performance may be achievable with data augmentation.

Full dataset with no augmentation: https://app.roboflow.com/pikavision/project2-dataset/3

### Results
Validating runs/detect/train6/weights/best.pt...
| Class      | Images | Instances | Precision (P) | Recall (R) | mAP@50 | mAP@50-95 |
|------------|--------|-----------|---------------|------------|--------|-----------|
| All        | 973    | 1019      | 0.937         | 0.948      | 0.968  | 0.811     |
| Cat        | 117    | 117       | 0.967         | 0.995      | 0.995  | 0.923     |
| Dog        | 241    | 241       | 0.967         | 0.979      | 0.993  | 0.867     |
| Drone      | 232    | 244       | 0.965         | 0.963      | 0.990  | 0.702     |
| Person     | 194    | 361       | 0.824         | 0.823      | 0.877  | 0.699     |
| Pikachu    | 56     | 56        | 0.965         | 0.978      | 0.986  | 0.860     |




