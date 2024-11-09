from os import path
from ultralytics import YOLO
import os

# wherever you're working
# path2pika = lambda *p: path.join(path.expanduser('~'), "Documents", "GitHub", "PikaVision", *p)
path2pika = lambda *p : path.join(os.getcwd(), *p)

def test_model(model : YOLO, sample_image):
    results = model.predict(
        source=path2pika(sample_image),
        conf=0.25,
    )

    for r in results:
        r.save(path2pika("the_detected_pikachu.jpeg"))


def train_pika_vision(model : YOLO) -> YOLO:
    results = model.train(
        data = path2pika('datasets','pikachu', 'data.yaml'),
        epochs = 100,
        imgsz = 640,
        save_dir = path2pika(),
    )
    return model, results

def pull_last_trained_model() -> YOLO:
    path2weights = path2pika("runs", "detect", "train", "weights", "last.pt")
    return YOLO(model=path2weights)

if __name__ == "__main__":
    model = YOLO(path2pika('runs', 'detect', 'Pikachu', 'weights', 'best.pt'))
    # model, results = train_pika_vision(model, 'data.yaml')

    # model = pull_last_trained_model()
    test_model(model, 'the_pikachu.jpeg') 