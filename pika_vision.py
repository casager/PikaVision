from os import path
from ultralytics import YOLO

# wherever you're working
path2pika = lambda *p: path.join(path.expanduser('~'), "Documents", "GitHub", "PikaVision", *p)

def test_model(model : YOLO):
    results = model.predict(
        source=path2pika("the_pikachu.jpeg"),
        conf=0.25,
    )

    for r in results:
        r.save(path2pika("the_detected_pikachu.jpeg"))


def train_pika_vision(model : YOLO) -> YOLO:
    model.train(
        data = path2pika('Dataset', 'data.yaml'),
        epochs = 2,
        imgsz = 640,
        save_dir = path2pika(),
    )
    return model

def pull_last_trained_model() -> YOLO:
    path2weights = path2pika("runs", "detect", "train", "weights", "last.pt")
    return YOLO(model=path2weights)

if __name__ == "__main__":
    model = YOLO("yolo11n.pt")
    # model = train_pika_vision(model)

    # model = pull_last_trained_model()
    test_model(model)