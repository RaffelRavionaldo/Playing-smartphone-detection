from ultralytics import YOLO
import torch

def main():
    # Clear the cache
    torch.cuda.empty_cache()

    # Load a model
    # model = YOLO("labels4.pt")  # load a partially trained model

    model = YOLO("yolov8n.pt")

    # Resume training
    results = model.train(
        batch=32,
        data='data.yaml',
        epochs=151,
        imgsz=640
        # resume=True
    )

if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    main()
