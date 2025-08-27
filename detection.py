import cv2
from ultralytics import YOLO

# Load YOLOv8m pretrained on COCO dataset
model = YOLO("yolo_models.pt")

# Input video
video_path = "your_input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video writer
out = cv2.VideoWriter(
    "path_to_save_video.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height)
)

def overlap_ratio(inner, outer):
    """Calculate the ratio of the inner box area (hp) to the outer box (person)."""
    xA = max(inner[0], outer[0])
    yA = max(inner[1], outer[1])
    xB = min(inner[2], outer[2])
    yB = min(inner[3], outer[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    innerArea = (inner[2] - inner[0]) * (inner[3] - inner[1])

    return interArea / float(innerArea + 1e-6) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference
    results = model(frame, conf=0.2, verbose=False)

    boxes = results[0].boxes.xyxy.cpu().numpy()  # [x1,y1,x2,y2]
    scores = results[0].boxes.conf.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().astype(int)

    persons = []
    phones = []

    # Separate persons and phones
    for box, score, cls in zip(boxes, scores, classes):
        if cls == 0:  # person
            persons.append(box)
        elif cls == 1:  # cell phone
            phones.append((box, score))
            
    # Check if phone it's in someones bbox
    for phone_box, phone_score in phones:
        for human_box in persons:
            if overlap_ratio(phone_box, human_box) > 0.3:
                x1, y1, x2, y2 = phone_box.astype(int)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"phone {phone_score:.2f}",
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
                break  # Only need one match

    out.write(frame)
    cv2.imshow("Result", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
