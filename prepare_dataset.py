import cv2
import os
import glob
import random
import shutil

# Paths
video_dir = "video_test"
output_dir = "train_phone_smartphone"
train_dir = os.path.join(output_dir, "train")
test_dir = os.path.join(output_dir, "test")

# Make directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Collect all videos
video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
              glob.glob(os.path.join(video_dir, "*.avi")) + \
              glob.glob(os.path.join(video_dir, "*.mov"))

all_images = []

for vid_idx, video_path in enumerate(video_files):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count < 30:
        step = 1
    else:
        step = frame_count // 30  # sample frames evenly

    frame_idx = 0
    saved_frames = 0

    while cap.isOpened() and saved_frames < 30:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            filename = f"video{vid_idx:03d}_frame{frame_idx:05d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            all_images.append(filepath)
            saved_frames += 1

        frame_idx += 1

    cap.release()

# Shuffle all collected images
random.shuffle(all_images)

# Split train (95%) and test (5%)
split_idx = int(0.95 * len(all_images))
train_files = all_images[:split_idx]
test_files = all_images[split_idx:]

# Move files into train/ and test/ folders
for f in train_files:
    shutil.move(f, os.path.join(train_dir, os.path.basename(f)))

for f in test_files:
    shutil.move(f, os.path.join(test_dir, os.path.basename(f)))

print(f"Total images: {len(all_images)}")
print(f"Train: {len(train_files)} | Test: {len(test_files)}")