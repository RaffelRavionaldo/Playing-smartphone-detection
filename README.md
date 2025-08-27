# Install

Git clone this repo and install the library we need : 

```
pip install opencv-python==4.11.0.86
pip install ultralytics==8.3.162
```

optional, you can install PyTorch to run it on your GPU. To install it, you can read this website: https://pytorch.org/get-started/locally/

# Prepare data (Optional)

If my models do not work well on your video, you can retrain the models. First one prepare the videos in a folder, then run my prepare_data.py. It will take ~30 frames per video, and then you can label it with your favorite tools. My suggestion you can try Roboflow or use this repo: https://github.com/HumanSignal/labelImg

Don't forget to change line 8 to the path of your video. If you need more images to be taken for the video, change the number 30 in my video to a bigger number.

# Running the code

Run the detection.py to detect if someone hold the smartphone on their hand. laps etc.

change line 5, 8, and 18 to your path of model and videos.

In this code, I determine if someone is playing smartphone or not, if the overlaps between the smartphone bbox and person bbox are more than the threshold (0.3). You can change the threshold value for more/less sensitive.
