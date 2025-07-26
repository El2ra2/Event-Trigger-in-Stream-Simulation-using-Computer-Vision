from ultralytics import YOLO
import cv2
import json
import matplotlib.pyplot as plt

# Loading the yolov10 model, the video, and defining integers and lists
model = YOLO('yolov10n.pt')

#loading video
video_path = './input_video.mp4'
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)          # get the fps of the video

frame_skip = 3                           # number of skipped frames
frame_index = 0                          # integer defining the number of the frame every loop
scale_percent = 50                       # percetage the video will be resized to
all_detections = []                      # stores detected labels, box-coordinates, conf. scores to pass into JSON
time_list = []                           # stores the timestamps of people detected
people_count_list =[]                    # stores the number of people detected

ret = True
while ret:                                                                                   # reads every frame
    ret, frame = cap.read()
    if ret:
        results = model.track(frame, persist = True)                                # detects and tracks objects
        frame_ = results[0].plot()                                     # plots the detected objects on the image

        height, width, _ = frame_.shape                                   # resizing the image to fit the screen
        new_height = int(height * scale_percent / 100)
        new_width = int(width * scale_percent / 100)
        resized = cv2.resize(frame_,(new_width,new_height), interpolation=cv2.INTER_AREA)

        if frame_index % frame_skip == 0:                                    # executes every skip of frames = 5
            frame_detections = []                                                     # stores frame information
            person_detected = 0                                             # stores number of people each frame

            for r in results:                                        # loop stores objects detected in the frame
                for box in r.boxes:
                    cls_id = int(box.cls)
                    label = model.model.names[cls_id]
                    conf = float(box.conf)

                    if (label == 'person') & (conf > 0.5): # if person is detected & confidence greater than 50%
                        person_detected += 1
                        if person_detected >=3:              # if more than 3 people, stores timestamp and count
                            detection_time = frame_index / fps
                            frame_detections.append([label, detection_time, conf])
                            time_list.append(detection_time)
                            people_count_list.append(person_detected)

            if frame_detections:                                              #  If more than 3 people detected,

                frame_json = {
                    "frame number": frame_index,
                    "time (seconds)": detection_time,
                    "frame detections": frame_detections
                }
                all_detections.append(frame_json)                                    # updates frame detections,

                text_width = int(new_width/5)
                text_height = int(new_height *3/4)
                resized = cv2.putText(resized, "TOO MANY PEOPLE!!!",                     # and triggers an alert
                                      (text_width,text_height),
                                      cv2.FONT_HERSHEY_SIMPLEX , 0.7 , (0 , 0 , 255) , 2)

        cv2.imshow('frame', resized)                               # outputs the image in a window named 'frame'
        if cv2.waitKey(25) & 0xFF == ord('q'):                              # and waits for 25 frames = 1 second
            break
        path = str('./output_images/frame_' + str(frame_index) + '.jpg')
        cv2.imwrite(path, resized)                                                      # stores annonated frame

        frame_index += 1                                                                 # updates to next frame

# Storing the detections in json file
with open("detections.json","w") as f:
    json.dump(all_detections, f, indent = 2)

# Creating video file for every third frame,
# upto the last frame, while also updating the progress in the console
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter('./output_video/video.mp4', fourcc, 30, (new_width,new_height))

for j in range(0, frame_index):
    path = str('./output_images/frame_' + str(j) + '.jpg')
    if path is None:
        break
    img = cv2.imread(path)
    if j % 300 == 0:                             # Prints the progress to console every 300 frames
        print("Processing...    frames done:"+str(j)+"/"+str(frame_index))
    video.write(img)
    j += 1

# Releasing video windows
video.release()
cap.release()
cv2.destroyAllWindows()

# Plotting counts of people (y-axis) vs. their timestamps (x-axis)
plt.figure(figsize=(10, 6))
plt.bar(time_list,people_count_list)
plt.title("Time of crowd detection")
plt.ylabel("Count of people")
plt.xlabel("Time (in seconds)")
plt.tight_layout()
plt.savefig("Crowd detection bar chart")
plt.show()