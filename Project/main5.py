from collections import defaultdict
import cv2
import numpy as np

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Open the video file
video_path = "people-walking.mp4"
cap = cv2.VideoCapture(video_path)
# Store the track history and overall movement direction for each track id
track_history = defaultdict(lambda: [])
overall_direction = {}

# Direction counters
direction_counts = {
    "Up": 0,
    "Down": 0,
    "Left": 0,
    "Right": 0,
    "Northwest": 0,
    "Northeast": 0,
    "Southwest": 0,
    "Southeast": 0,
}

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if success:
        results = model.track(frame, persist=True)
        # print(results)

        # Check if any boxes are detected in the current frame
        if results[0].boxes.id is not None:
            # Get the boxes and track IDs
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Plot the tracks and determine object direction
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = box
                x, y = (x1 + x2) / 2, (y1 + y2) / 2  # x, y center point
                track = track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

                    y_coords = [coord[1] for coord in track]
                    y_change = y_coords[-1] - y_coords[0]
                    x_coords = [coord[0] for coord in track]
                    x_change = x_coords[-1] - x_coords[0]

                    if y_change < 0:
                        if x_change < 0:
                            if abs(x_change/y_change) > 0 and abs(x_change/y_change) <= 0.57:
                                direction = "Up"
                            elif abs(x_change/y_change) >= 1.73:
                                direction = "Left"    
                            else:
                                direction = "Northwest"  
                        if x_change > 0:
                            if abs(x_change/y_change) > 0 and abs(x_change/y_change) <= 0.57:
                                direction = "Up"
                            elif abs(x_change/y_change) >= 1.73:
                                direction = "Right"    
                            else:
                                direction = "Northeast"    
                    elif y_change > 0:
                        if x_change < 0:
                            if abs(y_change/x_change) > 0 and abs(y_change/x_change) <= 0.57:
                                direction = "Left"
                            elif abs(y_change/x_change) >= 1.73:
                                direction = "Down"    
                            else:
                                direction = "Southwest"  
                        if x_change > 0:
                            if abs(y_change/x_change) > 0 and abs(y_change/x_change) <= 0.57:
                                direction = "Right"
                            elif abs(y_change/x_change) >= 1.73:
                                direction = "Down"    
                            else:
                                direction = "Southeast"                       
                    else:
                        if x_change == 0:
                            direction = "Up"
                        if x_change > 0:
                            direction = "Down"

                    # Update overall movement direction for the track id
                    overall_direction[track_id] = direction
            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                annotated_frame,
                [points],
                isClosed=False,
                color=(230, 230, 230),
                thickness=10,
            )
            # Display the annotated frame
            cv2.imshow("YOLOv8 Tracking", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    else : 
        break

# Increment the direction count only for the last direction of each track id
for track_id, last_direction in overall_direction.items():
    direction_counts[last_direction] += 1

# Print the direction counts
print("Direction Counts:")
for direction, count in direction_counts.items():
    print(f"{direction}: {count}")

cap.release()
cv2.destroyAllWindows()
