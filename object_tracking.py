import cv2
import numpy as np
import math


class ObjectDetection:
    def __init__(self, weights_path="dnn_model/yolov4.weights", cfg_path="dnn_model/yolov4.cfg"):
        print("Loading Object Detection")
        print("Running opencv dnn with YOLOv4")
        self.nmsThreshold = 0.4
        self.confThreshold = 0.5
        self.image_size = 608

        # Load Network
        net = cv2.dnn.readNet(weights_path, cfg_path)

        # Enable GPU CUDA (if available)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.model = cv2.dnn_DetectionModel(net)

        self.classes = []
        self.load_class_names()
        self.colors = np.random.uniform(0, 255, size=(80, 3))

        self.model.setInputParams(size=(self.image_size, self.image_size), scale=1 / 255)

    def load_class_names(self, classes_path="dnn_model/classes.txt"):
        with open(classes_path, "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.colors = np.random.uniform(0, 255, size=(80, 3))
        return self.classes

    def detect(self, frame):
        class_ids, scores, boxes = self.model.detect(frame, nmsThreshold=self.nmsThreshold, confThreshold=self.confThreshold)
        filtered_boxes = []
        for i, class_id in enumerate(class_ids):
            # Filtrar solo animales (ajusta según tus clases)
            if self.classes[class_id] in ["animal", "cow", "sheep", "bird"]:
                filtered_boxes.append(boxes[i])
        return filtered_boxes


# Initialize Object Detection
od = ObjectDetection()

# Open video file
video_path = "AdobeStock_377383696.mov"
cap = cv2.VideoCapture(video_path)

# Exit if video not opened
if not cap.isOpened():
    print(f"Could not open video: {video_path}")
    exit()
else:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize variables
count = 0
tracking_objects = {}
track_id = 0
counted_animals = set()  # To store unique IDs of counted animals
animal_count = 0  # Counter for animals entering the frame

# Video writer
video_output_file_name = "tracking_animals_full_frame.mp4"
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*"XVID"), fps, (width, height))

# Dictionary to store object states and trajectories
object_states = {}  # Tracks whether an object has been counted
trajectory_history = {}  # Stores trajectory history for each object
max_frames_lost = 10  # Maximum frames an object can be lost before being removed

# Distance threshold for object association (normalized to frame size)
distance_threshold = 0.05 * math.hypot(width, height)

# Function to calculate distance between two points
def calculate_distance(pt1, pt2):
    return math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])


# Network of nodes (coordinates)
node_network = {}  # Stores nodes and their connections
persistent_connections = {}  # Stores persistent connections between nodes
network_update_interval = 10  # Update the network every 10 frames
frame_counter = 0  # Counter to track frames
relation_count = 0  # Counter for unique relations
counted_relations = set()  # To store unique relations already counted


# Function to update the network of nodes
def update_node_network(center_points):
    global node_network
    new_nodes = {}

    # Add new nodes to the network
    for idx, pt in enumerate(center_points):
        node_id = f"node_{idx}_{frame_counter}"  # Unique ID for each node
        new_nodes[node_id] = {"point": pt, "connections": []}

    # Connect nodes based on proximity
    for node_id, node_data in new_nodes.items():
        pt = node_data["point"]
        for other_node_id, other_node_data in new_nodes.items():
            if node_id != other_node_id:
                distance = calculate_distance(pt, other_node_data["point"])
                if distance < distance_threshold:
                    node_data["connections"].append(other_node_id)

    # Merge new nodes into the existing network
    node_network.update(new_nodes)


# Function to update persistent connections
def update_persistent_connections(node_network):
    global persistent_connections

    # Iterate over the current connections
    for node_id, node_data in node_network.items():
        for connected_node in node_data["connections"]:
            connection_key = tuple(sorted((node_id, connected_node)))  # Order to avoid duplicates

            # Increment the persistence counter
            if connection_key in persistent_connections:
                persistent_connections[connection_key] += 1
            else:
                persistent_connections[connection_key] = 1

    # Remove non-persistent connections
    persistent_connections = {k: v for k, v in persistent_connections.items() if v >= 10}  # Example: Minimum persistence of 10 frames


# Function to count unique relations
def count_unique_relations(persistent_connections):
    global counted_relations, relation_count

    for connection_key in persistent_connections.keys():
        if connection_key not in counted_relations:
            relation_count += 1
            counted_relations.add(connection_key)

    return relation_count


# Main loop
while True:
    ret, frame = cap.read()
    count += 1
    frame_counter += 1
    if not ret:
        break

    # Detect objects on frame
    center_points_cur_frame = []
    boxes = od.detect(frame)
    for box in boxes:
        (x, y, w, h) = box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)
        center_points_cur_frame.append((cx, cy))

        # Draw rectangle around detected object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update the network every 10 frames
    if frame_counter % network_update_interval == 0:
        update_node_network(center_points_cur_frame)
        update_persistent_connections(node_network)

    # Track objects
    tracking_objects_copy = tracking_objects.copy()
    center_points_cur_frame_copy = center_points_cur_frame.copy()

    for object_id, pt in tracking_objects_copy.items():
        # Find the closest point in the current frame
        closest_distance = float('inf')
        closest_point = None

        for pt_cur in center_points_cur_frame_copy:
            distance = calculate_distance(pt, pt_cur)
            if distance < distance_threshold and distance < closest_distance:
                closest_distance = distance
                closest_point = pt_cur

        # Update object position if within a reasonable distance
        if closest_distance < distance_threshold:
            tracking_objects[object_id] = closest_point
            trajectory_history[object_id].append(closest_point)

            # Remove the closest point from the list if it exists
            if closest_point in center_points_cur_frame:
                center_points_cur_frame.remove(closest_point)

            # Reset lost frame counter
            if object_id in object_states:
                object_states[object_id]["frames_lost"] = 0

            # Check if the object is entering the frame (not yet counted)
            if object_id not in counted_animals:
                animal_count += 1
                counted_animals.add(object_id)
                print(f"Animal {object_id} entered the frame. Total count: {animal_count}")
        else:
            # Increment lost frame counter
            if object_id in object_states:
                object_states[object_id]["frames_lost"] += 1

            # Remove object if lost for too long
            if object_id in object_states and object_states[object_id]["frames_lost"] > max_frames_lost:
                tracking_objects.pop(object_id)
                object_states.pop(object_id, None)
                trajectory_history.pop(object_id, None)

    # Add new objects
    for pt in center_points_cur_frame:
        tracking_objects[track_id] = pt
        trajectory_history[track_id] = [pt]  # Initialize trajectory history
        object_states[track_id] = {"frames_lost": 0}  # New objects start with no lost frames
        track_id += 1

    # Draw circles, IDs, and trajectories for tracked objects
    for object_id, pt in tracking_objects.items():
        # Draw object ID
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

        # Draw trajectory
        if object_id in trajectory_history:
            for i in range(1, len(trajectory_history[object_id])):
                cv2.line(frame, trajectory_history[object_id][i - 1], trajectory_history[object_id][i], (255, 0, 0), 2)

    # Count unique relations
    unique_relation_count = count_unique_relations(persistent_connections)

    # Display the animal count and relation count
    cv2.putText(frame, f"Animal Count: {animal_count}", (10, 30), 0, 1, (255, 100, 0), 2)
    cv2.putText(frame, f"Relation Count: {unique_relation_count}", (10, 60), 0, 1, (255, 25cla0, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)
    video_out.write(frame)

    # Exit on 'ESC' key
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release resources
cap.release()
video_out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Total animals counted: {animal_count}")
print(f"Total unique relations counted: {relation_count}")