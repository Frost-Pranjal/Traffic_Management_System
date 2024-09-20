#!/usr/bin/python3

import cv2 
import pygame
import numpy as np
import os

# Load YOLOv3

cfg_path = "/home/frost/Projects/Traffic_Management_System/yolov3.cfg"
weights_path = "/home/frost/Projects/Traffic_Management_System/yolov3.weights"

net = cv2.dnn.readNet(weights_path, cfg_path)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Function to perform object detection
def detect_objects(frame):
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 2)

    return frame

# Function to detect lanes
def detect_lanes(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Define region of interest (ROI) for lane detection
    height, width = edges.shape
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width, height//2), (0, height//2)]], dtype=np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Apply Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=50)
    
    # Draw detected lanes on original frame
    lane_img = np.zeros_like(frame)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lane_img, (x1, y1), (x2, y2), (0, 255, 0), 5)
    
    # Combine detected lanes with original frame
    result = cv2.addWeighted(frame, 0.8, lane_img, 1, 0)
    
    return result

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Object Detection and Lane Detection")

# Fonts
font = pygame.font.SysFont(None, 24)

# Input box
input_box = pygame.Rect(200, 50, 400, 30)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
text = ''
path = ''

# Initialize video capture
cap = None

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            # If the user clicks on the input box
            if input_box.collidepoint(event.pos):
                # Toggle the active variable
                active = not active
            else:
                active = False
            # Change the color of the input box
            color = color_active if active else color_inactive
        if event.type == pygame.KEYDOWN:
            if active:
                if event.key == pygame.K_RETURN:
                    # Process the video path
                    if os.path.isfile(text):
                        cap = cv2.VideoCapture(text)
                    else:
                        print("Invalid file path")
                    text = ''
                elif event.key == pygame.K_BACKSPACE:
                    text = text[:-1]
                else:
                    text += event.unicode

    # Clear the screen
    screen.fill((30, 30, 30))

    # Render the input box and text
    pygame.draw.rect(screen, color, input_box, 2)
    txt_surface = font.render(text, True, (255, 255, 255))
    width = max(200, txt_surface.get_width() + 10)
    input_box.w = width
    screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
    pygame.draw.rect(screen, (255, 255, 255), input_box, 2)

    # Read and process video frames
    if cap:
        ret, frame = cap.read()
        if ret:
            # Perform object detection
            frame = detect_objects(frame)
            
            # Perform lane detection
            frame = detect_lanes(frame)

            # Convert frame to Pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_surface = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")

            # Display processed frame
            screen.blit(processed_surface, (0, 100))

    # Update the display
    pygame.display.flip()

# Release video capture
if cap:
    cap.release()

pygame.quit()
