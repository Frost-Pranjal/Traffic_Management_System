#!/usr/bin/python3

import cv2
import pygame
import numpy as np
import os

# Load Haar cascade classifiers for car detection
car_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_car.xml')

# Check if the cascade classifier is loaded successfully
if car_cascade.empty():
    print("Error: Unable to load Haar cascade classifier.")
    exit()

# Function to perform object detection using Haar cascade classifiers
def detect_objects(frame):
    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect cars using Haar cascade classifiers
    cars = car_cascade.detectMultiScale(gray, 1.1, 3)

    # Draw bounding boxes around detected cars
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

# Function to detect lanes and suggest lane
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
    
    # Store detected lane points
    left_lane_points = []
    right_lane_points = []

    # Draw detected lanes on original frame and collect lane points
    for line in lines:
        x1, y1, x2, y2 = line[0]
        slope = (y2 - y1) / (x2 - x1)
        if slope < 0:  # Left lane
            left_lane_points.extend([(x1, y1), (x2, y2)])
        elif slope > 0:  # Right lane
            right_lane_points.extend([(x1, y1), (x2, y2)])
    
    # Fit lines to left and right lane points
    if left_lane_points:
        left_line = np.polyfit([x[0] for x in left_lane_points], [y[1] for y in left_lane_points], 1)
    else:
        left_line = None
    
    if right_lane_points:
        right_line = np.polyfit([x[0] for x in right_lane_points], [y[1] for y in right_lane_points], 1)
    else:
        right_line = None

    # Determine suggested lane
    if left_line is not None and right_line is not None:
        suggested_lane = "Center"
    elif left_line is not None:
        suggested_lane = "Left"
    elif right_line is not None:
        suggested_lane = "Right"
    else:
        suggested_lane = "Unknown"
    
    # Draw suggested lane on original frame
    lane_img = np.zeros_like(frame)
    cv2.putText(lane_img, f"Suggested Lane: {suggested_lane}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Combine detected lanes with original frame
    result = cv2.addWeighted(frame, 0.8, lane_img, 1, 0)
    
    return result

# Initialize Pygame
pygame.init()

# Set up display
screen_width = 800
screen_height = 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Lane Suggest Software")

# Cream color (RGB)
cream_color = (245, 230, 200)
black_color = (0, 0, 0)

# Fonts
font = pygame.font.SysFont(None, 24)

# Input box
input_box = pygame.Rect(170, 50, 400, 30)
color_inactive = pygame.Color('lightskyblue3')
color_active = pygame.Color('dodgerblue2')
color = color_inactive
active = False
text = '/home/frost/Projects/Traffic_Management_System/videoplayback.mp4'  # Default path
path = ''

# Enter button
button = pygame.Rect(710, 50, 80, 30)
button_color = pygame.Color('dodgerblue2')

# Label for input file path
label_font = pygame.font.SysFont(None, 24)
label_text = label_font.render("Input File Path:", True, black_color)

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
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button.collidepoint(event.pos):
                if os.path.isfile(text):
                    cap = cv2.VideoCapture(text)
                else:
                    print("Invalid file path")
                text = ''

    # Fill the screen with cream color
    screen.fill(cream_color)

    # Render the input box and text
    pygame.draw.rect(screen, color, input_box, 2)
    txt_surface = font.render(text, True, black_color)
    width = max(200, txt_surface.get_width() + 10)
    input_box.w = width
    screen.blit(label_text, (45,55))
    screen.blit(txt_surface, (input_box.x + 5, input_box.y + 5))
    pygame.draw.rect(screen, (255, 255, 255), input_box, 2)

    # Render the Enter button
    pygame.draw.rect(screen, button_color, button)
    button_text = font.render("Enter", True, black_color)
    screen.blit(button_text, (727, 55))

    # Read and process video frames
    if cap:
        ret, frame = cap.read()
        if ret:
            # Perform object detection
            frame = detect_objects(frame)
            
            # Perform lane detection and suggest lane
            frame = detect_lanes(frame)

            # Convert frame to Pygame surface
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed_surface = pygame.image.frombuffer(frame.tostring(), frame.shape[1::-1], "RGB")

            # Display processed frame
            screen.blit(processed_surface, (170, 150))

    # Update the display
    pygame.display.flip()

# Release video capture
if cap:
    cap.release()

pygame.quit()
