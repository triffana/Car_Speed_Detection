
import cv2
import numpy as np
from time import sleep

# Constants
largura_min = 80  # Minimum width of the rectangle
altura_min = 80  # Minimum height of the rectangle
offset = 6  # Allowed error between pixels
pos_linha = 550  # Position of the counting line
delay = 60  # FPS of the video
scale_factor = 0.05  # Scale factor (meters per pixel), adjust this based on calibration

# Variables
detec = []
carros = 0
vehicle_ids = {}  # To assign unique IDs to vehicles
min_distance_moving_threshold = 10  # Minimum distance threshold to consider vehicle moving
speed_smoothing_window = 5  # Number of frames for speed smoothing
speed_history = {}  # To store historical speeds for smoothing

def pega_centro(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx, cy

def get_vehicle_id(center):
    for vid, (pos, _) in vehicle_ids.items():
        if np.linalg.norm(np.array(center) - np.array(pos)) < 100:  # Distance threshold
            return vid
    return None

def compute_smoothed_speed(vehicle_id, speed_kmph):
    if vehicle_id not in speed_history:
        speed_history[vehicle_id] = []
    speed_history[vehicle_id].append(speed_kmph)
    if len(speed_history[vehicle_id]) > speed_smoothing_window:
        speed_history[vehicle_id].pop(0)
    return np.mean(speed_history[vehicle_id])

cap = cv2.VideoCapture('video.mp4')
subtracao = cv2.createBackgroundSubtractorMOG2()

while True:
    ret, frame1 = cap.read()
    if not ret:
        break
    tempo = float(1 / delay)
    sleep(tempo)
    
    grey = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(grey, (3, 3), 5)
    img_sub = subtracao.apply(blur)
    dilat = cv2.dilate(img_sub, np.ones((5, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilatada = cv2.morphologyEx(dilat, cv2.MORPH_CLOSE, kernel)
    dilatada = cv2.morphologyEx(dilatada, cv2.MORPH_CLOSE, kernel)
    contorno, h = cv2.findContours(dilatada, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
   
    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (255, 127, 0), 3)
    
    smoothed_speed = 0  # Initialize smoothed_speed to avoid NameError
    
    for i, c in enumerate(contorno):
        (x, y, w, h) = cv2.boundingRect(c)
        validar_contorno = (w >= largura_min) and (h >= altura_min)
        if not validar_contorno:
            continue

        cv2.rectangle(frame1, (x, y), (x + w, y + h), (0, 255, 0), 2)
        centro = pega_centro(x, y, w, h)
        detec.append(centro)
        cv2.circle(frame1, centro, 4, (0, 0, 255), -1)

        vehicle_id = get_vehicle_id(centro)
        if vehicle_id is None:
            vehicle_id = len(vehicle_ids) + 1
            vehicle_ids[vehicle_id] = (centro, cv2.getTickCount())
        else:
            start_position, start_time = vehicle_ids[vehicle_id]
            end_time = cv2.getTickCount()
            time_elapsed = (end_time - start_time) / cv2.getTickFrequency()  # Time in seconds
            
            # Calculate distance traveled
            distance = np.linalg.norm(np.array(centro) - np.array(start_position))
            distance_meters = distance * scale_factor  # Convert to meters
            if distance > min_distance_moving_threshold:  # Filter out small distances
                print(f"Distance traveled: {distance:.2f} pixels ({distance_meters:.2f} meters)")
                if time_elapsed > 0:
                    speed_mps = distance_meters / time_elapsed  # Speed in meters per second
                    speed_kmph = speed_mps * 3.6  # Convert speed to kilometers per hour
                else:
                    speed_kmph = 0
                
                smoothed_speed = compute_smoothed_speed(vehicle_id, speed_kmph)
                
                # Check if vehicle crossed the line
                if y < (pos_linha + offset) and y > (pos_linha - offset):
                    carros += 1
                    cv2.line(frame1, (25, pos_linha), (1200, pos_linha), (0, 127, 255), 3)
                    print(f"Car detected. Speed: {smoothed_speed:.2f} km/h")
                    vehicle_ids.pop(vehicle_id, None)
                    
                vehicle_ids[vehicle_id] = (centro, start_time)

    cv2.putText(frame1, "VEHICLE COUNT: " + str(carros), (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
    cv2.putText(frame1, f"Speed: {smoothed_speed:.2f} km/h", (450, 130), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 5)
    cv2.imshow("Video Original", frame1)
    cv2.imshow("Detectar", dilatada)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
