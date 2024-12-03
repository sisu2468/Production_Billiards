#pyinstaller -F --paths=C:\Users\fjdhu\OneDrive\ドキュメント\Billiards_System\.venv\Lib\site-packages analysis_system.py --onefile --hide-console hide-early
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, PhotoImage
from PIL import Image, ImageDraw, ImageTk
import matplotlib.pyplot as plt
import math

file_path = None
spin = cv2.imread('english_spin.png', cv2.IMREAD_UNCHANGED)
english_spin = cv2.resize(spin, (150, 150))
spin_angle = 0.00000
spin_mouse_coordinates =  (-100005, 0)
# Global variable for video capture
cap = None
video_running = None  # Store the ID of the after() callback
select_count = 0

left_bar_image = cv2.imread('strength_left_bar.png', cv2.IMREAD_UNCHANGED)
strength_left_bar = cv2.resize(left_bar_image, (250, 30))

right_bar_image = cv2.imread('strength_right_bar.png', cv2.IMREAD_UNCHANGED)
strength_right_bar = cv2.resize(right_bar_image, (250, 30))

global english_spin_status
english_spin_status = False  
    
def show_main_screen():
    main_screen.pack()
    exercise_button.pack_forget()  # Hide the button after it's clicked
    rank_button.pack_forget()  # Hide the button after it's clicked

def show_rank_screen():
    main_screen.pack()
    exercise_button.pack_forget()  # Hide the button after it's clicked
    rank_button.pack_forget()  # Hide the button after it's clicked

def display_image(image):
    # Convert the image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert the image to PIL format
    image_pil = Image.fromarray(image_rgb)
    # Convert the image to ImageTk format
    image_tk = ImageTk.PhotoImage(image_pil)
    image_label.config(image=image_tk)
    image_label.image = image_tk

def return_origin():
    main_screen.pack_forget()
    exercise_button.pack(side=tk.LEFT, padx=10)
    rank_button.pack(side=tk.LEFT, padx=10)
    # canvas.delete("all")


# Function to generate a white image
def generate_white_board(width, height):
    white_image = np.ones((height, width, 3), dtype=np.uint8) * 255
    return white_image
cap = cv2.VideoCapture(1)  # 0 for default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1400)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
# Function to display the live video feed from the camera
def display_video_feed():
    global cap, video_running
    def update_frame():
        global video_running
        ret, frame = cap.read()
        if ret:
            # resized_frame = cv2.resize(frame, (1300, 700))
            # Convert the frame to RGB format
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            tk_image = ImageTk.PhotoImage(image=pil_image)
            
            # Display the frame in the Tkinter window
            image_label.config(image=tk_image)
            image_label.image = tk_image
            
        # Schedule the next frame update to keep the video running
        video_running = root.after(10, update_frame)  # 10 ms delay for smoother video
    
    update_frame()

# Function to display the white board
def display_white_board():
    white_image = generate_white_board(1180, 700)
    image_rgb = cv2.cvtColor(white_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    tk_image = ImageTk.PhotoImage(image=pil_image)
    image_label.config(image=tk_image)
    image_label.image = tk_image
    
def orientation(p, q, r):
    """
    Calculate orientation of three points (p, q, r).
    """
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0:
        return 0  # Collinear
    return 1 if val > 0 else -1  # Clockwise or Counterclockwise

def on_segment(p, q, r):
    """
    Check if point q lies on line segment pr.
    """
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def intersect_point(seg1, seg2):
    """
    Find the intersection point of two line segments.
    """
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    # Check for general intersection
    if (o1 != o2 and o3 != o4):
        # Calculate intersection point
        x_intersect = ((p1[0] * q1[1] - p1[1] * q1[0]) * (p2[0] - q2[0]) - (p1[0] - q1[0]) * (p2[0] * q2[1] - p2[1] * q2[0])) / ((p1[0] - q1[0]) * (p2[1] - q2[1]) - (p1[1] - q1[1]) * (p2[0] - q2[0]))
        y_intersect = ((p1[0] * q1[1] - p1[1] * q1[0]) * (p2[1] - q2[1]) - (p1[1] - q1[1]) * (p2[0] * q2[1] - p2[1] * q2[0])) / ((p1[0] - q1[0]) * (p2[1] - q2[1]) - (p1[1] - q1[1]) * (p2[0] - q2[0]))
        return (int(x_intersect), int(y_intersect))

    # Check for special cases of collinear segments with overlapping endpoints
    if (o1 == 0 and on_segment(p1, p2, q1)):
        return p2
    if (o2 == 0 and on_segment(p1, q2, q1)):
        return q2
    if (o3 == 0 and on_segment(p2, p1, q2)):
        return p1
    if (o4 == 0 and on_segment(p2, q1, q2)):
        return q1

    return None  # No intersection

#second intersect line
# Function to calculate the slope of a line
def calculate_slope(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    return (y2 - y1) / (x2 - x1)

# Function to calculate the equation of a line given a point and slope
def line_equation_from_point_slope(point, slope):
    x, y = point
    return slope, y - slope * x

def Normalize(vector):
  magnitude = math.sqrt(vector[0]**2 + vector[1]**2)
  return (vector[0]/magnitude, vector[1]/magnitude)

def dot(u, v):
   return sum(a*b for a,b in zip(u,v))

# Function to find the intersection point of a line with the boundary of the image
def find_intersection_with_boundary(point, slope, image_width, image_height):
    x, y = point
    # If the slope is close to zero, the line is approximately horizontal
    if abs(slope) < 1e-6:
        return x, 0 if y < 0 else image_height
    # If the slope is close to infinity, the line is approximately vertical
    if abs(slope) > 1e6:
        return 0 if x < 0 else image_width, y
    # Calculate intersection with top or bottom boundary
    if y < 0:
        return x - y / slope, 0
    elif y > image_height:
        return x + (image_height - y) / slope, image_height
    # Calculate intersection with left or right boundary
    if x < 0:
        return 0, y - x * slope
    elif x > image_width:
        return image_width, y + (image_width - x) * slope
    return None

def angle_between_lines(point1, point2, line_segment):
    # Calculate direction vectors of the two line segments
    vec1 = (point2[0] - point1[0], point2[1] - point1[1])
    vec2 = (line_segment[1][0] - line_segment[0][0], line_segment[1][1] - line_segment[0][1])
    
    # Calculate dot product
    dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    
    # Calculate magnitudes
    magnitude_vec1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
    magnitude_vec2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
    
    # Calculate angle in radians
    angle_radians = math.acos(dot_product / (magnitude_vec1 * magnitude_vec2))
    
    # Convert angle to degrees
    angle_degrees = math.degrees(angle_radians)
    # if angle_degrees > 90:
    #     angle_degrees = 180 - angle_degrees
    
    return angle_degrees
        
def angle_to_line(start_point, angle_deg, length, status):
    # Convert angle from degrees to radians
    # if status == 2 and angle_deg > 90:
    #     angle_deg = 180 - angle_deg
    angle_rad = math.radians(angle_deg)
    

    # Calculate the x and y components of the vector
    dx = length * math.cos(angle_rad)
    dy = length * math.sin(angle_rad)
    if (status == 1 and angle_deg > 90): 
        end_point = (int(start_point[0] - abs(dx)), int(start_point[1] + abs(dy)))
    elif (status == 1 and angle_deg <= 90): 
        end_point = (int(start_point[0] + abs(dx)), int(start_point[1] + abs(dy)))
    elif (status == 2 and angle_deg <= 90):
        end_point = (int(start_point[0] - abs(dy)), int(start_point[1] + abs(dx)))
    elif (status == 2 and angle_deg > 90): 
        end_point = (int(start_point[0] - abs(dy)), int(start_point[1] - abs(dx)))
    elif (status == 3 and angle_deg <= 90):
        end_point = (int(start_point[0] - abs(dx)), int(start_point[1] - abs(dy)))
    elif (status == 3 and angle_deg > 90): 
        end_point = (int(start_point[0] + abs(dx)), int(start_point[1] - abs(dy)))
    elif (status == 4 and angle_deg > 90):
        # dx = length * math.cos(angle_rad - 90)
        # dy = length * math.sin(angle_rad - 90)
        end_point = (int(start_point[0] + abs(dx)), int(start_point[1] + abs(dy)))
    elif (status == 4 and angle_deg <= 90):
        end_point = (int(start_point[0] + abs(dy)), int(start_point[1] - abs(dx)))
        
    return end_point

def second_intersect_point(line1, line2):
    x1, y1 = line1[0]
    x2, y2 = line1[1]
    x3, y3 = line2[0]
    x4, y4 = line2[1]
    
    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    # Check if the lines are parallel (denominator equals zero)
    if denominator == 0:
        return None  # No intersection
    
    # Calculate the intersection point
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
    
    if (min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)):
        return int(x), int(y)
    else: return None

def projection_point(white_ball, cushion_line):
    x_w, y_w = white_ball
    (x1, y1), (x2, y2) = cushion_line
    
    # Calculate the direction vector of the cushion line
    line_vec = np.array([x2 - x1, y2 - y1])
    line_vec_norm = line_vec / np.linalg.norm(line_vec)
    
    # Calculate the vector from the first point of the line to the white ball
    ball_vec = np.array([x_w - x1, y_w - y1])
    
    # Project the ball vector onto the line vector
    proj_length = np.dot(ball_vec, line_vec_norm)
    proj_vec = proj_length * line_vec_norm
    
    # Calculate the projection point
    proj_point = np.array([x1, y1]) + proj_vec
    
    # Convert the projection point to integer coordinates
    proj_point_int = proj_point.astype(int)
    return proj_point_int

def calculate_collision(nearest_pocket_coordinates, ten_units_away_point, real_ball_point, ball_radius):
    def vector_length(v):
        return np.sqrt(np.sum(v ** 2))

    def unit_vector(v):
        return v / vector_length(v)

    nearest_pocket_coordinates = np.array(nearest_pocket_coordinates)
    ten_units_away_point = np.array(ten_units_away_point)
    real_ball_point = np.array(real_ball_point)

    direction = unit_vector(ten_units_away_point - nearest_pocket_coordinates)
    
    # Calculate the vector from the cue ball to the stationary ball
    cue_to_ball = real_ball_point - nearest_pocket_coordinates
    
    # Project the cue_to_ball vector onto the direction of the cue ball's path
    projection_length = np.dot(cue_to_ball, direction)
    
    # Calculate the closest point on the cue ball's path to the stationary ball
    closest_point = nearest_pocket_coordinates + direction * projection_length
    
    # Calculate the distance from the closest point to the stationary ball
    distance_to_ball = vector_length(real_ball_point - closest_point)
    
    if distance_to_ball > 2 * ball_radius:
        print("No collision: the cue ball path does not intersect the stationary ball.")
        return None
    
    # Calculate the distance from the closest point to the collision point
    collision_distance = np.sqrt((2 * ball_radius)**2 - distance_to_ball**2)
    
    # The collision point is further along the path by collision_distance
    collision_point = closest_point - direction * collision_distance
    
    return collision_point

def find_perpendicular_line(collision_point, real_ball_point):
    def vector_length(v):
        return np.sqrt(np.sum(v ** 2))

    def unit_vector(v):
        return v / vector_length(v)

    collision_point = np.array(collision_point)
    real_ball_point = np.array(real_ball_point)

    # Calculate the direction vector from collision_point to real_ball_point
    direction = real_ball_point - collision_point
    
    # Find a perpendicular vector in 2D (swap and negate one component)
    if direction[0] != 0 or direction[1] != 0:
        perp_direction = np.array([-direction[1], direction[0]])
    else:
        # If the direction vector is zero, use a default perpendicular vector
        perp_direction = np.array([1, 0])

    # Normalize the perpendicular vector to get the unit vector
    perp_unit_vector = unit_vector(perp_direction)

    return perp_unit_vector


# If there is a ball in the way when the cue ball bounces
def draw_parallel_line_if_needed(image, nearest_pocket_coordinates, ten_units_away_point, text_number, parellel_diatance, line_status):
    # Calculate the direction vector from ten_units_away_point to nearest_pocket_coordinates
    direction_vector = (nearest_pocket_coordinates[0] - ten_units_away_point[0], 
                        nearest_pocket_coordinates[1] - ten_units_away_point[1])

    # Normalize the direction vector
    direction_vector_length = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
    normalized_direction_vector = (direction_vector[0] / direction_vector_length, 
                                   direction_vector[1] / direction_vector_length)

    # Calculate the perpendicular vector
    perpendicular_vector = (-normalized_direction_vector[1], normalized_direction_vector[0])

    # Offset the line by 10 units using the perpendicular vector
    offset_point_1 = (nearest_pocket_coordinates[0] + parellel_diatance * perpendicular_vector[0], 
                      nearest_pocket_coordinates[1] + parellel_diatance * perpendicular_vector[1])
    offset_point_2 = (ten_units_away_point[0] + parellel_diatance * perpendicular_vector[0], 
                      ten_units_away_point[1] + parellel_diatance * perpendicular_vector[1])
    offset_point_3 = (nearest_pocket_coordinates[0] - parellel_diatance * perpendicular_vector[0], 
                      nearest_pocket_coordinates[1] - parellel_diatance * perpendicular_vector[1])
    offset_point_4 = (ten_units_away_point[0] - parellel_diatance * perpendicular_vector[0], 
                      ten_units_away_point[1] - parellel_diatance * perpendicular_vector[1])

    # Count the number of billiard balls between nearest_pocket_coordinates and ten_units_away_point
    balls_between = 0
    real_ball_point = []
    for ball_number, (ball_x, ball_y) in text_number:
        if ball_number != '10':
            # Vector from ten_units_away_point to ball_coordinates
            ball_vector = (ball_x - ten_units_away_point[0], 
                        ball_y - ten_units_away_point[1])
            
            # Project ball_vector onto direction_vector to find the closest point on the line
            t = ((ball_vector[0] * direction_vector[0] + ball_vector[1] * direction_vector[1]) / (direction_vector_length ** 2))
            
            # If t is between 0 and 1, the ball is between ten_units_away_point and nearest_pocket_coordinates
            if 0 <= t <= 1:
                # Closest point on the line to the ball
                closest_point = (ten_units_away_point[0] + t * direction_vector[0], 
                                ten_units_away_point[1] + t * direction_vector[1])
                
                # Distance from the ball to the line
                distance_to_line = math.sqrt((ball_x - closest_point[0]) ** 2 + 
                                            (ball_y - closest_point[1]) ** 2)
                
                # If the distance is less than the radius, the ball is effectively "on" the line
                if distance_to_line <= 15:
                    balls_between += 1                    
                    distrupt_point = closest_point
                    real_ball_point = (ball_x, ball_y)
                    
    if line_status == 1:
        # If there are more than two balls between the points, draw the parallel line
        if balls_between > 1:
            cv2.line(image, (int(offset_point_1[0]), int(offset_point_1[1])), 
                    (int(offset_point_2[0]), int(offset_point_2[1])), (0, 204, 0), 2)
            cv2.line(image, (int(offset_point_3[0]), int(offset_point_3[1])), 
                    (int(offset_point_4[0]), int(offset_point_4[1])), (0, 204, 0), 2)
    elif line_status == 2:
        # If there are more than two balls between the points, draw the parallel line
        if balls_between > 0:
            cv2.line(image, (int(offset_point_1[0]), int(offset_point_1[1])), 
                    (int(offset_point_2[0]), int(offset_point_2[1])), (0, 204, 0), 2)
            cv2.line(image, (int(offset_point_3[0]), int(offset_point_3[1])), 
                    (int(offset_point_4[0]), int(offset_point_4[1])), (0, 204, 0), 2)
            
            collision_point = calculate_collision(nearest_pocket_coordinates, ten_units_away_point, real_ball_point, 10)

            perpendicular_unit_vector = find_perpendicular_line(collision_point, real_ball_point)
            
            line_point1 = collision_point - 100 * perpendicular_unit_vector
            direction_angle = angle_between_lines(nearest_pocket_coordinates, ten_units_away_point, (collision_point, line_point1))
            if direction_angle > 90.00:
                line_point1 = collision_point + 100 * perpendicular_unit_vector
            
            cv2.circle(image, (int(collision_point[0]), int(collision_point[1])), 10, (104, 120, 255), 2)
            
            cv2.line(image, (int(collision_point[0]), int(collision_point[1])), (int(line_point1[0]), int(line_point1[1])), (255, 255, 255), 2)
            
            reall_ball_dir = ((collision_point[0] - real_ball_point[0]), (collision_point[1] - real_ball_point[1]))
            real_ball_point = ((real_ball_point[0] - reall_ball_dir[0] * 5), (real_ball_point[1] - reall_ball_dir[1] * 5))
            print("closet_point", real_ball_point)
            
            cv2.line(image, (int(collision_point[0]), int(collision_point[1])), (int(real_ball_point[0]), int(real_ball_point[1])), (255, 255, 255), 2)
    elif line_status == 3:
        if balls_between > 0:
            return True
        else:
            return False
    elif line_status == 4:
        if balls_between > 1:
            print("ball_number")
            return True
        else:
            return False
def find_projection_point (pocket_lines, ten_units_away_point, white_ball_coordinates, cur_ball_line):
    idx = 0
    first_cushion_points = []
    first_cushion_point = []
    for line in pocket_lines:
        angle_min = 999
        
        ten_units_projection = projection_point(ten_units_away_point, line)
        white_units_projection = projection_point(white_ball_coordinates, line)
        
        if ten_units_projection[0] > white_units_projection[0] or ten_units_projection[1] > white_units_projection[1]:
            swap_point = ten_units_projection
            ten_units_projection = white_units_projection
            white_units_projection = swap_point
        
        if idx % 2 == 0:
            start_projection_point = ten_units_projection[0]
            second_projection_point = white_units_projection[0]
        else:
            start_projection_point = ten_units_projection[1]
            second_projection_point = white_units_projection[1]
        for i in range(start_projection_point, second_projection_point + 1) :
            if cur_ball_line ==1:
                second_angle = 180
            elif idx %2 == 0:
                second_angle = angle_between_lines((i, ten_units_projection[1]), ten_units_away_point, cur_ball_line)
            else :
                second_angle = angle_between_lines((ten_units_projection[0], i), ten_units_away_point, cur_ball_line)    
            # if second_angle > 70:
            if idx %2 == 0:
                first_projection_angle = angle_between_lines(white_ball_coordinates, (i, ten_units_projection[1]), line)
                second_projection_angle = angle_between_lines((i, ten_units_projection[1]), ten_units_away_point, line)
            else :
                first_projection_angle = angle_between_lines(white_ball_coordinates, (ten_units_projection[0], i), line)
                second_projection_angle = angle_between_lines((ten_units_projection[0], i), ten_units_away_point, line)
            angle_different = int(abs(first_projection_angle - second_projection_angle))
            if angle_min > angle_different and angle_different < 10:
                if idx %2 == 0:
                    first_cushion_point = (i, ten_units_projection[1])
                else:
                    first_cushion_point = (ten_units_projection[0], i)
                projection_compare_angle = angle_between_lines(ten_units_away_point, first_cushion_point, line)
                # if projection_compare_angle < 108:
                #     first_cushion_point = []
                angle_min = angle_different
            if angle_min == 0 or angle_min < angle_different:
                break
        if first_cushion_point:
            first_cushion_points.append(first_cushion_point)
        idx += 1
    return first_cushion_points

mouse_coordinates = (633, 350)
# Function to analyze the image
def analyze_image(image, hsv, pocket_name, parellel_diatance, english_spin_status, mouse_coordinates):
    # canvas.delete("all")
    global spin_angle, spin_mouse_coordinates
    # Define HSV ranges for each color
    color_ranges = {
        "yellow": ([11, 114, 150], [100, 255, 255], 80),
        # "red": ([105, 200, 160], [179, 255, 255], 30),
        "green": ([40, 100, 60], [90, 255, 255], 10),
        "white": ([0, 0, 196], [49, 69, 255], 50),
        "blue": ([105, 80, 103], [128, 255, 255], 40),
        "orange": ([5, 139, 210], [12, 255, 255], 40),
        "purple": ([0, 58, 218], [9, 170, 255], 40),
        "maroon": ([8, 102, 136], [13, 255, 219], 30),
        "red": ([0, 170, 80], [1, 255, 255], 30),
        "nineyellow": ([13, 79, 110], [62, 201, 255], 30),   #change
        "black": ([0, 0, 0], [179, 255, 60], 80),
    }
    
    # Define the trapezoid coordinates
    trapezoid_points = np.array([
        [78, 86],  # Polaris
        [1167, 95],  # Terra
        [1174, 636], # Mars
        [74, 638]    # Sirius
    ], dtype=np.int32)

    # Draw lines between each pair of points to form the trapezoid
    # cv2.line(image, tuple(trapezoid_points[0]), tuple(trapezoid_points[1]), (0, 255, 0), 2)  # Polaris to Terra
    # cv2.line(image, tuple(trapezoid_points[1]), tuple(trapezoid_points[2]), (0, 255, 0), 2)  # Terra to Mars
    # cv2.line(image, tuple(trapezoid_points[2]), tuple(trapezoid_points[3]), (0, 255, 0), 2)  # Mars to Sirius
    # cv2.line(image, tuple(trapezoid_points[3]), tuple(trapezoid_points[0]), (0, 255, 0), 2)  # Sirius to Polaris

    # Define the mask for each color
    masks = {}
    for color, (lower, upper, colorarea) in color_ranges.items():
        mask = np.ones_like(hsv[:, :, 0], dtype=np.uint8) * 255  # Create a white mask
        
        # Create a trapezoidal mask by filling the defined points
        trapezoid_mask = np.zeros_like(mask, dtype=np.uint8)
        cv2.fillPoly(trapezoid_mask, [trapezoid_points], 255)  # Fill trapezoid area with white
        
        # Apply the trapezoid mask to the color range mask
        color_mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        masked_color = cv2.bitwise_and(color_mask, color_mask, mask=trapezoid_mask)
        
        # Store the result in the masks dictionary
        masks[color] = (masked_color, colorarea)
        
    # Find contours for each color
    contours_by_color = {}
    for color, (mask, colorarea) in masks.items():
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours_by_color[color] = (contours, colorarea)    
        # Display the mask in a window with the color name
        # if color == "red":
        #     cv2.imshow(f'Mask - {colorarea}', mask, )

        # Print the total mask area in pixels for that color
        mask_area = cv2.countNonZero(mask)
        # print(f"Mask area for {color}: {mask_area} pixels")

    # Set numbers for each color
    numbers_by_color = {
        "yellow": 1,
        "blue": 2,
        "red": 3,
        "purple": 4,
        "orange": 5,
        "green": 6,
        "maroon": 7,
        "black": 8,
        "nineyellow": 9,
        "white": 10
    }

    check = []  # Assuming check is a list
    text_number = []
    pocket_ball_number = []   #ball number in pocket
    # Iterate through contours
    first_ball_number = 10000
    first_coordinates = {}
    second_ball_number = 10000
    for color, (contours, colorarea) in contours_by_color.items():
        area = 0.000
        area_comp = 1000000000
        for contour_one in contours:
            if area < cv2.contourArea(contour_one):
                area = cv2.contourArea(contour_one)
                contour = contour_one
        if color == "nineyellow":
            for contour_one in contours:
                if cv2.contourArea(contour_one) > colorarea and area_comp > cv2.contourArea(contour_one):
                    area_comp = cv2.contourArea(contour_one)
                    contour = contour_one
            if area_comp != 1000000000:
                area = area_comp
            print("area", area)
        # if color == 'red':
        #     print("aaaa", contour)
        #     for contour_one in contours:
        #         area_comp = cv2.contourArea(contour_one)
        #         contour = contour_one
        #         red_x, red_y, w, h = cv2.boundingRect(contour)
        #         if area_comp > 10:
        #             print('redarea', area_comp)
        #             cv2.putText(image, str(area_comp), (red_x, red_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 4)  # Draw only the circle outline
        # cv2.imshow("mask", mask)
        if area > colorarea: 
            (x, y), _ = cv2.minEnclosingCircle(contour)
            point = (int(x), int(y))
            close_to_existing_point = False
            # Check if the new point is close to any existing point in check
            # for stored_point in check:
            #     if abs(point[0] - stored_point[0]) < 10 and abs(point[1] - stored_point[1]) < 10:
            #         close_to_existing_point = True
            #         break
            if not close_to_existing_point:
                check.append(point)
                # cv2.circle(image, point, 1, (0, 0, 255), 2)  # Draw only the circle outline
                # cv2.putText(image, str(point), (int(x)-5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
                if color == 'white':
                    white_ball_coordinates = point
                # elif numbers_by_color[color] < first_ball_number:
                elif numbers_by_color[color] < 7:
                    # if color != "red":
                    first_ball_number = numbers_by_color[color]
                    first_coordinates = point
                elif numbers_by_color[color] < second_ball_number:
                    second_ball_number = numbers_by_color[color]
                    second_coordinates = point

                pocket_ball_number.append(numbers_by_color[color])
                # if color != "red":
                text_number.append((str(numbers_by_color[color]), point))
                # cv2.putText(image, str(area), (int(x)-5, int(y)+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
    print("text_number", text_number, "first_coordinates", first_coordinates)
    miss_ball_number = []
    i = 1
    for i in range(1, 10):
        ball_flag = True
        for ball_number in pocket_ball_number:
            if i == ball_number:
                ball_flag = False
        if ball_flag:
            miss_ball_number.append(i)
            
    ball_count = 1
    ball_color = {
        1: "#e9ae6a",
        2: "#0033cc",
        3: "#e11b58",
        4: "#190e4a",
        5: "#eb533b",
        6: "#03525e",
        7: "#6c335f",
        8: "#262b3b",
        9: "#fff490",
    }
    # Assuming `miss_ball_number` contains the list of ball numbers
    for ball_number in miss_ball_number:
        # Calculate the center coordinates of the circle based on the ball number
        x_center =  22 # Adjust as needed based on your layout
        y_center = 525 - ball_count * 40   # Adjust as needed based on your layout
        ball_count = ball_count + 1
        # Draw the circle on the Canvas
        circle_radius = 15
        # canvas.create_oval(x_center - circle_radius, y_center - circle_radius, 
        #                 x_center + circle_radius, y_center + circle_radius, 
        #                 fill=ball_color[ball_number], outline=ball_color[ball_number])
        # canvas.create_text(x_center, y_center, text=str(ball_number), fill='white', width=10, font=("Arial", 12, "bold"))
    
    # Define pocket locations (assuming (x, y) coordinates)
    pocket_locations = {
        "Polaris": (83, 97),
        "Jupiter": (625, 92),
        "Terra": (1162, 97),
        "Sirius": (80, 635),
        "Venus": (625, 635),
        "Mars": (1165, 632),
    }
    cv2.circle(image, (83, 92), 2, (255, 255, 255), 5)
    cv2.circle(image, (625, 92), 2, (255, 255, 255), 5)
    cv2.circle(image, (1162, 92), 2, (255, 255, 255), 5)
    cv2.circle(image, (80, 635), 2, (255, 255, 255), 5)
    cv2.circle(image, (625, 635), 2, (255, 255, 255), 5)
    cv2.circle(image, (1165, 632), 2, (255, 255, 255), 5)
    # Display Pocket Text
    cv2.putText(image, "P", (65, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "J", (620, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "T", (1185, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "S", (55, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "V", (620, 665), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    cv2.putText(image, "M", (1180, 650), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    pocket_lines = [
            [(85, 97), (1158, 105)],
            [(1158, 105), (1166, 625)],
            [(1166, 625), (75, 625)],
            [(75, 625), (85, 97)],
    ]

    cv2.line(image, (85, 97), (1158, 105), (0, 255, 0), 2)  # Polaris to Terra
    cv2.line(image, (1158, 105), (1166, 625), (0, 255, 0), 2)  # Polaris to Terra
    cv2.line(image, (1166, 625), (75, 625), (0, 255, 0), 2)  # Polaris to Terra
    cv2.line(image, (75, 625), (85, 97), (0, 255, 0), 2)  # Polaris to Terra
    
    # cv2.circle(image, (916, 30), 2, (255, 255, 255), 5)
    # cv2.circle(image, (1055, 30), 2, (255, 255, 255), 5)
    
    # cv2.circle(image, (213, 672), 2, (255, 255, 255), 5)
    # cv2.circle(image, (355, 672), 2, (255, 255, 255), 5)
    # cv2.circle(image, (495, 672), 2, (255, 255, 255), 5)
    # cv2.circle(image, (775, 668), 2, (255, 255, 255), 5)
    # cv2.circle(image, (916, 665), 2, (255, 255, 255), 5)
    # cv2.circle(image, (1055, 665), 2, (255, 255, 255), 5)
    
    # cv2.circle(image, (31, 211), 2, (255, 255, 255), 5)
    # cv2.circle(image, (33, 352), 2, (255, 255, 255), 5)
    # cv2.circle(image, (33, 494), 2, (255, 255, 255), 5)
    
    # cv2.circle(image, (1232, 208), 2, (255, 255, 255), 5)
    # cv2.circle(image, (1232, 347), 2, (255, 255, 255), 5)
    # cv2.circle(image, (1232, 487), 2, (255, 255, 255), 5)
    
    # nearest_second_pocket_distance = float('inf')  # Initialize with a large value
    # nearest_second_pocket_coordinates = None
    # if second_coordinates:
    #     for pocket, coordinates in pocket_locations.items():
    #         # Calculate the distance between the first coordinates and the current pocket coordinates
    #         distance = math.sqrt((second_coordinates[0] - coordinates[0])**2 + (second_coordinates[1] - coordinates[1])**2)
    #         if distance < nearest_second_pocket_distance:
    #             nearest_second_pocket_distance = distance
    #             nearest_second_pocket_coordinates = coordinates
    #     # Calculate the direction vector from second_coordinates to nearest_second_pocket_coordinates
    #     second_direction_vector = (nearest_second_pocket_coordinates[0] - second_coordinates[0], nearest_second_pocket_coordinates[1] - second_coordinates[1])
    #     # Determine the start and end points for the line
    #     start_point = (second_coordinates[0] - 1400 * second_direction_vector[0], second_coordinates[1] - 1400 * second_direction_vector[1])
    #     end_point = (second_coordinates[0] + 1400 * second_direction_vector[0], second_coordinates[1] + 1400 * second_direction_vector[1])
    #     line_length = 0.000000
    #     for line in pocket_lines:
    #         intersection = intersect_point((nearest_second_pocket_coordinates, start_point), line)
    #         if intersection:
    #             line_com_length = math.sqrt((intersection[0] - nearest_second_pocket_coordinates[0])**2 + (intersection[1] - nearest_second_pocket_coordinates[1])**2)
    #             if line_length < line_com_length:
    #                 line_length = line_com_length
    #                 second_intersection_points = intersection
    #     print("second_interse", second_intersection_points)
    #     cv2.line(image, second_intersection_points, start_point, (255, 255, 255), 10)
        
    #     # Draw the infinite line passing through second_coordinates and nearest_second_pocket_coordinates
    #     # cv2.line(image, nearest_second_pocket_coordinates, second_coordinates, (0, 150, 255), 2)
        
    #     # Calculate the central point between second_coordinates and second_intersection_points
    #     central_point = (int((second_coordinates[0] + second_intersection_points[0]) / 2), 
    #                     int((second_coordinates[1] + second_intersection_points[1]) / 2))
    #     print("central_point", central_point)
                
    #     line_distance = math.sqrt((second_coordinates[0] - second_intersection_points[0])**2 + (second_coordinates[1] - second_intersection_points[1])**2)

    nearest_pocket_distance = float('inf')  # Initialize with a large value
    nearest_pocket_coordinates = None
    if first_coordinates:
        if pocket_name == 'aaa':
            for pocket, coordinates in pocket_locations.items():
                # Calculate the distance between the first coordinates and the current pocket coordinates
                distance = math.sqrt((first_coordinates[0] - coordinates[0])**2 + (first_coordinates[1] - coordinates[1])**2)
                if distance < nearest_pocket_distance:
                    nearest_pocket_distance = distance
                    nearest_pocket_coordinates = coordinates
        else :
            nearest_pocket_coordinates = pocket_locations[pocket_name]
        
        # Calculate the direction vector from first_coordinates to nearest_pocket_coordinates
        direction_vector = (nearest_pocket_coordinates[0] - first_coordinates[0], nearest_pocket_coordinates[1] - first_coordinates[1])

        # Determine the start and end points for the line
        first_start_point = (first_coordinates[0] - 1400 * direction_vector[0], first_coordinates[1] - 1400 * direction_vector[1])
        end_point = (first_coordinates[0] + 1400 * direction_vector[0], first_coordinates[1] + 1400 * direction_vector[1])
        
        # Normalize the direction vector
        direction_vector_length = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
        normalized_direction_vector = (direction_vector[0] / direction_vector_length, direction_vector[1] / direction_vector_length)

        # Calculate the coordinates of a point 10 units away from first_coordinates along the direction vector
        ten_units_away_point = (int(first_coordinates[0] - 24.2 * normalized_direction_vector[0]), int(first_coordinates[1] - 24.2 * normalized_direction_vector[1]))
        # cv2.line(image, perpendicular_point, perpendicular_end_point, (0, 0, 255), 2)
        print('white_ball_coordinates', white_ball_coordinates)
        white_direction_vector = (white_ball_coordinates[0] - ten_units_away_point[0], white_ball_coordinates[1] - ten_units_away_point[1])
        white_start_point = (ten_units_away_point[0] - 1400 * white_direction_vector[0], ten_units_away_point[1] - 1400 * white_direction_vector[1])
        white_end_point = (ten_units_away_point[0] + 1400 * white_direction_vector[0], ten_units_away_point[1] + 1400 * white_direction_vector[1])        
        # cv2.line(image, white_start_point, ten_units_away_point, (150, 0, 255), 2)
        # cv2.line(image, ten_units_away_point, white_end_point, (150, 0, 255), 2)
        # cv2.circle(image, (first_cushion_points[idx][0], first_cushion_points[idx][1]), 10, (104, 120, 255), 2)
        
        
        # Calculate the direction vector from ten_units_away_point to nearest_pocket_coordinates
        direction_ten_vector = (nearest_pocket_coordinates[0] - ten_units_away_point[0], 
                                nearest_pocket_coordinates[1] - ten_units_away_point[1])

        # Normalize the direction vector
        direction_ten_vector_length = math.sqrt(direction_ten_vector[0]**2 + direction_ten_vector[1]**2)
        normalized_direction_vector = ((direction_ten_vector[0] / direction_ten_vector_length) * 1400, 
                                    (direction_ten_vector[1] / direction_ten_vector_length) * 1400)

        # Calculate the new point 1400 units away from ten_units_away_point in the opposite direction
        new_point = (int(ten_units_away_point[0] - normalized_direction_vector[0]), 
                        int(ten_units_away_point[1] - normalized_direction_vector[1]))
        draw_parallel_line_if_needed(image, nearest_pocket_coordinates, ten_units_away_point, text_number, parellel_diatance, 1)

        # Draw the infinite line passing through first_coordinates and nearest_pocket_coordinates
        cur_ball_line = [ten_units_away_point, nearest_pocket_coordinates]
        
        white_angle = angle_between_lines(white_ball_coordinates, ten_units_away_point, cur_ball_line)
        print("white_angle", white_angle)                
        
        first_cushion_point = None
        distrupt_ball1 = False
        distrupt_ball2 = False            
        if white_angle <= 98.000:
            distrupt_ball1 = draw_parallel_line_if_needed(image, first_coordinates, nearest_pocket_coordinates, text_number, 15, 4)
            if distrupt_ball1:
                hit_ball_cushion = find_projection_point(pocket_lines, nearest_pocket_coordinates, first_coordinates, 1)
                # Calculate the direction vector from first_coordinates to nearest_pocket_coordinates
                direction_vector = (hit_ball_cushion[0][0] - first_coordinates[0], hit_ball_cushion[0][1] - first_coordinates[1])

                # Determine the start and end points for the line
                first_start_point = (first_coordinates[0] - 1400 * direction_vector[0], first_coordinates[1] - 1400 * direction_vector[1])
                end_point = (first_coordinates[0] + 1400 * direction_vector[0], first_coordinates[1] + 1400 * direction_vector[1])
                
                # Normalize the direction vector
                direction_vector_length = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
                normalized_direction_vector = (direction_vector[0] / direction_vector_length, direction_vector[1] / direction_vector_length)

                # Calculate the coordinates of a point 10 units away from first_coordinates along the direction vector
                ten_units_away_point = (int(first_coordinates[0] - 23 * normalized_direction_vector[0]), int(first_coordinates[1] - 23 * normalized_direction_vector[1]))
                # cv2.line(image, perpendicular_point, perpendicular_end_point, (0, 0, 255), 2)
                
                white_angle1 = angle_between_lines(white_ball_coordinates, ten_units_away_point, (first_coordinates, hit_ball_cushion[0]))
                
                white_direction_vector = (ten_units_away_point[0] - white_ball_coordinates[0], ten_units_away_point[1] - white_ball_coordinates[1])
                white_start_point = (white_ball_coordinates[0] - 1400 * white_direction_vector[0], white_ball_coordinates[1] - 1400 * white_direction_vector[1])
                white_end_point = (white_ball_coordinates[0] + 1400 * white_direction_vector[0], white_ball_coordinates[1] + 1400 * white_direction_vector[1])        

                # Calculate the direction vector from ten_units_away_point to hit_ball_cushion
                direction_ten_vector = (hit_ball_cushion[0][0] - ten_units_away_point[0], 
                                        hit_ball_cushion[0][1] - ten_units_away_point[1])

                # Normalize the direction vector
                direction_ten_vector_length = math.sqrt(direction_ten_vector[0]**2 + direction_ten_vector[1]**2)
                normalized_direction_vector = ((direction_ten_vector[0] / direction_ten_vector_length) * 1400, 
                                            (direction_ten_vector[1] / direction_ten_vector_length) * 1400)

                # Calculate the new point 1400 units away from ten_units_away_point in the opposite direction
                new_point = (int(ten_units_away_point[0] - normalized_direction_vector[0]), 
                                int(ten_units_away_point[1] - normalized_direction_vector[1]))
                
                cv2.circle(image, hit_ball_cushion[0], 10, (104, 120, 255), 2)
                # cv2.line(image, first_coordinates, hit_ball_cushion[0], (255, 255, 255), 2)
                cv2.line(image, first_coordinates, nearest_pocket_coordinates, (255, 255, 255), 2)
                cv2.line(image, white_ball_coordinates, ten_units_away_point, (255, 255, 255), 2)
            else :
                cv2.line(image, nearest_pocket_coordinates, first_coordinates, (255, 255, 255), 2)
                cv2.line(image, white_ball_coordinates, ten_units_away_point, (255, 255, 255), 2)
        else :
            distrupt_ball1 = False
            distrupt_ball2 = False
            idx = -1
            first_cushion_points = find_projection_point(pocket_lines, ten_units_away_point, white_ball_coordinates, cur_ball_line)
            # cv2.circle(image, first_cushion_points[0], 10, (104, 120, 255), 2)
            # cv2.circle(image, first_cushion_points[1], 10, (104, 120, 255), 2)
            # cv2.circle(image, first_cushion_points[2], 10, (104, 120, 255), 2)
            # cv2.circle(image, first_cushion_points[3], 10, (104, 120, 255), 2)

            ten_cushion_length = 9999999.0000
            for i in range(len(first_cushion_points)):
                distrupt_ball1 = draw_parallel_line_if_needed(image, white_ball_coordinates, (first_cushion_points[i][0], first_cushion_points[i][1]), text_number, 15, 3)
                distrupt_ball2 = draw_parallel_line_if_needed(image, (first_cushion_points[i][0], first_cushion_points[i][1]), ten_units_away_point, text_number, 15, 3)
                between_length = math.sqrt((first_cushion_points[i][0] - ten_units_away_point[0])**2 + (first_cushion_points[i][1] - ten_units_away_point[1])**2)
                if distrupt_ball1 == False and distrupt_ball2 == False and between_length < ten_cushion_length:
                    idx = i
                distrupt_ball1 = False
                distrupt_ball1 = False

            # if distrupt_ball1 or distrupt_ball2:
            if idx == -1:
                hit_ball_cushion = find_projection_point(pocket_lines, nearest_pocket_coordinates, first_coordinates, 1)
                print("hitball", hit_ball_cushion[0])
                # Calculate the direction vector from first_coordinates to nearest_pocket_coordinates
                direction_vector = (hit_ball_cushion[0][0] - first_coordinates[0], hit_ball_cushion[0][1] - first_coordinates[1])

                # Determine the start and end points for the line
                first_start_point = (first_coordinates[0] - 1400 * direction_vector[0], first_coordinates[1] - 1400 * direction_vector[1])
                end_point = (first_coordinates[0] + 1400 * direction_vector[0], first_coordinates[1] + 1400 * direction_vector[1])
                
                # Normalize the direction vector
                direction_vector_length = math.sqrt(direction_vector[0]**2 + direction_vector[1]**2)
                normalized_direction_vector = (direction_vector[0] / direction_vector_length, direction_vector[1] / direction_vector_length)

                # Calculate the coordinates of a point 10 units away from first_coordinates along the direction vector
                ten_units_away_point = (int(first_coordinates[0] - 23 * normalized_direction_vector[0]), int(first_coordinates[1] - 23 * normalized_direction_vector[1]))
                # cv2.line(image, perpendicular_point, perpendicular_end_point, (0, 0, 255), 2)
                
                white_angle1 = angle_between_lines(white_ball_coordinates, ten_units_away_point, (first_coordinates, hit_ball_cushion[0]))
                print("white", white_angle1)
                
                white_direction_vector = (ten_units_away_point[0] - white_ball_coordinates[0], ten_units_away_point[1] - white_ball_coordinates[1])
                # white_start_point = (white_ball_coordinates[0] - 1400 * white_direction_vector[0], white_ball_coordinates[1] - 1400 * white_direction_vector[1])
                # white_end_point = (white_ball_coordinates[0] + 1400 * white_direction_vector[0], white_ball_coordinates[1] + 1400 * white_direction_vector[1])        

                # Calculate the direction vector from ten_units_away_point to hit_ball_cushion[0]
                direction_ten_vector = (hit_ball_cushion[0][0] - ten_units_away_point[0], 
                                        hit_ball_cushion[0][1] - ten_units_away_point[1])

                # Normalize the direction vector
                direction_ten_vector_length = math.sqrt(direction_ten_vector[0]**2 + direction_ten_vector[1]**2)
                normalized_direction_vector = ((direction_ten_vector[0] / direction_ten_vector_length) * 1400, 
                                            (direction_ten_vector[1] / direction_ten_vector_length) * 1400)

                # Calculate the new point 1400 units away from ten_units_away_point in the opposite direction
                new_point = (int(ten_units_away_point[0] - normalized_direction_vector[0]), 
                                int(ten_units_away_point[1] - normalized_direction_vector[1]))
                
                cv2.circle(image, hit_ball_cushion[0], 10, (104, 120, 255), 2)
                cv2.line(image, first_coordinates, hit_ball_cushion[0], (255, 255, 255), 2)
                cv2.line(image, hit_ball_cushion[0], nearest_pocket_coordinates, (255, 255, 255), 2)
                cv2.line(image, white_ball_coordinates, ten_units_away_point, (255, 255, 255), 2)
            else:
                cv2.line(image, nearest_pocket_coordinates, ten_units_away_point, (255, 255, 255), 2)
                cv2.circle(image, (first_cushion_points[idx][0], first_cushion_points[idx][1]), 10, (104, 120, 255), 2)
                cv2.line(image, white_ball_coordinates, (first_cushion_points[idx][0], first_cushion_points[idx][1]), (255, 255, 255), 2)
                cv2.line(image, (first_cushion_points[idx][0], first_cushion_points[idx][1]), ten_units_away_point, (255, 255, 255), 2)            
                first_cushion_point = (first_cushion_points[idx][0], first_cushion_points[idx][1])
                
                
        # cv2.line(image, cushion_line[0], cushion_line[1], (255, 255, 255), 2)
                
        #///////////////あーここがぴったりだ、力加減だけ注意して打ってみよう。       
                
        if first_cushion_point:
            # Calculate the direction vector from first_cushion_point to ten_units_away_point
            direction_vector = (ten_units_away_point[0] - first_cushion_point[0], 
                                ten_units_away_point[1] - first_cushion_point[1])
        else:
            # Calculate the direction vector of the line segment
            direction_vector = (end_point[0] - first_start_point[0], end_point[1] - first_start_point[1])

        # Calculate the direction vector from nearest_pocket_coordinates to new_point
        pocket_to_new_vector = (new_point[0] - nearest_pocket_coordinates[0], 
                                new_point[1] - nearest_pocket_coordinates[1])
        spin_central_points = (633, 350)
        left_central_points = (383, 350)
        right_central_points = (883, 350)
        if english_spin_status:
            # Extract alpha channel
            alpha_s = english_spin[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            
            alpha_s_strength = strength_left_bar[:, :, 3] / 255.0
            alpha_l_strength = 1.0 - alpha_s_strength
                    
            # Define the region of interest (ROI) on the background image
            x1, y1 = spin_central_points[0]-75, spin_central_points[1]-75
            x2, y2 = spin_central_points[0]+75, spin_central_points[1]+75
            
            left_x1, left_y1 = left_central_points[0]-125, left_central_points[1]-15
            left_x2, left_y2 = left_central_points[0]+125, left_central_points[1]+15
            
            right_x1, right_y1 = right_central_points[0]-125, right_central_points[1]-15
            right_x2, right_y2 = right_central_points[0]+125, right_central_points[1]+15

            # Blend images
            for c in range(0, 3):  # Loop over color channels
                image[y1:y2, x1:x2, c] = (alpha_s * english_spin[:, :, c] + alpha_l * image[y1:y2, x1:x2, c])
            
            for c in range(0, 3):  # Loop over color channels
                image[left_y1:left_y2, left_x1:left_x2, c] = (alpha_s_strength * strength_left_bar[:, :, c] + alpha_l_strength * image[left_y1:left_y2, left_x1:left_x2, c])
            
            for c in range(0, 3):  # Loop over color channels
                image[right_y1:right_y2, right_x1:right_x2, c] = (alpha_s_strength * strength_right_bar[:, :, c] + alpha_l_strength * image[right_y1:right_y2, right_x1:right_x2, c])

            spin_distance = math.sqrt((mouse_coordinates[0] - spin_central_points[0]) ** 2 +
                                    (mouse_coordinates[1] - spin_central_points[1]) ** 2)
            angle_Vector = (-pocket_to_new_vector[1], pocket_to_new_vector[0])
            
            if spin_distance <= 75.00:
                if spin_central_points != mouse_coordinates:
                    spin_mouse_coordinates = mouse_coordinates
                    cv2.circle(image, mouse_coordinates, 3, (255, 0, 255), 3)
                    # Direction vectors
                    direction_vector_1 = (0, 75)  # (633, 350) to (633, 425)
                    direction_vector_2 = (mouse_coordinates[0] - spin_central_points[0], mouse_coordinates[1] - spin_central_points[1])

                    # Calculate the dot product
                    dot_product = direction_vector_1[0] * direction_vector_2[0] + direction_vector_1[1] * direction_vector_2[1]

                    # Calculate magnitudes
                    magnitude_1 = math.sqrt(direction_vector_1[0] ** 2 + direction_vector_1[1] ** 2)
                    magnitude_2 = math.sqrt(direction_vector_2[0] ** 2 + direction_vector_2[1] ** 2)

                    # Calculate the cosine of the angle
                    cos_theta = dot_product / (magnitude_1 * magnitude_2)

                    # Calculate the angle in radians
                    angle_radians = math.acos(cos_theta)

                    # Convert angle to degrees
                    spin_angle_degrees = math.degrees(angle_radians)
                    
                    print(".......", spin_angle_degrees)
                
                    spin_angle = (spin_angle_degrees / 90) * 20
                    if mouse_coordinates[0] > 633:
                        spin_angle *= -1
                    # Convert angle to radians if it's in degrees
                    angle_radians = math.radians(spin_angle)
                    
                    # cv2.namedWindow('image')
                    # cv2.setMouseCallback('image',draw_circle)
                    
                    # Calculate the rotated vector components
                    perpendicular_vector = (angle_Vector[0] * math.cos(angle_radians) - angle_Vector[1] * math.sin(angle_radians)
                                    , angle_Vector[0] * math.sin(angle_radians) + angle_Vector[1] * math.cos(angle_radians))
                else:
                    perpendicular_vector = angle_Vector
            else:
                perpendicular_vector = angle_Vector
            # cv2.rectangle(image, (333, 340), (333+1, 360), 0, 1)
            # cv2.rectangle(image, (933, 340), (933 + 1, 360), 0, 1)
            # cv2.putText(image, str(70), (324, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
            # cv2.putText(image, str(70), (924, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
            if mouse_coordinates[1] >= 335 and mouse_coordinates[1] <= 365:
                if mouse_coordinates[0] >= 270 and mouse_coordinates[0] <= 496:
                    left_distance = mouse_coordinates[0] - left_central_points[0]
                    cv2.rectangle(image, (mouse_coordinates[0], 340), (mouse_coordinates[0]+1, 360), 0, 1)
                    cv2.rectangle(image, (right_central_points[0] - left_distance, 340), (right_central_points[0] - left_distance + 1, 360), 0, 1)
                    
                    strength_val = int(50 - (left_distance / 25) * 10)
                    strength_angle = (strength_val / 100) * 10
                    
                    cv2.putText(image, str(strength_val), (mouse_coordinates[0]-10, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
                    cv2.putText(image, str(strength_val), (right_central_points[0] - left_distance-10, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
                    if spin_angle != 0.00000:
                    #     perpendicular_vector = perpendicular_vector_spin
                        angle_radians = math.radians(spin_angle + strength_angle)
                        cv2.circle(image, spin_mouse_coordinates, 3, (255, 0, 255), 3)
                        perpendicular_vector = (angle_Vector[0] * math.cos(angle_radians) - angle_Vector[1] * math.sin(angle_radians)
                                    , angle_Vector[0] * math.sin(angle_radians) + angle_Vector[1] * math.cos(angle_radians))
                    else:
                        angle_radians = math.radians(strength_angle)
                        cv2.circle(image, spin_mouse_coordinates, 3, (255, 0, 255), 3)
                        perpendicular_vector = (angle_Vector[0] * math.cos(angle_radians) - angle_Vector[1] * math.sin(angle_radians)
                                    , angle_Vector[0] * math.sin(angle_radians) + angle_Vector[1] * math.cos(angle_radians))
                if mouse_coordinates[0] >= 770 and mouse_coordinates[0] <= 996:
                    right_distance = mouse_coordinates[0] - right_central_points[0]
                    cv2.rectangle(image, (left_central_points[0] - right_distance, 340), (left_central_points[0] - right_distance + 1, 360), 0, 1)
                    cv2.rectangle(image, (mouse_coordinates[0], 340), (mouse_coordinates[0]+1, 360), 0, 1)
                    
                    strength_val = int(50 + (right_distance / 25) * 10)
                    strength_angle = (strength_val / 100) * 10
                    
                    cv2.putText(image, str(strength_val), (mouse_coordinates[0]-10, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
                    cv2.putText(image, str(strength_val), (left_central_points[0] - right_distance-10, 378), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Draw only the circle outline
                    if spin_angle != 0.00000:
                    #     perpendicular_vector = perpendicular_vector_spin
                        angle_radians = math.radians(spin_angle + strength_angle)
                        perpendicular_vector = (angle_Vector[0] * math.cos(angle_radians) - angle_Vector[1] * math.sin(angle_radians)
                                    , angle_Vector[0] * math.sin(angle_radians) + angle_Vector[1] * math.cos(angle_radians))
                    else:
                    #     perpendicular_vector = perpendicular_vector_spin
                        angle_radians = math.radians(strength_angle)
                        perpendicular_vector = (angle_Vector[0] * math.cos(angle_radians) - angle_Vector[1] * math.sin(angle_radians)
                                    , angle_Vector[0] * math.sin(angle_radians) + angle_Vector[1] * math.cos(angle_radians))
        else:  
            # Calculate the perpendicular direction vector to the direction from nearest_pocket_coordinates to new_point
            perpendicular_vector = (-pocket_to_new_vector[1], pocket_to_new_vector[0])

        # Normalize the perpendicular vector
        perpendicular_vector_length = math.sqrt(perpendicular_vector[0]**2 + perpendicular_vector[1]**2)
        normalized_perpendicular_vector = (perpendicular_vector[0] / perpendicular_vector_length, 
                                        perpendicular_vector[1] / perpendicular_vector_length)

        # Calculate the direction vector between ten_units_away_point and white_ball_coordinates
        ten_to_white_vector = (white_ball_coordinates[0] - ten_units_away_point[0], 
                            white_ball_coordinates[1] - ten_units_away_point[1])

        # Adjust the normalized perpendicular vector if necessary to ensure bisector_point is in the opposite direction
        dot_product = (normalized_perpendicular_vector[0] * ten_to_white_vector[0] + 
                    normalized_perpendicular_vector[1] * ten_to_white_vector[1])

        if dot_product > 0:
            # If the dot product is positive, the vectors point in the same direction, so invert the perpendicular vector
            normalized_perpendicular_vector = (-normalized_perpendicular_vector[0], -normalized_perpendicular_vector[1])

        bisector_point = (int(ten_units_away_point[0] + 1400 * normalized_perpendicular_vector[0]), 
                    int(ten_units_away_point[1] + 1400 * normalized_perpendicular_vector[1]))
        
        if first_cushion_point and distrupt_ball1 == False:
            second_cushion_angle = angle_between_lines(first_cushion_point, ten_units_away_point, (ten_units_away_point, bisector_point))
            print("second_cushion_angle", second_cushion_angle)
            if second_cushion_angle > 95:
                bisector_point = (int(ten_units_away_point[0] - 1400 * normalized_perpendicular_vector[0]), 
                    int(ten_units_away_point[1] - 1400 * normalized_perpendicular_vector[1]))
        
        # # Calculate the direction vector from ten_units_away_point to white_end_point
        # vec_white = (white_end_point[0] - ten_units_away_point[0], white_end_point[1] - ten_units_away_point[1])

        # # Calculate the direction vector from ten_units_away_point to start_point
        # vec_start = (first_start_point[0] - ten_units_away_point[0], first_start_point[1] - ten_units_away_point[1])

        # # Normalize the direction vectors
        # mag_white = math.sqrt(vec_white[0]**2 + vec_white[1]**2)
        # normalized_vec_white = (vec_white[0] / mag_white, vec_white[1] / mag_white)

        # mag_start = math.sqrt(vec_start[0]**2 + vec_start[1]**2)
        # normalized_vec_start = (vec_start[0] / mag_start, vec_start[1] / mag_start)

        # # Calculate the sum of the normalized direction vectors
        # bisector_vec = (normalized_vec_white[0] + normalized_vec_start[0], normalized_vec_white[1] + normalized_vec_start[1])

        # # Normalize the bisector vector
        # mag_bisector = math.sqrt(bisector_vec[0]**2 + bisector_vec[1]**2)
        # normalized_bisector = (bisector_vec[0] / mag_bisector, bisector_vec[1] / mag_bisector)        

        # Calculate a point on the bisector line
        # bisector_point = (int(ten_units_away_point[0] + normalized_bisector[0] * 1400), int(ten_units_away_point[1] + normalized_bisector[1] * 1400))
        # Find intersection points
        intersection_points = []
        intersection_line = []
        line_status = 1
        print("intersection", bisector_point)
        for line in pocket_lines:
            intersection = intersect_point((ten_units_away_point, bisector_point), line)
            if intersection:
                intersection_points.append(intersection)
                intersection_line.append(line)
                break
            line_status = line_status + 1
        bisector_point = intersection_points[0]
        # Calculate the angle between the line formed by ten_units_away_point and bisector_point
        angle = angle_between_lines(ten_units_away_point, bisector_point, intersection_line[0])
        
        second_line_coordinates = angle_to_line(bisector_point, angle, 1400, line_status)
        # cv2.circle(image, bisector_point, 20, (104, 120, 255), 2)
        
        # Define the length of each dash and gap
        dash_length = 2
        gap_length = 2
        
        # Calculate the length of the line segment
        line_length = math.sqrt((bisector_point[0] - ten_units_away_point[0])**2 + (bisector_point[1] - ten_units_away_point[1])**2)

        # Calculate the number of dashes needed
        num_dashes = int(line_length / (dash_length + gap_length))
        print("num", num_dashes, line_length)
        # Calculate the step size for drawing dashes
        step_x = (bisector_point[0] - ten_units_away_point[0]) / num_dashes
        step_y = (bisector_point[1] - ten_units_away_point[1]) / num_dashes
        cv2.circle(image, bisector_point, 10, (255, 255, 100), 2)
        # Draw the dotted line
        current_point = ten_units_away_point
        part_length = gap_length  # Consider the initial gap
        for i in range(num_dashes):
            start_point = (int(current_point[0]), int(current_point[1]))
            end_point = (int(current_point[0] + dash_length * step_x), int(current_point[1] + dash_length * step_y))
            part_length = part_length + math.sqrt((start_point[0] - end_point[0])**2 + (start_point[1] - end_point[1])**2) +  7  # Add the length of the current dash and gap
            
            # Check if the end point exceeds bisector_point
            if part_length > line_length:            
                break
            cv2.line(image, start_point, end_point, (255, 255, 255), 2)
            
            current_point = (current_point[0] + (dash_length + gap_length) * step_x, current_point[1] + (dash_length + gap_length) * step_y)
        
        # distrupt ball parallel line draw
        draw_parallel_line_if_needed(image, ten_units_away_point, end_point, text_number, parellel_diatance, 2)
        
        
        # Calculate distances between bisector_point and each pocket location
        distances = {name: math.sqrt((bisector_point[0] - loc[0])**2 + (bisector_point[1] - loc[1])**2) for name, loc in pocket_locations.items()}
        # Find the closest pocket location
        closest_pocket = min(distances, key=distances.get)
        # Check if the distance to the closest pocket location is within the specified range (e.g., 20 units)
        if distances[closest_pocket] > 20:
            
            # if second_coordinates:
            #     second_line_intersection_point = None
            #     for line in pocket_lines:
            #         intersection = intersect_point((bisector_point, second_line_coordinates), line)
            #         if intersection and intersection != bisector_point:
            #             second_line_intersection_point = intersection
                
                # line1 = (second_coordinates, second_intersection_points)
                # line2 = (bisector_point,second_line_coordinates)
                # stable_intersect_point = second_intersect_point(line1, line2)
                # if stable_intersect_point:
                #     cv2.circle(image, stable_intersect_point, 10, (100, 150, 120), 2)
                # else: stable_intersect_point = second_line_intersection_point
                # # Get the dimensions of the foreground image
                # foreground_height, foreground_width, _ = foreground_image.shape
                # x, y = stable_intersect_point[0] - 99, stable_intersect_point[1] - 32
                # print("stable_intersect_point", x, y)
                
                # if x < 0: x = 80
                # elif x > 1050: x = 1000
                # if y < 80: y = 80
                # elif y > 550: y = 550
                # # Define the region where you want to place the image
                # y1, y2 = y, y + foreground_height
                # x1, x2 = x, x + foreground_width

                # # Blend the foreground image onto the background image
                # alpha_s = foreground_image[:, :, 3] / 255.0
                # alpha_l = 1.0 - alpha_s
                
                # for c in range(0, 3):
                #     image[y1:y2, x1:x2, c] = (alpha_s * foreground_image[:, :, c] +
                #                                         alpha_l * image[y1:y2, x1:x2, c])
                # cv2.circle(image, second_coordinates, 10, (255, 255, 255), 2)  # Draw only the circle outline

            line_length1 = math.sqrt((bisector_point[0] - second_line_coordinates[0])**2 + (bisector_point[1] - second_line_coordinates[1])**2)

            # Calculate the number of dashes needed
            num_dashes = int(line_length1 / (dash_length + gap_length))

            # Calculate the step size for drawing dashes
            step_x = (second_line_coordinates[0] - bisector_point[0]) / num_dashes
            step_y = (second_line_coordinates[1] - bisector_point[1]) / num_dashes

            # Draw the dotted line
            current_point = bisector_point
            part_length = gap_length  # Consider the initial gap
            for i in range(num_dashes):
                start_point = (int(current_point[0]), int(current_point[1]))
                bisector_point = (int(current_point[0] + dash_length * step_x), int(current_point[1] + dash_length * step_y))
                part_length = part_length + math.sqrt((start_point[0] - bisector_point[0])**2 + (start_point[1] - bisector_point[1])**2) +  7  # Add the length of the current dash and gap
                
                cv2.line(image, start_point, bisector_point, (255, 255, 255), 2)
                
                # Check if the end point exceeds bisector_point
                if part_length > line_length1:
                    break
                
                current_point = (current_point[0] + (dash_length + gap_length) * step_x, current_point[1] + (dash_length + gap_length) * step_y)
                        
        cv2.circle(image, ten_units_away_point, 10, (0, 0, 153), 2)
        
        cv2.circle(image, first_coordinates, 10, (0, 255, 255), 2)  # Draw only the circle outline
        for text, point in text_number:
            cv2.putText(image, text, (point[0]-5, point[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        for point in check:
            cv2.circle(image, point, 1, (0, 0, 255), 2)  # Draw only the circle outline
            
        

            
    # Display image with circles and numbers
    # Convert image to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Convert image to PIL format
    pil_image = Image.fromarray(image_rgb)
    # Convert PIL image to Tkinter PhotoImage
    tk_image = ImageTk.PhotoImage(image=pil_image)

    # Update the image label with the new image
    image_label.config(image=tk_image)
    image_label.image = tk_image

pocket_color = 'aaa'
pocket_pos = pocket_color
# Function to capture the current frame from the live video and analyze it
def select_image_file():
    global cap, video_running, select_count
    if cap:
        # Capture the current frame from the live video
        ret, frame = cap.read()
        print("video", video_running)
        if select_count == 1:
            display_video_feed()
            select_count = 0
        elif video_running:
            root.after_cancel(video_running)
            select_count = 1
            if ret:
                # Resize the frame to 1180x700
                # resized_frame = cv2.resize(frame, (1180, 700))
                
                # Convert the frame to HSV
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Analyze the frame
                analyze_image(frame, hsv, pocket_color, 10, english_spin_status, mouse_coordinates)
                
                # Convert the frame to RGB for display in Tkinter
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image_rgb)
                tk_image = ImageTk.PhotoImage(image=pil_image)
                
                # Display the frame in the Tkinter window
                image_label.config(image=tk_image)
                image_label.image = tk_image
            else:
                print("Failed to capture frame from video")


# # Function to select image file
# def select_image_file():
#     global file_path
#     file_path = filedialog.askopenfilename()
#     if file_path:
#         # Read the selected image file
#         image = cv2.imread(file_path)
#         # Resize the image
#         # image = cv2.resize(img, (1000, 550))
#         # Convert image to HSV
#         hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
#         # Analyze the image
#         analyze_image(image, hsv, pocket_color, 10, english_spin_status, mouse_coordinates)

# Function to handle pocket selection
def select_pocket(pocket):
    global file_path, pocket_pos, cap, video_running
    pocket_pos = pocket
    print("////////////////////////")
    print(f"Selected pocket: {pocket_pos}")
    if cap:
        # Capture the current frame from the live video
        ret, frame = cap.read()
        if video_running:
            root.after_cancel(video_running)

        if ret:
            # Resize the frame to 1180x700
            # resized_frame = cv2.resize(frame, (1180, 700))
            
            # Convert the frame to HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Analyze the frame
            analyze_image(frame, hsv, pocket, 10, english_spin_status, mouse_coordinates)
    # if file_path:
    #     # Read the selected image file
    #     image = cv2.imread(file_path)
    #     # Resize the image
    #     # image = cv2.resize(img, (1000, 550))
    #     # Convert image to HSV
    #     hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #     image = cv2.resize(image, (1180, 600))
    #     # Display the image
    #     # cv2.imshow("Selected Image", image)
    #     # Analyze the image
    #     analyze_image(image, hsv, pocket, 10, english_spin_status, mouse_coordinates)
        
def input_distance():
    global file_path
    global pocket_pos
    if file_path:
        distance = float(entry.get())  # Get the value entered in the entry widget
        # Read the selected image file
        image = cv2.imread(file_path)
        # Resize the image
        # image = cv2.resize(img, (1000, 550))
        # Convert image to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (1180, 600))
        # Display the image
        # cv2.imshow("Selected Image", image)
        # Analyze the image
        analyze_image(image, hsv, pocket_pos, distance, english_spin_status, mouse_coordinates)

def show_hit_ball():
    global file_path
    global english_spin_status
    english_spin_status = not english_spin_status
    if file_path:
        image = cv2.imread(file_path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.resize(image, (1180, 600))
        analyze_image(image, hsv, pocket_pos, 10, english_spin_status, mouse_coordinates)
        
def on_left_click(event):
    global file_path
    x, y = event.x, event.y
    print(f"Left mouse button clicked at x={x}, y={y}")
    if file_path:
        image = cv2.imread(file_path)
        image = cv2.resize(image, (1180, 600))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        analyze_image(image, hsv, pocket_pos, 10, english_spin_status, (x, y))
        
# Create the main Tkinter window
root = tk.Tk()
root.title("ビリヤード解析システム")
root.geometry("1400x900")

# Main screen widgets
main_screen = tk.Frame(root)
select_file_button = tk.Button(main_screen, text="キャプチャ", command=select_image_file, bg="#99ffff", width=20, font=("Arial", 14, 'bold'))
select_file_button.pack(side=tk.TOP, pady=10)

return_button = tk.Button(main_screen, text="戻る", bg="#99ffff", width=20, font=("Arial", 14, 'bold'), command=return_origin)
return_button.pack(side=tk.TOP, padx=10)  # Adjust the pady value as needed for spacing

pocket_selection_frame = tk.Frame(main_screen)
pockets = ["Polaris", "Sirius", "Jupiter", "Venus", "Terra", "Mars"]
for pocket in pockets:
    pocket_button = tk.Button(pocket_selection_frame, text=pocket, command=lambda p=pocket: select_pocket(p), bg="#66a3ff",  width=10, font=("Arial", 12))
    pocket_button.pack(side=tk.LEFT, padx=15, pady=10)
pocket_selection_frame.pack()

image_label = tk.Label(main_screen)
image_label.pack(side=tk.LEFT, padx=10, pady=10)

display_video_feed()

# hit_point_button = tk.Button(main_screen, text="表示", width=5, bg='#b8b894', font=("Semi-serif", 12, "bold"), command=show_hit_ball, compound='left')
# hit_point_button.pack(pady=10)

# label = tk.Label(main_screen, text="通り:", font=("Semi-serif", 15, "bold"))
# label.pack(pady=5)

# entry = tk.Entry(main_screen, width=5, font=("Semi-serif", 15))
# entry.pack(pady=5)

# # Bind the <Return> key event to the input_distance function
# entry.bind("<Return>", lambda event: input_distance())


# canvas = tk.Canvas(main_screen, width=40, height=500, bg='#1a75ff')
# canvas.pack(side=tk.RIGHT, padx=10, pady=10)

# Button to exercise to the secondary screen
exercise_button = tk.Button(root, text="フリー練習モード", width=58, height=30, bg='#b8b894', font=("Semi-serif", 15, "bold"),  command=show_main_screen)
exercise_button.pack(side=tk.LEFT, padx=10)

rank_button = tk.Button(root, text="ランクアップ試験モード", width=58, height=30, bg='#ff9999', font=("Semi-serif", 15, "bold"), command=show_rank_screen)
rank_button.pack(side=tk.LEFT, padx=10)

# input_button = tk.Button(root, text="Open Input Window", width=30, height=2, bg='#ff9999', font=("Semi-serif", 15, "bold"), command=create_input_window)
# input_button.pack(pady=20)

# Bind mouse motion event to the on_mouse_move function
root.bind("<Button-1>", on_left_click)

# Run the Tkinter event loop
root.mainloop()

# Release the video capture object when the window is closed
cap.release()