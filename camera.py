import cv2
from tkinter import Tk, Label
from PIL import Image, ImageTk

# Initialize Tkinter window
root = Tk()
image_label = Label(root)
image_label.pack()

# Camera setup with desired resolution
cap = cv2.VideoCapture(1)  # 0 for default camera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1180)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
video_running = None
select_count = 1  # Toggle for video feed

def display_video_feed():
    """Displays the live video feed resized to 1180x700."""
    global video_running
    
    def update_frame():
        global video_running
        ret, frame = cap.read()
        
        if ret:
            # Convert the frame to RGB for displaying in Tkinter
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            tk_image = ImageTk.PhotoImage(image=pil_image)

            # Update the Tkinter label with the frame
            image_label.config(image=tk_image)
            image_label.image = tk_image

        # Schedule the next frame update
        video_running = root.after(10, update_frame)
    
    update_frame()

# Function to toggle the video feed
def select_image_file():
    global cap, video_running, select_count
    if cap:
        ret, frame = cap.read()
        print("video", video_running)
        
        if select_count == 1:
            # Start displaying the video feed
            display_video_feed()
            select_count = 0
        elif video_running:
            # Stop the video feed
            root.after_cancel(video_running)
            select_count = 1

# Initialize the video feed
select_image_file()

# Run the Tkinter main loop
root.mainloop()

# Release the camera when the window is closed
cap.release()
