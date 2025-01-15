import os
import cv2
import numpy as np
import shutil
import threading
import time
import datetime
from tkinter import Tk, Label, Button, filedialog, messagebox, Frame, LEFT, RIGHT, BOTH, ttk
from PIL import Image, ImageTk
from ultralytics import YOLO
from tkinter.ttk import Progressbar
from twilio.rest import Client
from tkinter import Toplevel, Listbox

# ------------------- Helper Functions -------------------
def send_sms(phone_number, message):
    account_sid = ''  # Replace with your Account SID
    auth_token = ''  # Replace with your Auth Token
    twilio_phone_number = ''  # Replace with your Twilio phone number

    client = Client(account_sid, auth_token)

    try:
        message = client.messages.create(
            body=message,
            from_=twilio_phone_number,
            to=f'+{phone_number}'  # Include country code
        )
        print(f"SMS sent to {phone_number}")
    except Exception as e:
        print(f"Failed to send SMS: {e}")


def clear_folder(folder_path):
    """Clears the specified folder, creating it if it doesn't exist."""
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)

def save_frame(frame, frame_index, output_folder):
    """Saves the given frame to a timestamped subfolder in the history folder."""
    # Get the current date and time
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    history_folder = os.path.join("history", timestamp)
    os.makedirs(history_folder, exist_ok=True)

    # Save the frame
    filename = f"accident_frame_{frame_index:04d}.jpg"
    save_path = os.path.join(history_folder, filename)
    cv2.imwrite(save_path, frame)

    print(f"Frame saved: {save_path}")
    return history_folder

def detect_accidents(video_path, output_folder):
    """
    Detects potential accidents in a video using optical flow and saves frames
    with significant motion to the specified folder.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Unable to open video file."

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return False, "Couldn't read the first frame from the video."

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    motion_threshold = 1.5
    motion_limit = 400
    consecutive_frames = 3
    accident_frame_count = 0
    frame_index = 0

    clear_folder(output_folder)  # Ensure output folder is clean

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_index += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        significant_motion = (mag > motion_threshold)
        significant_pixels = np.count_nonzero(significant_motion)

        if significant_pixels > motion_limit:
            accident_frame_count += 1
        else:
            accident_frame_count = 0

        if accident_frame_count >= consecutive_frames:
            # Save frame in optic flow output folder
            filename = f"optic_flow_frame_{frame_index:04d}.jpg"
            save_path = os.path.join(output_folder, filename)
            cv2.imwrite(save_path, frame)
            print(f"Frame saved: {save_path}")

        prev_gray = gray

    cap.release()
    return True, f"Optical flow completed. Frames saved in {output_folder}"

def process_with_yolo(input_folder, output_folder, model_path="best.pt"):
    """
    Processes images using the YOLO model and saves the output
    in the yolo_output folder. Creates a subfolder in the history folder
    only if accidents are detected.
    """
    # Load the YOLOv8 model
    model = YOLO(model_path)

    # Set confidence threshold
    confidence_threshold = 0.85

    # Ensure YOLO output folder is clean
    clear_folder(output_folder)

    accident_detected = False
    history_folder = None

    # Process each image in the input folder
    for image_name in os.listdir(input_folder):
        input_image_path = os.path.join(input_folder, image_name)

        # Ensure it's an image file
        if not image_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
            print(f"Skipping non-image file: {image_name}")
            continue

        # Run inference on the image
        results = model(input_image_path, conf=confidence_threshold)

        # Iterate through results and filter by 'accident' or 'severe_accident' labels
        for result in results:
            detected = False
            for box in result.boxes:
                # Safely extract the class index
                class_index = int(box.cls.item())  # Convert tensor to integer index
                label = result.names[class_index]  # Get the label name
                if label in ["Accident", "severe-accident"]:  # Check if label matches
                    detected = True
                    accident_detected = True  # Update flag
                    break

            if detected:
                # Create history folder if it doesn't exist
                if not history_folder:
                    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                    history_folder = os.path.join("history", timestamp)
                    os.makedirs(history_folder, exist_ok=True)

                # Save the annotated frame in YOLO output folder
                annotated_frame = result.plot()  # Get annotated frame
                save_path = os.path.join(output_folder, f"processed_{image_name}")
                cv2.imwrite(save_path, annotated_frame)

                # Log the frame in the history folder
                history_save_path = os.path.join(history_folder, f"processed_{image_name}")
                cv2.imwrite(history_save_path, annotated_frame)
                print(f"Processed and saved: {save_path}")
                break  # Exit loop as we only need to save one frame per image if conditions match

    if not accident_detected:
        print("No accidents detected. History folder not created.")

    return output_folder

# ------------------- GUI Code -------------------
import threading

class AccidentDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Accident Detection GUI")
        self.root.geometry("1000x800")
        self.root.configure(bg="#001f3f")  # Deep Blue background

        self.video_path = None
        self.image_list = []
        self.current_image_index = 0
        
        # Header Label
        self.header = Label(
            root, 
            text="Accident Detection using Optical Flow and YOLO", 
            font=("Arial", 20, "bold"), 
            bg="#001f3f",  # Deep Blue
            fg="#ff4500"   # Orange-Red
        )
        self.header.pack(pady=10)

        # Buttons Frame
        self.button_frame = Frame(root, bg="#001f3f")  # Deep Blue
        self.button_frame.pack(pady=10)

        self.add_video_button = Button(
            self.button_frame, 
            text="Add Video", 
            command=self.add_video, 
            width=15, 
            bg="#ff4500",  # Orange-Red
            fg="white", 
            font=("Arial", 12, "bold")
        )
        self.add_video_button.pack(side=LEFT, padx=10)

        self.run_pipeline_button = Button(
            self.button_frame, 
            text="Run Detection", 
            command=self.start_pipeline_thread, 
            width=15, 
            bg="#ffa500",  # Yellow-Orange
            fg="white", 
            font=("Arial", 12, "bold")
        )
        self.run_pipeline_button.pack(side=LEFT, padx=10)

        self.refresh_button = Button(
            self.button_frame, 
            text="Refresh", 
            command=self.refresh_app, 
            width=15, 
            bg="#ff4500",  # Orange-Red
            fg="white", 
            font=("Arial", 12, "bold")
        )
        self.refresh_button.pack(side=LEFT, padx=10)

        self.view_history_button = Button(
            self.button_frame,
            text="View History",
            command=self.view_history,
            width=15,
            bg="#ffa500",  # Yellow-Orange
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.view_history_button.pack(side=LEFT, padx=10)



        # Progress Label
        self.progress_label = Label(
            root, 
            text="Progress: Waiting for input...", 
            font=("Arial", 14), 
            bg="#001f3f",  # Deep Blue
            fg="#ffa500"   # Yellow-Orange
        )
        self.progress_label.pack(pady=10)

        # Progress Bar
        self.progress_bar = Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress_bar.pack(pady=10)

        # Result Label
        self.result_label = Label(
            root, 
            text="Result: None", 
            font=("Arial", 16), 
            bg="#001f3f",  # Deep Blue
            fg="#ffa500"   # Yellow-Orange
        )
        self.result_label.pack(pady=10)

        # Image Display
        self.image_label = Label(root, bg="#003366", fg="white", font=("Arial", 24, "bold"), text="FastAid Accident Detection Software", anchor="center")
        self.image_label.pack(pady=10, fill=BOTH, expand=True)

        # Navigation Buttons
        self.nav_frame = Frame(root, bg="#001f3f")  # Deep Blue
        self.nav_frame.pack(pady=10)

        self.prev_button = Button(
            self.nav_frame, 
            text="Previous", 
            command=self.show_previous_image, 
            width=10, 
            bg="#ffa500",  # Yellow-Orange
            fg="#001f3f",  # Deep Blue Text
            font=("Arial", 12, "bold")
        )
        self.prev_button.pack(side=LEFT, padx=20)

        self.next_button = Button(
            self.nav_frame, 
            text="Next", 
            command=self.show_next_image, 
            width=10, 
            bg="#ffa500",  # Yellow-Orange
            fg="#001f3f",  # Deep Blue Text
            font=("Arial", 12, "bold")
        )
        self.next_button.pack(side=LEFT, padx=20)

    def add_video(self):
        self.video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi")])
        if self.video_path:
            self.progress_label.config(text=f"Progress: Video loaded: {os.path.basename(self.video_path)}")
            messagebox.showinfo("Video Selected", f"Video loaded: {self.video_path}")

    def view_history(self):
        """Opens a new window to display the list of history subfolders."""
        # Create a new window
        history_window = Toplevel(self.root)
        history_window.title("Accident History")
        history_window.geometry("600x400")

        Label(history_window, text="Select a folder to view frames:", font=("Arial", 14)).pack(pady=10)

        # Listbox to display subfolders
        folder_listbox = Listbox(history_window, font=("Arial", 12), width=50, height=15)
        folder_listbox.pack(pady=10)

        # Populate the Listbox with subfolders
        history_folder = "history"
        if os.path.exists(history_folder):
            subfolders = [f for f in os.listdir(history_folder) if os.path.isdir(os.path.join(history_folder, f))]
            for subfolder in subfolders:
                folder_listbox.insert("end", subfolder)
        else:
            os.makedirs(history_folder, exist_ok=True)

        def load_selected_folder():
            """Loads the selected folder and displays frames."""
            selected = folder_listbox.curselection()
            if selected:
                subfolder = folder_listbox.get(selected[0])
                full_path = os.path.join(history_folder, subfolder)
                self.load_images(full_path)  # Make sure you have a method to load images
                history_window.destroy()  # Close the history window

        # Button to confirm selection
        Button(history_window, text="Open", command=load_selected_folder, font=("Arial", 12), bg="#ff4500", fg="white").pack(pady=10)

    def start_pipeline_thread(self):
        """Start the detection pipeline in a separate thread."""
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        
        # Start a new thread for running the pipeline
        threading.Thread(target=self.run_pipeline, daemon=True).start()
        
    def update_progress(self, message, progress=0):
        """Update the progress label and bar."""
        self.progress_label.config(text=f"Progress: {message}")
        self.progress_bar["value"] = progress
        self.root.update_idletasks()

    def run_pipeline(self):
        """Run the detection pipeline and update the GUI."""
        try:
            self.update_progress("Running optic flow detection...", progress=10)
            optic_flow_output = "optic_flow_output"
            yolo_output = "yolo_output"

            # Run optical flow detection
            success, message = detect_accidents(self.video_path, optic_flow_output)
            if not success:
                self.update_progress("Error during optic flow detection.", progress=10)
                messagebox.showerror("Error", message)
                return

            self.update_progress("Running YOLO detection...", progress=50)
            # Run YOLO detection and get the output folder
            yolo_output_folder = process_with_yolo(optic_flow_output, yolo_output, model_path="best.pt")

            self.update_progress("Detection completed. Loading results...", progress=100)
            self.load_images(yolo_output_folder)

            self.update_progress("Done. You can navigate the results.", progress=100)

            # After detection, send SMS if accidents are detected
            if os.listdir(yolo_output):  # Check if YOLO output folder is not empty
                phone_number = ""
                message = f"Accident Detected near RIST at {time.strftime('%Y-%m-%d %H:%M:%S')}"
                send_sms(phone_number, message)

        except Exception as e:
            self.update_progress(f"Error: {str(e)}")
            
    def load_images(self, folder):
        self.image_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if self.image_list:
            self.current_image_index = 0
            self.display_image(self.image_list[self.current_image_index])
        else:
            self.result_label.config(text="Result: No Accident Detected")

    def display_default_message(self):
        """Display the default message on the image label."""
        self.image_label.config(
            text="FastAid Accident Detection Software",  # Default message
            fg="white",  # Text color
            font=("Arial", 24, "bold"),  # Font style
            anchor="center",  # Text alignment
            image=None  # Ensure no image is displayed
        )
    def display_image(self, image_path):
        """Display an image on the image label."""
        try:
            img = Image.open(image_path)
            img = img.resize(
                (self.image_label.winfo_width(), self.image_label.winfo_height()),
                Image.Resampling.LANCZOS  # Replaces Image.ANTIALIAS
            )
            self.img = ImageTk.PhotoImage(img)  # Store a reference to avoid garbage collection
            self.image_label.config(image=self.img, text="")  # Display the image and remove text
        except Exception as e:
            print(f"Error displaying image: {e}")
            self.display_default_message()  # Fallback to default message

    def show_next_image(self):
        if self.image_list and self.current_image_index < len(self.image_list) - 1:
            self.current_image_index += 1
            self.display_image(self.image_list[self.current_image_index])

    def show_previous_image(self):
        if self.image_list and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_image(self.image_list[self.current_image_index])

    def refresh_app(self):
        """Reset the application state."""
        self.video_path = None
        self.image_list = []
        self.current_image_index = 0
        self.image_label.config(image="")
        self.update_progress("Progress: Waiting for input...", progress=0)
        self.result_label.config(text="Result: None")
        messagebox.showinfo("App Refreshed", "Application state has been reset.")
        self.display_default_message()  # Fallback to default message

# Launch the GUI
root = Tk()
app = AccidentDetectionApp(root)
root.mainloop()
