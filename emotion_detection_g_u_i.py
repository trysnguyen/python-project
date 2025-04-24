# emotion_gui.py
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from emotion_detector import EmotionDetector

class EmotionDetectionGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Emotion Detection System")
        self.root.geometry("1200x800")
        
        self.detector = EmotionDetector()
        self.setup_gui()
        self.is_camera_running = False
        self.cap = None

    def setup_gui(self):
        # Create main containers
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)
        
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        
        # Control buttons
        ttk.Button(self.control_frame, text="Select Image", 
                  command=self.select_image).pack(side=tk.LEFT, padx=5)
        
        self.camera_button = ttk.Button(self.control_frame, text="Start Camera", 
                                      command=self.toggle_camera)
        self.camera_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Show Statistics", 
                  command=self.show_statistics).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(self.control_frame, text="Reset Stats", 
                  command=self.reset_statistics).pack(side=tk.LEFT, padx=5)
        
        # Create display area
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack(expand=True, fill=tk.BOTH)
        
        # Statistics window
        self.stats_window = None
        self.stats_canvas = None

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff")])
        
        if file_path:
            image = cv2.imread(file_path)
            if image is not None:
                processed_image, faces = self.detector.detect_emotion(image)
                self.display_image(processed_image)
            else:
                messagebox.showerror("Error", "Could not load the selected image")

    def toggle_camera(self):
        if self.is_camera_running:
            self.stop_camera()
            self.camera_button.config(text="Start Camera")
        else:
            self.start_camera()
            self.camera_button.config(text="Stop Camera")

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if self.cap.isOpened():
            self.is_camera_running = True
            self.update_camera()
        else:
            messagebox.showerror("Error", "Could not access the camera")

    def stop_camera(self):
        self.is_camera_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def update_camera(self):
        if self.is_camera_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                processed_frame, faces = self.detector.detect_emotion(frame)
                self.display_image(processed_frame)
            self.root.after(10, self.update_camera)

    def display_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        # Resize image while maintaining aspect ratio
        display_size = (800, 600)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.video_label.config(image=photo)
        self.video_label.image = photo

    def show_statistics(self):
        if self.stats_window is None or not tk.Toplevel.winfo_exists(self.stats_window):
            self.stats_window = tk.Toplevel(self.root)
            self.stats_window.title("Emotion Statistics")
            self.stats_window.geometry("600x400")
            
            # Create new figure
            fig, ax = plt.subplots(figsize=(8, 6))
            stats = self.detector.get_emotion_stats()
            
            # Create bar chart
            bars = ax.bar(stats.keys(), stats.values())
            ax.set_title("Detected Emotions Distribution")
            ax.set_xlabel("Emotions")
            ax.set_ylabel("Count")
            plt.xticks(rotation=45)
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom')
            
            # Create canvas and show plot
            self.stats_canvas = FigureCanvasTkAgg(fig, master=self.stats_window)
            self.stats_canvas.draw()
            self.stats_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def reset_statistics(self):
        self.detector.reset_stats()
        messagebox.showinfo("Statistics Reset", "Emotion statistics have been reset")
        if self.stats_window is not None and tk.Toplevel.winfo_exists(self.stats_window):
            self.show_statistics()  # Refresh the statistics window if it's open

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.root.mainloop()

    def on_closing(self):
        self.stop_camera()
        self.root.destroy()

if __name__ == "__main__":
    app = EmotionDetectionGUI()
    app.run()