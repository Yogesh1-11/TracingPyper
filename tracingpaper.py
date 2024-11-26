##########edge detection app
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk

class EdgeModificationApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()

    def setup_ui(self):
        self.root.title("Edge Modification App")
        self.canvas = tk.Canvas(self.root)
        self.canvas.pack(side=tk.LEFT, padx=10, pady=10)
        self.setup_buttons()
        self.setup_controls()
        self.modified_edges = None
        self.drawing = False
        self.eraser_enabled = False
        self.draw_enabled = False
        self.eraser_size = 5
        self.sensitivity = 100
        self.dilation_scale_value = 1
        self.eraser_points = set()
        self.drawn_points = set()
        self.prev_x = None
        self.prev_y = None
        self.setup_bindings()

    def setup_buttons(self):
        buttons = [tk.Button(self.root, text="Load Image", command=self.load_image),
                   tk.Button(self.root, text="Detect Edges", command=self.detect_edges),
                   tk.Button(self.root, text="Eraser", command=self.enable_eraser),
                   tk.Button(self.root, text="Draw", command=self.enable_draw),
                   tk.Button(self.root, text="Save Overlay", command=self.save_overlay)]

        for button in buttons:
            button.pack(pady=10)

    def setup_controls(self):
        self.size_label = tk.Label(self.root, text="Eraser/Draw Size:")
        self.size_label.pack()
        self.size_scale = tk.Scale(self.root, from_=1, to=50, orient=tk.HORIZONTAL, length=200, command=self.update_eraser_size)
        self.size_scale.set(5)
        self.size_scale.pack()
        self.color_label = tk.Label(self.root, text="Detected Edge Color:")
        self.color_label.pack()
        self.color_options = ["Blue", "Black", "White", "Red", "Green", "Yellow"]
        self.selected_color = tk.StringVar(value="Yellow")
        self.color_dropdown = ttk.Combobox(self.root, textvariable=self.selected_color, values=self.color_options)
        self.color_dropdown.pack()
        self.sensitivity_label = tk.Label(self.root, text="Edge Sensitivity:")
        self.sensitivity_label.pack()
        self.sensitivity_scale = tk.Scale(self.root, from_=1, to=500, orient=tk.HORIZONTAL, length=200, command=self.update_sensitivity)
        self.sensitivity_scale.set(100)
        self.sensitivity_scale.pack()
        self.dilation_label = tk.Label(self.root, text="Dilation Scale:")
        self.dilation_label.pack()
        self.dilation_scale = tk.Scale(self.root, from_=1, to=10, orient=tk.HORIZONTAL, length=200, command=self.update_dilation)
        self.dilation_scale.set(1)
        self.dilation_scale.pack()

        # New controls for manual resizing
        self.length_label = tk.Label(self.root, text="Length:")
        self.length_label.pack()
        self.length_entry = tk.Entry(self.root)
        self.length_entry.pack()

        self.breadth_label = tk.Label(self.root, text="Breadth:")
        self.breadth_label.pack()
        self.breadth_entry = tk.Entry(self.root)
        self.breadth_entry.pack()

        # Button to apply manual resizing
        self.resize_button = tk.Button(self.root, text="Resize Manually", command=self.resize_image_manual)
        self.resize_button.pack()

        # Autofill button for current values
        self.autofill_button = tk.Button(self.root, text="Autofill Current Values", command=self.autofill_values)
        self.autofill_button.pack()

    def setup_bindings(self):
        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.add_point)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)
        self.root.bind("<BackSpace>", self.discard_points)
        self.root.bind("<Return>", self.add_points)
        self.root.bind("<Up>", self.update_dilation)
        self.root.bind("<Down>", self.update_dilation)
        
        # Add these lines to handle resizing
        self.canvas.bind("<B3-Motion>", self.resize_image)
        self.canvas.bind("<ButtonRelease-3>", self.stop_resizing)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            self.image = cv2.imread(file_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.show_image()
            # Autofill current values after loading an image
            self.autofill_values()

    def show_image(self):
        img = Image.fromarray(self.image)
        img = ImageTk.PhotoImage(img)
        self.canvas.config(width=img.width(), height=img.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img)
        self.canvas.img = img

    def detect_edges(self):
        if hasattr(self, 'image'):
            gray_image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray_image, 0.5 * self.sensitivity, 1.5 * self.sensitivity)
            self.modify_edges(edges)

    def modify_edges(self, edges):
        dilation_kernel_size = int(self.dilation_scale_value)
        dilated_edges = cv2.dilate(edges, np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8), iterations=1)

        for point in self.eraser_points:
            x, y = point
            roi = dilated_edges[y - self.eraser_size:y + self.eraser_size, x - self.eraser_size:x + self.eraser_size]
            roi[:] = 0

        color_options_dict = {"Blue": (255, 0, 0), "Black": (0, 0, 0), "White": (255, 255, 255),
                              "Red": (255, 0, 0), "Green": (0, 255, 0), "Yellow": (255, 255, 0)}

        selected_color_bgr = color_options_dict[self.selected_color.get()]
        self.modified_edges = cv2.cvtColor(dilated_edges, cv2.COLOR_GRAY2RGB)
        self.modified_edges[dilated_edges != 0] = selected_color_bgr

        self.image_with_edges = cv2.addWeighted(self.image, 1, np.zeros_like(self.image), 0, 0)
        self.image_with_edges = cv2.addWeighted(self.image_with_edges, 0.8, self.modified_edges, 0.2, 0)

        self.show_image_with_edges()

    def show_image_with_edges(self):
        img_with_edges = Image.fromarray(self.image_with_edges)
        img_with_edges = ImageTk.PhotoImage(img_with_edges)
        self.canvas.config(width=img_with_edges.width(), height=img_with_edges.height())
        self.canvas.create_image(0, 0, anchor=tk.NW, image=img_with_edges)
        self.canvas.img = img_with_edges

    def start_drawing(self, event):
        if self.eraser_enabled or self.draw_enabled:
            self.drawing = True
            if self.draw_enabled:
                self.draw(event)
        # Store initial coordinates for resizing
        self.prev_x = event.x
        self.prev_y = event.y

    def add_point(self, event):
        if self.drawing:
            if self.draw_enabled:
                self.draw(event)
            else:
                x, y = event.x, event.y
                if self.eraser_enabled:
                    self.eraser_points.add((x, y))
                    self.canvas.create_oval(x - self.eraser_size, y - self.eraser_size, x + self.eraser_size, y + self.eraser_size, outline="red", width=2)
                else:
                    self.drawn_points.add((x, y))
                    self.canvas.create_oval(x - self.eraser_size, y - self.eraser_size, x + self.eraser_size, y + self.eraser_size, outline="black", width=2)

    def stop_drawing(self, event):
        self.drawing = False

    def discard_points(self, event):
        if self.modified_edges is not None:
            edges = self.modified_edges.copy()
            self.clear_points(edges)
            self.modified_edges = edges.copy()
            self.image_with_edges = cv2.addWeighted(self.image, 1.0, np.zeros_like(self.image), 0.0, 0)
            self.image_with_edges = cv2.addWeighted(self.image_with_edges, 0.8, self.modified_edges, 0.2, 0)
            self.show_image_with_edges()

    def clear_points(self, edges):
        points_set = self.eraser_points if self.eraser_enabled else self.drawn_points
        for point in points_set:
            x, y = point
            cv2.circle(edges, (x, y), self.eraser_size, (0, 0, 0), -1)

        points_set.clear()

    def enable_eraser(self):
        self.eraser_enabled = not self.eraser_enabled
        self.draw_enabled = False
        self.update_cursor()

    def enable_draw(self):
        self.eraser_enabled = False
        self.draw_enabled = not self.draw_enabled
        self.update_cursor()

    def update_cursor(self):
        cursor_type = "arrow" if not (self.eraser_enabled or self.draw_enabled) else "pencil" if self.draw_enabled else "arrow"
        self.canvas.config(cursor=cursor_type)

    def update_eraser_size(self, value):
        self.eraser_size = int(value)

    def update_sensitivity(self, value):
        self.sensitivity = int(value)

    def update_dilation(self, event=None):
        if hasattr(self, 'image'):
            self.dilation_scale_value = int(self.dilation_scale.get())
            self.detect_edges()

    def resize_image(self, event):
        if hasattr(self, 'image') and self.prev_x is not None and self.prev_y is not None:
            delta_x = event.x - self.prev_x
            delta_y = event.y - self.prev_y
            self.prev_x = event.x
            self.prev_y = event.y

            new_width = self.canvas.winfo_width() + delta_x
            new_height = self.canvas.winfo_height() + delta_y

            if new_width > 0 and new_height > 0:
                self.image = cv2.resize(self.image, (new_width, new_height))
                self.show_image()

    def stop_resizing(self, event):
        self.prev_x = None
        self.prev_y = None

    def resize_image_manual(self):
        if hasattr(self, 'image'):
            try:
                new_length = int(self.length_entry.get())
                new_breadth = int(self.breadth_entry.get())
                if new_length > 0 and new_breadth > 0:
                    self.image = cv2.resize(self.image, (new_length, new_breadth))
                    self.show_image()
            except ValueError:
                print("Invalid input for length or breadth")

    def draw(self, event):
        if self.draw_enabled:
            x, y = event.x, event.y
            self.drawn_points.add((x, y))
            self.canvas.create_oval(x - self.eraser_size, y - self.eraser_size, x + self.eraser_size, y + self.eraser_size, outline="black", width=2)

    def add_points(self, event):
        if self.draw_enabled:
            if self.modified_edges is not None:
                edges = self.modified_edges.copy()
                self.modify_edges(edges)
                self.drawn_points.clear()
                self.modified_edges = edges.copy()
                self.image_with_edges = cv2.addWeighted(self.image, 1.0, np.zeros_like(self.image), 0.0, 0)
                self.image_with_edges = cv2.addWeighted(self.image_with_edges, 0.8, self.modified_edges, 0.2, 0)
                self.show_image_with_edges()

    def save_overlay(self):
        if hasattr(self, 'image_with_edges'):
            alpha_channel = np.zeros((self.modified_edges.shape[0], self.modified_edges.shape[1]), dtype=np.uint8)
            alpha_channel[self.modified_edges[:, :, 0] != 0] = 255
            color_options_dict = {"Blue": (255, 0, 0), "Black": (0, 0, 0), "White": (255, 255, 255),
                                  "Red": (255, 0, 0), "Green": (0, 255, 0), "Yellow": (255, 255, 0)}
            selected_color_bgr = color_options_dict[self.selected_color.get()]
            result_image = np.zeros_like(self.modified_edges, dtype=np.uint8)
            result_image[self.modified_edges[:, :, 0] != 0] = selected_color_bgr
            for point in self.eraser_points.union(self.drawn_points):
                x, y = point
                cv2.circle(result_image, (x, y), self.eraser_size, selected_color_bgr, -1)
            alpha_channel = np.expand_dims(alpha_channel, axis=2)
            result_image = np.concatenate((result_image, alpha_channel), axis=2)
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if save_path:
                cv2.imwrite(save_path, result_image)

    # New method to autofill current values
    def autofill_values(self):
        if hasattr(self, 'image'):
            current_length, current_breadth = self.image.shape[1], self.image.shape[0]
            self.length_entry.delete(0, tk.END)
            self.length_entry.insert(0, str(current_length))
            self.breadth_entry.delete(0, tk.END)
            self.breadth_entry.insert(0, str(current_breadth))

if __name__ == "__main__":
    root = tk.Tk()
    app = EdgeModificationApp(root)
    root.mainloop()
