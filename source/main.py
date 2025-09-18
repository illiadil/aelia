#####################################################################################
#####------ Aircraft Emergency Landing-site Identification and Assessment ------#####
#####################################################################################

# Import librairies
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw, ImageFont
import requests
from skimage.io import imread, imsave
from skimage import img_as_float, color
import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from models import transunet_model, unet_model
import lnir as lnir
import math
import srtm

# Define GUI main class
class AELIA_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AELIA - Aircraft Emergency Landing-site Identification and Assessment")

        # --- Fixed window size (Left image 512x512 + Right panel 300 width) ---
        window_width = 512 + 300
        window_height = 512
        self.root.geometry(f"{window_width}x{window_height}")
        self.root.resizable(False, False)  # disable resizing

        # --- Main Layout: Left = Image, Right = Controls ---
        self.left_frame = tk.Frame(root, bg="black", width=512, height=512)
        self.left_frame.pack(side="left", fill="y")
        self.left_frame.pack_propagate(False)

        self.right_frame = tk.Frame(root, width=300, height=512, bg="#f0f0f0")
        self.right_frame.pack(side="right", fill="y")
        self.right_frame.pack_propagate(False)

        # --- Image Panel (fixed 512x512) ---
        self.image_panel = tk.Label(self.left_frame, text="Load a map to begin", 
                                    bg="gray", fg="white", width=512, height=512)
        self.image_panel.pack(fill="both", expand=True)

        # --- Notebook for Tabs ---
        self.notebook = ttk.Notebook(self.right_frame)
        self.notebook.pack(fill="both", expand=True, padx=5, pady=5)

        # --- Create four Tabs ---
        self.create_map_tab()
        self.create_segmentation_tab()
        self.create_detection_tab()
        self.create_assessment_tab()

        # --- Status Bar ---
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w")
        status_bar.pack(side="bottom", fill="x")

        # --- Image variable and elevation data ---
        self.img = None
        self.elevation_data = srtm.get_data()

    # ---------- Tabs ----------
    
    # ---------- Create map tab ----------
    def create_map_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Map")

        self.load_image_button = tk.Button(tab, text="Load Image", command=self.load_image)
        self.load_image_button.pack(pady=10, fill="x")
        
        tk.Label(tab, text="Map Coordinates:").pack(anchor="w", pady=5)
        
        tk.Label(tab, text="  Top-Left (longitude):").pack(anchor="w", pady=5)
        self.longitude1_entry = tk.Entry(tab)
        self.longitude1_entry.pack(fill="x")
        tk.Label(tab, text="  Top-Left (latitude):").pack(anchor="w", pady=5)
        self.latitude1_entry = tk.Entry(tab)
        self.latitude1_entry.pack(fill="x")
        tk.Label(tab, text="  Bottom-Right (longitude):").pack(anchor="w", pady=5)
        self.longitude2_entry = tk.Entry(tab)
        self.longitude2_entry.pack(fill="x")
        tk.Label(tab, text="  Bottom-Right (latitude):").pack(anchor="w", pady=5)
        self.latitude2_entry = tk.Entry(tab)
        self.latitude2_entry.pack(fill="x")
        
        btn_set_coord = tk.Button(tab, text="Set coordinates", command=self.set_coordinates)
        btn_set_coord.pack(pady=10, fill="x")
        
        btn_exit = tk.Button(tab, text="Exit", command=self.root.quit)
        btn_exit.pack(pady=10, fill="x")

    # ---------- Create segmentation tab ----------
    def create_segmentation_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Segmentation")

        tk.Label(tab, text="Segmentation Method:").pack(anchor="w", pady=5)
        self.seg_method = ttk.Combobox(tab, values=["U-Net","TransUnet","Mask R-CNN","FCN","FPSNet"], state="readonly")
        self.seg_method.current(0)
        self.seg_method.pack(fill="x")

        tk.Label(tab, text="Threshold:").pack(anchor="w", pady=5)
        self.threshold_entry = tk.Entry(tab)
        self.threshold_entry.insert(0, "0.5")
        self.threshold_entry.pack(fill="x")

        btn_weights = tk.Button(tab, text="Load model weights", command=self.load_weight)
        btn_weights.pack(pady=10, fill="x")

        btn_segment = tk.Button(tab, text="Run segmentation", command=self.run_segmentation)
        btn_segment.pack(pady=10, fill="x")

    # ---------- Create terrain detection tab ----------
    def create_detection_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Detection")

        tk.Label(tab, text="Aircraft winspan (m):").pack(anchor="w", pady=5)
        self.winspang = tk.Entry(tab)
        self.winspang.insert(0, "5")
        self.winspang.pack(fill="x")
        tk.Label(tab, text="Safety factor:").pack(anchor="w", pady=5)
        self.safety = tk.Entry(tab)
        self.safety.insert(0, "1.2")
        self.safety.pack(fill="x")
        tk.Label(tab, text="Image resolution (Pixels Per Meter):").pack(anchor="w", pady=5)
        self.ppm = tk.Entry(tab)
        self.ppm.insert(0, "3")
        self.ppm.pack(fill="x")
        tk.Label(tab, text="Number of proposals:").pack(anchor="w", pady=5)
        self.nb_proposals = tk.Entry(tab)
        self.nb_proposals.insert(0, "3")
        self.nb_proposals.pack(fill="x")
        tk.Label(tab, text="Search depth (%):").pack(anchor="w", pady=0)
        self.search_dept = tk.Scale(tab, from_=0, to=100, length=250,orient=tk.HORIZONTAL)
        self.search_dept.pack(pady=0)
        
        btn_detect = tk.Button(tab, text="Detect Available Landing", command=self.detect_sites)
        btn_detect.pack(pady=10, fill="x")

    # ---------- Create terrain assessment tab ----------
    def create_assessment_tab(self):
        tab = ttk.Frame(self.notebook)
        self.notebook.add(tab, text="Assessment")
        
        tk.Label(tab, text="Runway Min Length (m):").pack(anchor="w", pady=5)
        self.min_length_entry = tk.Entry(tab)
        self.min_length_entry.insert(0, "200")
        self.min_length_entry.pack(fill="x")

        tk.Label(tab, text="Mesh step size (px):").pack(anchor="w", pady=5)
        self.mesh_step_entry = tk.Entry(tab)
        self.mesh_step_entry.insert(0, "5")
        self.mesh_step_entry.pack(fill="x")

        tk.Label(tab, text="Max allowed slope (Degrees °):").pack(anchor="w", pady=5)
        self.max_slope_entry = tk.Entry(tab)
        self.max_slope_entry.insert(0, "10")
        self.max_slope_entry.pack(fill="x")

        tk.Label(tab, text="Max allowed average slope (Degrees °):").pack(anchor="w", pady=5)
        self.max_average_slope_entry = tk.Entry(tab)
        self.max_average_slope_entry.insert(0, "5")
        self.max_average_slope_entry.pack(fill="x")

        btn_assess = tk.Button(tab, text="Assess Sites", command=self.assess_sites)
        btn_assess.pack(pady=10, fill="x")
        
        # Create a Treeview widget for showing the terrain assessment resutls
        table = ttk.Treeview(tab)

        # Define the treeview columns
        table['columns'] = ('Site #', 'Max slope (°)', 'Average slope (°)','Length (m)')
        
        self.result_table = table
        
        # Format the columns
        table.column('#0', width=0, stretch=tk.NO)
        table.column('Site #', anchor=tk.W, width=30)
        table.column('Max slope (°)', anchor=tk.W, width=50)
        table.column('Average slope (°)', anchor=tk.W, width=50)
        table.column('Length (m)', anchor=tk.W, width=50)

        # Create the headings
        table.heading('#0', text='', anchor=tk.W)
        table.heading('Site #', text='Site #', anchor=tk.W)
        table.heading('Max slope (°)', text='Max slope (°)', anchor=tk.W)
        table.heading('Average slope (°)', text='Average slope (°)', anchor=tk.W)
        table.heading('Length (m)', text='Length (m)', anchor=tk.W)
        
        # Pack the table
        table.pack(expand=True, fill=tk.BOTH)
            
        
    # ---------- Functions ----------
        
    # ---------- Method to fill the assessment table, once done ----------
    def fill_assessment_table(self):
        tree = self.result_table
        # Clear existing data
        for item in tree.get_children():
            tree.delete(item)
        # Insert new data
        for row in self.sites_data:
            tree.insert('', 'end', values=row)
            
    # ---------- Method to set the reference coordinates : top-left and bottom-right ----------
    def set_coordinates(self):
        self.longitude1 = float(self.longitude1_entry.get())
        self.latitude1 = float(self.latitude1_entry.get())
        self.longitude2 = float(self.longitude2_entry.get())
        self.latitude2 = float(self.latitude2_entry.get())
        print("Coordinates 1 : " + str(self.latitude1) + "/" + str(self.longitude1) )
        print("Coordinates 2 : " + str(self.latitude2) + "/" + str(self.longitude2) )
        messagebox.showinfo("Coordinates","Reference coordinates loaded")
        
        
    # ---------- Method to load the image from a selection popup and show in the interface ----------
    def load_image(self):
        filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.tif")])
        if filepath:
            self.img = Image.open(filepath)
            self.img = self.img.resize((512, 512))  # fixed size
            img_tk = ImageTk.PhotoImage(self.img)
            self.image_panel.config(image=img_tk, text="")
            self.image_panel.image = img_tk
            self.raw_images = np.zeros((4,256,256,3), dtype=np.uint8)
            # divide the image into four sub-image of 256x256
            self.raw_images[0] = imread(filepath)[0:256,0:256,:3]
            self.raw_images[1] = imread(filepath)[0:256,256:512,:3]
            self.raw_images[2] = imread(filepath)[256:512,0:256,:3]
            self.raw_images[3] = imread(filepath)[256:512,256:512,:3]
            self.status_var.set(f"Loaded: {filepath}")

    # ---------- Method to load the segmentation weights ----------
    def load_weight(self):
        if self.img:
            method = self.seg_method.get()
            #U-Net, TransUnet, Segformer, Mask R-CNN, FCN, FPSNet
            if method == "U-Net":
                messagebox.showinfo("Segmentation", f"Loading {method} weights")
                unet_model.load_weights(r"weights\Unet.weights.h5")
                messagebox.showinfo("Segmentation","Loading complete")
            elif method in ["Segformer","TransUnet","Mask R-CNN","FCN","FPSNet"]:
                messagebox.showinfo(method,"Weight unavailable for this model")
        else:
            messagebox.showwarning("Error", "Please load a map first.")

    # ---------- Method to apply the segmentation model on the four sub-images ----------
    def run_segmentation(self):
        if self.img:
            threshold = float( self.threshold_entry.get() )
            self.predictions = unet_model.predict(self.raw_images)
            self.segmented = np.zeros((512, 512, 3))
            self.segmented[0:256,0:256,:3] = self.raw_images[0].copy()
            self.segmented[0:256,256:512,:3] = self.raw_images[1].copy()
            self.segmented[256:512,0:256,:3] = self.raw_images[2].copy()
            self.segmented[256:512,256:512,:3] = self.raw_images[3].copy()
            self.prediction = np.zeros((512, 512))
            self.prediction[0:256,0:256] = ( self.predictions[0] > threshold ).reshape((256,256))
            self.prediction[0:256,256:512] = ( self.predictions[1] > threshold ).reshape((256,256))
            self.prediction[256:512,0:256] = ( self.predictions[2] > threshold ).reshape((256,256))
            self.prediction[256:512,256:512] = ( self.predictions[3] > threshold ).reshape((256,256))
            print(self.segmented.shape)
            print(self.prediction.shape)

            self.segmented[:,:,1] = self.segmented[:,:,1] + 50 * self.prediction
            imsave("tmp/temp.png", self.segmented.astype(np.uint8))
            self.img = Image.open("tmp/temp.png")
            img_tk = ImageTk.PhotoImage(self.img)
            self.image_panel.config(image=img_tk, text="")
            self.image_panel.image = img_tk
            messagebox.showinfo("Segmentation","Segmentation complete")
        else:
            messagebox.showwarning("Error", "Please load a map first.")

    # ---------- Method check if a point is inside a given rectangle ----------
    def is_point_in_rectangle(self,px, py, x1, y1, x2, y2, x3, y3, x4, y4):
        s1 = (py - y1)*(x2 - x1) - (px - x1)*(y2 - y1)
        s2 = (py - y2)*(x3 - x2) - (px - x2)*(y3 - y2)
        s3 = (py - y3)*(x4 - x3) - (px - x3)*(y4 - y3)
        s4 = (py - y4)*(x1 - x4) - (px - x4)*(y1 - y4)

        return s1 <= 0 and s2 <= 0 and s3 <= 0 and s4 <= 0

    # ---------- Method to apply the LNIR (Longest Inscribed Rectangle) method on the segmentation masks ----------
    def detect_sites(self):
        if self.img:
            print(str(self.search_dept.get() ))
            self.ppm_value = float(self.ppm.get())
            min_width = int(float(self.winspang.get())*self.ppm_value*float(self.safety.get()))
            self.prediction_temp = self.prediction.copy()
            terrain = Image.open("tmp/temp.png")
            terrain.save("tmp/temp2.png")
            search_angle = 1+ 5*int(1 - self.search_dept.get() / 100)
            print("search angle : "+ str(search_angle) + "°")
            self.n_proposals = int(self.nb_proposals.get())
            self.detected_site = [None] * self.n_proposals
            for i in range(self.n_proposals):
                self.root.update()
                print("Searching site #"+str(i+1) )
                self.detected_site[i] = lnir.check_all_angles(self.prediction_temp,search_angle,min_width)
                im = Image.fromarray(self.prediction_temp.copy()*255)
                im = im.convert("L")
                im.save("tmp/"+str(i)+"_input.png")
                x1 = self.detected_site[i][0][0]
                y1 = self.detected_site[i][0][1]
                x2 = self.detected_site[i][1][0]
                y2 = self.detected_site[i][1][1]
                x3 = self.detected_site[i][2][0]
                y3 = self.detected_site[i][2][1]
                x4 = self.detected_site[i][3][0]
                y4 = self.detected_site[i][3][1]
                for k in range(512):
                    for j in range(512):
                        if self.is_point_in_rectangle(j,k, x1, y1, x2, y2, x3, y3, x4, y4):
                            self.prediction_temp[k,j] = 0
                terrain = Image.open("tmp/temp2.png")
                terrain_selection = ImageDraw.Draw(terrain)
                terrain_selection.polygon(self.detected_site[i], fill ="#ADD8E6", outline ="#3B3B3B")
                terrain.save("tmp/temp2.png")
            self.img = Image.open("tmp/temp2.png")
            img_tk = ImageTk.PhotoImage(self.img)
            self.image_panel.config(image=img_tk, text="")
            self.image_panel.image = img_tk
        else:
            messagebox.showwarning("Error", "Please load a map first.")

    # ---------- Method to convert pixel coordinates to latitude longitude coordinates----------
    def pixels_to_lat_long(self,x,y):
        diff_long = self.longitude2 - self.longitude1
        diff_lat = self.latitude1 - self.latitude2
        long = x/512.0*diff_long + self.longitude1
        lat = self.latitude1 - y/512.0*diff_lat
        return (lat, long);

    # ---------- Method to assess each detected sites for 3 parameters : max slope, average slope and length ----------
    def assess_sites(self):
        if self.img:
            self.elevations = [None] * self.n_proposals
            self.sites_data = [None] * self.n_proposals
            min_length = int(self.min_length_entry.get())
            mesh_step = int(self.mesh_step_entry.get() )
            mesh_step_meters = float(self.ppm_value* mesh_step)
            allowed_max_slope = float(self.max_slope_entry.get())
            allowed_average_slope = float(self.max_average_slope_entry.get())
            print("allowed_max_slope : "+str(allowed_max_slope)+" / allowed_average_slope : "+str(allowed_average_slope)) 
            terrain = Image.open("tmp/temp.png")
            terrain.save("tmp/temp2.png")
            for i in range(self.n_proposals):
                print("Assessing site #"+str(i+1))
                x1 = self.detected_site[i][0][0]
                y1 = self.detected_site[i][0][1]
                x2 = self.detected_site[i][1][0]
                y2 = self.detected_site[i][1][1]
                x3 = self.detected_site[i][2][0]
                y3 = self.detected_site[i][2][1]
                x4 = self.detected_site[i][3][0]
                y4 = self.detected_site[i][3][1]
                terrain = Image.open("tmp/temp2.png")
                terrain_assesed = ImageDraw.Draw(terrain)
                L = math.sqrt( (x3-x2)**2 + (y3-y2)**2 )
                l = math.sqrt( (x1-x2)**2 + (y1-y2)**2 )
                N = int(L/mesh_step)
                M = int(l/mesh_step)
                self.elevations[i] = np.zeros((N, M))
                for n in range(1,N+1):
                    for m in range(1,M+1):
                        x = x2 + int((x3-x2)*mesh_step*n/L) + int((x1-x2)*mesh_step*m/l)
                        y = y2 + int((y3-y2)*mesh_step*n/L) + int((y1-y2)*mesh_step*m/l)
                        lat, long = self.pixels_to_lat_long(x,y)
                        elevation = self.elevation_data.get_elevation(lat, long)
                        self.elevations[i][n-1][m-1] = elevation
                        terrain_assesed.circle(xy=(x,y), radius=2, fill="red")
                grad_y, grad_x = np.gradient(self.elevations[i])
                # Calculate the magnitude of the gradient at each point
                slope_magnitude = np.sqrt(grad_x**2 + grad_y**2) / mesh_step_meters
                max_slope = round(math.degrees(math.atan(np.amax(slope_magnitude))), 2)
                # Calculate the average gradient magnitude
                average_slope = round(math.degrees(math.atan(np.mean(slope_magnitude))), 2)
                self.sites_data[i] = ( i+1, max_slope, average_slope, round(L) )
                print("Assessment of site #"+str(i+1)+" : Max slope = "+str(max_slope)+" / average slope = "+str(average_slope)+" / L = "+str(L))
                if L > min_length and max_slope < allowed_max_slope and average_slope < allowed_average_slope:
                    terrain_assesed.polygon(self.detected_site[i], fill ="#00ff00", outline ="#3B3B3B")
                else:
                    terrain_assesed.polygon(self.detected_site[i], fill ="#fc7070", outline ="#3B3B3B")
                try:
                    font = ImageFont.truetype("arial.ttf", size=20)
                except IOError:
                    font = ImageFont.load_default() # Fallback to default font if not found

                # Draw text
                terrain_assesed.text((int((x1+x3)/2)-8, int((y1+y3)/2)-8), "#"+str(i+1), fill=(0, 0, 0), font=font) # Black text
                terrain.save("tmp/temp2.png")
                self.img = Image.open("tmp/temp2.png")
                img_tk = ImageTk.PhotoImage(self.img)
                self.image_panel.config(image=img_tk, text="")
                self.image_panel.image = img_tk
            self.status_var.set("Assessment complete")
            self.fill_assessment_table()
        else:
            messagebox.showwarning("Error", "Please load a map first.")


if __name__ == "__main__":
    root = tk.Tk()
    app = AELIA_GUI(root)
    root.mainloop()