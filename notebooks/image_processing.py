import numpy as np
import cv2
from matplotlib.path import Path

def process(X, Y, noise_sensitivity = 0.8):
    traj_x, traj_y = None, None
    noise_threshold = int(round((1 - noise_sensitivity)*255))
    try:
        # Image to map data to
        img_size = (max(X), max(Y))
        img = np.zeros(img_size, dtype=np.uint8)
    
        # Normalize data and invert Y-axis (OpenCV uses top left corner as origin)
        x_norm = ((X - X.min()) / (X.max() - X.min()) * (img_size[1] - 1)).astype(int)
        y_norm = (img_size[0] - 1 - ((Y - Y.min()) / (Y.max() - Y.min()) * (img_size[0] - 1))).astype(int)        
        img[y_norm, x_norm] = 255
        
        # Image processing
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        closed_img = cv2.morphologyEx(blurred_img, cv2.MORPH_CLOSE, kernel)
        _, binary_img = cv2.threshold(closed_img, noise_threshold, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
        # Remove contours with length 0 and flatten
        contours = [c[:, 0, :] for c in contours if cv2.arcLength(c, closed=False) > 0]
        
        # Split into two arrays and filter out small contours (likely noise)
        contours = [(pts[:, 0], pts[:, 1]) for pts in contours if len(pts) > 10]
            
        # Recover trajectory points
        inside_mask = np.zeros(len(X), dtype=bool)
        for cx, cy in contours:
            # Scale countours back to original data scope
            scaled_x = cx * (X.max() - X.min()) / (img_size[1] - 1) + X.min()
            scaled_y = ((img_size[0] - 1) - cy) * (Y.max() - Y.min()) / (img_size[0] - 1) + Y.min()
            
            # Create polygon path from scaled contours
            trajectory_polygon = Path(np.column_stack((scaled_x, scaled_y)))
            
            # Append points inside the polygon
            contained_points = trajectory_polygon.contains_points(np.column_stack((X, Y)))
            inside_mask |= contained_points
        
        traj_x, traj_y = X[inside_mask], Y[inside_mask]
    except Exception as e:
        print(e)
        pass
    
    return traj_x, traj_y