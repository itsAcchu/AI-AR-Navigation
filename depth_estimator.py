import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2

class DepthEstimationModule:
    def __init__(self, model_type="MiDaS_small"):
        """
        Initialize depth estimation module
        
        Args:
            model_type (str): Type of MiDaS model to use ('MiDaS_small' for mobile)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load MiDaS model for depth estimation
        if model_type == "MiDaS_small":
            self.model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        self.model.to(self.device)
        self.model.eval()
        
        # Define transform for input preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Reduced size for better performance
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def estimate_depth(self, frame):
        """
        Estimate depth map from RGB image
        
        Args:
            frame (numpy.ndarray): RGB image [H, W, C]
            
        Returns:
            numpy.ndarray: Depth map where higher values = farther distance
        """
        try:
            # Resize frame for better performance
            h, w, _ = frame.shape
            if h > 480 or w > 640:
                frame = cv2.resize(frame, (640, 480))
            
            # Convert numpy image to tensor and normalize
            input_image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            input_tensor = self.transform(input_image).unsqueeze(0).to(self.device)
            
            # Run inference
            with torch.no_grad():
                depth_map = self.model(input_tensor)
            
            # Convert to numpy and resize to original resolution
            depth_map = depth_map.squeeze().cpu().numpy()
            depth_map = np.interp(depth_map, (depth_map.min(), depth_map.max()), (0, 10))  # Normalize to 0-10m range
            
            # Resize back to input frame dimensions
            depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))
            
            return depth_map
        except Exception as e:
            print(f"Error estimating depth: {e}")
            return np.zeros(frame.shape[:2], dtype=np.float32)
    
    def get_distance_to_objects(self, depth_map, object_boxes):
        """
        Calculate the distance to detected objects using the depth map
        
        Args:
            depth_map (numpy.ndarray): Estimated depth map
            object_boxes (list): List of bounding boxes [x1, y1, x2, y2]
            
        Returns:
            list: Estimated distances to each object in meters
        """
        distances = []
        for box in object_boxes:
            try:
                x1, y1, x2, y2 = [int(coord) for coord in box]
                
                # Make sure coordinates are within bounds
                x1 = max(0, min(x1, depth_map.shape[1] - 1))
                y1 = max(0, min(y1, depth_map.shape[0] - 1))
                x2 = max(0, min(x2, depth_map.shape[1] - 1))
                y2 = max(0, min(y2, depth_map.shape[0] - 1))
                
                # Skip if invalid box dimensions
                if x2 <= x1 or y2 <= y1:
                    distances.append(float('inf'))
                    continue
                
                lower_half_y1 = y1 + (y2 - y1) // 2
                region = depth_map[lower_half_y1:y2, x1:x2]
                
                if region.size > 10:
                    flat_region = region.flatten()
                    flat_region = np.sort(flat_region)
                    idx = max(0, min(len(flat_region) - 1, int(len(flat_region) * 0.1)))
                    distance = flat_region[idx]
                    distances.append(float(distance))
                else:
                    distances.append(float(np.median(depth_map)))
            except Exception as e:
                print(f"Error calculating distance: {e}")
                distances.append(float('inf'))
        
        return distances