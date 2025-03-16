import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
import cv2

class ARNavigationModel:
    def __init__(self, num_classes=91, pretrained=True):
        """
        Initialize the AR Navigation model with object detection capabilities.
        
        Args:
            num_classes (int): Number of object classes to detect
            pretrained (bool): Whether to use pretrained weights
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize the object detection model (Faster R-CNN with ResNet50 backbone)
        # Fix deprecated 'pretrained' parameter with 'weights'
        if pretrained:
            backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            backbone = torchvision.models.resnet50(weights=None)
            
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        backbone.out_channels = 2048
        
        # Define anchor generator for region proposals
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),),
            aspect_ratios=((0.5, 1.0, 2.0),)
        )
        
        # Initialize Faster R-CNN model with improved parameters for performance
        self.detection_model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=640,  # Reduced from 800 for better performance
            max_size=800,  # Reduced from 1333 for better performance
            box_score_thresh=0.70,  # Slightly increased threshold to reduce false positives
            rpn_pre_nms_top_n_test=500,  # Reduced for better performance
            rpn_post_nms_top_n_test=100  # Reduced for better performance
        )
        
        self.detection_model.to(self.device)
        self.detection_model.eval()
        
        # Labels mapping for COCO dataset
        self.coco_labels = {
            1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
            6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
            11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
            16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep',
            21: 'cow', 22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe',
            27: 'backpack', 28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase',
            34: 'frisbee', 35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
            39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
            43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
            49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
            54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
            59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
            64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
            73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
            78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
            84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
            89: 'hair drier', 90: 'toothbrush'
        }
        
        # Define priority objects for visually impaired navigation
        self.priority_objects = {
            'person': 0.9,
            'bicycle': 0.8,
            'car': 0.9,
            'motorcycle': 0.8,
            'bus': 0.9,
            'truck': 0.9,
            'traffic light': 0.95,
            'stop sign': 0.95,
            'bench': 0.7,
            'chair': 0.7,
            'potted plant': 0.6,
            'stairs': 0.95,
            'door': 0.9,
            'curb': 0.95,
            'pathway': 0.9
        }
        
    def detect_objects(self, image):
        """
        Detect objects in the input image.
        
        Args:
            image (torch.Tensor): Input image tensor [C, H, W]
            
        Returns:
            dict: Dictionary containing detection results with boxes, labels, and scores
        """
        try:
            # Add batch dimension
            image = image.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                predictions = self.detection_model(image)
                
            return predictions[0]
        except Exception as e:
            print(f"Error in object detection: {e}")
            return {'boxes': torch.tensor([]), 'labels': torch.tensor([]), 'scores': torch.tensor([])}
    
    def process_frame(self, frame):
        """
        Process a single video frame for object detection.
        
        Args:
            frame (numpy.ndarray): RGB image in numpy format [H, W, C]
            
        Returns:
            dict: Processed results with detected objects, priorities, and distances
        """
        try:
            # Handle potential memory issues by resizing large frames
            h, w, _ = frame.shape
            if h > 720 or w > 1280:
                scale = min(720/h, 1280/w)
                new_h, new_w = int(h * scale), int(w * scale)
                frame = cv2.resize(frame, (new_w, new_h))
            
            # Convert numpy image to tensor
            image = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            
            # Get detection results
            detections = self.detect_objects(image)
            
            # Extract results
            boxes = detections.get('boxes', torch.tensor([])).cpu().numpy()
            labels = detections.get('labels', torch.tensor([])).cpu().numpy()
            scores = detections.get('scores', torch.tensor([])).cpu().numpy()
            
            # Format results
            results = []
            for box, label, score in zip(boxes, labels, scores):
                if label in self.coco_labels:
                    label_name = self.coco_labels[label]
                    if score >= 0.65:
                        priority = self.priority_objects.get(label_name, 0.5)
                        results.append({
                            'label': label_name,
                            'box': box.tolist(),
                            'score': float(score),
                            'priority': priority,
                            'distance': 0.0  # Placeholder, will be updated with depth info
                        })
            
            # Sort results by priority and distance
            results.sort(key=lambda x: (x['priority'], -x['distance']), reverse=True)
            
            return {
                'detections': results,
                'frame_shape': frame.shape
            }
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {'detections': [], 'frame_shape': frame.shape}
    
    def add_custom_classes(self, new_model_path):
        """
        Load a fine-tuned model with additional custom classes relevant for navigation.
        
        Args:
            new_model_path (str): Path to the custom model weights
        """
        try:
            self.detection_model.load_state_dict(torch.load(new_model_path, map_location=self.device))
            max_id = max(self.coco_labels.keys())
            self.coco_labels[max_id + 1] = 'stairs'
            self.coco_labels[max_id + 2] = 'door'
            self.coco_labels[max_id + 3] = 'curb'
            self.coco_labels[max_id + 4] = 'pathway'
        except Exception as e:
            print(f"Error loading custom classes: {e}")