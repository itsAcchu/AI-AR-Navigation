import torch
import numpy as np
import math
from collections import deque

class NavigationModule:
    def __init__(self, object_detector, depth_estimator):
        """
        Initialize the navigation module for visually impaired assistance
        
        Args:
            object_detector: Object detection model instance
            depth_estimator: Depth estimation model instance
        """
        self.object_detector = object_detector
        self.depth_estimator = depth_estimator
        
        # Navigation state
        self.current_position = None  # GPS coordinates when available
        self.current_orientation = 0  # In degrees, 0 is North
        self.planned_path = []  # List of waypoints [lat, lon]
        
        # System parameters
        self.safe_distance = 1.5  # Minimum safe distance in meters
        self.warning_distance = 3.0  # Distance to start warnings
        self.critical_distance = 1.0  # Critical proximity for urgent alerts
        
        # Risk assessment parameters
        self.risk_threshold = 0.7  # Threshold for high-risk situations
        self.path_history = deque(maxlen=50)  # Store recent path points
        
        # Obstacle memory for tracking consistent obstacles
        self.obstacle_memory = {}  # {object_id: {position, last_seen, confidence}}
        self.memory_timeout = 10  # Seconds to keep an obstacle in memory
        
        # Navigation feedback states
        self.feedback_states = {
            'clear': {'priority': 0, 'message': 'Path clear'},
            'caution': {'priority': 1, 'message': 'Proceed with caution'},
            'warning': {'priority': 2, 'message': 'Obstacle ahead'},
            'danger': {'priority': 3, 'message': 'Stop immediately'},
            'crosswalk': {'priority': 2, 'message': 'Approaching crosswalk'},
            'turn_left': {'priority': 1, 'message': 'Turn left ahead'},
            'turn_right': {'priority': 1, 'message': 'Turn right ahead'}
        }
        
        # Current active navigation state
        self.current_state = 'clear'
        
    def set_destination(self, destination_coords):
        """
        Set navigation destination and plan path
        
        Args:
            destination_coords (tuple): (latitude, longitude) of destination
        
        Returns:
            bool: True if path planning was successful
        """
        if not self.current_position:
            return False
            
        # In a real implementation, this would call a path planning service
        # For now, creating a placeholder path
        self.planned_path = self._generate_sample_path(
            self.current_position, 
            destination_coords
        )
        
        return len(self.planned_path) > 0
    
    def _generate_sample_path(self, start, end):
        """
        Generate a sample path between two coordinates
        This is a placeholder for actual path planning
        
        Args:
            start (tuple): Starting coordinates (lat, lon)
            end (tuple): Ending coordinates (lat, lon)
            
        Returns:
            list: List of waypoints [(lat, lon), ...]
        """
        # Create a straight line path with 10 points
        path = []
        for i in range(11):
            factor = i / 10
            lat = start[0] + (end[0] - start[0]) * factor
            lon = start[1] + (end[1] - start[1]) * factor
            path.append((lat, lon))
        
        return path
    
    def update_position(self, lat, lon, heading=None):
        """
        Update the current position and orientation
        
        Args:
            lat (float): Current latitude
            lon (float): Current longitude
            heading (float): Current heading in degrees (optional)
        """
        new_position = (lat, lon)
        
        # Update position history for path tracking
        if self.current_position:
            self.path_history.append(self.current_position)
        
        self.current_position = new_position
        
        if heading is not None:
            self.current_orientation = heading
    
    def process_frame(self, frame):
        """
        Process a camera frame to extract navigation information
        
        Args:
            frame (numpy.ndarray): RGB image [H, W, C]
            
        Returns:
            dict: Navigation guidance information
        """
        try:
            # Get object detections
            detection_results = self.object_detector.process_frame(frame)
            
            # Calculate depth map
            depth_map = self.depth_estimator.estimate_depth(frame)
            
            # Get distances to detected objects
            object_boxes = [obj['box'] for obj in detection_results['detections']]
            distances = self.depth_estimator.get_distance_to_objects(depth_map, object_boxes)
            
            # Combine detection results with distance information
            for i, detection in enumerate(detection_results['detections']):
                if i < len(distances) and distances[i] is not None:
                    detection['measured_distance'] = distances[i]
            
            # Analyze the scene and determine guidance
            guidance = self._analyze_scene(detection_results, depth_map)
            
            return guidance
        except Exception as e:
            print(f"Error processing frame: {e}")
            return {
                'state': 'clear',
                'priority': 0,
                'message': 'System error',
                'risk_level': 0.0,
                'obstacles': []
            }
    
    def _analyze_scene(self, detection_results, depth_map):
        """
        Analyze scene to provide navigation guidance
        
        Args:
            detection_results (dict): Results from object detection
            depth_map (numpy.ndarray): Depth information
            
        Returns:
            dict: Navigation guidance information
        """
        # Extract detected objects
        detections = detection_results['detections']
        
        # Calculate risk level and identify obstacles
        risk_level, obstacles = self._calculate_risk_level(detections, depth_map)
        
        # Determine appropriate guidance
        guidance = self._determine_guidance(risk_level, obstacles)
        
        # Add current objects to obstacle memory
        self._update_obstacle_memory(detections)
        
        return guidance
    
    def _calculate_risk_level(self, detections, depth_map):
        """
        Calculate the current risk level based on detections and depth
        
        Args:
            detections (list): Detected objects with distances
            depth_map (numpy.ndarray): Complete depth map
            
        Returns:
            tuple: (risk_level, obstacle_list)
        """
        if not detections:
            return 0.0, []
        
        # Extract central area of depth map to focus on path ahead
        h, w = depth_map.shape
        center_region = depth_map[h//4:3*h//4, w//3:2*w//3]
        min_center_distance = center_region.min() if center_region.size > 0 else float('inf')
        
        # Calculate risk based on proximity of closest object
        risk_level = 0.0
        critical_obstacles = []
        
        # Check detected objects with high priority
        for detection in detections:
            priority = detection.get('priority', 0.5)
            distance = detection.get('measured_distance', detection.get('distance', float('inf')))
            
            # Calculate individual object risk
            if distance < self.critical_distance:
                obj_risk = 1.0 * priority
            elif distance < self.safe_distance:
                obj_risk = (self.safe_distance - distance) / (self.safe_distance - self.critical_distance) * priority
            else:
                obj_risk = 0.0
                
            # Only consider as obstacle if risk is significant
            if obj_risk > 0.3:
                critical_obstacles.append({
                    'label': detection['label'],
                    'distance': distance,
                    'risk': obj_risk,
                    'box': detection['box']
                })
                
            # Update overall risk level
            risk_level = max(risk_level, obj_risk)
        
        # Consider minimum distance in central area (path ahead)
        if min_center_distance < self.safe_distance:
            path_risk = (self.safe_distance - min_center_distance) / self.safe_distance
            risk_level = max(risk_level, path_risk * 0.8)  # 0.8 factor as it's not a specifically detected object
            
        return risk_level, critical_obstacles
    
    def _determine_guidance(self, risk_level, obstacles):
        """
        Determine appropriate guidance based on risk level and obstacles
        
        Args:
            risk_level (float): Calculated risk level (0.0 to 1.0)
            obstacles (list): List of detected obstacles
            
        Returns:
            dict: Guidance information
        """
        # Determine navigation state based on risk level
        if risk_level >= 0.9:
            nav_state = 'danger'
        elif risk_level >= 0.7:
            nav_state = 'warning'
        elif risk_level >= 0.3:
            nav_state = 'caution'
        else:
            nav_state = 'clear'
        
        # Override for specific obstacle types
        for obstacle in obstacles:
            if obstacle['label'] == 'traffic light' and obstacle['distance'] < 8.0:
                nav_state = 'crosswalk'
                break
        
        # Get feedback information for this state
        feedback = self.feedback_states[nav_state]
        
        # Build detailed guidance message
        guidance_message = feedback['message']
        
        if nav_state != 'clear' and obstacles:
            # Add details about closest obstacle
            closest = min(obstacles, key=lambda x: x['distance'])
            guidance_message += f". {closest['label']} {closest['distance']:.1f} meters ahead"
            
            # Add directional guidance if needed
            if closest['box'][0] < 0.4 * 640:  # Assuming 640px width, obstacle on left
                guidance_message += " on your left"
            elif closest['box'][0] > 0.6 * 640:  # Obstacle on right
                guidance_message += " on your right"
        
        # Construct final guidance information
        guidance = {
            'state': nav_state,
            'priority': feedback['priority'],
            'message': guidance_message,
            'risk_level': risk_level,
            'obstacles': obstacles
        }
        
        # If we have a planned path, add navigation directions
        if self.planned_path and len(self.planned_path) > 1:
            next_direction = self._get_next_direction()
            if next_direction:
                guidance['navigation'] = next_direction
        
        self.current_state = nav_state
        return guidance
    
    def _get_next_direction(self):
        """
        Get the next turn direction based on planned path
        
        Returns:
            dict: Direction information or None
        """
        if not self.current_position or len(self.planned_path) < 2:
            return None
            
        # Find the next waypoint
        current_idx = 0
        min_distance = float('inf')
        
        for i, waypoint in enumerate(self.planned_path):
            dist = self._haversine_distance(self.current_position, waypoint)
            if dist < min_distance:
                min_distance = dist
                current_idx = i
        
        # If we're at the last waypoint or near it
        if current_idx >= len(self.planned_path) - 1 or min_distance < 5:  # Within 5 meters
            return {
                'type': 'destination',
                'distance': min_distance,
                'message': f"Destination in {min_distance:.0f} meters"
            }
        
        # Get the next waypoint
        next_waypoint = self.planned_path[current_idx + 1]
        
        # Calculate bearing to next waypoint
        bearing = self._calculate_bearing(self.current_position, next_waypoint)
        
        # Convert to relative direction based on current orientation
        relative_angle = (bearing - self.current_orientation) % 360
        
        # Determine direction type
        if 315 <= relative_angle or relative_angle < 45:
            direction_type = 'straight'
        elif 45 <= relative_angle < 135:
            direction_type = 'right'
        elif 225 <= relative_angle < 315:
            direction_type = 'left'
        else:
            direction_type = 'turnaround'
        
        return {
            'type': direction_type,
            'distance': min_distance,
            'message': f"Go {direction_type}, {min_distance:.0f} meters"
        }
    
    def _update_obstacle_memory(self, detections):
        """
        Update obstacle memory with current detections
        
        Args:
            detections (list): Current frame detections
        """
        # Implement obstacle tracking logic here
        # This would associate detections across frames and maintain object persistence
        pass
    
    def _haversine_distance(self, point1, point2):
        """
        Calculate distance between two GPS points using Haversine formula
        
        Args:
            point1 (tuple): (latitude, longitude) pair
            point2 (tuple): (latitude, longitude) pair
            
        Returns:
            float: Distance in meters
        """
        # Earth's radius in meters
        R = 6371000
        
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def _calculate_bearing(self, point1, point2):
        """
        Calculate bearing from point1 to point2
        
        Args:
            point1 (tuple): (latitude, longitude) pair
            point2 (tuple): (latitude, longitude) pair
            
        Returns:
            float: Bearing in degrees
        """
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.degrees(math.atan2(y, x))
        return (bearing + 360) % 360
    
# File: navigation_system.py (add to the end of the file)

class ARNavigation:
    def __init__(self):
        """
        Initialize the AR Navigation system that integrates object detection,
        depth estimation, and navigation modules.
        """
        # Import required modules
        from object_detector import ARNavigationModel
        from depth_estimator import DepthEstimationModule
        
        # Initialize the object detector
        self.object_detector = ARNavigationModel(pretrained=True)
        
        # Initialize the depth estimator
        self.depth_estimator = DepthEstimationModule(model_type="MiDaS_small")
        
        # Initialize the navigation module
        self.navigation_module = NavigationModule(
            object_detector=self.object_detector,
            depth_estimator=self.depth_estimator
        )
        
        self.is_running = False
        print("AR Navigation system initialized")
    
    def start(self):
        """
        Start the AR navigation system.
        """
        self.is_running = True
        print("AR Navigation system started")
        
        # Initialize with a default position (can be updated with actual GPS later)
        self.navigation_module.update_position(37.7749, -122.4194, heading=0)
        
    def stop(self):
        """
        Stop the AR navigation system and release resources.
        """
        self.is_running = False
        print("AR Navigation system stopped")
    
    def process_camera_frame(self, frame):
        """
        Process a camera frame and provide navigation guidance.
        
        Args:
            frame (numpy.ndarray): RGB image from camera [H, W, C]
            
        Returns:
            dict: Navigation guidance information
        """
        if not self.is_running:
            return {'message': 'System not running'}
        
        # Process the frame using the navigation module
        guidance = self.navigation_module.process_frame(frame)
        
        return guidance
    
    def set_destination(self, lat, lon):
        """
        Set a navigation destination.
        
        Args:
            lat (float): Destination latitude
            lon (float): Destination longitude
            
        Returns:
            bool: True if destination was set successfully
        """
        return self.navigation_module.set_destination((lat, lon))
    
    def update_position(self, lat, lon, heading=None):
        """
        Update the current position and orientation.
        
        Args:
            lat (float): Current latitude
            lon (float): Current longitude
            heading (float): Current heading in degrees (optional)
        """
        self.navigation_module.update_position(lat, lon, heading)