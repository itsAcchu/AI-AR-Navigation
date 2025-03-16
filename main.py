import cv2
import numpy as np
import time
import signal
import sys
from navigation_system import ARNavigation

# Flag to control application running state
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C and other termination signals"""
    global running
    print("\nStopping application gracefully...")
    running = False

def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Initialize the AR Navigation system
    print("Initializing AR Navigation system...")
    ar_navigation = ARNavigation()

    # Start the system
    ar_navigation.start()

    # Simulate camera input (replace with real camera feed)
    print("Opening camera...")
    cap = cv2.VideoCapture(0)  # Use 0 for the default camera
    
    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # FPS calculation variables
    frame_count = 0
    start_time = time.time()
    
    while running:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        # Process the frame through the AR Navigation system
        guidance = ar_navigation.process_camera_frame(frame)
        
        # Display the guidance message on the frame
        cv2.putText(frame, guidance['message'], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the resulting frame
        cv2.imshow('AR Navigation', frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count >= 30:
            end_time = time.time()
            fps = frame_count / (end_time - start_time)
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # When everything is done, release the capture and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Stop the AR Navigation system
    ar_navigation.stop()
    print("Application stopped")

if __name__ == "__main__":
    main()