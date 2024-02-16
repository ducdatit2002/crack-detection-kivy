import cv2

# Path to the video file
video_url = 'http://example.com/path_to_video.mp4'

# Create a VideoCapture object
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read and display the video frame by frame
while True:
    ret, frame = cap.read()
    
    # Break the loop if we're at the end of the video file
    if not ret:
        print("Reached end of video.")
        break
    
    # Display the frame
    cv2.imshow('Video Playback', frame)
    
    # Press 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close display window
cap.release()
cv2.destroyAllWindows()
