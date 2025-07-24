from moviepy.editor import VideoFileClip
from fer import FER
import cv2
import threading
import time

def map_emotion_to_frame_rate(emotion):
    """
    Map emotions to frame rates.
    - Happy: Dncrease frame rate.
    - Sad: Iecrease frame rate.
    - Neutral or other: Normal frame rate.
    """
    if emotion == 'happy':
        return 15  # High frame rate for happy emotions
    elif emotion == 'sad':
        return 60  # Low frame rate for sad emotions
    else:
        return 30  # Normal frame rate for neutral or other emotions

def detect_emotion(webcam, emotion_detector, current_emotion):
    """
    Continuously detect emotions from the webcam.
    """
    while True:
        ret_webcam, frame_webcam = webcam.read()
        if not ret_webcam:
            break

        # Detect emotion from the webcam
        emotion_results = emotion_detector.detect_emotions(frame_webcam)
        if emotion_results:
            dominant_emotion = emotion_results[0]['emotions']
            current_emotion[0] = max(dominant_emotion, key=dominant_emotion.get)
        else:
            current_emotion[0] = 'neutral'

def dynamic_playback_with_audio_and_video(clip, current_emotion):
    """
    Play the video with dynamically controlled frame rates and synchronized audio.
    """
    original_fps = clip.fps  # Original FPS of the video
    print(f"Original FPS: {original_fps}")
    print("Press Ctrl+C to stop playback.")

    # Shared emotion variable
    emotion = 'neutral'

    # Capture the video for frame-based manipulation
    cap = cv2.VideoCapture(clip.filename)

    # Start audio playback in a separate thread for synchronization
    audio_thread = threading.Thread(target=lambda: clip.audio.preview())
    audio_thread.start()

    try:
        while True:
            # Read the next frame from the video
            ret_video, frame_video = cap.read()
            if not ret_video:
                break

            # Detect current emotion and adjust FPS
            emotion = current_emotion[0] if current_emotion[0] else 'neutral'
            adjusted_fps = map_emotion_to_frame_rate(emotion)
            adjusted_delay = max(0.001, 1 / adjusted_fps)  # Ensure non-zero delay

            # Overlay emotion and FPS information
            cv2.putText(frame_video, f"Emotion: {emotion}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame_video, f"FPS: {adjusted_fps}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display the video frame
            cv2.imshow("Video Playback", frame_video)

            # Simulate frame delay and allow quitting with 'q'
            if cv2.waitKey(int(adjusted_delay * 1000)) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("Playback stopped by user.")
    finally:
        cv2.destroyAllWindows()
        cap.release()
        audio_thread.join()  # Ensure audio thread is terminated

def process_video(video_path):
    """
    Process the video, handle emotion detection, and control playback.
    """
    emotion_detector = FER()
    webcam = cv2.VideoCapture(0)
    clip = VideoFileClip(video_path)

    # Shared variable for current emotion
    current_emotion = [None]

    # Start emotion detection in a separate thread
    emotion_thread = threading.Thread(target=detect_emotion, args=(webcam, emotion_detector, current_emotion))
    emotion_thread.start()

    # Play video with dynamic frame rate control and audio
    dynamic_playback_with_audio_and_video(clip, current_emotion)

    # Clean up
    clip.close()
    webcam.release()
    emotion_thread.join()

if __name__ == "__main__":
    video_path = "C:\\Users\\aarav\\Downloads\\paan.mp4"  # Replace with the path to your MP4 file

    if not video_path:
        print("Error: No video file path specified.")
    else:
        process_video(video_path)
