import cv2
import numpy as np
from tqdm import tqdm

VIDEO_PATH = r"C:\Users\aarav\Desktop\test3.mp4"
OUTPUT_VIDEO = "hyperlapse_output_stabilized_smoothed_9.mp4"
FRAME_STEP = 10
SMOOTHING_RADIUS = 30


def estimate_transform(prev_gray, curr_gray):
    prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    if prev_pts is None:
        return np.array([[1, 0, 0], [0, 1, 0]])

    curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    if curr_pts is None or status is None:
        return np.array([[1, 0, 0], [0, 1, 0]])

    idx = np.where((status == 1).flatten())[0]
    if len(idx) < 10:
        return np.array([[1, 0, 0], [0, 1, 0]])

    prev_pts = prev_pts[idx]
    curr_pts = curr_pts[idx]

    m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
    if m is None:
        return np.array([[1, 0, 0], [0, 1, 0]])
    return m



def smooth_trajectory(transforms, radius):
    transforms = np.array(transforms)
    smoothed = np.zeros_like(transforms)

    if len(transforms) < 2 * radius + 1:
        radius = max(1, len(transforms) // 2)

    kernel_size = 2 * radius + 1
    kernel = np.ones(kernel_size) / kernel_size

    for i in range(3):  # dx, dy, da
        padded = np.pad(transforms[:, i], (radius, radius), mode='edge')
        smoothed_vals = np.convolve(padded, kernel, mode='valid')
        smoothed[:, i] = smoothed_vals[:len(transforms)]  # match shape exactly

    return smoothed





def fix_border(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] // 2, s[0] // 2), 0, 1.04)
    return cv2.warpAffine(frame, T, (s[1], s[0]))


def create_stabilized_timelapse(video_path, output_path, frame_step=10, smoothing_radius=30):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    transforms = []
    frames = []
    prev_gray = None

    print("Reading and computing transforms...")
    for i in tqdm(range(n_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_step != 0:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            m = estimate_transform(prev_gray, gray)
            dx = m[0, 2]
            dy = m[1, 2]
            da = np.arctan2(m[1, 0], m[0, 0])
            transforms.append([dx, dy, da])
        frames.append(frame)
        prev_gray = gray

    cap.release()

    print("Smoothing camera trajectory...")
    smoothed_transforms = smooth_trajectory(transforms, smoothing_radius)


    print("Applying smoothed transforms and saving video...")
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    prev_to_cur_transform = np.zeros((2, 3))
    prev_to_cur_transform[:2, :2] = np.eye(2)
    prev_to_cur_transform[:, 2] = 0

    for i in tqdm(range(len(frames))):
        frame = frames[i]
        dx, dy, da = smoothed_transforms[i] if i < len(smoothed_transforms) else [0, 0, 0]
        m = np.array([
            [np.cos(da), -np.sin(da), dx],
            [np.sin(da), np.cos(da), dy]
        ])
        stabilized = cv2.warpAffine(frame, m, (width, height))
        stabilized = fix_border(stabilized)
        out.write(stabilized)

    out.release()
    print(f"\n Done! Saved to: {output_path}")


if __name__ == "__main__":
    create_stabilized_timelapse(VIDEO_PATH, OUTPUT_VIDEO, frame_step=FRAME_STEP, smoothing_radius=SMOOTHING_RADIUS)