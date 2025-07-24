import os
import csv
import subprocess
import librosa
import numpy as np
import shutil
import tensorflow as tf
import tensorflow_hub as hub
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============ CONFIGURATION ============
VIDEO_PATH = r"C:\Users\aarav\Downloads\videoplayback.mp4"
TMP_FOLDER = "temp_segments"
FINAL_OUTPUT = "output_hybrid_music_detection_A.mp4"
SCENE_LOG_CSV = "scene_log_2.csv"
MIN_SCENE_DURATION = 3.0
MUSIC_SPEED = 2.5

# ============ LOAD YAMNET ============
yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
class_map_path = tf.keras.utils.get_file(
    'yamnet_class_map.csv',
    'https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv'
)
class_names = [line.split(',')[2].strip() for line in open(class_map_path).readlines()[1:]]

# ============ DURATION ============
def get_video_duration(video_path):
    try:
        output = subprocess.check_output([
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of',
            'default=noprint_wrappers=1:nokey=1', video_path
        ], stderr=subprocess.DEVNULL).decode().strip()
        return float(output)
    except Exception as e:
        print(f"Error getting duration for {video_path}: {e}")
        return 0

# ============ SCENE DETECTION ============
def detect_scenes(video_path):
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector())
    video_manager.start()
    scene_manager.detect_scenes(video_manager)
    scene_list = scene_manager.get_scene_list()
    video_manager.release()
    return [(s.get_seconds(), e.get_seconds()) for s, e in scene_list]

# ============ AUDIO EXTRACTION ============
def extract_scene_audio(video_path, start, end, idx):
    scene_audio = os.path.join(TMP_FOLDER, f"scene_{idx:04d}.wav")
    subprocess.call([
        'ffmpeg', '-y', '-ss', str(start), '-to', str(end), '-i', video_path,
        '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', scene_audio
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return scene_audio

# ============ YAMNET MUSIC DETECTION ============
def is_music_yamnet(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        scores, embeddings, spectrogram = yamnet_model(y)
        scores_np = scores.numpy()
        mean_scores = np.mean(scores_np, axis=0)
        top_indices = np.argsort(mean_scores)[-10:]
        top_labels = [class_names[i].lower() for i in top_indices]

        # Must include "music"
        if not any("music" in lbl for lbl in top_labels):
            return False

        # Must NOT include action-related sounds
        disallowed = ["gunshot", "explosion", "speech", "shout", "bang", "scream"]
        if any(bad in lbl for lbl in top_labels for bad in disallowed):
            return False

        return True
    except Exception as e:
        print(f"YAMNet error: {e}")
        return False


# ============ LIBROSA MUSIC DETECTION ============
def is_music_librosa(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=None)
        if len(y) < sr:
            return False
        y_harm, y_perc = librosa.effects.hpss(y)
        hpr = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(np.abs(mfcc))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        contrast_mean = np.mean(contrast)
        return hpr > 1.2 and mfcc_mean > 25 and tempo > 70 and contrast_mean > 20
    except:
        return False

# ============ EXPORT SCENE ============
def export_scene(start, end, idx, speed, output_dir, tag):
    out_path = os.path.join(output_dir, f"{tag}_scene_{idx:04d}.mp4")
    video_filter = f"setpts={1/speed}*PTS" if speed != 1 else "null"
    if speed <= 2:
        audio_filter = f"atempo={speed}"
    else:
        audio_filter = "atempo=2.0,atempo={:.2f}".format(speed / 2)

    cmd = [
        'ffmpeg', '-y', '-ss', str(start), '-to', str(end), '-i', VIDEO_PATH,
        '-vf', video_filter, '-af', audio_filter,
        '-preset', 'fast', '-c:v', 'libx264', '-c:a', 'aac', out_path
    ]
    subprocess.call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return out_path

def export_wrapper(args):
    return export_scene(*args)

import matplotlib.pyplot as plt  # <- Move this to the top

def visualize_detected_music_scenes(predicted_intervals, video_duration, save_path="detected_music_timeline.png"):
    plt.figure(figsize=(15, 1.5))
    for (start, end) in predicted_intervals:
        plt.axvspan(start, end, color='red', alpha=0.6)
    plt.xlim(0, video_duration)
    plt.xlabel("Time (seconds)")
    plt.title("Detected Music Scenes Timeline")
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\n Music scene timeline saved to: {save_path}")


# ============ MAIN ============
if __name__ == "__main__":
    if os.path.exists(TMP_FOLDER):
        shutil.rmtree(TMP_FOLDER)
    os.makedirs(TMP_FOLDER)

    print(" Detecting scenes...")
    scenes = detect_scenes(VIDEO_PATH)
    scenes = [(s, e) for s, e in scenes if e - s >= MIN_SCENE_DURATION]

    print(" Analyzing scenes for music using hybrid (Librosa + YAMNet)...")
    music_flags = []
    rows = []
    for idx, (s, e) in enumerate(tqdm(scenes)):
        audio_path = extract_scene_audio(VIDEO_PATH, s, e, idx)
        is_music = is_music_yamnet(audio_path) or is_music_librosa(audio_path)
        tag = "music" if is_music else "main"
        rows.append([idx, round(s, 2), round(e, 2), round(e - s, 2), is_music, tag])
        music_flags.append(is_music)
        os.remove(audio_path)
    with open(SCENE_LOG_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Scene", "Start", "End", "Duration", "IsMusic", "Tag"])
        writer.writerows(rows)

    print(f" Scene log saved to: {SCENE_LOG_CSV}")

    music_scenes = [(s, e) for (s, e), flag in zip(scenes, music_flags) if flag]
    non_music_scenes = [(s, e) for (s, e), flag in zip(scenes, music_flags) if not flag]

    print(f" Non-music scenes: {len(non_music_scenes)}, Music scenes: {len(music_scenes)}")

    export_jobs = []
    for idx, (s, e) in enumerate(non_music_scenes):
        export_jobs.append((s, e, idx, 1.0, TMP_FOLDER, "main"))
    for idx, (s, e) in enumerate(music_scenes):
        export_jobs.append((s, e, idx, MUSIC_SPEED, TMP_FOLDER, "music"))

    print(" Exporting scenes...")
    with Pool(processes=min(cpu_count(), 6)) as pool:
        segment_paths = list(tqdm(pool.imap(export_wrapper, export_jobs), total=len(export_jobs)))

    segment_paths.sort(key=lambda p: (0 if "main" in p else 1, p))

    filelist_path = os.path.join(TMP_FOLDER, "filelist.txt")
    with open(filelist_path, "w") as f:
        for path in segment_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")

    print(" Merging all segments...")
    subprocess.call([
        'ffmpeg', '-y', '-f', 'concat', '-safe', '0',
        '-i', filelist_path, '-c', 'copy', FINAL_OUTPUT
    ])

    print(f"\n Done! Final output: {FINAL_OUTPUT}")

    # duration comparison
    original_duration = get_video_duration(VIDEO_PATH)
    output_duration = get_video_duration(FINAL_OUTPUT)
    
    print(f" Original Duration : {original_duration/60:.2f} minutes")
    print(f" Output Duration   : {output_duration/60:.2f} minutes")
    reduction = (1 - output_duration / original_duration) * 100
    print(f" Duration Reduced  : {reduction:.2f}%")

    # Warn
    if output_duration > original_duration * 1.1:
        print(" Warning: Output is significantly longer than original.")
    elif output_duration < original_duration * 0.3:
        print(" Warning: Output may be too short.")
    else:
        print(" Output duration is within expected range.")
    
    # 🟢 Now call the visualization function here
    visualize_detected_music_scenes(music_scenes, original_duration)


