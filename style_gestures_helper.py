from motion_visualizer.model_animator import create_video
from motion_visualizer.bvh2npy import convert_bvh2npy
import sys
import os
import subprocess

try:
    bvh_folder = sys.argv[1]
    epoch = sys.argv[2]
    sample_id = sys.argv[3]
except:
    print("USAGE: python style_gestures_helper BVH_FOLDER EPOCH SAMPLE_ID(0-19)")
    exit()

print("Result folder:", bvh_folder)
print("Sample id (0-19):", sample_id)

audio_folder = "/home/rajmund/StyleGestures/data/trinity/processed/visualization_test"
bvh_file = os.path.join(bvh_folder, f"sampled_{epoch}_temp100_1k_{sample_id}.bvh")
audio_file = "Recording_007_0{}.wav".format(sample_id if int(sample_id) >= 10 else f"0{sample_id}")
audio_file = os.path.join(audio_folder, audio_file)
npy_out = "temp.npy"
mp4_out = "temp.mp4"
print("bvh2npy")
# Extract 3D coordinates
convert_bvh2npy(bvh_file, npy_out)
# Visualize those 3D coordinates
print("createvideo")
create_video(npy_out, mp4_out)
print("ffmpeg")
# Add the audio to the video
command = f"ffmpeg -y -i {audio_file} -i temp.mp4 -c:v libx264 -c:a libvorbis -loglevel quiet -shortest generated_video.mp4"
subprocess.call(command.split())

print("\ndone")

# Remove temporary files
for ext in ["npy", "mp4"]:
    os.remove("temp." + ext)