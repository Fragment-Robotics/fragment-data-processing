from modal import App, CloudBucketMount, Image, Secret
from pathlib import Path

MOUNT_PATH = Path("/mnt/data")
R2_ENDPOINT_URL = "https://e6e3361866faa51825bc96ee2f7804c1.r2.cloudflarestorage.com"
HEIGHT = 1080
WIDTH = 1920
NUM_FRAMES = 50  # Number of frames to process
KEEP_TIME_DELTA = 1 / 2 # 2 Hz

# R2 secret
secret = Secret.from_name("cloudfare_r2")

# Setup image
image = (
    Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("numpy", "ffmpeg-python", "mcap")
)
# Main app
app = App(name="mcap-processor", image=image)

# Setup cloud bucket mount
mount = CloudBucketMount(
    bucket_name="strider",
    bucket_endpoint_url=R2_ENDPOINT_URL,
    secret=secret,
)

# Test function
@app.function(
    volumes={MOUNT_PATH: mount},
    memory=(8192, 16384),
)
def process_mcaps():
    import ffmpeg
    import numpy as np
    from mcap.reader import make_reader

    # Get the list of files
    files = list(MOUNT_PATH.glob("**/*.mcap"))

    # Mcap reader
    reader = make_reader(Path(files[0]).open("rb"))

    # Configure FFmpeg to suppress output text
    ffmpeg_args = {'loglevel': 'quiet', 'hide_banner': None}

    # Initialize FFmpeg input and output buffers outside the loop
    input_buffer = ffmpeg.input('pipe:', format='h264')
    output_buffer = (
        ffmpeg.output(input_buffer, 'pipe:', format='rawvideo', pix_fmt='rgb24', **ffmpeg_args)
        .global_args('-loglevel', 'error')
    )

    batch_frames = []
    batch_times = []
    left_frames = []
    right_frames = []
    last_time = 0
    for i, (schema, channel, msg) in enumerate(reader.iter_messages(topics="/nw/perception/front_camera/stereo/image_compressed")):
        if i < NUM_FRAMES:
            batch_frames.append(msg.data)
            batch_times.append(msg.publish_time)
            continue

        # Decode the frame using FFmpeg
        out, _ = ffmpeg.run(output_buffer, capture_stdout=True, input=b''.join(batch_frames))
        
        # Convert the output buffer to a numpy array
        decoded_frame = np.frombuffer(out, np.uint8).reshape(-1, HEIGHT, WIDTH * 2, 3)

        # Determine keep indices
        keep_indices = []
        for j, t in enumerate(batch_times):
            if ((t - last_time) / 1e9 ) > KEEP_TIME_DELTA:
                keep_indices.append(j)
                last_time = t
        decoded_frame = decoded_frame[keep_indices]
        
        # Break into left and right frames
        left_frame = decoded_frame[:, :, :WIDTH, :]
        right_frame = decoded_frame[:, :, WIDTH:, :]
        
        # Process the left and right frames as needed
        # ...
        left_frames.extend(left_frame)
        right_frames.extend(right_frame)

        break

    print(f"Processed {len(left_frames)} frames")


@app.local_entrypoint()
def main():
    process_mcaps.remote()
    print("Done!")