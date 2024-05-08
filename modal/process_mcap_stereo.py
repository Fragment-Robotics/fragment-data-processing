from pathlib import Path

from modal import App, CloudBucketMount, Image, Secret

MOUNT_PATH = Path("/mnt/data")
IMAGE_PATH = Path("/mnt/data/images")
R2_ENDPOINT_URL = "https://e6e3361866faa51825bc96ee2f7804c1.r2.cloudflarestorage.com"
HEIGHT = 1080
WIDTH = 1920
KEEP_TIME_DELTA = 5  # once every 5 seconds

# R2 secret
secret = Secret.from_name("cloudfare_r2")

# Setup image
image = (
    Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg", "libswscale-dev", "libavcodec-dev", "libavformat-dev", "git", "cmake")
    .pip_install("uv")
    .run_commands("uv pip install --system numpy mcap loguru pillow")
    .run_commands("git clone https://github.com/DaWelter/h264decoder.git && cd h264decoder && uv pip install --system .")
)

# Main app
app = App(name="mcap-processor", image=image)

# Setup cloud bucket mount
mount = CloudBucketMount(
    bucket_name="strider",
    bucket_endpoint_url=R2_ENDPOINT_URL,
    secret=secret,
)

@app.function(
    volumes={MOUNT_PATH: mount},
    memory=(4192, 8192),
)
def process_mcaps(mcap_file: Path):
    import h264decoder
    import numpy as np
    from loguru import logger
    from mcap.reader import make_reader
    from PIL import Image

    # Mcap reader
    logger.info(f"Processing {mcap_file}")
    reader = make_reader(mcap_file.open("rb"))

    # H264 decoder
    decoder = h264decoder.H264Decoder()

    left_frames = []
    right_frames = []
    frame_times = []
    last_time = 0
    for i, (schema, channel, msg) in enumerate(
        reader.iter_messages(topics="/nw/perception/front_camera/stereo/image_compressed")
    ):
        # Decode message
        framedatas = decoder.decode(msg.data)
        
        for framedata in framedatas:
            (frame, w, h, ls) = framedata
            if frame is not None:
                # Skip frames that are too close together
                time_diff = (msg.publish_time - last_time) / 1e9
                if time_diff < KEEP_TIME_DELTA:
                    continue
                else:
                    last_time = msg.publish_time
                
                # Convert frame to numpy array
                frame = np.frombuffer(frame, dtype=np.ubyte, count=len(frame))
                frame = frame.reshape((h, ls//3, 3))
                frame = frame[:,:w,:]
                
                # Store left and right frames
                left_frames.append(frame[:, :WIDTH, :])
                right_frames.append(frame[:, WIDTH:, :])
                frame_times.append(msg.publish_time)

        # # Save images to disk
        # logger.info(f"Saving {len(lfs)} frames")
        # for j, (lf, rf) in enumerate(zip(lfs, rfs)):
        #     left_frames.append(IMAGE_PATH / f"{mcap_file.stem}_left_{j}.png")
        #     right_frames.append(IMAGE_PATH / f"{mcap_file.stem}_right_{j}.png")
        #     Image.from_array(lf).save(left_frames[-1])
        #     Image.from_array(rf).save(right_frames[-1])

    logger.info(f"Saved {len(left_frames)} frames at time diffs:")
    last_time = frame_times[0]
    for time in frame_times:
        logger.info(f"  {(time - last_time) / 1e9}s")
        last_time = time
    return len(left_frames)


@app.function(volumes={MOUNT_PATH: mount})
def get_files():
    from loguru import logger

    files = list(MOUNT_PATH.glob("**/*.mcap"))
    logger.info(f"Found {len(files)} MCAP files")
    return files


@app.local_entrypoint()
def main():
    files = get_files.remote()
    total_frames = sum(process_mcaps.map(files))
    print("Processed", total_frames, "stereo frames")