from pathlib import Path

from modal import App, CloudBucketMount, Image, Secret

MOUNT_PATH = Path("/mnt/data")
IMAGE_PATH = Path("/mnt/data/images")
R2_ENDPOINT_URL = "https://e6e3361866faa51825bc96ee2f7804c1.r2.cloudflarestorage.com"
HEIGHT = 1080
WIDTH = 1920
NUM_FRAMES = 500  # Number of frames to process
KEEP_TIME_DELTA = 5  # once every 10 seconds

# R2 secret
secret = Secret.from_name("cloudfare_r2")

# Setup image
image = Image.debian_slim().apt_install("ffmpeg").pip_install("numpy", "ffmpeg-python", "mcap", "loguru", "pillow")
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
    import ffmpeg
    import numpy as np
    from loguru import logger
    from mcap.reader import make_reader
    from PIL import Image

    # Mcap reader
    logger.info(f"Processing {mcap_file}")
    reader = make_reader(mcap_file.open("rb"))

    # Configure FFmpeg to suppress output text
    ffmpeg_args = {"loglevel": "quiet", "hide_banner": None}

    # Initialize FFmpeg input and output buffers outside the loop
    input_buffer = ffmpeg.input("pipe:", format="h264")
    output_buffer = ffmpeg.output(input_buffer, "pipe:", format="rawvideo", pix_fmt="rgb24", **ffmpeg_args).global_args(
        "-loglevel", "error"
    )

    batch_frames = []
    batch_times = []
    left_frames = []
    right_frames = []
    last_time = 0
    for i, (schema, channel, msg) in enumerate(
        reader.iter_messages(topics="/nw/perception/front_camera/stereo/image_compressed")
    ):
        if i < NUM_FRAMES:
            batch_frames.append(msg.data)
            batch_times.append(msg.publish_time)
            continue
        logger.info(f"Processing batch {i}")

        # Decode the frame using FFmpeg
        out, _ = ffmpeg.run(output_buffer, capture_stdout=True, input=b"".join(batch_frames))

        # Convert the output buffer to a numpy array
        decoded_frames = np.frombuffer(out, np.uint8).reshape(-1, HEIGHT, WIDTH * 2, 3)
        logger.info(f"Decoded frames shape: {decoded_frames.shape}")

        # Determine keep indices
        keep_indices = []
        for j, t in enumerate(batch_times):
            if ((t - last_time) / 1e9) > KEEP_TIME_DELTA and j < len(decoded_frames):
                keep_indices.append(j)
                last_time = t

        logger.info(f"Keeping {len(keep_indices)} frames")
        decoded_frames = decoded_frames[keep_indices]

        # Break into left and right frames
        lfs = decoded_frames[:, :, :WIDTH, :]
        rfs = decoded_frames[:, :, WIDTH:, :]

        left_frames.append(lfs)
        right_frames.append(rfs)

        # # Save images to disk
        # logger.info(f"Saving {len(lfs)} frames")
        # for j, (lf, rf) in enumerate(zip(lfs, rfs)):
        #     left_frames.append(IMAGE_PATH / f"{mcap_file.stem}_left_{j}.png")
        #     right_frames.append(IMAGE_PATH / f"{mcap_file.stem}_right_{j}.png")
        #     Image.from_array(lf).save(left_frames[-1])
        #     Image.from_array(rf).save(right_frames[-1])

        break

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