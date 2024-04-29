# import the necessary packages
from deffcode import FFdecoder
import cv2

# define suitable FFmpeg parameter
ffparams = {
    "-vcodec": None,  # skip source decoder and let FFmpeg chose
    "-enforce_cv_patch": True, # enable OpenCV patch for YUV(NV12) frames
    "-ffprefixes": [
        "-vsync",
        "0",  # prevent duplicate frames
        "-hwaccel",
        "cuda",  # accelerator
        "-hwaccel_output_format",
        "cuda",  # output accelerator
    ],
    "-custom_resolution": "null",  # discard source `-custom_resolution`
    "-framerate": "null",  # discard source `-framerate`
    "-vf": "scale_cuda=640:360,"  # scale to 640x360 in GPU memory
    + "fps=60.0,"  # framerate 60.0fps in GPU memory
    + "hwdownload,"  # download hardware frames to system memory
    + "format=nv12",  # convert downloaded frames to NV12 pixel format
}

# initialize and formulate the decoder with `foo.mp4` source
decoder = FFdecoder(
    "/home/johnny/Projects/small-fast-detector/tracker/videos/demo.mp4",
    frame_format="null",  # discard source frame pixel format
    verbose=True, # enable verbose output
    **ffparams # apply various params and custom filters
)

decoder = decoder.formulate()

# grab the NV12 frame from the decoder
for frame in decoder.generateFrame():

    # check if frame is None
    if frame is None:
        break

    # convert it to `BGR` pixel format,
    # since imshow() method only accepts `BGR` frames
    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)

    # {do something with the BGR frame here}

    # Show output window
    cv2.imshow("Output", frame)

    # check for 'q' key if pressed
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break


# close output window
cv2.destroyAllWindows()

# terminate the decoder
decoder.terminate()
