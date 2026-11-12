# Thin overlay on the gsplat/ROCm training image that adds the deps needed for
# Orbbec/RealSense rosbag-mcap scenes + Open3D live odometry. Avoids the full
# (~30 min) gsplat recompile. The base Dockerfile already lists these for the next
# full build; this image just makes them available now.
#
#   podman build -f docker/orbbec_deps.Dockerfile \
#     --build-arg BASE_IMAGE=eligsir:rocm-base \
#     -t eligsir:orbbec-deps .
# The base image is deliberately supplied by the caller. The former default
# was a developer-local, legacy image name and was not a portable dependency.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
RUN pip install --no-cache-dir rosbags open3d \
 && (pip install --no-cache-dir pyrealsense2 || echo "pyrealsense2 unavailable; skipping")
