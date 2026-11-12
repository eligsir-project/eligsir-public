import os

import numpy as np
from PIL import Image

from scene.readers.common import CameraInfo, SceneInfo, fetchPlyFlexible, getNerfppNorm, storePly
from eligsir.utils.graphics_utils import focal2fov
from eligsir.utils.rgbd_frames import c2w_to_camera_rt
from eligsir.utils.sh_utils import SH2RGB


def is_realityscan_mobile(path: str) -> bool:
    has_sfm = os.path.exists(os.path.join(path, "sfm.txt"))
    has_xmp = any(f.endswith(".xmp") for f in os.listdir(path)) if os.path.isdir(path) else False
    return has_sfm and has_xmp


def readRealityScanSceneInfo(
    path: str,
    eval: bool,
    llffhold: int = 8,
    num_pts: int = 100_000,
) -> SceneInfo:
    from realityscan_importer import load_project

    rs_scene = load_project(path, undistort=True)

    cam_infos = []
    for uid, cam in enumerate(rs_scene.cameras):
        R, T = c2w_to_camera_rt(cam.world_from_camera)
        image = Image.open(cam.image_path)
        cam_infos.append(
            CameraInfo(
                uid=uid,
                R=R,
                T=T,
                FovX=focal2fov(cam.fx, cam.width),
                FovY=focal2fov(cam.fy, cam.height),
                image=image,
                image_path=str(cam.image_path),
                image_name=cam.uuid,
                width=cam.width,
                height=cam.height,
                fx=cam.fx,
                fy=cam.fy,
                cx=cam.cx,
                cy=cam.cy,
            )
        )

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    if rs_scene.point_cloud is not None:
        ply_path = os.path.join(path, "realityscan_pcd.ply")
        storePly(ply_path, rs_scene.point_cloud.xyz, rs_scene.point_cloud.rgb)
    else:
        ply_path = os.path.join(path, "realityscan_random.ply")
        xyz = (
            np.random.random((num_pts, 3))
            * nerf_normalization["radius"] * 3 * 2
            - (nerf_normalization["radius"] * 3)
        )
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    pcd = fetchPlyFlexible(ply_path)

    return SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
