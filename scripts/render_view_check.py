#!/usr/bin/env python3
"""Render a cameras.json view (and a parallax-offset novel view) from a gsplat PLY."""
import argparse, json, numpy as np, torch
from PIL import Image
from plyfile import PlyData
from scene.gaussian_model import GaussianModel as LegacyGaussianModel
from scene.gsplat_model import GsplatGaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from eligsir.utils.graphics_utils import focal2fov
from arguments import get_default_pipeline_params


def resolve_gaussian_model_cls(model_layout):
    return GsplatGaussianModel if model_layout == "gsplat" else LegacyGaussianModel

def save_img(t, path):
    arr=(t.clamp(0,1).permute(1,2,0).cpu().numpy()*255).astype("uint8")
    Image.fromarray(arr).save(path)

def cam_from_json(e, offset_m):
    rot=np.array(e["rotation"],dtype=np.float64)      # W2C rot
    pos=np.array(e["position"],dtype=np.float64)      # cam center world
    R_c2w=rot.T
    pos=pos + offset_m * R_c2w[:,0]                    # slide along cam right
    W2C=np.eye(4); W2C[:3,:3]=rot; W2C[:3,3]=pos
    Rt=np.linalg.inv(W2C)
    R=Rt[:3,:3].T; T=Rt[:3,3]
    w,h=int(e["width"]),int(e["height"]); fx,fy,cx,cy=e["fx"],e["fy"],e["cx"],e["cy"]
    img=torch.zeros((3,h,w),dtype=torch.float32)
    return Camera(0,R,T,focal2fov(fx,w),focal2fov(fy,h),img,None,"v",0,fx=fx,fy=fy,cx=cx,cy=cy)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ply",required=True); ap.add_argument("--cameras",required=True)
    ap.add_argument("--index",type=int,default=-1); ap.add_argument("--offset",type=float,default=0.0)
    ap.add_argument("--tag",required=True)
    ap.add_argument("--model_layout",default="gsplat",choices=["gsplat","legacy"])
    a=ap.parse_args()
    pipe=get_default_pipeline_params()
    bg=torch.zeros(3,dtype=torch.float32,device="cuda")
    GaussianModelCls = resolve_gaussian_model_cls(a.model_layout)
    g=GaussianModelCls(3); g.load_ply(a.ply)
    cams=json.load(open(a.cameras)); idx=a.index if a.index>=0 else len(cams)//2
    cam=cam_from_json(cams[idx], a.offset)
    with torch.no_grad():
        im=render(cam,g,pipe,bg)["render"].clamp(0,1)
    save_img(im,f"/tmp/{a.tag}.png")
    black=(im.max(0).values<0.02).float().mean().item()
    blow=(im.min(0).values>0.95).float().mean().item()
    print(f"{a.tag}: N={g.get_xyz.shape[0]} cam={idx} off={a.offset} mean={im.mean().item():.3f} black_frac={black:.3f} blowout_frac={blow:.3f}")

if __name__=="__main__": main()
