import sys, numpy as np, torch, open3d as o3d
sys.path.insert(0,'/home/rajehyl/powerfoam')
def cd(P, G, tag):
    a=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); b=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(G))
    d1=np.asarray(a.compute_point_cloud_distance(b)); d2=np.asarray(b.compute_point_cloud_distance(a))
    print(f'{tag:<42} n={len(P):>8}  acc={d1.mean()*100:7.3f}cm  comp={d2.mean()*100:7.3f}cm  CD={((d1.mean()+d2.mean())/2)*100:7.3f}cm')
for s in ['scene0062_00','scene0097_00']:
    G=np.load(f'/home/rajehyl/scannet_gt/train/{s}/coord.npy').astype(np.float64)
    ck=torch.load(f'/home/rajehyl/gaussian_baseline_scannet/{s}/ckpts/ckpt_29999_rank0.pt',map_location='cpu',weights_only=False)['splats']
    cd(ck['means'].numpy().astype(np.float64), G, f'{s} GAUSSIAN centers')
    import configargparse, warp as wp
    from configs import Params, add_group
    from data_loader import DataHandler
    from powerfoam.scene import PowerfoamScene
    wp.init()
    for variant in ['frozen','nonfrozen']:
        p=configargparse.ArgParser(); add_group(p,Params); p.add_argument('-c','--config',is_config_file=True)
        a=p.parse_args(['-c',f'/home/rajehyl/powerfoam/output/scannet_{s}_{variant}/config.yaml'])
        dh=DataHandler(a); dh.reload('all',downsample=a.downsample[-1])
        m=PowerfoamScene(a); m.initialize_from_dataset(dh,device='cuda'); m.load_pt(f'/home/rajehyl/powerfoam/output/scannet_{s}_{variant}/model.pt')
        cd(m.points.detach().cpu().numpy().astype(np.float64), G, f'{s} FOAM {variant} cell centers')
