import json, glob, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ks, names, allbins = [], [], []
for f in sorted(glob.glob('artifacts/scannet/scaling_scene*.json')):
    d=json.load(open(f)); names.append(f.split('scaling_')[1].replace('.json','').replace('scene','').replace('_00',''))
    ks.append(d['pooled_k_percell']); allbins.append((np.array(d['bins']['r']), np.array(d['bins']['sup']), np.array(d['bins']['acc'])))
d0=json.load(open('artifacts/scannet/radius_scaling.json')) if glob.glob('artifacts/scannet/radius_scaling.json') else None
ks.append(1.51); names.append('0000')
ks=np.array(ks)
fig, ax = plt.subplots(1,3, figsize=(16,4.6))
ax[0].bar(range(len(ks)), ks, color=['#2f855a' if abs(k-2)<0.3 else '#a0aec0' for k in ks])
ax[0].axhline(2.0, color='#e53e3e', ls='--', label='surface (k=2)')
ax[0].axhline(3.0, color='#a0aec0', ls='-.', label='volume (k=3)')
ax[0].axhline(ks.mean(), color='#2b6cb0', ls=':', label=f'mean {ks.mean():.2f}±{ks.std():.2f}')
ax[0].set_xticks(range(len(ks))); ax[0].set_xticklabels(names, rotation=45, fontsize=7)
ax[0].set_ylabel('exponent k  (support ~ r^k)'); ax[0].set_title('per-scene scaling exponent'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis='y')
for (r,s,a),nm in zip(allbins,names):
    ax[1].loglog(r, s/s[0], alpha=.6, lw=1)
rr=np.linspace(0.02,0.09,50)
ax[1].loglog(rr, (rr/0.02)**2, 'r--', lw=2, label='slope 2 (surface)')
ax[1].loglog(rr, (rr/0.02)**3, ls='-.', color='#a0aec0', lw=2, label='slope 3 (volume)')
ax[1].set_xlabel('power radius r (m)'); ax[1].set_ylabel('support (normalized)'); ax[1].set_title('evidence vs radius, all scenes'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which='both')
for (r,s,a),nm in zip(allbins,names):
    ax[2].plot(r, a, alpha=.65, lw=1.2)
ax[2].set_xlabel('power radius r (m)'); ax[2].set_ylabel('point-weighted accuracy'); ax[2].set_title('accuracy vs radius (scene-dependent!)'); ax[2].grid(alpha=.3)
fig.suptitle(f'Radius scaling across {len(ks)} ScanNet scenes: evidence ~ r^{ks.mean():.2f} (surface law), accuracy relation NOT consistent')
fig.tight_layout(); fig.savefig('artifacts/scannet/radius_scaling_10scene.png', dpi=140)
print('wrote artifacts/scannet/radius_scaling_10scene.png')
print(f'k = {ks.mean():.2f} +/- {ks.std():.2f}; 95% CI approx [{ks.mean()-1.96*ks.std()/np.sqrt(len(ks)):.2f}, {ks.mean()+1.96*ks.std()/np.sqrt(len(ks)):.2f}]')
