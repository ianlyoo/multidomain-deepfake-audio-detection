"""r4 adapter: retain sub102 optimizer/backbone/exports with fold-local real draws."""
from __future__ import annotations
import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import ft_nii_multilabel as ml

class RealWindowDataset:
    def __init__(self,items,steps,batch,seed,view_share=.35,aug=True):
        self.mix=[i for i in items if i['source']!='real_domain']
        self.real=[i for i in items if i['source']=='real_domain']
        assert self.mix and self.real
        self.n=steps*batch;self.seed=seed;self.share=view_share
        self.aug=ml.ft.AugConfig(p_gain=.3,p_noise=.2,noise_snr_db=(15.,40.),p_mp3=.1) if aug else None
        # Balance domains/classes, then files. Avoid raw ITW count overwhelming
        # the smaller but important real-song/music panels.
        self.strata={}
        for i in self.real:
            family=i.get('panel') or i['id'].split(':')[0]
            key=(family,i['file_fake'])
            self.strata.setdefault(key,[]).append(i)
        self.keys=sorted(self.strata)
    def __len__(self):return self.n-getattr(self,'start',0)
    def __getitem__(self,idx):
        rng=np.random.default_rng([self.seed,idx+getattr(self,'start',0)])
        if rng.random()<self.share:
            pool=self.strata[self.keys[int(rng.integers(len(self.keys)))]]
        else:pool=self.mix
        i=pool[int(rng.integers(len(pool)))];n=i['n']
        start=int(rng.integers(0,n-ml.WIN+1)) if n>ml.WIN else 0
        x=ml.read_window(i['path'],start,n)
        if i['source']=='real_domain':
            labs=i['real_labels'];y=np.array([0 if v is None else v for v in labs],np.float32)
            m=np.array([v is not None for v in labs],np.float32)
        else:y,m=ml.window_labels(i,start,min(n,ml.WIN))
        if self.aug is not None:x=ml.ft.augment(x,self.aug,rng)
        return ml.ft.normalize(x),y,m

def main():
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--items',required=True)
    args,rest=p.parse_known_args()
    ml.ITEMS=Path(args.items);ml.WindowDataset=RealWindowDataset
    def lease(step,cap_gb):
        ml.LEASE.parent.mkdir(parents=True,exist_ok=True)
        ml.LEASE.write_text(json.dumps(dict(project='deepvoice',job='sub111',pid=os.getpid(),
            cap_gb=cap_gb,ram_gb=5,checkpoint=str(step),updated=time.strftime('%Y-%m-%dT%H:%M:%S'))),encoding='utf-8')
    ml.write_lease=lease
    def resources(min_ram_gb,min_vram_gb,**kwargs):
        ram,vram=ml.ft.free_resources()
        if ml.REQUEST.exists() or ram<min_ram_gb or vram is None or vram<min_vram_gb:
            ml.release_lease();print('RESOURCE_PAUSE',ram,vram,flush=True);raise SystemExit(3)
        print('RESOURCES',ram,vram,flush=True)
        return ram,vram
    ml.ft.wait_for_resources=resources
    # Spawn workers resolve this module's class normally; no shared file edits.
    import torch
    torch.set_num_threads(4)
    return ml.main(rest)
if __name__=='__main__':sys.exit(main())
