"""Import-safe copy of render_local.py load/clip/level (sub102 overlap replay)."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,os,random,subprocess,time,sys
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import butter,sosfiltfilt,resample_poly
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
CAT=pd.read_csv(ROOT/'source_catalog.csv',low_memory=False).fillna('').set_index('source_id')
MAN=ROOT/'render_manifest.jsonl';ERR=ROOT/'render_errors.jsonl'
SR=16000
def digest(b):return hashlib.sha256(b).hexdigest()
@lru_cache(maxsize=32)
def load(sid):
 path=str(CAT.loc[sid,'path'])
 proc=subprocess.run([FFMPEG,'-nostdin','-v','error','-threads','1','-i',path,'-t','30','-ac','1','-ar','16000','-f','f32le','pipe:1'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=75)
 if proc.returncode:raise RuntimeError('decode '+sid+' '+proc.stderr.decode(errors='replace')[:200])
 x=np.frombuffer(proc.stdout,dtype='<f4').copy()
 if len(x)<SR//2 or not np.isfinite(x).all():raise ValueError('invalid '+sid)
 return x
def clip(x,n,rng):
 if len(x)<n:x=np.tile(x,(n+len(x)-1)//len(x))
 start=rng.randrange(len(x)-n+1)
 return x[start:start+n].copy()
def level(x,target=.07):
 rms=np.sqrt(np.mean(x.astype(np.float64)**2)+1e-9)
 return x*(target/rms)
