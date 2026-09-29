from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import csv,hashlib,json,os,random,subprocess,time,sys
import numpy as np,pandas as pd,soundfile as sf
from scipy.signal import butter,sosfiltfilt,resample_poly
from resources import check,append,atomic_json
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'deps'))
import imageio_ffmpeg
FFMPEG=imageio_ffmpeg.get_ffmpeg_exe()
AUDIO=ROOT/'rendered';AUDIO.mkdir(exist_ok=True)
CAT=pd.read_csv(ROOT/'source_catalog.csv',low_memory=False).fillna('').set_index('source_id')
PLAN=pd.read_csv(ROOT/'mixture_plan.csv',low_memory=False).fillna('')
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
def mp3(x):
 p=subprocess.run([FFMPEG,'-nostdin','-v','error','-f','f32le','-ar','16000','-ac','1','-i','pipe:0','-c:a','libmp3lame','-b:a','64k','-f','mp3','pipe:1'],input=x.astype('<f4').tobytes(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 if p.returncode:raise RuntimeError('mp3 encode '+p.stderr.decode(errors='replace')[:200])
 q=subprocess.run([FFMPEG,'-nostdin','-v','error','-i','pipe:0','-ac','1','-ar','16000','-f','f32le','pipe:1'],input=p.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
 if q.returncode:raise RuntimeError('mp3 decode '+q.stderr.decode(errors='replace')[:200])
 y=np.frombuffer(q.stdout,dtype='<f4').copy()
 if len(y)<len(x):y=np.pad(y,(0,len(x)-len(y)))
 return y[:len(x)]
sos=butter(4,[300,3400],btype='bandpass',fs=SR,output='sos')
def telephone(x):
 y=sosfiltfilt(sos,x);y=resample_poly(y,1,2);y=np.clip(y,-1,1)
 u=np.sign(y)*np.log1p(255*np.abs(y))/np.log1p(255)
 q=np.rint((u+1)*127.5).clip(0,255).astype(np.uint8)
 u=q.astype(np.float32)/127.5-1
 y=np.sign(u)*np.expm1(np.abs(u)*np.log1p(255))/255
 z=resample_poly(y,2,1)
 return z[:len(x)] if len(z)>=len(x) else np.pad(z,(0,len(x)-len(z)))
def render(row):
 rng=random.Random(int(row.seed));n=int(round(float(row.duration_s)*SR))
 speech=load(row.speech_id) if row.speech_id else None
 music=load(row.music_id) if row.music_id else None
 partial=load(row.partial_fake_id) if row.partial_fake_id else None
 if speech is not None:speech=level(clip(speech,n,rng))
 splice_start='';splice_seconds=''
 if partial is not None:
  sec=rng.uniform(1,min(4,float(row.duration_s)-1));k=int(sec*SR)
  start=rng.randrange(SR//2,max(SR//2+1,n-k-SR//2))
  fake=level(clip(partial,k,rng));fade=min(320,k//4)
  if fade:
   alpha=np.linspace(0,1,fade,dtype=np.float32)
   speech[start:start+fade]=speech[start:start+fade]*(1-alpha)+fake[:fade]*alpha
   speech[start+fade:start+k-fade]=fake[fade:-fade]
   speech[start+k-fade:start+k]=fake[-fade:]*(1-alpha)+speech[start+k-fade:start+k]*alpha
  else:speech[start:start+k]=fake
  splice_start=round(start/SR,3);splice_seconds=round(k/SR,3)
 if music is not None:music=level(clip(music,n,rng))
 if speech is None:x=music
 elif music is None:x=speech
 elif row.kind.startswith('overlap_'):
  cut=n//3;gain=10**(-float(row.snr_db)/20);x=np.zeros(n,dtype=np.float32)
  if rng.random()<.5:
   x[:2*cut]=speech[:2*cut];x[cut:]+=music[cut:]*gain
  else:
   x[:2*cut]=music[:2*cut]*gain;x[cut:]+=speech[cut:]
 elif row.kind.startswith('seq_'):
  cut=n//2
  x=np.concatenate([speech[:cut],music[cut:]]) if row.order=='speech_then_music' else np.concatenate([music[:cut],speech[cut:]])
  fade=min(160,cut//4);x[cut-fade:cut+fade]*=np.abs(np.linspace(-1,1,2*fade,dtype=np.float32))
 else:
  srms=np.sqrt(np.mean(speech.astype(np.float64)**2)+1e-9)
  mrms=np.sqrt(np.mean(music.astype(np.float64)**2)+1e-9)
  gain=(srms/(10**(float(row.snr_db)/20)))/mrms
  x=speech+music*gain
 x=np.asarray(x,dtype=np.float32)
 if row.channel=='telephone':x=telephone(x).astype(np.float32)
 elif row.channel=='mp3_64k':x=mp3(x)
 peak=float(np.max(np.abs(x)))
 if peak>.98:x=x*(.98/peak)
 assert len(x)==n and np.isfinite(x).all()
 out=AUDIO/(row.id+'.flac');tmp=out.with_suffix('.flac.tmp')
 sf.write(tmp,x,SR,format='FLAC',subtype='PCM_16');os.replace(tmp,out)
 return dict(id=row.id,path=str(out),sha256=digest(out.read_bytes()),bytes=out.stat().st_size,samples=n,seconds=n/SR,split=row.split,kind=row.kind,channel=row.channel,voice_fake=int(row.voice_fake),music_fake=int(row.music_fake),file_fake=int(row.file_fake),voice_present=int(row.voice_present),music_present=int(row.music_present),speech_id=row.speech_id,music_id=row.music_id,partial_fake_id=row.partial_fake_id,seed=int(row.seed),snr_db=row.snr_db,order=row.order,splice_start_s=splice_start,splice_seconds=splice_seconds)

check('render_start',initial=True)
done={json.loads(line)['id'] for line in MAN.read_text(encoding='utf-8').splitlines()} if MAN.exists() else set()
rows=[row for row in PLAN.itertuples(index=False) if row.id not in done]
started=time.time()
with ThreadPoolExecutor(max_workers=3) as pool:
 for base in range(0,len(rows),100):
  for rec in pool.map(render,rows[base:base+100]):append(MAN,rec);done.add(rec['id'])
  print('RENDER',len(done),'of',len(PLAN),'elapsed',round(time.time()-started,1),flush=True)
  if len(done)%1000==0 or base+100>=len(rows):
   atomic_json(ROOT/('render_checkpoint_%05d.json'%len(done)),dict(completed=len(done),planned=len(PLAN),elapsed_s=time.time()-started,resources=check('render_'+str(len(done)))))
atomic_json(ROOT/'render_complete.json',dict(files=len(done),elapsed_s=time.time()-started))
print('RENDER_COMPLETE',len(done),flush=True)
