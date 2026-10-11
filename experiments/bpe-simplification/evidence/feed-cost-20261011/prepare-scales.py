"""Reproduce real Chinese Wikipedia nested prefixes; never repeat source text."""
import argparse,hashlib,json,re,urllib.request
from pathlib import Path
import pyarrow.parquet as pq

REVISION='b04c8d1ceb2f5cd4588862100d08de323dccfbaa'
SHARD_SHA='853ddf4a138c792ad7386d6de831788f96326c03e1e3349f7a315498392e8b56'
URL=f'https://huggingface.co/datasets/wikimedia/wikipedia/resolve/{REVISION}/20231101.zh/train-00000-of-00006.parquet'
EXPECTED={128:(134217639,'7dd7de3f8ec76913c510fb0840818b6749b12fbdf503c997932df2e444a7af0b'),256:(268435162,'61537f422d250f93ed669193faec992a516ce049b41ad33fb02b52064780ed98'),384:(402653152,'cee7cc37a8a25450d0bf02955c98fb544fb2ed536a27ad55acfae1a8a9a614a5'),512:(536870289,'a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f')}
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--shard',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--download',action='store_true');args=parser.parse_args()
 if not args.shard.exists():
  if not args.download:raise FileNotFoundError('Supply cached verified shard or explicitly use --download')
  args.shard.parent.mkdir(parents=True,exist_ok=True)
  with urllib.request.urlopen(URL) as source,args.shard.open('wb') as target:__import__('shutil').copyfileobj(source,target)
 assert args.shard.stat().st_size==587109027 and digest(args.shard)==SHARD_SHA
 args.out.mkdir(exist_ok=True,parents=True);space=re.compile(r'\s+')
 for size,(expected_bytes,expected_sha) in EXPECTED.items():
  dest=args.out/f'zh{size}.txt';total=lines=0;done=False
  with dest.open('wb') as output:
   for batch in pq.ParquetFile(args.shard).iter_batches(batch_size=256,columns=['text']):
    for article in batch.column(0).to_pylist():
     for paragraph in (article or '').split('\n\n'):
      line=(space.sub(' ',paragraph).strip()+'\n').encode('utf-8')
      if not 32<=len(line)<=8192:continue
      if total+len(line)>size*2**20:done=True;break
      output.write(line);total+=len(line);lines+=1
     if done:break
    if done:break
  sha=digest(dest);assert (total,sha)==(expected_bytes,expected_sha)
  print(json.dumps(dict(path=str(dest),bytes=total,sha256=sha,lines=lines)),flush=True)
if __name__=='__main__':main()
