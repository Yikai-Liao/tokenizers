"""Convert buffered monotonic-clock events to native Perfetto TrackEvents.
Append an optional kernel trace; its clock snapshots align monotonic and boot time.
"""
import argparse,json
from pathlib import Path
from perfetto.protos.perfetto.trace.perfetto_trace_pb2 import Trace
SCHEMA={
 'training':['workers'], 'round':['vocab_before','rules','raw_positions'],
 'prepare':['rules','raw_positions'], 'commit':['event_chunks','completed_births'],
 'prepare.job':['job','tasks','raw_positions','worker'],
 'prepare.task':['rank','raw_positions','whole_rule','direct_births','contiguous'],
 'prepare.scan':['rank','raw_positions','matched_positions','contiguous'],
 'prepare.finish':['rank','direct_births','births','touched_groups'],
 'commit.owner':['owner','changes','birth_refs','completed_births','worker'],
 'commit.group':['birth_refs'], 'commit.counts':['changes'], 'commit.completed':['completed_births'],
 'commit.reduce':['birth_refs'], 'commit.aggregate':['bucket','fragments'],
 'commit.encode_publish':['bucket','groups','positions','encoded_groups'],
 'commit.prefix':['heap_entries','prefix_entries'], 'prepare.assemble':['jobs'],
 'prepare.aa':['raw_positions'], 'prepare.aa_validate':['raw_positions'],
 'prepare.aa_choose':['matched_positions'], 'prepare.aa_job':['job','matched_positions'],
}
def convert(src,dst,system=None):
 pid,buffers=json.loads(Path(src).read_text());trace=Trace()
 process_uuid=pid<<32
 p=trace.packet.add();p.trusted_packet_sequence_id=pid;p.track_descriptor.uuid=process_uuid
 p.track_descriptor.process.pid=pid;p.track_descriptor.process.process_name='BPE training'
 events=[]
 for tid,records in buffers:
  p=trace.packet.add();p.trusted_packet_sequence_id=tid;p.track_descriptor.uuid=process_uuid|tid
  p.track_descriptor.parent_uuid=process_uuid;p.track_descriptor.thread.pid=pid;p.track_descriptor.thread.tid=tid
  p.track_descriptor.thread.thread_name='BPE worker '+str(tid)
  for record in records:
   events.append((record['ts'],0,-record['dur'],tid,record))
   events.append((record['ts']+record['dur'],1,record['dur'],tid,record))
 for ts,end,_,tid,r in sorted(events,key=lambda x:x[:4]):
  p=trace.packet.add();p.timestamp=ts;p.timestamp_clock_id=3;p.trusted_packet_sequence_id=tid
  e=p.track_event;e.track_uuid=process_uuid|tid;e.type=2 if end else 1
  if not end:
   e.name=r['name'];e.categories.append('bpe')
   e.debug_annotations.add(name='round',uint_value=r['round'])
   for key,value in zip(SCHEMA.get(r['name'],[]),r['fields']):
    e.debug_annotations.add(name=key,uint_value=value)
 with Path(dst).open('wb') as out:
  if system: out.write(Path(system).read_bytes())
  out.write(trace.SerializeToString())
 print(json.dumps({'trace':str(dst),'pid':pid,'threads':len(buffers),'slices':len(events)//2,'bytes':Path(dst).stat().st_size}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('src');p.add_argument('dst');p.add_argument('--system');a=p.parse_args();convert(a.src,a.dst,a.system)
