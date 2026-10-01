//! Standalone initial index kernels; no Trainer or merge speed claim.
use ahash::RandomState;
use std::{collections::HashMap, io::{BufRead, BufReader}, time::Instant};
mod radsort_u64;
type Map<K,V> = HashMap<K,V,RandomState>;
fn map<K,V>() -> Map<K,V> { Map::with_hasher(RandomState::with_seeds(1,2,3,4)) }
const NONE:u32=u32::MAX;
#[repr(C)] #[derive(Clone,Copy,Default)] struct Record { key:u32, pos:u32 }
unsafe extern "C" { fn permuted_sort(records:*mut Record,n:usize); }
struct Corpus { slots:Vec<u32>, pivots:Vec<(u32,u64)>, bounds:Vec<usize>, one:Vec<bool>, alphabet:usize, edges:usize, words:usize }
impl Corpus {
    fn from_words(words:Vec<(String,u64)>) -> Self {
        let mut present=vec![false;0x110000];
        for (s,_) in &words {for ch in s.chars(){present[ch as usize]=true;}}
        let chars:Vec<char>=present.iter().enumerate().filter_map(|(i,&seen)|seen.then(||char::from_u32(i as u32).unwrap())).collect();
        drop(present);assert!(chars.len()<=65536);
        let mut ids=vec![NONE;0x110000];
        for (id,&ch) in chars.iter().enumerate() { ids[ch as usize]=id as u32; }
        let n=1+words.iter().map(|(s,_)|s.chars().count()+1).sum::<usize>();
        let mut slots=Vec::with_capacity(n);slots.push(NONE);
        let mut pivots=Vec::with_capacity(words.len());let mut edges=0;
        for (s,w) in &words {
            pivots.push((slots.len() as u32,*w));
            let old=slots.len();slots.extend(s.chars().map(|ch|ids[ch as usize]));
            edges+=(slots.len()-old).saturating_sub(1);slots.push(NONE);
        }
        let mut bounds=Vec::with_capacity(n.div_ceil(256)+1);let mut pivot=0;
        for b in 0..=n.div_ceil(256) { while pivot<pivots.len() && (pivots[pivot].0 as usize)<(b*256).min(n) {pivot+=1;}bounds.push(pivot); }
        let mut one=vec![true;n.div_ceil(256)];
        for (i,&(start,w)) in pivots.iter().enumerate() { if w!=1 { let end=pivots.get(i+1).map_or(n,|x|x.0 as usize);if (start as usize)<end {for b in start as usize/256..=(end-1)/256 {one[b]=false;}} } }
        Self {slots,pivots,bounds,one,alphabet:chars.len(),edges,words:words.len()}
    }
    fn weight(&self,p:u32)->u64 {
        let b=p as usize/256;if self.one[b] {return 1;}
        let lo=self.bounds[b];let hi=self.bounds[b+1];
        let i=lo+self.pivots[lo..hi].partition_point(|x|x.0<=p);self.pivots[i-1].1
    }
    fn scan(&self,mut f:impl FnMut(u32,u32,u64)) {
        let mut wi=0;let mut w=0;
        for p in 0..self.slots.len().saturating_sub(1) {
            if wi<self.pivots.len() && self.pivots[wi].0 as usize==p {w=self.pivots[wi].1;wi+=1;}
            let a=self.slots[p];let b=self.slots[p+1];
            if a!=NONE && b!=NONE { f(a* self.alphabet as u32+b,p as u32,w); }
        }
    }
}
struct Index {keys:Vec<u32>,freq:Vec<u64>,counts:Vec<u32>,starts:Vec<usize>,pos:Vec<u32>}
impl Index {
    fn allocate(keys:Vec<u32>,freq:Vec<u64>,counts:Vec<u32>)->Self {
        let mut starts=Vec::with_capacity(keys.len()+1);starts.push(0);
        for &c in &counts {starts.push(starts.last().unwrap()+c as usize);}
        let pos=vec![0;*starts.last().unwrap()];Self{keys,freq,counts,starts,pos}
    }
    fn hash(&self)->u64 {
        let mut order:Vec<_>=(0..self.keys.len()).collect();order.sort_unstable_by_key(|&i|self.keys[i]);
        let mut h=0xcbf29ce484222325_u64;
        for i in order {
            for v in [self.keys[i] as u64,self.freq[i],self.counts[i] as u64] {h=(h^v).wrapping_mul(0x100000001b3);}
            let p=&self.pos[self.starts[i]..self.starts[i+1]];
            assert!(p.windows(2).all(|w|w[0]<w[1]));
            for &v in p {h=(h^v as u64).wrapping_mul(0x100000001b3);}
        }h
    }
    fn validate(&self,c:&Corpus,floor:u64) {
        let mut oracle:Map<u32,(u64,Vec<u32>)>=map();
        c.scan(|key,p,w|{let entry=oracle.entry(key).or_default();entry.0=entry.0.checked_add(w).unwrap();entry.1.push(p);});
        oracle.retain(|_,v|v.0>=floor);assert_eq!(oracle.len(),self.keys.len());
        for (i,&key) in self.keys.iter().enumerate() {
            let (freq,pos)=oracle.remove(&key).unwrap();assert_eq!(freq,self.freq[i]);assert_eq!(pos.len(),self.counts[i] as usize);
            assert_eq!(pos,&self.pos[self.starts[i]..self.starts[i+1]]);
        } assert!(oracle.is_empty());
    }
    fn bytes(&self)->usize {self.keys.capacity()*4+self.freq.capacity()*8+self.counts.capacity()*4+self.starts.capacity()*8+self.pos.capacity()*4}
}
fn standard_sort(r:&mut Vec<Record>) {
    let mut tmp=vec![Record::default();r.len()];
    for shift in [0,8,16,24] {
        let mut counts=[0usize;256];for x in r.iter(){counts[((x.key>>shift)&255)as usize]+=1;}
        let mut next=[0usize;256];let mut n=0;for b in 0..256{next[b]=n;n+=counts[b];}
        for &x in r.iter(){let b=((x.key>>shift)&255)as usize;tmp[next[b]]=x;next[b]+=1;}
        std::mem::swap(r,&mut tmp);
    }
}
fn sorted(c:&Corpus,floor:u64,rad:bool)->(Index,usize) {
    let mut records=Vec::with_capacity(c.edges);c.scan(|key,pos,_|records.push(Record{key,pos}));
    if rad {unsafe {permuted_sort(records.as_mut_ptr(),records.len());}}else{standard_sort(&mut records);}
    let mut keys=Vec::new();let mut freq=Vec::new();let mut counts=Vec::new();let mut ranges=Vec::new();let mut i=0;
    while i<records.len(){let key=records[i].key;let begin=i;let mut f=0u64;while i<records.len()&&records[i].key==key{f=f.checked_add(c.weight(records[i].pos)).unwrap();i+=1;}
        if f>=floor{keys.push(key);freq.push(f);counts.push((i-begin)as u32);ranges.push((begin,i));}}
    let mut out=Index::allocate(keys,freq,counts);for (k,&(b,e)) in ranges.iter().enumerate(){for (dst,r) in out.pos[out.starts[k]..out.starts[k+1]].iter_mut().zip(&records[b..e]){*dst=r.pos;}}
    let sort_extra=if rad{512*512*8+9*(512+c.edges/512)+8192}else{8*c.edges};
    let peak=(8*c.edges+sort_extra).max(8*c.edges+out.bytes()+ranges.capacity()*16);
    (out,peak)
}
fn hash_direct(c:&Corpus,floor:u64)->(Index,usize){
    let mut lookup:Map<u32,usize>=map();let mut keys=Vec::new();let mut freq:Vec<u64>=Vec::new();let mut counts=Vec::new();
    c.scan(|key,_,w|{let i=*lookup.entry(key).or_insert_with(||{let i=keys.len();keys.push(key);freq.push(0);counts.push(0);i});counts[i]+=1;freq[i]=freq[i].checked_add(w).unwrap();});
    let total=keys.len();let mut keepkeys=Vec::new();let mut keepfreq=Vec::new();let mut keepcounts=Vec::new();
    for i in 0..total{if freq[i]>=floor{let j=keepkeys.len();lookup.insert(keys[i],j);keepkeys.push(keys[i]);keepfreq.push(freq[i]);keepcounts.push(counts[i]);}else{lookup.insert(keys[i],usize::MAX);}}
    drop(keys);drop(freq);drop(counts);
    let mut out=Index::allocate(keepkeys,keepfreq,keepcounts);let mut next=out.starts[..out.keys.len()].to_vec();
    c.scan(|key,p,_|{let i=lookup[&key];if i!=usize::MAX{out.pos[next[i]]=p;next[i]+=1;}});
    // Hashbrown bucket/control allocation bound derived from load factor; allocator metadata omitted.
    let lookupbytes=(lookup.capacity()*8).div_ceil(7).next_power_of_two()*17;
    let peak=out.bytes()+next.capacity()*8+lookupbytes;
    (out,peak)
}
fn sorted_rust(c:&Corpus,floor:u64)->(Index,usize) {
    let mut records=Vec::with_capacity(c.edges);c.scan(|key,pos,_|records.push((key as u64)<<32|pos as u64));
    let scratch=radsort_u64::sort(&mut records);
    let mut keys=Vec::new();let mut freq=Vec::new();let mut counts=Vec::new();let mut ranges=Vec::new();let mut i=0;
    while i<records.len(){let key=(records[i]>>32)as u32;let begin=i;let mut f=0u64;while i<records.len()&&(records[i]>>32)as u32==key{f=f.checked_add(c.weight(records[i]as u32)).unwrap();i+=1;}
        if f>=floor{keys.push(key);freq.push(f);counts.push((i-begin)as u32);ranges.push((begin,i));}}
    let mut out=Index::allocate(keys,freq,counts);for (k,&(b,e)) in ranges.iter().enumerate(){for (dst,&r) in out.pos[out.starts[k]..out.starts[k+1]].iter_mut().zip(&records[b..e]){*dst=r as u32;}}
    let peak=(8*c.edges+scratch).max(8*c.edges+out.bytes()+ranges.capacity()*16);(out,peak)
}
fn bitset_rank(c:&Corpus,floor:u64)->(Index,usize){
    let universe=c.alphabet*c.alphabet;let mut bits=vec![0u64;universe.div_ceil(64)];c.scan(|key,_,_|bits[key as usize/64]|=1<<(key%64));
    let mut prefix=Vec::with_capacity(bits.len());let mut keys=Vec::new();let mut rank=0u32;
    for (b,&word) in bits.iter().enumerate(){prefix.push(rank);let mut rest=word;while rest!=0{let bit=rest.trailing_zeros();keys.push((b*64)as u32+bit);rank+=1;rest&=rest-1;}}
    let locate=|key:u32|->usize{let b=key as usize/64;let bit=key%64;(prefix[b]+(bits[b]&((1u64<<bit)-1)).count_ones())as usize};
    let mut freq=vec![0u64;keys.len()];let mut counts=vec![0u32;keys.len()];c.scan(|key,_,w|{let i=locate(key);counts[i]+=1;freq[i]=freq[i].checked_add(w).unwrap();});
    let mut remap=vec![usize::MAX;keys.len()];let mut keepkeys=Vec::new();let mut keepfreq=Vec::new();let mut keepcounts=Vec::new();
    for i in 0..keys.len(){if freq[i]>=floor{remap[i]=keepkeys.len();keepkeys.push(keys[i]);keepfreq.push(freq[i]);keepcounts.push(counts[i]);}}
    drop(keys);drop(freq);drop(counts);let mut out=Index::allocate(keepkeys,keepfreq,keepcounts);let mut next=out.starts[..out.keys.len()].to_vec();
    c.scan(|key,p,_|{let i=remap[locate(key)];if i!=usize::MAX{out.pos[next[i]]=p;next[i]+=1;}});
    let peak=out.bytes()+next.capacity()*8+bits.capacity()*8+prefix.capacity()*4+remap.capacity()*8;(out,peak)
}
fn rss()->(usize,usize){let text=std::fs::read_to_string("/proc/self/status").unwrap();let value=|label:&str|text.lines().find(|l|l.starts_with(label)).unwrap().split_whitespace().nth(1).unwrap().parse::<usize>().unwrap()*1024;(value("VmRSS:"),value("VmHWM:"))}
fn run(c:&Corpus,mode:&str,floor:u64)->Index{let (before_rss,before_hwm)=rss();let start=Instant::now();let (index,peak)=match mode{"radix"=>sorted(c,floor,false),"radsort"=>sorted(c,floor,true),"radsort-rust"=>sorted_rust(c,floor),"hash2"=>hash_direct(c,floor),"rank3"=>bitset_rank(c,floor),_=>panic!("unknown mode")};let ms=start.elapsed().as_secs_f64()*1000.;let (rss,hwm)=rss();let hash=index.hash();println!("{{\"mode\":\"{mode}\",\"alphabet\":{},\"words\":{},\"slots\":{},\"edges\":{},\"pairs\":{},\"retained_edges\":{},\"index_ms\":{ms:.3},\"before_index_rss_bytes\":{before_rss},\"before_index_hwm_bytes\":{before_hwm},\"rss_bytes\":{rss},\"hwm_bytes\":{hwm},\"modeled_index_peak_bytes\":{peak},\"final_index_bytes\":{},\"checksum\":\"{hash:016x}\",\"floor\":{floor}}}",c.alphabet,c.words,c.slots.len(),c.edges,index.keys.len(),index.pos.len(),index.bytes());index}
fn main(){let args:Vec<_>=std::env::args().collect();assert!(args.len()>=3,"MODE INPUT [FLOOR] [verify]");let mode=&args[1];let floor=args.get(3).map_or(2,|v|v.parse().unwrap());
    if mode=="selftest" {for words in [vec![],vec![("A".into(),1)],vec![("AAAAA\r\n".into(),1),("AAA".into(),3),("B\n".into(),0),("".into(),7),("\r\n".into(),2)],vec![("A".repeat(3000),5),("A\n".into(),1)]]{let c=Corpus::from_words(words);for floor in [1,2,9,100000]{for mode in ["radix","radsort","radsort-rust","hash2","rank3"]{run(&c,mode,floor).validate(&c,floor);}}}eprintln!("boundary/floor/weights/AA/long-word selftests passed");return;}
    let reader=BufReader::new(std::fs::File::open(&args[2]).unwrap());let mut words:Vec<(String,u64)>=Vec::new();let mut seen:Map<String,usize>=map();let mut reader=reader;loop{let mut line=String::new();if reader.read_line(&mut line).unwrap()==0{break;}if let Some(&i)=seen.get(&line){words[i].1+=1;}else{let i=words.len();seen.insert(line.clone(),i);words.push((line,1));}}drop(seen);
    let c=Corpus::from_words(words);let out=run(&c,mode,floor);if args.get(4).is_some_and(|s|s=="verify"){out.validate(&c,floor);eprintln!("oracle verified");}
}
