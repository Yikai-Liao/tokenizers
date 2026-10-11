use serde_json::json;
use std::{fs,mem,time::Instant};

#[inline(never)]
pub(super) fn freeze<T>(name: &str, values: Vec<T>) -> Box<[T]> {
    let len=values.len(); let capacity=values.capacity(); let old=values.as_ptr() as usize;
    let begin=Instant::now();
    let result=values.into_boxed_slice();
    let seconds=begin.elapsed().as_secs_f64();
    eprintln!("BPE_STORAGE {}",json!({"event":"freeze","name":name,"len":len,"capacity":capacity,
        "element_bytes":size_of::<T>(),"potential_spare_bytes":(capacity-len)*size_of::<T>(),
        "old_pointer":old,"new_pointer":result.as_ptr() as usize,"pointer_changed":old!=result.as_ptr() as usize,
        "seconds":seconds,"container_bytes":mem::size_of_val(&result),"boxed":true}));
    result
}

pub(super) fn phase(name: &str) {
    let status=fs::read_to_string("/proc/self/status").unwrap();
    let read=|key: &str| status.lines().find(|s|s.starts_with(key)).unwrap().split_whitespace().nth(1).unwrap().parse::<usize>().unwrap();
    eprintln!("BPE_STORAGE {}",json!({"event":"phase","name":name,"rss_kib":read("VmRSS:"),"hwm_kib":read("VmHWM:")}));
}

pub(super) fn growth<T>(name: &str, old: usize, values: &Vec<T>) {
    if old!=values.capacity() {eprintln!("BPE_STORAGE {}",json!({"event":"capacity_growth","name":name,"old_capacity":old,"capacity":values.capacity(),"len":values.len(),"element_bytes":size_of::<T>()}));}
}

pub(super) fn sizes(values: &[(&str,usize)]) {
    eprintln!("BPE_STORAGE {}",json!({"event":"sizes","values":values}));
}
