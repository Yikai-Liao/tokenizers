mod baseline;
mod local;
mod direct;
mod word_counts;
use std::{hint::black_box,time::Instant};
fn cpu()->f64 { let mut r=std::mem::MaybeUninit::<libc::rusage>::uninit();assert_eq!(unsafe{libc::getrusage(libc::RUSAGE_SELF,r.as_mut_ptr())},0);let r=unsafe{r.assume_init()};r.ru_utime.tv_sec as f64+r.ru_utime.tv_usec as f64*1e-6+r.ru_stime.tv_sec as f64+r.ru_stime.tv_usec as f64*1e-6 }
fn main(){
 tk_encode::parallelism::set_parallelism(true);
 println!("case,workers,rep,arm,wall_seconds,cpu_seconds,unique");
 for (case,n,distinct,work) in [("cheap-repeat",500_000,false,0),("cheap-distinct-zh",200_000,true,0),("expensive-short",4,true,8_000_000),("cheap-short",4,true,0)] {
  let inputs:Vec<String>=(0..n).map(|i|if distinct{format!("中文词{i}")}else{format!("word{}",i%64)}).collect();
  for workers in [1,4]{
   let pool=rayon::ThreadPoolBuilder::new().num_threads(workers).build().unwrap();let mut reference=None;
   for rep in 0..7 {
    let arms=if rep%2==0{[0,1,2]}else{[2,1,0]};
    for arm in arms {
     let process=|s:&str| -> tk_encode::Result<Vec<String>>{let mut x=17u64;for i in 0..work{x=black_box(x.wrapping_mul(6364136223846793005).wrapping_add(i));}black_box(x);Ok(vec![s.to_owned()])};
     let start=Instant::now();let cpu_start=cpu();
     let result=pool.install(||match arm{0=>baseline::count(inputs.iter(),&process),1=>local::count(inputs.iter(),&process),_=>direct::count(inputs.iter(),&process)}).unwrap();
     let elapsed=start.elapsed().as_secs_f64();let cpu_elapsed=cpu()-cpu_start;
     if let Some(expected)=&reference{assert_eq!(&result,expected);}else{reference=Some(result.clone());}
     println!("{case},{workers},{rep},{},{elapsed},{cpu_elapsed},{}",["baseline","local","direct"][arm],result.len());
    }
   }
  }
 }
}
