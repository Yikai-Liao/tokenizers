use serde::Deserialize;
pub type Error = Box<dyn std::error::Error + Send + Sync>;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Job {
    pub protocol_version: u32,
    pub attempt_id: String,
    pub build_id: String,
    pub input_id: String,
    pub input: String,
    pub output: String,
    pub workers: usize,
    pub pretokenizer: String,
    pub trainer: TrainerConfig,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainerConfig {
    pub vocab_size: usize,
    pub min_frequency: u64,
    pub prefix: Option<String>,
    pub suffix: Option<String>,
    pub max_token_length: Option<usize>,
}

pub fn cpu_seconds() -> Result<f64, Error> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) } != 0 {
        return Err(std::io::Error::last_os_error().into());
    }
    let usage = unsafe { usage.assume_init() };
    Ok((usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) as f64
        + (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) as f64 / 1e6)
}
