//! Training-stage work counters and a renderer with an elapsed-time heartbeat.
//! Algorithms report completed chunks. Rendering never runs in a token loop.
use std::sync::{
    Arc,
    atomic::{AtomicU64, Ordering},
    mpsc,
};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use tk_encode::{Result, utils::progress::ProgressFormat};

pub(crate) struct TrainingProgress {
    renderer: Option<(mpsc::Sender<Message>, JoinHandle<()>)>,
}
enum Message {
    Stage(Arc<Stage>),
    Stop,
}
enum Work {
    Known(u64),
    Merges(u64),
}
struct Stage {
    name: &'static str,
    work: Work,
    completed: AtomicU64,
    vocabulary: AtomicU64,
    started: Instant,
}
#[derive(Clone, Default)]
pub(crate) struct WorkProgress(Option<Arc<Stage>>);
impl TrainingProgress {
    pub(crate) fn new(enabled: bool, format: ProgressFormat) -> Result<Self> {
        if !enabled || format == ProgressFormat::Silent {
            return Ok(Self { renderer: None });
        }
        let (sender, receiver) = mpsc::channel();
        let renderer = std::thread::Builder::new()
            .name("tokenizer-progress".into())
            .spawn(move || render(receiver, format))?;
        Ok(Self {
            renderer: Some((sender, renderer)),
        })
    }
    pub(crate) fn stage(&self, name: &'static str, total: usize) -> WorkProgress {
        self.start(name, Work::Known(total as u64), 0)
    }
    pub(crate) fn merges(&self, target: usize, vocabulary: usize) -> WorkProgress {
        self.start(
            "Compute merges",
            Work::Merges(target as u64),
            vocabulary as u64,
        )
    }
    fn start(&self, name: &'static str, work: Work, vocabulary: u64) -> WorkProgress {
        let Some((sender, _)) = &self.renderer else {
            return WorkProgress::default();
        };
        let stage = Arc::new(Stage {
            name,
            work,
            completed: AtomicU64::new(0),
            vocabulary: AtomicU64::new(vocabulary),
            started: Instant::now(),
        });
        // The renderer is an observer. A closed output thread does not invalidate training.
        if sender.send(Message::Stage(Arc::clone(&stage))).is_err() {
            return WorkProgress::default();
        }
        WorkProgress(Some(stage))
    }
}
impl WorkProgress {
    pub(crate) fn complete(&self, amount: usize) {
        if let Some(stage) = &self.0 {
            stage.completed.fetch_add(amount as u64, Ordering::Relaxed);
        }
    }
    pub(crate) fn learned(&self, rules: usize, vocabulary: usize) {
        if let Some(stage) = &self.0 {
            stage.completed.store(rules as u64, Ordering::Relaxed);
            stage.vocabulary.store(vocabulary as u64, Ordering::Relaxed);
        }
    }
}
impl Drop for TrainingProgress {
    fn drop(&mut self) {
        if let Some((sender, renderer)) = self.renderer.take() {
            let _ = sender.send(Message::Stop);
            let _ = renderer.join();
        }
    }
}
// The renderer samples coarse counters, so estimating rate adds no clocks or
// synchronization to token processing. Smooth over two seconds; idle samples
// let the estimate reflect a slow chunk rather than keeping a stale fast ETA.
struct RateEstimate {
    observed: Instant,
    completed: u64,
    per_second: Option<f64>,
}
impl RateEstimate {
    fn new(started: Instant) -> Self {
        Self {
            observed: started,
            completed: 0,
            per_second: None,
        }
    }
    fn remaining(&mut self, completed: u64, total: u64) -> Option<f64> {
        let now = Instant::now();
        let interval = now.duration_since(self.observed);
        // An immediate stage message can already contain completed work. Wait
        // for a heartbeat-sized sample before seeding the smoothed rate.
        if interval < Duration::from_millis(250) {
            return None;
        }
        let seconds = interval.as_secs_f64();
        if seconds > 0.0 {
            let rate = (completed - self.completed) as f64 / seconds;
            self.per_second = match self.per_second {
                Some(previous) => {
                    let alpha = 1.0 - (-seconds / 2.0).exp();
                    Some(previous + alpha * (rate - previous))
                }
                None if completed > self.completed => Some(rate),
                None => None,
            };
        }
        self.observed = now;
        self.completed = completed;
        self.per_second
            .filter(|&rate| rate > 0.0 && completed < total)
            .map(|rate| (total - completed) as f64 / rate)
    }
}
fn render(receiver: mpsc::Receiver<Message>, format: ProgressFormat) {
    let mut current: Option<Arc<Stage>> = None;
    let mut rate = RateEstimate::new(Instant::now());
    #[cfg(feature = "progressbar")]
    let bar = if format == ProgressFormat::Indicatif {
        let bar = indicatif::ProgressBar::new(0);
        bar.set_style(
            indicatif::ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {msg} {wide_bar} {pos}/{len}")
                .expect("the training progress template is a constant"),
        );
        Some(bar)
    } else {
        None
    };
    let show = |stage: &Stage, finished: bool, rate: &mut RateEstimate| {
        let completed = stage.completed.load(Ordering::Relaxed);
        let vocabulary = stage.vocabulary.load(Ordering::Relaxed);
        let elapsed = stage.started.elapsed().as_secs_f64();
        let eta = match stage.work {
            Work::Known(total) => rate.remaining(completed, total),
            Work::Merges(_) => None,
        };
        if format == ProgressFormat::JsonLines {
            match stage.work {
                Work::Known(total) => {
                    eprintln!(
                        "{}",
                        serde_json::json!({"stage":stage.name,"current":completed,"total":total,
                        "elapsed_seconds":elapsed,"stage_eta_seconds":eta,"finished":finished})
                    );
                }
                Work::Merges(target) => eprintln!(
                    "{}",
                    serde_json::json!({"stage":stage.name,"learned_rules":completed,
                    "vocabulary_size":vocabulary,"vocabulary_target":target,"elapsed_seconds":elapsed,"finished":finished})
                ),
            }
        }
        #[cfg(feature = "progressbar")]
        if let Some(bar) = &bar {
            match stage.work {
                Work::Known(total) => {
                    bar.set_length(total);
                    bar.set_position(completed);
                    bar.set_message(match eta {
                        Some(seconds) => format!("{} (stage ETA {:.0}s)", stage.name, seconds),
                        None => stage.name.to_owned(),
                    });
                }
                Work::Merges(target) => {
                    bar.set_length(0);
                    bar.set_position(0);
                    bar.set_message(format!(
                        "{}: {} rules, vocabulary {}/{}",
                        stage.name, completed, vocabulary, target
                    ));
                }
            }
            bar.tick();
        }
    };
    loop {
        match receiver.recv_timeout(Duration::from_millis(250)) {
            Ok(Message::Stage(stage)) => {
                if let Some(previous) = current.take() {
                    show(&previous, true, &mut rate);
                }
                #[cfg(feature = "progressbar")]
                if let Some(bar) = &bar {
                    bar.reset_elapsed();
                }
                rate = RateEstimate::new(stage.started);
                show(&stage, false, &mut rate);
                current = Some(stage);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {
                if let Some(stage) = &current {
                    show(stage, false, &mut rate);
                }
            }
            Ok(Message::Stop) | Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    if let Some(stage) = current {
        show(&stage, true, &mut rate);
    }
    #[cfg(feature = "progressbar")]
    if let Some(bar) = bar {
        bar.finish_and_clear();
    }
}
