"""Instrument a disposable source checkout; no trainer commit is modified."""
from pathlib import Path
from tree_sitter import Language,Parser
import tree_sitter_rust
PARSER=Parser(Language(tree_sitter_rust.language()))
BENCH=None

OBSERVER = r'''
//! Benchmark-copy-only phase boundaries. This file is never committed to the PR series.
use std::time::{SystemTime, UNIX_EPOCH};

pub(super) struct Phase(&'static str);

pub(super) fn event(name: &str, action: &str) {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    assert_eq!(unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) },0);
    let peak = unsafe { usage.assume_init() }.ru_maxrss;
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    let field = |name: &str| status.lines().find(|s| s.starts_with(name))
        .and_then(|s|s.split_whitespace().nth(1)).unwrap().parse::<u64>().unwrap();
    let ns = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
    eprintln!("BPE_REVIEW_PHASE {{\"phase\":\"{name}\",\"action\":\"{action}\",\"unix_ns\":{ns},\"rss_kib\":{},\"hwm_kib\":{},\"swap_kib\":{},\"ru_maxrss_kib\":{peak}}}",
        field("VmRSS:"),field("VmHWM:"),field("VmSwap:"));
}

impl Phase {
    pub(super) fn new(name: &'static str) -> Self {
        event(name, "begin");
        Self(name)
    }
}

impl Drop for Phase {
    fn drop(&mut self) { event(self.0, "end"); }
}
'''

def owner_of(node, data):
    while node:
        if node.type == 'impl_item':
            field = node.child_by_field_name('type')
            return data[field.start_byte:field.end_byte].decode() if field else None
        node = node.parent
    return None

def instrument():
    bpe = BENCH / 'tokenizers/tk-train/src/trainers/bpe'
    for path in bpe.glob('*.rs'):
        if path.name in ('word.rs', 'parity_trainer.rs', 'tests.rs', 'review_phase.rs'):
            continue
        data = path.read_bytes()
        root = PARSER.parse(data).root_node
        edits = []

        def walk(node):
            if node.type == 'function_item':
                name = node.child_by_field_name('name')
                name = data[name.start_byte:name.end_byte].decode()
                owner = owner_of(node, data)
                phase = None
                if name in ('compute_alphabet', 'select_alphabet'):
                    phase = 'alphabet_selection'
                elif name == 'initialize' and owner == 'Vocabulary':
                    phase = 'vocabulary_initialization'
                elif name == 'build' and owner == 'InitialTokenIds':
                    phase = 'symbol_id_tables'
                elif name == 'tokenize_words' or (name == 'new' and owner == 'Corpus') or name == 'materialize':
                    phase = 'corpus_materialization'
                elif name == 'build' and owner and owner.split('<', 1)[0] == 'CorpusPlan':
                    phase = 'corpus_layout'
                elif name == 'count_pairs' or (name == 'build' and owner == 'PairIndex'):
                    phase = 'initial_pair_count'
                elif name == 'retire' and owner == 'PairIndex':
                    phase = 'index_release'
                elif name in ('into_parts', 'into_vocab') and owner == 'Vocabulary':
                    phase = 'model_publication'
                path_to_observer = 'review_phase' if path.name == 'mod.rs' else 'super::review_phase'
                body = node.child_by_field_name('body')
                if phase and body:
                    edits.append((body.start_byte+1, f'\nlet _review_phase = {path_to_observer}::Phase::new("{phase}");\n'.encode()))
                if name == 'update_progress' and body and path.name == 'mod.rs':
                    edits.append((body.start_byte+1, b'\nif message == "Compute merges" { review_phase::event("merge_loop", "begin"); }\n'))
                if name == 'finalize_progress' and body and path.name == 'mod.rs':
                    edits.append((body.start_byte+1, b'\nif stage == "Compute merges" { review_phase::event("merge_loop", "end"); }\n'))
            for child in node.named_children:
                walk(child)

        walk(root)
        for at, value in sorted(edits, reverse=True):
            data = data[:at] + value + data[at:]
        text = data.decode()
        if path.name == 'mod.rs':
            if text.startswith('#!'):
                at = text.index('\n')+1
                text = text[:at]+'mod review_phase;\n'+text[at:]
            else:
                text = 'mod review_phase;\n'+text
            # Old trainers publish parts directly in the coordinator.
            for marker in ['        let vocab: Vocab =', '        let merges: Merges =']:
                if marker in text and 'Vocabulary' not in text:
                    text = text.replace(marker, '        let _review_output = review_phase::Phase::new("model_publication");\n'+marker, 1)
                    break
            text = text.replace('            drop(corpus);', '            { let _review_release = review_phase::Phase::new("corpus_release"); drop(corpus); }')
        path.write_text(text)
    (bpe / 'review_phase.rs').write_text(OBSERVER)
    manifest=BENCH/'tokenizers/tk-train/Cargo.toml'
    manifest.write_text(manifest.read_text().replace('[dependencies]', '[dependencies]\nlibc = \"0.2\"',1))
