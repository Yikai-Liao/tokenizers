"""Copy the fixed source and instrument diagnostics outside production builds."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

BPE = Path('tokenizers/tk-train/src/trainers/bpe')


def replace(path, old, new):
    text = path.read_text()
    assert text.count(old) == 1, (path, old, text.count(old))
    path.write_text(text.replace(old, new))


def prepare(root, dest, diagnostic):
    assert not dest.exists(), dest
    for name in ['tk-train', 'tk-encode', 'tk-convert', 'tk-serialize', 'bitcannon']:
        shutil.copytree(root / 'tokenizers' / name, dest / 'tokenizers' / name,
                        ignore=shutil.ignore_patterns('target'))
    runner = Path('experiments/bpe-simplification/runner')
    shutil.copytree(root / runner, dest / runner, ignore=shutil.ignore_patterns('target'))
    (dest / 'baseline.json').write_text(json.dumps({
        'sha': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'diff': subprocess.check_output(['git', '-C', str(root), 'diff', 'HEAD', '--', 'tokenizers'], text=True),
        'diagnostic': diagnostic}, indent=2))
    if diagnostic:
        corpus = dest / BPE / 'corpus.rs'
        merge = dest / BPE / 'merge.rs'
        with corpus.open('a') as stream:
            stream.write('''
// Diagnostic-only, local to a fresh reader. No shared counter in the hot loop.
#[derive(Default)]
struct LookupStats {
    input: u64, valid: u64, hits: u64, first: u64, expired: u64,
    single: u64, unit: u64, binary: u64, comparisons: u64,
    extra_comparisons: u64, first_index_sum: u64, suffix_sum: u64,
    suffix_comparisons: u64, suffix_queries: u64, first_position: usize, last_position: usize, duplicates: u64,
    jumps: [u64; 4], index: Option<usize>, runs: usize,
}
impl Drop for LookupStats {
    fn drop(&mut self) {
        eprintln!("LOOKUP_DIAG {}", serde_json::json!({"kind":"fresh", "input":self.input,
            "valid":self.valid,"hits":self.hits,"first":self.first,"expired":self.expired,
            "single":self.single,"unit":self.unit,"binary":self.binary,
            "comparisons":self.comparisons,"extra_comparisons":self.extra_comparisons,
            "first_index_sum":self.first_index_sum,"suffix_sum":self.suffix_sum,
            "jumps":self.jumps,"runs":self.runs,"suffix_comparisons":self.suffix_comparisons,
            "suffix_queries":self.suffix_queries,"first_position":self.first_position,
            "last_position":self.last_position,"duplicates":self.duplicates}));
    }
}
''')
        replace(corpus, 'let matcher = self.fixed_matcher(pair);',
                'let matcher = self.fixed_matcher(pair);\n        let mut stats = LookupStats::default();\n        stats.runs = self.weight_runs.len();')
        replace(corpus, 'let matched = matcher(p)?;', '''stats.input += 1;
            let matched = matcher(p)?;
            if stats.valid == 0 { stats.first_position = p; }
            else if stats.last_position == p { stats.duplicates += 1; }
            stats.last_position = p;
            stats.valid += 1;''')
        replace(corpus, 'Some(lookup) if p < lookup.end => lookup,\n                _ => self.weight(p),', '''Some(lookup) if p < lookup.end => { stats.hits += 1; lookup },
                _ => {
                    if stats.index.is_none() { stats.first += 1; } else { stats.expired += 1; }
                    let previous = stats.index;
                    let lookup = self.diagnostic_weight(p, &mut stats);
                    if let Some(old) = previous {
                        let jump = stats.index.unwrap() - old;
                        stats.jumps[match jump { 1 => 0, 2..=4 => 1, 5..=16 => 2, _ => 3 }] += 1;
                    } else { stats.first_index_sum += stats.index.unwrap() as u64; }

                    lookup
                },''')
        replace(corpus, '    fn word(&self, position: usize) -> usize {', '''    fn diagnostic_weight(&self, position: usize, stats: &mut LookupStats) -> WeightLookup {
        let single = self.weight_runs.len() == 1;
        let unit = !single && self.unit_weight_range.as_ref().is_some_and(|r| r.contains(&position));
        let i = if single { stats.single += 1; 0 } else if unit {
            stats.unit += 1;
            // Extra lookup exists solely to diagnose the existing unit fast path.
            self.weight_runs.partition_point(|run| {
                stats.extra_comparisons += 1; run.start <= position
            }) - 1
        } else {
            stats.binary += 1;
            self.weight_runs.partition_point(|run| {
                stats.comparisons += 1; run.start <= position
            }) - 1
        };
        if !single && !unit && let Some(old) = stats.index {
            stats.suffix_queries += 1;
            stats.suffix_sum += (self.weight_runs.len() - old - 1) as u64;
            let found = old + self.weight_runs[old + 1..].partition_point(|run| {
                stats.suffix_comparisons += 1; run.start <= position
            });
            assert_eq!(found, i);
        }
        stats.index = Some(i);
        WeightLookup { weight: self.weight_runs[i].weight,
            end: self.weight_runs.get(i + 1).map_or(self.len(), |run| run.start) }
    }

    fn word(&self, position: usize) -> usize {''')
        replace(merge, '                drop(positions);', '''                eprintln!("LOOKUP_DIAG {}", serde_json::json!({"kind":"aa_selection",
                    "input":original_positions_len,"accepted":accepted_starts.len()}));
                drop(positions);''')
        replace(merge, '        scan(&mut neighbors, &mut starts)?;', '''        scan(&mut neighbors, &mut starts)?;
        eprintln!("LOOKUP_DIAG {}", serde_json::json!({"kind":"accepted","matches":starts.len()}));''')
        replace(merge, '                    for &word in words {\n                        self.scan_word(&rule, word, limit, neighbors, starts)?;\n                    }', '''                    let mut reuse = [0u64; 8];
                    for &word in words {
                        self.scan_word(&rule, word, limit, neighbors, starts, &mut reuse)?;
                    }
                    eprintln!("LOOKUP_DIAG {}", serde_json::json!({"kind":"reuse",
                        "whole":self.whole_words(),"words":reuse[0],"zero":reuse[1],
                        "one":reuse[2],"multi":reuse[3],"single":reuse[4],"unit":reuse[5],
                        "binary":reuse[6],"comparisons":reuse[7]}));''')
        replace(merge, '        starts: &mut SmallVec<[usize; 2]>,\n    ) -> Result<()> {',
                '        starts: &mut SmallVec<[usize; 2]>,\n        reuse: &mut [u64; 8],\n    ) -> Result<()> {')
        replace(merge, '        let weight = self.weight(range.start).weight;', '''        let before = starts.len();
        let mut stats = super::LookupStats::default();
        let weight = self.diagnostic_weight(range.start, &mut stats).weight;
        reuse[4] += stats.single; reuse[5] += stats.unit;
        reuse[6] += stats.binary; reuse[7] += stats.comparisons;
        // Suppress fresh-reader reporting for this temporary reuse probe.
        std::mem::forget(stats);''')
        replace(merge, '        Ok(())\n    }\n}\n\nenum FreshBatch', '''        reuse[0] += 1;
        reuse[match starts.len() - before { 0 => 1, 1 => 2, _ => 3 }] += 1;
        Ok(())
    }
}

enum FreshBatch''')
        replace(merge, '                for (matched, weight) in self.corpus.fresh_matches(&mut positions, pair) {', '''                let decoder_bytes = std::mem::size_of_val(&positions);
                let reader = self.corpus.fresh_matches(&mut positions, pair);
                eprintln!("LOOKUP_DIAG {}", serde_json::json!({"kind":"layout",
                    "aa":pair.0==pair.1,"reader":std::mem::size_of_val(&reader),
                    "decoder":decoder_bytes,
                    "cache":std::mem::size_of::<super::WeightLookup>(),
                    "option_cache":std::mem::size_of::<Option<super::WeightLookup>>() }));
                for (matched, weight) in reader {''')
    (dest / 'source-hashes.json').write_text(json.dumps({str(p.relative_to(dest)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in dest.rglob('*') if p.is_file()}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path); p.add_argument('dest', type=Path)
    p.add_argument('--diagnostic', action='store_true')
    a = p.parse_args(); prepare(a.root.resolve(), a.dest.resolve(), a.diagnostic)
