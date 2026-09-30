//! Streaming upper bounds before training; no corpus replication or feed tables.
use std::collections::HashSet;
use std::io::{BufRead, BufReader, Write};

fn main() -> std::io::Result<()> {
    let args: Vec<_> = std::env::args().collect();
    let target: usize = args[3].parse().unwrap();
    let input = BufReader::new(std::fs::File::open(&args[1])?);
    let mut output = std::io::BufWriter::new(std::fs::File::create(&args[2])?);
    let mut bytes = 0;
    let mut symbols = 0;
    let mut lines = 0;
    let mut edges = 0;
    let mut alphabet = HashSet::new();
    let mut pairs = HashSet::new();
    for line in input.split(b'\n') {
        let mut line = line?;
        line.push(b'\n');
        if bytes + line.len() > target { break; }
        bytes += line.len();
        output.write_all(&line)?;
        let text = std::str::from_utf8(&line).unwrap();
        let mut prior = None;
        let mut chars = 0_usize;
        for c in text.chars() {
            alphabet.insert(c);
            if let Some(a) = prior { pairs.insert((a,c)); }
            prior = Some(c);
            chars += 1;
        }
        symbols += chars;
        edges += chars.saturating_sub(1);
        lines += 1;
    }
    output.flush()?;
    println!("{{\"bytes\":{bytes},\"lines_upper\":{lines},\"slots_upper\":{},\"edges_upper\":{edges},\"pairs_upper\":{},\"alphabet\":{}}}",symbols+lines+1,pairs.len(),alphabet.len());
    Ok(())
}
