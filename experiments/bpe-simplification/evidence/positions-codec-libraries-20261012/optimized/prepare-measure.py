from pathlib import Path
r=Path(__file__).resolve().parent;old=Path('/tmp/bpe-positions-codec-libraries-results-20261012')
s=(old/'measure.py').read_text();start=s.index('cases=');end=s.index('\nfor case,arms in cases:');s=s[:start]+"cases=[(case,['hand','fmt-zero-push','fmt-tls-push','gimli-inline']) for case in ['en-whitespace-w6','en-bytelevel_regex-w6','zh-bytelevel_regex-w1']]"+s[end:];(r/'measure.py').write_text(s)
s=(old/'disassemble.py').read_text().replace("['hand','gimli','fmt-slice','fmt-array']","['hand','fmt-zero-push','fmt-tls-push','gimli-inline']");(r/'disassemble.py').write_text(s)
s=(old/'summarize.py').read_text().replace("['hand','gimli','fmt-slice','fmt-array']","['hand','fmt-zero-push','fmt-tls-push','gimli-inline']");(r/'summarize.py').write_text(s)
