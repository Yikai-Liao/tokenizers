from pathlib import Path
old=Path('/tmp/bpe-positions-encapsulation-final-results-20261012');r=Path(__file__).resolve().parent
s=(old/'measure-overhead.py').read_text();start=s.index('cases=');end=s.index('\nfor case,arms in cases:')
s=s[:start]+"cases=[(case,['hand','gimli','fmt-slice','fmt-array']) for case in ['en-whitespace-w6','en-bytelevel_regex-w6','zh-bytelevel_regex-w1','zh-whitespace-w1']]"+s[end:]
s=s.replace("ref=r/'normal'/f'{case}-r0-a'","ref=Path('/tmp/bpe-positions-encapsulation-final-results-20261012')/'normal'/f'{case}-r0-a'")
s=s.replace("r/'overhead'","r/'normal'").replace("r/'overhead.json'","r/'normal.json'")
s=s.replace("'Overhead comparisons complete'","'Library comparisons complete'")
(r/'measure.py').write_text(s)
