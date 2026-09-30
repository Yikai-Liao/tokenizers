# Optimization commit screen: H → I

Single-call comparison under the same 512 MiB H→I screen. Each candidate ran once; treat timings as a screening result, not a repeatable performance estimate.

## Result

| Metric | H: weight-one-buckets | I: commit-direct-assemble | I − H |
|---|---:|---:|---:|
| Commit phase | 5.971 s | 6.876 s | +0.905 s (+15.2%) |
| Fused prepare | 8.142 s | 9.970 s | +1.828 s |
| Delta | 8.459 s | 10.364 s | +1.905 s |
| Train | 26.187 s | 28.538 s | +2.351 s (+9.0%) |
| Wall elapsed | 30.692 s | 32.776 s | +2.084 s (+6.8%) |
| Initialize / merge | 6.215 / 15.557 s | 5.708 / 18.611 s | −0.508 / +3.054 s |

I did not improve the primary commit metric in this screen. Initialization was about 0.5 s faster, while merge was about 3.1 s slower. Initial route/sort/group/install measured 930/2,094/272/737 ms for H and 849/1,745/369/711 ms for I. I reported a 2 MiB peak commit descriptor allocation.

## Equivalence and resource gates

Both runs used the same 536,870,289-byte input (SHA-256 `a0d40d4102cba1e933f25e5ccd17552d2eaebaec4ef770bc39d8e46e740b4d4f`), `none` pre-tokenizer, reference backend, vocab 50,000, minimum frequency 2, four initialization and four merge workers, atomic corpus enabled, and `parallel_u32_flat32` layout. Model SHA-256 matched (`d50fb836e342e1a301bb56d5ab53c0b79fae38f51a03051c6d6b02ab6cf862cd`); both produced 50,000 vocab entries and 29,243 merges. Symbols, edges, pairs, initial corpus/posting bytes, and weight-bucket bitmap bytes also matched. The one-weight bucket count differed by 13 (797,834 vs. 797,847 of 805,039; 99.105% in each).

Peak RSS was 4,648,488 KiB (H) and 4,648,300 KiB (I). I's minimum available memory was 3,735,662,592 bytes and sampled process swap was zero. Both passed the 512 MiB screen's resource and output checks.

## Provenance and artifacts

- H commit `00216d914186e42458d45e72276b13c700749c6a`; binary SHA-256 `97d890912374da73ab5f70f4c14ab6a296cfd395053b21817494302cf04f0cfd`.
- I commit `e3a1954ccdce4b80ea2865792088b1fe82fc352c`; binary SHA-256 `65ef01f4b64fdf7207addbb6f15801af600dd222c709fefd9fbcdc65af4ee9de`.
- Raw results, environment metadata, stdout, and stderr are in `optimization-commit-screen.{h,i}.{jsonl,environment.json,stdout,stderr}` in this directory. The combined model signature is in `optimization-commit-screen.signature.json`.
