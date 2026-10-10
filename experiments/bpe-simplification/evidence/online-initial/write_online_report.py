"""Generate the experiment report from raw results, then archive small evidence."""
from pathlib import Path
import json, statistics, shutil, hashlib
OUT = Path('/root/code/tokenizers-simplification-results/online-initial')
REPO = Path('/root/code/tokenizers-workspaces/bpe-simplification')
DEST = REPO / 'experiments/bpe-simplification'
rows = [json.loads(l) for l in (OUT / 'runs.jsonl').read_text().splitlines()]
small = json.loads((OUT / 'small-ablation-summary.json').read_text())
selected = json.loads((OUT / 'selected-summary.json').read_text())
assert len(selected) == 8, 'Contrast is incomplete'
assert all(r['valid'] for r in rows if r['kind'] in ('selected-phases', 'phases-pipeline', 'prepare-ablation', 'prepare-detail', 'phases'))
parts = []
def add(s): parts.append(s.strip() + '\n')
def table(headers, items):
    add('| ' + ' | '.join(headers) + ' |\n|' + '|'.join('---' for _ in headers) + '|\n' + '\n'.join('| ' + ' | '.join(map(str,r)) + ' |' for r in items))
def value(r, stage, key='cpu_seconds'):
    return next(p[key] for p in r['phases'] if p['name']==stage)
def find(case, kind, arm):
    rr=[r for r in rows if r['valid'] and (r['case'],r['kind'],r['arm'])==(case,kind,arm)]
    assert len(rr)==1, (case,kind,arm,len(rr))
    return rr[0]
add('''# 在线初始压缩与 prepare 优化：逐项证据

本实验选择 **整词分块＋生产者内压缩，不加 wave 屏障；再加入规则索引和有序写入**。规则索引是 prepare 的主要收益来源。有序写入仅增加 3 行，在已有索引时，两种规模的 prepare CPU 均进一步降低约 2–3%，因此一起保留。候选位于独立工作树，生产 Rust 尚未合入原 2100 行分支。

完整候选为 **2185 行生产／797 行测试**，符合 2200／800 预算；比原版本多 85／17 行。诊断观察器和 runner 不进入生产补丁。下列数字有不同计时边界：初始阶段峰值、整个训练峰值、prepare CPU 和完整训练墙钟分别报告。''')
add('''## 先判断初始阶段哪些组合有价值

四个实验臂共用同一二进制、相同的整词分块、临时压缩片段和最终编码。raw-all 先收集全部块再压缩；online-all 在每个生产者结束时立即压缩并释放原始列表；wave 在每四个块后加入屏障。每个正常块上限为 2²⁴ 个 resident slot；超过上限的单个词独立处理。频率准入仍使用全局完整计数。

表内为两轮观测的中位数；CPU 为进程 CPU，峰值单位 GiB。原 2100 行版本到 raw-all 的差额属于分块与片段表示这一组变更，未进一步拆成单个机制的因果贡献。''')
items=[]
for case in ('zh-256MiB','zh-512MiB'):
    for arm in ('baseline','raw-all','online-all','raw-wave','online-wave'):
        rr=[r for r in rows if r['valid'] and (r['case'],r['kind'],r['arm'])==(case,'diagnostic',arm)]
        assert len(rr)==2
        med=lambda f:statistics.median(f(r) for r in rr)
        items.append([case,arm,f"{med(lambda r:r['initial_phase'][0]['wall_seconds']):.3f}",f"{med(lambda r:r['initial_phase'][0]['cpu_seconds']):.3f}",f"{med(lambda r:r['initial_phase'][0]['new_process_hwm_kib'])/1048576:.3f}",f"{med(lambda r:r['metrics']['process_hwm_kib_before_validation'])/1048576:.3f}"])
table(['输入','臂','初始 wall s','初始 CPU s','初始进程峰值 GiB','训练峰值 GiB'],items)
initial_effects=[]
for case in ('zh-256MiB','zh-512MiB'):
    measured={}
    for arm in ('baseline','raw-all','online-all','online-wave'):
        rr=[r for r in rows if r['valid'] and (r['case'],r['kind'],r['arm'])==(case,'diagnostic',arm)]
        measured[arm]={k:statistics.median(r['initial_phase'][0][k] for r in rr) for k in ('wall_seconds','cpu_seconds','new_process_hwm_kib')}
    for label,before,after in [('分块与片段表示组','baseline','raw-all'),('已有分块，再加在线压缩','raw-all','online-all'),('已有在线压缩，再加 wave','online-all','online-wave')]:
        initial_effects.append([case,label,*[f"{100*(measured[after][k]/measured[before][k]-1):+.1f}%" for k in ('wall_seconds','cpu_seconds','new_process_hwm_kib')]])
table(['输入','条件变化','初始 wall 变化','初始 CPU 变化','初始峰值变化'],initial_effects)
add('''在线压缩在无 wave 时显著降低初始峰值；wave 对 raw 路径也有效。但 online-all 与 online-wave 的初始峰值几乎一致：在线释放已经控制了同时驻留的原始块。wave 的额外调度屏障因此省略，整个训练峰值的微小变动不作为另加机制的依据。

1 GiB 的原 2100 行诊断臂触发了自身 swap 保护（16 KiB），该次完整训练时间与峰值对比排除；其已完成的初始阶段仅保留为部分日志。1 GiB 旧矩阵只完成部分轮次；旧计划中的无插桩对照被用户要求的阶段分析取代，未完成项不补写为结果。''')
add('''## 把此前的训练差距拆到阶段

此前无插桩 main 与仅初始段插桩候选的跨轮次墙钟差约 23%，不能直接分摊。下表使用随后同一种十阶段观察器、同一 1 GiB 样本的正序配对；按用户要求取消了反序。main 为复杂实现，初始候选为 2165 行在线压缩版本，尚未加入本轮 prepare 优化。''')
a=find('zh-1024MiB','phases','main');b=find('zh-1024MiB','phases','candidate')
items=[]
for x,y in zip(a['phases'],b['phases']):
    assert x['name']==y['name']
    items.append([x['name'],f"{x['wall_seconds']:.3f}",f"{y['wall_seconds']:.3f}",f"{y['wall_seconds']-x['wall_seconds']:+.3f}",f"{x['cpu_seconds']:.3f}",f"{y['cpu_seconds']:.3f}",f"{y['cpu_seconds']-x['cpu_seconds']:+.3f}"])
for label,key,unit in [('未覆盖间隙','wall_seconds','train_seconds'),('未覆盖 CPU','cpu_seconds','train_cpu_seconds')]:
    av=a['metrics'][unit]-sum(p[key] for p in a['phases']);bv=b['metrics'][unit]-sum(p[key] for p in b['phases'])
    items.append([label,*( [f'{av:.3f}',f'{bv:.3f}',f'{bv-av:+.3f}','—','—','—'] if key=='wall_seconds' else ['—','—','—',f'{av:.3f}',f'{bv:.3f}',f'{bv-av:+.3f}'])])
table(['阶段','main wall','候选 wall','Δ wall s','main CPU','候选 CPU','Δ CPU s'],items)
add(f"同观察器完整训练 wall {a['metrics']['train_seconds']:.3f} → {b['metrics']['train_seconds']:.3f} s（+{100*(b['metrics']['train_seconds']/a['metrics']['train_seconds']-1):.1f}%），CPU {a['metrics']['train_cpu_seconds']:.3f} → {b['metrics']['train_cpu_seconds']:.3f} s。prepare 是最大增加项；初始索引阶段反而更快。未覆盖间隙包含协调和观察器等，不能全部归给算法。")
add('''## prepare 核心差异与细分插桩

main 使用可复用的 `SelectedRuleIndex`：head／tail 直接按 token ID 索引，只有共享端点才回退到精确 pair 哈希。main 还把多个小规则放入同一作业，并使用可复用的窄位置 scratch。初始候选在扫描中查询普通 pair 哈希，并通过 `Builder::push` 重复验证有序性；两者的作业组织和事件构建方式也不同。

这里的主要差距出现在 merge 前准备的扫描／收集，并非 radsort。main 初始索引的通用排序路径与 prepare 是不同阶段；ByteLevel 的有限字母表另有专用索引路径。

细分观察器用 joined 进程 CPU 计整个并行段，用线程 CPU 计每个工作段；没有逐 occurrence 的时钟或原子累加。工作线程墙钟会重叠，不能相加当作阶段墙钟，线程子阶段也已包含于 worker total。''')
a=find('zh-1024MiB','prepare-detail','main');b=find('zh-1024MiB','prepare-detail','candidate')
da={p['name']:p for p in a['prepare_detail']['stages']};db={p['name']:p for p in b['prepare_detail']['stages']}
table(['细分','CPU 时钟','main CPU s','候选 CPU s','Δ CPU s'],[[key,da[key]['cpu_clock'],f"{da[key]['cpu_seconds']:.3f}",f"{db[key]['cpu_seconds']:.3f}",f"{db[key]['cpu_seconds']-da[key]['cpu_seconds']:+.3f}"] for key in ('selected_index','ordinary_joined','aa_joined','worker_total','worker_setup','scan_collect','finish_encode','postjoin_gather')])
extra=value(b,'prepare')-value(a,'prepare');scan=db['scan_collect']['cpu_seconds']-da['scan_collect']['cpu_seconds']
add(f"这一对 prepare CPU {value(a,'prepare'):.3f} → {value(b,'prepare'):.3f} s；扫描／收集增加 {scan:.3f} s，约占 prepare 增量 {100*scan/extra:.1f}%。两者都访问 591,821,980 个位置，匹配 568,585,911 次；差异来自每次访问成本及作业组织，而不是少做了输入工作。main 普通作业 20,261 个，初始候选 49,950 个；本轮没有单独消融小规则分组，因此不把它的贡献写成已测百分比。worker setup 的放置边界也不完全一致，主判断使用 scan 与 joined prepare。")
add('''## 各项优化单独及叠加收益

C＝2165 行在线初始候选，O＝仅有序写入，L＝仅规则索引，OL＝二者。每规模每臂一次，顺序 C→O→L→OL；属于探索结果。表中正数表示 CPU 节省。条件收益以被加优化前的那个版本为分母。''')
table(['输入','C prepare CPU','O','L','OL','C→O','C→L','C→OL','L→OL','O→OL'],[[r['case'],*[f"{r['metrics'][x]['prepare_cpu']:.3f}" for x in ('control','ordered','lookup','combined')],*[f"{r['effects']['prepare_cpu'][x]:.3f}" for x in ('ordered_gain','lookup_gain','combined_gain','ordered_added_to_lookup','lookup_added_to_ordered')]] for r in small])
table(['输入','O 单独收益','L 单独收益','组合收益','已有 L 再加 O','已有 O 再加 L','交互 CPU s'],[[r['case'],*[f"{r['effects']['prepare_cpu'][x]:.1f}%" for x in ('ordered_gain_percent','lookup_gain_percent','combined_gain_percent')],f"{100*r['effects']['prepare_cpu']['ordered_added_to_lookup']/r['metrics']['lookup']['prepare_cpu']:.1f}%",f"{100*r['effects']['prepare_cpu']['lookup_added_to_ordered']/r['metrics']['ordered']['prepare_cpu']:.1f}%",f"{r['effects']['prepare_cpu']['interaction_gain']:+.3f}"] for r in small])
add('''交互项定义为 O＋L−C−OL；非零意味着两项收益不能简单相加。本轮 1 GiB 控制臂的 CPU 偏高，O 单独收益从 512 MiB 的 0.4% 跳到 16.2%，因此不能将 16.2% 认定为稳定因果收益。L 的主力判断结合了更小规模、扫描细分与两种条件对比；O 的保留理由是已有 L 后约 2–3% 的同向增量和仅 3 行的代价，不是单独臂的大数字。

L 增加 17 行生产／4 行测试：只对唯一 head 直接查 counterpart／replacement；共享 head 使用原精确哈希，tail gate 先拒绝不可能的左侧命中，保留边界与别名语义。O 仅在单调 snapshot 写入的两个调用处使用有序追加；通用追加与最终 encoder 继续验证。组合没有新增公共模式或配置。''')
table(['输入','版本','训练 wall s','训练 CPU s','prepare wall s','扫描线程 CPU s','训练峰值 GiB'],[[r['case'],a,*[f"{r['metrics'][a][k]:.3f}" for k in ('train_wall','train_cpu','prepare_wall','scan_thread_cpu')],f"{r['metrics'][a]['hwm_kib']/1048576:.3f}"] for r in small for a in ('control','ordered','lookup','combined')])
add('''512 MiB 完整训练 CPU 的组合节省只有 1.7%，墙钟节省 2.1%；不能把 prepare 的 9.2% 当作完整训练收益。1 GiB 完整训练 CPU 单次节省 15.8%，受到控制臂偏慢影响。O 加到 L 后，512 MiB 完整墙钟略增 0.116 s，1 GiB 减少 1.567 s；这里不声称稳定的完整训练墙钟加速。''')
add('''## ByteLevel 与 Whitespace：在相同分词器内比较

同一英文／中文 256 MiB 原始语料分别经两种预分词器处理。ByteLevel 使用 regex 和 byte→Unicode 映射、完整 byte 字母表；Whitespace 使用 Unicode 词边界和自然字母表。两种输入的训练难度不同，所以比较候选相对同预分词器 main 的开销，再看二者差异。core 读取预先冻结的 WordCounts；pipeline 包括公共 feed 和 train。表中的 feed CPU 用 pipeline CPU 减 train CPU 推导，包含两段之间的协调间隙；runner 原始输出没有独立 feed CPU。每配置每臂 core／pipeline 各一次，没有反序。''')
core={r['case']:r for r in selected if r['kind']=='selected-phases'}
table(['语言','ByteLevel 候选/main CPU','Whitespace 候选/main CPU','Whitespace 初始索引额外 CPU','占全部额外 CPU'],[[lang,f"{core[lang+'-256MiB']['relative_to_main_percent']['candidate']['train_cpu_seconds']:+.1f}%",f"{core[lang+'-whitespace-256MiB']['relative_to_main_percent']['candidate']['train_cpu_seconds']:+.1f}%",f"{core[lang+'-whitespace-256MiB']['metrics']['candidate']['initial_index_cpu_seconds']-core[lang+'-whitespace-256MiB']['metrics']['main']['initial_index_cpu_seconds']:.3f} s",f"{100*(core[lang+'-whitespace-256MiB']['metrics']['candidate']['initial_index_cpu_seconds']-core[lang+'-whitespace-256MiB']['metrics']['main']['initial_index_cpu_seconds'])/(core[lang+'-whitespace-256MiB']['metrics']['candidate']['train_cpu_seconds']-core[lang+'-whitespace-256MiB']['metrics']['main']['train_cpu_seconds']):.1f}%"] for lang in ('en','zh')])
add('''英文 ByteLevel 的相对 CPU 开销更高，中文却是 Whitespace 明显更高；不能推出 ByteLevel 普遍放大差距。四份 main 日志均记录 ByteLevel 走 bounded、Whitespace 走 keyed。中文 Whitespace 的最大差额在初始索引，已不是 prepare；main keyed 路径以 radix::sort_by_key 排序（从 radsort 的 radixsort_permuted.c 移植），简化候选使用块内 pair 累积和临时片段压缩。没有单独关闭 radix 的消融，无法把该阶段的全部差额都算作 radsort 的贡献。

英文 prepared corpus 不到一个 2²⁴-slot 块，候选初始索引仅有一个生产者；其初始墙钟高于 main，而 CPU 差额小得多。固定大块在小型输入上的并行粒度也是实际代价，本轮没有调小块大小。

因此这个候选可以显著改善原简化实现的中文 ByteLevel 内存，但它不是覆盖所有预分词器、可替代 main 的性能方案。特别是中文 Whitespace 的通用初始索引差距依然很大。''')
for kind,label in [('selected-phases','core'),('phases-pipeline','pipeline')]:
    add('### ' + label)
    items=[]
    for r in selected:
        if r['kind']!=kind: continue
        for arm in ('main','baseline','candidate'):
            m=r['metrics'][arm]
            items.append([r['case'],arm,f"{m['train_seconds']:.3f}",f"{m['train_cpu_seconds']:.3f}",f"{m['initial_index_wall_seconds']:.3f}",f"{m['initial_index_cpu_seconds']:.3f}",f"{m['prepare_wall_seconds']:.3f}",f"{m['prepare_cpu_seconds']:.3f}",f"{m['process_hwm_kib_before_validation']/1048576:.3f}"])
    table(['输入','版本','训练 wall','训练 CPU','初始 wall','初始 CPU','prepare wall','prepare CPU','峰值 GiB'],items)
    if kind=='phases-pipeline':
        table(['输入','版本','feed wall s','feed CPU s','pipeline wall s','pipeline CPU s'],[[r['case'],arm,*[f"{r['metrics'][arm][k]:.3f}" for k in ('feed_seconds','derived_feed_cpu_seconds','pipeline_seconds','pipeline_cpu_seconds')]] for r in selected if r['kind']==kind for arm in ('main','baseline','candidate')])
    table(['输入','版本','相对 main 训练 wall','训练 CPU','初始 CPU','prepare CPU','峰值'],[[r['case'],arm,*[f"{r['relative_to_main_percent'][arm][k]:+.1f}%" for k in ('train_seconds','train_cpu_seconds','initial_index_cpu_seconds','prepare_cpu_seconds','process_hwm_kib_before_validation')]] for r in selected if r['kind']==kind for arm in ('baseline','candidate')])
    if kind=='phases-pipeline':
        table(['输入','候选/main pipeline wall','候选/main pipeline CPU'],[[r['case'],*[f"{r['relative_to_main_percent']['candidate'][k]:+.1f}%" for k in ('pipeline_seconds','pipeline_cpu_seconds')]] for r in selected if r['kind']==kind])
add('''### 候选相对原 2100 行版本的取舍
+
+下面的分母是同组 baseline，而非 main。新机制的内存收益比完整训练速度收益更一致；英文固定大块的初始并行度代价、中文 Whitespace 的时间回退均保留在结论中。feed 的实现相同，其单次差额不归给这些训练优化。''')
table(['输入','模式','训练 wall 变化','训练 CPU 变化','prepare CPU 变化','峰值变化'],[[r['case'],r['kind'],*[f"{100*(r['metrics']['candidate'][k]/r['metrics']['baseline'][k]-1):+.1f}%" for k in ('train_seconds','train_cpu_seconds','prepare_cpu_seconds','process_hwm_kib_before_validation')]] for r in selected])
add('''## 正确性、复现边界与交付

全部有效训练检查完整 vocabulary ID 与有序 merge 列表。实际 vocab／merge 数随初始字母表变化，在各次输出中记录，不能把 ByteLevel 的 49,744 merges 套到 Whitespace。选定对照不得出现 swap 或并行 cargo／rustc／其他 benchmark。观察器固定十阶段，release opt3／fat LTO／codegen-units 1，workers 4，affinity 0–3，min_frequency 2，Cargo.lock 和 source／binary／patch SHA 均已记录。

宿主是共享 VM；检查只排除编译器与 benchmark 并发，不能宣称宿主完全静默。CPU 与墙钟都受时间段影响，单样本百分比不等于置信区间。1 GiB partial 与取消的反序、被取代的无插桩计划均留有原始标记。未运行的计划不会出现在成功计数中。

干净组合候选已通过默认与无默认特性完整 native 测试（各 17＋1 doctest）、Clippy `-D warnings`、严格 provenance Miri（2 tests）及格式／行数检查。索引初版的 separator 越界由小规模语义测试捕获，在构建性能二进制前已修复；失败日志与修正后验证都保留。

生产补丁：[small-combined.patch](evidence/online-initial/small-combined.patch)。补丁基于 `7e77262b`，只修改生产实现和所需语义测试，可在独立树审阅与应用。未自动合入生产分支。当前报告和证据是本轮实验交付。

原始结果：[runs.jsonl](evidence/online-initial/runs.jsonl)；逐项摘要：[small-ablation-summary.json](evidence/online-initial/small-ablation-summary.json)；分词器摘要：[selected-summary.json](evidence/online-initial/selected-summary.json)；候选构建：[selected-full-manifest.json](evidence/online-initial/selected-full-manifest.json)。''')
(DEST / 'ONLINE_COMPRESSION.md').write_text('\n'.join(parts))
evidence = DEST / 'evidence/online-initial'
evidence.mkdir(parents=True,exist_ok=True)
inventory=[]
def copy(p, rel=None):
    rel=rel or p.relative_to(OUT)
    target=evidence/rel;target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(p,target)
    inventory.append(dict(file=str(rel),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_path=str(p)))
for p in sorted(OUT.iterdir()):
    if p.is_file() and p.suffix in ('.json','.jsonl','.py','.patch','.txt'):
        copy(p)
for p in sorted((OUT / 'validation').iterdir()):
    if p.is_file():copy(p)
for p in sorted((OUT / 'runs').rglob('*')):
    if p.is_file() and p.name in ('job.json','result.json','stdout.json','stderr.log','CANCELLED_BY_USER.json','HALTED_BY_USER.json'):
        copy(p)
    elif p.is_file() and p.name=='samples.json':
        inventory.append(dict(file=str(p.relative_to(OUT)),external_only=True,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),original_path=str(p)))
for p in (OUT/'one-gib/input.json', OUT/'one-gib/raw/manifest.json'):
    if p.exists():copy(p)
for p in (OUT/'pretokenizers').rglob('*'):
    if p.is_file() and p.name in ('job.json','stdout.json','stderr.log'):copy(p)
(evidence/'archive-inventory.json').write_text(json.dumps(dict(files=inventory,external_large_artifacts='Input corpora, WordCounts, models, binaries and detailed RSS samples remain at their immutable manifest paths; no duplicate large fixtures in git.'),indent=2))
print('Report and evidence written',DEST/'ONLINE_COMPRESSION.md')
