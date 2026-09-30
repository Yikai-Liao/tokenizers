# BPE trainer variant: corpus-direct

Entry points: original `BpeTrainer::train_vocab()`, `do_train()` and `Trainer::train()`.

- Algorithm: parallel.
- Corpus token ID and posting width: u32.
- Atomic corpus: False.
- Initialization workers: same as merge workers.
- Merge workers: existing tk_encode parallelism controls; disabled parallelism selects 1.
- Empty affixes use the guarded endpoint algorithm; nonempty affixes retain the compatible cohort path.
- Historical comparison helpers are module-private.
- Base snapshot: 8c968e10; fixed PR comparator: 6ac0de5359d9e0e1ed0608422575a360ef91b908.

Experiments and provenance: central `/root/code/tokenizers/benchmarks/hf-bpe/EXPERIMENT_LOG.md`.

The local benchmark supports only the original `reference` API route; legacy indexed features remain in the historical snapshot.

Optimization parent: `bpe/parallel-count4` at `b6a28768`.

Corpus construction now measures ordered word regions in parallel, writes each
region directly into its final exclusive slice, and combines boundary metadata
after the join. Alphabet selection and canonical ID assignment retain the original
behavior; nonempty affixes retain the generic path. Initialization worker controls
apply to construction and pair counting. No additional full corpus is retained.

Private statistics split alphabet, region measurement, final allocation and fill
times. The current round plan and measurements are maintained centrally in
`/root/code/tokenizers/benchmarks/hf-bpe/OPTIMIZATION_PLAN.md`.

Candidate C parent: `98ca7fc1` (corpus-parallel).

Unlimited alphabet collection uses worker Unicode presence bitmaps, followed by
canonical character ordering; limited alphabets keep the original selector.
A read-only Unicode-to-ID table removes per-position string hashing.
MaybeUninit final storage is initialized by exclusive word regions and converted
only after complete coverage and joins; its private unsafe conversion is documented
and tested. Alphabet temporary arrays and ID table capacity are measured separately.

## 匹配控制项

本分支从 corpus-direct `fbdc0b2b` 派生，仅把原接口 plain 路径的 `atomic_corpus` 改为 true。初始化仍采用 C 的并行 alphabet、直接查询和独占直接填充；merge 保留排序 Plan 与 delta 两阶段算法。用于与融合候选隔离算法收益。

## 融合候选

flat 非AA Atomic 批次按规则/位置连续分工，一次只读遍历完成校验与邻边 delta，保存4字节有效位置；所有读取 join 后并行共享 Atomic 写入。AA、多块及非原子配置保持旧路径。出生key唯一规则生产者与任务次序维持 posting 有序；提交阶段加 debug 验证。计时 fused_prepare_ms 同时覆盖 filter/delta，不能直接和旧 delta_ms 比较。

## B2 查询热点

从融合B派生，权重查询增加每256位置一个u32 pivot下界目录，桶内精确查询；selected边界增加head/tail直接表，重复符号通过小哈希表回退。字母表/语料构造、初始pair计数、选择和提交算法均保持父版本。目录在初始化结束后、merge计时中创建一次，内存单列统计；Selected是每批临时表。

## D 初始radix分组

从B2 `a0832c48` 派生，仅更换flat且完整初始ID域<=65536的pair初始化；其它配置旧路径。两次顺序扫描精确路由8字节code/position记录，稳定radix按初始u16+u16 pair分组（公开corpus/posting仍u32），先加权/低频剪枝再精确预留最终posting与owner表。merge协议不变。weight目录提取为共享私有metadata模块，初始化计数构建后merge复用。新增临时route/scratch/group容量与全部阶段计时，峰值RSS实测为准。

## DE组合

D `d15c18cc` 合入E `35eaf03c` 的同一SmallPosting批量接口与owner提交改动，并将radix初始posting安装改为从record尾部读取、倒序直接填入预留区域。初始count/radix/频率/剪枝和merge prepare/rewrite不改；直接影响posting install与owner commit。slot/metadata内存表示保持D，全部发布len在写完后。count=3仍遵循SmallPosting最小4槽heap，capacity按实际报告。
