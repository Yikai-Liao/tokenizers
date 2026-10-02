# BPE trainer affixes: primary-source notes

Checked on 2026-10-02. Here, “affix” means Hugging Face Tokenizers BPE's `continuing_subword_prefix` and/or `end_of_word_suffix`. The strings can be used by more than one tokenizer algorithm, so the source and role matter.

## Findings

The Tokenizers 0.23.2 Rust `BpeTrainer` config initializes both affix fields to `None`; its setters configure them explicitly ([versioned trainer source](https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/tokenizers/src/models/bpe/trainer.rs)). The official BPE trainer API exposes both as optional parameters ([trainer API](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#trainers)). The official Quicktour trains BPE without either setting and shows `</w>` as an optional end-of-word suffix ([Quicktour](https://huggingface.co/docs/tokenizers/python/latest/quicktour.html#training-your-own-tokenizer)). This documents defaults and supported use, not population-level frequency.

`##` is the documented default `continuing_subword_prefix` for `WordPieceTrainer` ([trainer API](https://huggingface.co/docs/tokenizers/python/latest/api/reference.html#tokenizers.trainers.WordPieceTrainer)); the WordPiece course uses it to mark continuation pieces ([WordPiece course](https://huggingface.co/learn/llm-course/chapter6/6?fw=pt)). The same course says that `train_new_from_iterator()` uses BPE internally because the library does not implement WordPiece training, even though the source tokenizer is WordPiece ([course note](https://huggingface.co/learn/llm-course/chapter6/6?fw=pt#wordpiece-tokenization)). Thus `##` is a WordPiece convention, but it can occur in a BPE-based retraining route; neither fact makes it the default for `BpeTrainer`.

The versioned `CharBPETokenizer.train` and `train_from_iterator` wrappers default `suffix="</w>"` and pass it as `end_of_word_suffix` ([Tokenizers 0.23.2 wrapper source](https://raw.githubusercontent.com/huggingface/tokenizers/v0.23.2/bindings/python/py_src/tokenizers/implementations/char_level_bpe.py)). This is a wrapper default; the lower-level `BpeTrainer` default remains `None`.

Classic subword-nmt marks the final character with `</w>` in the learned word symbols ([subword-nmt implementation](https://github.com/rsennrich/subword-nmt/blob/master/subword_nmt/learn_bpe.py); [Sennrich et al., 2016](https://aclanthology.org/P16-1162/)). GPT-2's released encoder uses byte-to-Unicode mapping and byte-level BPE merges ([OpenAI GPT-2 encoder](https://github.com/openai/gpt-2/blob/master/src/encoder.py)); RoBERTa describes byte-level BPE ([original paper](https://arxiv.org/abs/1907.11692)). These examples are useful context for distinct BPE setups, not prevalence evidence about trainer affixes.

SentencePiece uses `▁` (U+2581) for escaped whitespace and a dummy leading whitespace in normalization; its docs describe whitespace handling and piece constraints, not the Tokenizers BPE `continuing_subword_prefix` field ([normalization](https://github.com/google/sentencepiece/blob/master/doc/normalization.md); [piece constraints](https://github.com/google/sentencepiece/blob/master/doc/piece_constraints.md)).

The Hub BERT `tokenizer.json` stores `continuing_subword_prefix="##"`; the GPT-2 artifact stores empty affix fields ([BERT artifact](https://huggingface.co/bert-base-uncased/raw/main/tokenizer.json); [GPT-2 artifact](https://huggingface.co/gpt2/raw/main/tokenizer.json)). These serialized files describe the loaded model's behavior. They do not record the original training command or establish how common a configuration is.

## Benchmark choice

Use no affix as the lower-level `BpeTrainer` default baseline; retain `##` as a diagnostic BPE prefix, `</w>` as a historical suffix and wrapper example, and both together as a composition case. Use absent private-use markers only as controls for marker/alphabet reasoning. No usage share or “most common” claim follows from these sources.

For speed work, focus first on generic initialization, posting/candidate storage, and repeated cohort work that preserves semantics for any valid affix. A certificate tied to a particular marker or alphabet can only be an additional conditionally applicable path.

Source URLs, access date, and the claim each supports are also recorded in [`results/affix-usage-research/sources.json`](results/affix-usage-research/sources.json).
