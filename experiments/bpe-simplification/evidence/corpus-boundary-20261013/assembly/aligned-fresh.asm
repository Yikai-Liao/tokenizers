
/tmp/bpe-corpus-binaries-20261013/aligned-candidate-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000e4950 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>>:
   e4950:	push   %rbp
   e4951:	push   %r15
   e4953:	push   %r14
   e4955:	push   %r13
   e4957:	push   %r12
   e4959:	push   %rbx
   e495a:	sub    $0x3e8,%rsp
   e4961:	mov    0x428(%rsp),%r14
   e4969:	mov    %rsi,0x138(%rsp)
   e4971:	mov    %rcx,0x178(%rsp)
   e4979:	mov    %r8,0x180(%rsp)
   e4981:	mov    (%r14),%rbp
   e4984:	movzbl 0x0(%rbp),%eax
   e4988:	test   %al,%al
   e498a:	je     e4a18 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc8>
   e4990:	mov    0x8(%r14),%rax
   e4994:	xorps  %xmm0,%xmm0
   e4997:	movups %xmm0,0x78(%rsp)
   e499c:	mov    0x18(%r14),%rcx
   e49a0:	movq   $0x0,0x40(%rsp)
   e49a9:	movq   $0x8,0x48(%rsp)
   e49b2:	movq   $0x0,0x50(%rsp)
   e49bb:	mov    %rbp,0x58(%rsp)
   e49c0:	mov    %rax,0x60(%rsp)
   e49c5:	lea    0x68(%rsp),%r15
   e49ca:	movq   $0x0,0x68(%rsp)
   e49d3:	movq   $0x4,0x70(%rsp)
   e49dc:	movq   $0x4,0x88(%rsp)
   e49e8:	movq   $0x0,0x90(%rsp)
   e49f4:	mov    %rcx,0x98(%rsp)
   e49fc:	lea    0x40(%rsp),%rsi
   e4a01:	call   f3470 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e4a06:	add    $0x3e8,%rsp
   e4a0d:	pop    %rbx
   e4a0e:	pop    %r12
   e4a10:	pop    %r13
   e4a12:	pop    %r14
   e4a14:	pop    %r15
   e4a16:	pop    %rbp
   e4a17:	ret
   e4a18:	mov    %r9,%r12
   e4a1b:	mov    %rdi,0x208(%rsp)
   e4a23:	mov    0x420(%rsp),%rbx
   e4a2b:	shr    $1,%rsi
   e4a2e:	cmp    %r8,%rsi
   e4a31:	jae    e6004 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16b4>
   e4a37:	mov    0x8(%r14),%rax
   e4a3b:	xorps  %xmm0,%xmm0
   e4a3e:	movups %xmm0,0x1d8(%rsp)
   e4a46:	mov    0x18(%r14),%rcx
   e4a4a:	movq   $0x0,0x1a0(%rsp)
   e4a56:	movq   $0x8,0x1a8(%rsp)
   e4a62:	movq   $0x0,0x1b0(%rsp)
   e4a6e:	mov    %rbp,0x118(%rsp)
   e4a76:	mov    %rbp,0x1b8(%rsp)
   e4a7e:	mov    %rax,0x238(%rsp)
   e4a86:	mov    %rax,0x1c0(%rsp)
   e4a8e:	lea    0x1c8(%rsp),%r15
   e4a96:	movq   $0x0,0x1c8(%rsp)
   e4aa2:	movq   $0x4,0x1d0(%rsp)
   e4aae:	movq   $0x4,0x1e8(%rsp)
   e4aba:	movq   $0x0,0x1f0(%rsp)
   e4ac6:	mov    %rcx,0x258(%rsp)
   e4ace:	mov    %rcx,0x1f8(%rsp)
   e4ad6:	imul   $0x70,%rbx,%rax
   e4ada:	add    %r12,%rax
   e4add:	mov    %rax,0x260(%rsp)
   e4ae5:	movq   $0x0,0x188(%rsp)
   e4af1:	movq   $0x8,0x190(%rsp)
   e4afd:	movq   $0x0,0x198(%rsp)
   e4b09:	mov    $0x8,%eax
   e4b0e:	mov    %rax,0x240(%rsp)
   e4b16:	lea    0x21b73(%rip),%rax        # 106690 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3318>
   e4b1d:	movq   %rax,%xmm0
   e4b22:	movdqa %xmm0,0x3d0(%rsp)
   e4b2b:	lea    0x40(%rsp),%rbp
   e4b30:	lea    0x222c1(%rip),%rax        # 106df8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a80>
   e4b37:	mov    %rax,0x10(%rsp)
   e4b3c:	movq   $0x0,0x248(%rsp)
   e4b48:	jmp    e4bc0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x270>
   e4b4a:	nopw   0x0(%rax,%rax,1)
   e4b50:	mov    0x190(%rsp),%rax
   e4b58:	mov    %rax,0x240(%rsp)
   e4b60:	imul   $0x38,%rbx,%rax
   e4b64:	mov    0x2e8(%rsp),%rcx
   e4b6c:	mov    0x240(%rsp),%rdx
   e4b74:	mov    %rcx,0x30(%rdx,%rax,1)
   e4b79:	movdqu 0x2b8(%rsp),%xmm0
   e4b82:	movdqu 0x2c8(%rsp),%xmm1
   e4b8b:	movdqu 0x2d8(%rsp),%xmm2
   e4b94:	movdqu %xmm2,0x20(%rdx,%rax,1)
   e4b9a:	movdqu %xmm1,0x10(%rdx,%rax,1)
   e4ba0:	movdqu %xmm0,(%rdx,%rax,1)
   e4ba5:	inc    %rbx
   e4ba8:	mov    %rbx,0x248(%rsp)
   e4bb0:	mov    %rbx,0x198(%rsp)
   e4bb8:	lea    0x1c8(%rsp),%r15
   e4bc0:	cmp    0x260(%rsp),%r12
   e4bc8:	je     e6025 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e4bce:	mov    (%r12),%r14d
   e4bd2:	mov    0x4(%r12),%edx
   e4bd7:	mov    0x8(%r12),%ecx
   e4bdc:	movups 0x60(%r12),%xmm0
   e4be2:	movaps %xmm0,0x340(%rsp)
   e4bea:	movups 0x50(%r12),%xmm0
   e4bf0:	movaps %xmm0,0x330(%rsp)
   e4bf8:	movdqu 0x10(%r12),%xmm0
   e4bff:	movdqu 0x20(%r12),%xmm1
   e4c06:	movdqu 0x30(%r12),%xmm2
   e4c0d:	movups 0x40(%r12),%xmm3
   e4c13:	movaps %xmm3,0x320(%rsp)
   e4c1b:	movdqa %xmm2,0x310(%rsp)
   e4c24:	movdqa %xmm1,0x300(%rsp)
   e4c2d:	movdqa %xmm0,0x2f0(%rsp)
   e4c36:	mov    0x258(%rsp),%rax
   e4c3e:	mov    (%rax),%rdi
   e4c41:	mov    0x8(%rax),%rsi
   e4c45:	mov    %rsi,0x160(%rsp)
   e4c4d:	mov    0x10(%rax),%rax
   e4c51:	mov    %rax,0x100(%rsp)
   e4c59:	mov    %rdi,0x120(%rsp)
   e4c61:	mov    (%rdi),%rbx
   e4c64:	mov    0x50(%rbx),%r8
   e4c68:	lea    0x350(%rsp),%rdi
   e4c70:	mov    %r14d,%esi
   e4c73:	mov    %rdx,0x20(%rsp)
   e4c78:	mov    %ecx,0x174(%rsp)
   e4c7f:	mov    %r15,%r9
   e4c82:	call   f4640 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::new>
   e4c87:	movq   $0x0,0xf0(%rsp)
   e4c93:	mov    0x50(%rbx),%rsi
   e4c97:	cmp    %r14,%rsi
   e4c9a:	jbe    e66b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d68>
   e4ca0:	mov    0x20(%rsp),%rdi
   e4ca5:	cmp    %rdi,%rsi
   e4ca8:	jbe    e66bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d6b>
   e4cae:	mov    0x48(%rbx),%rax
   e4cb2:	mov    %r14,0x130(%rsp)
   e4cba:	mov    (%rax,%r14,8),%rcx
   e4cbe:	mov    %rcx,0x128(%rsp)
   e4cc6:	mov    (%rax,%rdi,8),%rax
   e4cca:	mov    %rax,0x168(%rsp)
   e4cd2:	lea    0x2f0(%rsp),%rax
   e4cda:	mov    %rax,0x268(%rsp)
   e4ce2:	mov    %rbx,0x270(%rsp)
   e4cea:	mov    %rbx,0x38(%rsp)
   e4cef:	mov    %rbx,0x278(%rsp)
   e4cf7:	mov    $0x80,%edi
   e4cfc:	call   *0x22a3e(%rip)        # 107740 <malloc@GLIBC_2.2.5>
   e4d02:	test   %rax,%rax
   e4d05:	je     e66c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d79>
   e4d0b:	mov    %rax,%rbx
   e4d0e:	add    $0x70,%r12
   e4d12:	mov    %r12,0xf8(%rsp)
   e4d1a:	pxor   %xmm0,%xmm0
   e4d1e:	lea    0x150(%rsp),%rax
   e4d26:	movdqu %xmm0,(%rax)
   e4d2a:	movq   $0x10,0x140(%rsp)
   e4d36:	mov    %rbx,0x148(%rsp)
   e4d3e:	lea    0x268(%rsp),%rax
   e4d46:	mov    %rax,0x40(%rsp)
   e4d4b:	movq   $0x10,0x48(%rsp)
   e4d54:	mov    $0x10,%r12d
   e4d5a:	xor    %r14d,%r14d
   e4d5d:	xor    %r15d,%r15d
   e4d60:	mov    %rbp,%rdi
   e4d63:	call   f6e20 <<core::iter::adapters::take::Take<&mut core::iter::adapters::inspect::Inspect<core::iter::adapters::map::Map<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#1}>> as core::iter::traits::iterator::Iterator>::next>
   e4d68:	cmp    $0x1,%rax
   e4d6c:	jne    e4f18 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x5c8>
   e4d72:	cmp    $0xffffffffffffffff,%r15
   e4d76:	je     e648e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b3e>
   e4d7c:	mov    %rdx,%rbp
   e4d7f:	mov    %r12,%rax
   e4d82:	sub    %r15,%rax
   e4d85:	jbe    e4d90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x440>
   e4d87:	mov    %r15,%r13
   e4d8a:	jmp    e4e7e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e4d8f:	nop
   e4d90:	je     e4e18 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x4c8>
   e4d96:	mov    %r12,%rsi
   e4d99:	mov    %r15,%r13
   e4d9c:	mov    %r14,%rcx
   e4d9f:	cmp    %rax,%r14
   e4da2:	jbe    e4e70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x520>
   e4da8:	mov    %r12,%rdx
   e4dab:	sub    %rcx,%rdx
   e4dae:	mov    %r13,%rax
   e4db1:	sub    %rdx,%rax
   e4db4:	cmp    %rax,%rdx
   e4db7:	jbe    e4de8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x498>
   e4db9:	mov    %rsi,%rcx
   e4dbc:	sub    %r12,%rcx
   e4dbf:	cmp    %rax,%rcx
   e4dc2:	jb     e4de8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x498>
   e4dc4:	mov    0x148(%rsp),%rbx
   e4dcc:	lea    (%rbx,%r12,8),%rdi
   e4dd0:	shl    $0x3,%rax
   e4dd4:	mov    %rsi,%r12
   e4dd7:	mov    %rbx,%rsi
   e4dda:	mov    %rax,%rdx
   e4ddd:	call   *0x2294d(%rip)        # 107730 <memcpy@GLIBC_2.14>
   e4de3:	jmp    e4e7e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e4de8:	mov    %rsi,%r15
   e4deb:	sub    %rdx,%r15
   e4dee:	mov    0x148(%rsp),%rbx
   e4df6:	mov    %rsi,%r12
   e4df9:	lea    (%rbx,%r14,8),%rsi
   e4dfd:	lea    (%rbx,%r15,8),%rdi
   e4e01:	shl    $0x3,%rdx
   e4e05:	call   *0x2292d(%rip)        # 107738 <memmove@GLIBC_2.2.5>
   e4e0b:	mov    %r15,0x150(%rsp)
   e4e13:	mov    %r15,%r14
   e4e16:	jmp    e4e7e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e4e18:	mov    $0x1,%edx
   e4e1d:	mov    $0x8,%ecx
   e4e22:	mov    $0x8,%r8d
   e4e28:	lea    0x140(%rsp),%rdi
   e4e30:	mov    %r15,%rsi
   e4e33:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e4e38:	mov    0x140(%rsp),%rsi
   e4e40:	mov    0x150(%rsp),%r14
   e4e48:	mov    0x158(%rsp),%r13
   e4e50:	mov    %r12,%rax
   e4e53:	sub    %r13,%rax
   e4e56:	mov    %r14,%rcx
   e4e59:	cmp    %rax,%r14
   e4e5c:	ja     e4da8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x458>
   e4e62:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e4e70:	mov    0x148(%rsp),%rbx
   e4e78:	mov    %rsi,%r12
   e4e7b:	mov    %rcx,%r14
   e4e7e:	lea    (%r14,%r13,1),%rax
   e4e82:	cmp    %r12,%rax
   e4e85:	mov    $0x0,%ecx
   e4e8a:	cmovae %r12,%rcx
   e4e8e:	sub    %rcx,%rax
   e4e91:	mov    %rbp,(%rbx,%rax,8)
   e4e95:	lea    0x1(%r13),%r15
   e4e99:	mov    %r15,0x158(%rsp)
   e4ea1:	cmp    %r12,%r15
   e4ea4:	lea    0x40(%rsp),%rbp
   e4ea9:	jae    e4d60 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x410>
   e4eaf:	mov    %r14,0x18(%rsp)
   e4eb4:	add    %r13,%r14
   e4eb7:	inc    %r14
   e4eba:	nopw   0x0(%rax,%rax,1)
   e4ec0:	mov    %rbp,%rdi
   e4ec3:	call   f6e20 <<core::iter::adapters::take::Take<&mut core::iter::adapters::inspect::Inspect<core::iter::adapters::map::Map<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#1}>> as core::iter::traits::iterator::Iterator>::next>
   e4ec8:	cmp    $0x1,%rax
   e4ecc:	jne    e4f10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x5c0>
   e4ece:	cmp    %r12,%r14
   e4ed1:	mov    $0x0,%eax
   e4ed6:	cmovae %r12,%rax
   e4eda:	shl    $0x3,%rax
   e4ede:	mov    %rbx,%rcx
   e4ee1:	sub    %rax,%rcx
   e4ee4:	mov    %rdx,(%rcx,%r14,8)
   e4ee8:	inc    %r15
   e4eeb:	inc    %r14
   e4eee:	cmp    %r15,%r12
   e4ef1:	jne    e4ec0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x570>
   e4ef3:	mov    %r12,0x158(%rsp)
   e4efb:	mov    %r12,%r15
   e4efe:	mov    0x18(%rsp),%r14
   e4f03:	jmp    e4d60 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x410>
   e4f08:	nopl   0x0(%rax,%rax,1)
   e4f10:	mov    %r15,0x158(%rsp)
   e4f18:	mov    0x128(%rsp),%rsi
   e4f20:	mov    0x168(%rsp),%rdi
   e4f28:	add    %rsi,%rdi
   e4f2b:	movups 0x140(%rsp),%xmm0
   e4f33:	mov    0x150(%rsp),%rax
   e4f3b:	mov    0x158(%rsp),%rcx
   e4f43:	movaps %xmm0,0x280(%rsp)
   e4f4b:	mov    %rax,0x290(%rsp)
   e4f53:	mov    %rcx,0x298(%rsp)
   e4f5b:	mov    0x278(%rsp),%rax
   e4f63:	lea    0x2a0(%rsp),%rcx
   e4f6b:	mov    %rax,0x10(%rcx)
   e4f6f:	movups 0x268(%rsp),%xmm0
   e4f77:	movups %xmm0,(%rcx)
   e4f7a:	mov    0x2b0(%rsp),%rax
   e4f82:	lea    0x80(%rsp),%rdx
   e4f8a:	mov    %rax,0x30(%rdx)
   e4f8e:	movdqa 0x280(%rsp),%xmm0
   e4f97:	mov    0x290(%rsp),%rax
   e4f9f:	mov    0x298(%rsp),%rcx
   e4fa7:	movdqa 0x2a0(%rsp),%xmm1
   e4fb0:	movdqu %xmm1,0x20(%rdx)
   e4fb5:	mov    %rax,0x10(%rdx)
   e4fb9:	mov    %rcx,0x18(%rdx)
   e4fbd:	movdqu %xmm0,(%rdx)
   e4fc1:	movq   $0x0,0x40(%rsp)
   e4fca:	mov    0x38(%rsp),%rax
   e4fcf:	mov    %rax,0x58(%rsp)
   e4fd4:	mov    %rsi,0x60(%rsp)
   e4fd9:	mov    %rdi,0x68(%rsp)
   e4fde:	mov    0x130(%rsp),%rcx
   e4fe6:	mov    %ecx,0x70(%rsp)
   e4fea:	mov    0x20(%rsp),%rcx
   e4fef:	mov    %ecx,0x74(%rsp)
   e4ff3:	mov    %rax,0x78(%rsp)
   e4ff8:	mov    0x80(%rsp),%rax
   e5000:	mov    %rax,0xd8(%rsp)
   e5008:	mov    0x98(%rsp),%rbx
   e5010:	test   %rbx,%rbx
   e5013:	je     e5e20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e5019:	mov    0x120(%rsp),%rax
   e5021:	mov    0x8(%rax),%rax
   e5025:	mov    %rax,0x120(%rsp)
   e502d:	mov    0x100(%rsp),%rcx
   e5035:	movzbl (%rcx),%eax
   e5038:	mov    %al,0xf(%rsp)
   e503c:	mov    0x10(%rcx),%rax
   e5040:	mov    %rax,0xd0(%rsp)
   e5048:	mov    0x8(%rcx),%rax
   e504c:	mov    %rax,0x108(%rsp)
   e5054:	mov    0x160(%rsp),%rdx
   e505c:	mov    0x18(%rdx),%rax
   e5060:	mov    %rax,0x228(%rsp)
   e5068:	mov    0x28(%rdx),%rax
   e506c:	mov    %rax,0x218(%rsp)
   e5074:	mov    0x20(%rdx),%rax
   e5078:	mov    %rax,0x210(%rsp)
   e5080:	mov    (%rdx),%rax
   e5083:	mov    %rax,0x230(%rsp)
   e508b:	mov    0x8(%rdx),%rax
   e508f:	mov    %rax,0x110(%rsp)
   e5097:	mov    0x50(%rcx),%rax
   e509b:	mov    %rax,0x220(%rsp)
   e50a3:	mov    0x48(%rcx),%rax
   e50a7:	mov    %rax,0x250(%rsp)
   e50af:	mov    0x90(%rsp),%r14
   e50b7:	mov    0x58(%rsp),%r12
   e50bc:	mov    0x60(%rsp),%rax
   e50c1:	mov    %rax,0x128(%rsp)
   e50c9:	mov    0x70(%rsp),%eax
   e50cd:	mov    %eax,0x168(%rsp)
   e50d4:	mov    0x68(%rsp),%rax
   e50d9:	mov    %rax,0x130(%rsp)
   e50e1:	mov    0x74(%rsp),%eax
   e50e5:	mov    %eax,0x160(%rsp)
   e50ec:	mov    0xd8(%rsp),%r10
   e50f4:	mov    %r12,0x100(%rsp)
   e50fc:	jmp    e5116 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7c6>
   e50fe:	xchg   %ax,%ax
   e5100:	cmp    0x160(%rsp),%eax
   e5107:	je     e57cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe7f>
   e510d:	test   %rbx,%rbx
   e5110:	je     e5e20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e5116:	mov    %r14,%rax
   e5119:	lea    0x1(%r14),%rcx
   e511d:	cmp    %r10,%rcx
   e5120:	mov    %r10,0x20(%rsp)
   e5125:	mov    %r10,%rcx
   e5128:	mov    $0x0,%edx
   e512d:	cmovb  %rdx,%rcx
   e5131:	neg    %rcx
   e5134:	add    %rcx,%r14
   e5137:	inc    %r14
   e513a:	mov    %r14,0x90(%rsp)
   e5142:	mov    %rbx,%r13
   e5145:	dec    %r13
   e5148:	mov    %r13,0x98(%rsp)
   e5150:	mov    0x88(%rsp),%rcx
   e5158:	mov    0xa0(%rsp),%rbp
   e5160:	mov    (%rcx,%rax,8),%rax
   e5164:	mov    %rax,0x18(%rsp)
   e5169:	mov    0x58(%rbp),%rax
   e516d:	mov    $0x0,%r15d
   e5173:	test   %rax,%rax
   e5176:	je     e5516 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc6>
   e517c:	dec    %rax
   e517f:	mov    %rax,0x58(%rbp)
   e5183:	lea    0x8(%rbp),%rax
   e5187:	lea    0x10(%rbp),%r8
   e518b:	mov    0x0(%rbp),%rsi
   e518f:	test   %rsi,%rsi
   e5192:	je     e51cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x87f>
   e5194:	mov    (%rax),%rdx
   e5197:	test   %rdx,%rdx
   e519a:	je     e51c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x877>
   e519c:	mov    0x18(%rbp),%r9
   e51a0:	test   $0x7f,%r9b
   e51a4:	je     e53ba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa6a>
   e51aa:	movsbq (%rsi),%rbx
   e51ae:	test   %rbx,%rbx
   e51b1:	js     e5285 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x935>
   e51b7:	add    (%r8),%rbx
   e51ba:	mov    %rbx,(%r8)
   e51bd:	mov    $0x1,%edi
   e51c2:	jmp    e52ef <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x99f>
   e51c7:	movq   $0x0,0x0(%rbp)
   e51cf:	mov    0x40(%rbp),%r15
   e51d3:	test   %r15,%r15
   e51d6:	je     e55c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc73>
   e51dc:	mov    0x48(%rbp),%r11
   e51e0:	cmp    %r11,%r15
   e51e3:	je     e55c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc73>
   e51e9:	lea    0x20(%r15),%rbx
   e51ed:	mov    %rbx,0x40(%rbp)
   e51f1:	mov    0x50(%rbp),%r9
   e51f5:	mov    (%r15),%rcx
   e51f8:	dec    %rcx
   e51fb:	cmp    %rcx,%r9
   e51fe:	jne    e5219 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x8c9>
   e5200:	movq   $0x1,0x0(%rbp)
   e5208:	pxor   %xmm0,%xmm0
   e520c:	movdqu %xmm0,(%rax)
   e5210:	mov    %r9,0x18(%rbp)
   e5214:	jmp    e5454 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb04>
   e5219:	mov    %rcx,%r10
   e521c:	shr    $0x7,%r10
   e5220:	test   %r9,%r9
   e5223:	je     e52f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9a7>
   e5229:	mov    %ecx,%eax
   e522b:	and    $0x7f,%eax
   e522e:	cmp    $0x1,%rax
   e5232:	mov    %r10,%rsi
   e5235:	sbb    $0xffffffffffffffff,%rsi
   e5239:	cmp    $0x2,%rsi
   e523d:	jb     e66a8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d58>
   e5243:	mov    0x18(%r15),%rdi
   e5247:	mov    %rdi,%rdx
   e524a:	cmp    $0x11,%rdi
   e524e:	jb     e5254 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x904>
   e5250:	mov    0x10(%r15),%rdx
   e5254:	lea    0x8(,%rsi,8),%rax
   e525c:	cmp    %rdx,%rax
   e525f:	ja     e64cc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b7c>
   e5265:	mov    %r9,%rdx
   e5268:	shr    $0x7,%rdx
   e526c:	cmp    %rsi,%rdx
   e526f:	jae    e66b3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d63>
   e5275:	cmp    $0x11,%rdi
   e5279:	jb     e52fb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9ab>
   e527f:	mov    0x8(%r15),%rax
   e5283:	jmp    e52ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9af>
   e5285:	cmp    $0x1,%rdx
   e5289:	je     e64f5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba5>
   e528f:	and    $0x7f,%ebx
   e5292:	lea    -0x1(%rdx),%r10
   e5296:	mov    $0x7,%ecx
   e529b:	xor    %edi,%edi
   e529d:	mov    %rbx,%rax
   e52a0:	movzbl 0x1(%rsi,%rdi,1),%r11d
   e52a6:	cmp    $0x3f,%ecx
   e52a9:	jne    e52b5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x965>
   e52ab:	cmp    $0x1,%r11b
   e52af:	ja     e64f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba9>
   e52b5:	movzbl %r11b,%ebx
   e52b9:	test   %bl,%bl
   e52bb:	jns    e52d6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x986>
   e52bd:	and    $0x7f,%ebx
   e52c0:	shl    %cl,%rbx
   e52c3:	or     %rbx,%rax
   e52c6:	add    $0x7,%ecx
   e52c9:	inc    %rdi
   e52cc:	cmp    %rdi,%r10
   e52cf:	jne    e52a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x950>
   e52d1:	jmp    e64f5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba5>
   e52d6:	shl    %cl,%rbx
   e52d9:	or     %rax,%rbx
   e52dc:	add    $0x2,%rdi
   e52e0:	add    (%r8),%rbx
   e52e3:	mov    %rbx,(%r8)
   e52e6:	cmp    %rdx,%rdi
   e52e9:	ja     e6482 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b32>
   e52ef:	sub    %rdi,%rdx
   e52f2:	jmp    e53d3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa83>
   e52f7:	xor    %edi,%edi
   e52f9:	jmp    e5304 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9b4>
   e52fb:	lea    0x8(%r15),%rax
   e52ff:	mov    0x8(%rax,%rdx,8),%rdi
   e5304:	and    $0x7f,%ecx
   e5307:	cmp    $0x1,%rcx
   e530b:	sbb    $0xffffffffffffffff,%r10
   e530f:	mov    $0x8,%esi
   e5314:	cmp    $0x2,%r10
   e5318:	jae    e5338 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9e8>
   e531a:	mov    0x18(%r15),%rcx
   e531e:	mov    %rcx,%rax
   e5321:	cmp    $0x11,%rcx
   e5325:	jb     e5360 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa10>
   e5327:	mov    0x10(%r15),%rax
   e532b:	mov    %rax,%rdx
   e532e:	sub    %rsi,%rdx
   e5331:	jae    e536c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa1c>
   e5333:	jmp    e64bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b6b>
   e5338:	dec    %r10
   e533b:	bsr    %r10,%rcx
   e533f:	not    %ecx
   e5341:	mov    $0xffffffffffffffff,%rax
   e5348:	shr    %cl,%rax
   e534b:	lea    0x10(,%rax,8),%rsi
   e5353:	mov    0x18(%r15),%rcx
   e5357:	mov    %rcx,%rax
   e535a:	cmp    $0x11,%rcx
   e535e:	jae    e5327 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9d7>
   e5360:	mov    %rax,%rdx
   e5363:	sub    %rsi,%rdx
   e5366:	jb     e64bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b6b>
   e536c:	cmp    %rdi,%rdx
   e536f:	jb     e64c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b70>
   e5375:	cmp    $0x11,%rcx
   e5379:	jb     e5381 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa31>
   e537b:	mov    0x8(%r15),%r15
   e537f:	jmp    e5385 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa35>
   e5381:	add    $0x8,%r15
   e5385:	add    %rdi,%rsi
   e5388:	add    %r15,%rsi
   e538b:	sub    %rdi,%rdx
   e538e:	mov    %rsi,0x0(%rbp)
   e5392:	mov    %rdx,0x8(%rbp)
   e5396:	movq   $0x0,0x10(%rbp)
   e539e:	mov    %r9,0x18(%rbp)
   e53a2:	je     e5454 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb04>
   e53a8:	movq   $0x0,0x50(%rbp)
   e53b0:	test   $0x7f,%r9b
   e53b4:	jne    e51aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x85a>
   e53ba:	cmp    $0x7,%rdx
   e53be:	jbe    e64ab <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b5b>
   e53c4:	mov    (%rsi),%rbx
   e53c7:	mov    %rbx,(%r8)
   e53ca:	add    $0xfffffffffffffff8,%rdx
   e53ce:	mov    $0x8,%edi
   e53d3:	add    %rdi,%rsi
   e53d6:	mov    %rsi,0x0(%rbp)
   e53da:	mov    %rdx,0x8(%rbp)
   e53de:	inc    %r9
   e53e1:	mov    %r9,0x18(%rbp)
   e53e5:	mov    0xa8(%rsp),%rax
   e53ed:	cmp    0x10(%rax),%rbx
   e53f1:	jae    e6465 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b15>
   e53f7:	mov    0xb0(%rsp),%rax
   e53ff:	mov    0x10(%rax),%rcx
   e5403:	cmpl   $0x1,(%rax)
   e5406:	jne    e5500 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbb0>
   e540c:	cmp    %rcx,%rbx
   e540f:	jae    e5510 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc0>
   e5415:	lea    0x0(,%rbx,4),%rcx
   e541d:	jmp    e5509 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbb9>
   e5422:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5430:	mov    $0x1,%esi
   e5435:	xor    %edx,%edx
   e5437:	mov    %rsi,0x0(%rbp)
   e543b:	mov    %rdx,0x8(%rbp)
   e543f:	pxor   %xmm0,%xmm0
   e5443:	movdqu %xmm0,(%r8)
   e5448:	mov    %r9,%rbx
   e544b:	test   %rdx,%rdx
   e544e:	jne    e5605 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xcb5>
   e5454:	movq   $0x0,0x0(%rbp)
   e545c:	cmp    %r11,%rbx
   e545f:	je     e55bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc6b>
   e5465:	lea    0x20(%rbx),%r9
   e5469:	mov    %r9,0x40(%rbp)
   e546d:	mov    (%rbx),%rax
   e5470:	dec    %rax
   e5473:	je     e5430 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xae0>
   e5475:	mov    %eax,%ecx
   e5477:	and    $0x7f,%ecx
   e547a:	shr    $0x7,%rax
   e547e:	cmp    $0x1,%rcx
   e5482:	sbb    $0xffffffffffffffff,%rax
   e5486:	mov    $0x8,%edi
   e548b:	cmp    $0x2,%rax
   e548f:	jae    e54b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb60>
   e5491:	mov    0x18(%rbx),%rcx
   e5495:	mov    %rcx,%rax
   e5498:	cmp    $0x11,%rcx
   e549c:	jb     e54d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb88>
   e549e:	mov    0x10(%rbx),%rax
   e54a2:	mov    %rax,%rdx
   e54a5:	sub    %rdi,%rdx
   e54a8:	jae    e54e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb94>
   e54aa:	jmp    e6446 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1af6>
   e54af:	nop
   e54b0:	dec    %rax
   e54b3:	bsr    %rax,%rcx
   e54b7:	not    %ecx
   e54b9:	mov    $0xffffffffffffffff,%rax
   e54c0:	shr    %cl,%rax
   e54c3:	lea    0x10(,%rax,8),%rdi
   e54cb:	mov    0x18(%rbx),%rcx
   e54cf:	mov    %rcx,%rax
   e54d2:	cmp    $0x11,%rcx
   e54d6:	jae    e549e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb4e>
   e54d8:	mov    %rax,%rdx
   e54db:	sub    %rdi,%rdx
   e54de:	jb     e6446 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1af6>
   e54e4:	cmp    $0x11,%rcx
   e54e8:	jb     e54f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xba0>
   e54ea:	mov    0x8(%rbx),%rsi
   e54ee:	jmp    e54f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xba7>
   e54f0:	add    $0x8,%rbx
   e54f4:	mov    %rbx,%rsi
   e54f7:	add    %rdi,%rsi
   e54fa:	jmp    e5437 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xae7>
   e54ff:	nop
   e5500:	cmp    %rcx,%rbx
   e5503:	jae    e5510 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc0>
   e5505:	lea    (%rbx,%rbx,1),%rcx
   e5509:	add    0x8(%rax),%rcx
   e550d:	prefetcht0 (%rcx)
   e5510:	mov    $0x1,%r15d
   e5516:	lea    (%r15,%r13,1),%rax
   e551a:	mov    0x20(%rsp),%r10
   e551f:	cmp    %r10,%rax
   e5522:	jbe    e568e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd3e>
   e5528:	mov    %r10,%rax
   e552b:	sub    %r13,%rax
   e552e:	cmp    %rax,%r15
   e5531:	ja     e561f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xccf>
   e5537:	mov    %r10,%r12
   e553a:	mov    %r14,%rcx
   e553d:	cmp    %rax,%r14
   e5540:	jbe    e5680 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd30>
   e5546:	mov    0x20(%rsp),%rdi
   e554b:	mov    %rdi,%rdx
   e554e:	sub    %rcx,%rdx
   e5551:	mov    %r13,%rax
   e5554:	sub    %rdx,%rax
   e5557:	cmp    %rax,%rdx
   e555a:	jbe    e5588 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc38>
   e555c:	mov    %r12,%rcx
   e555f:	sub    %rdi,%rcx
   e5562:	cmp    %rax,%rcx
   e5565:	jb     e5588 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc38>
   e5567:	mov    0x88(%rsp),%rsi
   e556f:	lea    (%rsi,%rdi,8),%rdi
   e5573:	shl    $0x3,%rax
   e5577:	mov    %rax,%rdx
   e557a:	call   *0x221b0(%rip)        # 107730 <memcpy@GLIBC_2.14>
   e5580:	mov    %r12,%r10
   e5583:	jmp    e5686 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd36>
   e5588:	mov    %r12,%rbp
   e558b:	sub    %rdx,%rbp
   e558e:	mov    0x88(%rsp),%rax
   e5596:	lea    (%rax,%r14,8),%rsi
   e559a:	lea    (%rax,%rbp,8),%rdi
   e559e:	shl    $0x3,%rdx
   e55a2:	call   *0x22190(%rip)        # 107738 <memmove@GLIBC_2.2.5>
   e55a8:	mov    %rbp,0x90(%rsp)
   e55b0:	mov    %r12,%r10
   e55b3:	mov    %rbp,%r14
   e55b6:	jmp    e5686 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd36>
   e55bb:	movq   $0x0,0x50(%rbp)
   e55c3:	cmpq   $0x0,0x20(%rbp)
   e55c8:	je     e55eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc9b>
   e55ca:	add    $0x20,%rbp
   e55ce:	mov    %rbp,%rdi
   e55d1:	call   e2770 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   e55d6:	mov    %rdx,%rbx
   e55d9:	cmp    $0x1,%rax
   e55dd:	je     e53e5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa95>
   e55e3:	movq   $0x0,0x0(%rbp)
   e55eb:	xor    %r15d,%r15d
   e55ee:	lea    (%r15,%r13,1),%rax
   e55f2:	mov    0x20(%rsp),%r10
   e55f7:	cmp    %r10,%rax
   e55fa:	jbe    e568e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd3e>
   e5600:	jmp    e5528 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbd8>
   e5605:	xor    %r9d,%r9d
   e5608:	movq   $0x0,0x50(%rbp)
   e5610:	test   $0x7f,%r9b
   e5614:	jne    e51aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x85a>
   e561a:	jmp    e53ba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa6a>
   e561f:	mov    $0x1,%edx
   e5624:	mov    $0x8,%ecx
   e5629:	mov    $0x8,%r8d
   e562f:	lea    0x80(%rsp),%rdi
   e5637:	mov    %r13,%rsi
   e563a:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e563f:	mov    0x80(%rsp),%r12
   e5647:	mov    0x90(%rsp),%r14
   e564f:	mov    0x98(%rsp),%r13
   e5657:	mov    0x20(%rsp),%rax
   e565c:	sub    %r13,%rax
   e565f:	mov    %r12,0xd8(%rsp)
   e5667:	mov    %r14,%rcx
   e566a:	cmp    %rax,%r14
   e566d:	ja     e5546 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbf6>
   e5673:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5680:	mov    %r12,%r10
   e5683:	mov    %rcx,%r14
   e5686:	mov    0x100(%rsp),%r12
   e568e:	lea    (%r14,%r13,1),%rax
   e5692:	cmp    %r10,%rax
   e5695:	mov    $0x0,%ecx
   e569a:	cmovae %r10,%rcx
   e569e:	sub    %rcx,%rax
   e56a1:	mov    %r10,%rcx
   e56a4:	sub    %rax,%rcx
   e56a7:	cmp    %r15,%rcx
   e56aa:	jae    e56c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd70>
   e56ac:	mov    0x88(%rsp),%rcx
   e56b4:	cmp    %rax,%r10
   e56b7:	jne    e56ce <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd7e>
   e56b9:	mov    %rbx,(%rcx)
   e56bc:	jmp    e56d2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd82>
   e56be:	xchg   %ax,%ax
   e56c0:	cmp    $0x1,%r15
   e56c4:	jne    e56e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd90>
   e56c6:	mov    0x88(%rsp),%rcx
   e56ce:	mov    %rbx,(%rcx,%rax,8)
   e56d2:	mov    $0x1,%ebx
   e56d7:	jmp    e56e2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd92>
   e56d9:	nopl   0x0(%rax)
   e56e0:	xor    %ebx,%ebx
   e56e2:	add    %r13,%rbx
   e56e5:	mov    %rbx,0x98(%rsp)
   e56ed:	mov    (%r12),%rax
   e56f1:	mov    0x10(%r12),%rsi
   e56f6:	cmp    $0x1,%rax
   e56fa:	jne    e5720 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xdd0>
   e56fc:	mov    0x18(%rsp),%r9
   e5701:	cmp    %rsi,%r9
   e5704:	jae    e668d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d3d>
   e570a:	mov    0x8(%r12),%rcx
   e570f:	mov    (%rcx,%r9,4),%edx
   e5713:	jmp    e5747 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xdf7>
   e5715:	data16 cs nopw 0x0(%rax,%rax,1)
   e5720:	mov    0x18(%rsp),%r9
   e5725:	cmp    %rsi,%r9
   e5728:	jae    e6681 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d31>
   e572e:	mov    0x8(%r12),%rcx
   e5733:	movzwl (%rcx,%r9,2),%edx
   e5738:	cmp    $0xffff,%dx
   e573c:	movzwl %dx,%edx
   e573f:	mov    $0xffffffff,%edi
   e5744:	cmove  %edi,%edx
   e5747:	mov    0x128(%rsp),%r8
   e574f:	cmp    %r8,0x130(%rsp)
   e5757:	sete   %dil
   e575b:	test   %r8,%r8
   e575e:	sete   %r8b
   e5762:	cmp    0x168(%rsp),%edx
   e5769:	setne  %dl
   e576c:	or     %dil,%r8b
   e576f:	or     %dl,%r8b
   e5772:	jne    e510d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7bd>
   e5778:	mov    0x130(%rsp),%rdx
   e5780:	lea    (%r9,%rdx,1),%rbp
   e5784:	cmp    %rsi,%rbp
   e5787:	jae    e510d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7bd>
   e578d:	mov    0x128(%rsp),%rdx
   e5795:	lea    (%r9,%rdx,1),%rdi
   e5799:	test   %al,%al
   e579b:	je     e57ae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe5e>
   e579d:	cmp    %rsi,%rdi
   e57a0:	jae    e6690 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e57a6:	mov    (%rcx,%rdi,4),%eax
   e57a9:	jmp    e5100 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7b0>
   e57ae:	cmp    %rsi,%rdi
   e57b1:	jae    e6684 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e57b7:	movzwl (%rcx,%rdi,2),%eax
   e57bb:	cmp    $0xffff,%ax
   e57bf:	movzwl %ax,%eax
   e57c2:	mov    $0xffffffff,%ecx
   e57c7:	cmove  %ecx,%eax
   e57ca:	jmp    e5100 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7b0>
   e57cf:	mov    %r14,0x20(%rsp)
   e57d4:	cmpl   $0x1,0x40(%rsp)
   e57d9:	jne    e57ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xeaf>
   e57db:	mov    0x50(%rsp),%rdx
   e57e0:	cmp    %rdx,%r9
   e57e3:	jae    e57ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xeaf>
   e57e5:	mov    0x48(%rsp),%r13
   e57ea:	lea    0x1c8(%rsp),%r15
   e57f2:	mov    0x230(%rsp),%r14
   e57fa:	jmp    e58ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e57ff:	mov    0x78(%rsp),%rax
   e5804:	mov    0x68(%rax),%rcx
   e5808:	mov    0x70(%rax),%rsi
   e580c:	cmp    $0x1,%rsi
   e5810:	lea    0x1c8(%rsp),%r15
   e5818:	mov    0x230(%rsp),%r14
   e5820:	jne    e582f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xedf>
   e5822:	mov    0x8(%rcx),%r13
   e5826:	mov    0x10(%rax),%rdx
   e582a:	jmp    e58ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e582f:	cmpl   $0x1,0x18(%rax)
   e5833:	jne    e5851 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf01>
   e5835:	mov    0x18(%rsp),%rdx
   e583a:	cmp    0x20(%rax),%rdx
   e583e:	jb     e5851 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf01>
   e5840:	mov    0x28(%rax),%rdx
   e5844:	mov    $0x1,%r13d
   e584a:	cmp    %rdx,0x18(%rsp)
   e584f:	jb     e58ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e5851:	test   %rsi,%rsi
   e5854:	je     e6727 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dd7>
   e585a:	mov    %rsi,%rdi
   e585d:	xor    %edx,%edx
   e585f:	mov    0x18(%rsp),%r11
   e5864:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5870:	mov    %rdx,%r8
   e5873:	mov    %rdi,%r9
   e5876:	shr    $1,%r9
   e5879:	add    %r9,%rdx
   e587c:	mov    %rdx,%r10
   e587f:	shl    $0x4,%r10
   e5883:	cmp    %r11,(%rcx,%r10,1)
   e5887:	cmova  %r8,%rdx
   e588b:	sub    %r9,%rdi
   e588e:	cmp    $0x1,%rdi
   e5892:	ja     e5870 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf20>
   e5894:	mov    %rdx,%rdi
   e5897:	shl    $0x4,%rdi
   e589b:	cmp    (%rcx,%rdi,1),%r11
   e589f:	sbb    $0xffffffffffffffff,%rdx
   e58a3:	lea    -0x1(%rdx),%rdi
   e58a7:	cmp    %rsi,%rdi
   e58aa:	jae    e6730 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1de0>
   e58b0:	shl    $0x4,%rdi
   e58b4:	mov    0x8(%rcx,%rdi,1),%r13
   e58b9:	cmp    %rsi,%rdx
   e58bc:	jae    e5826 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xed6>
   e58c2:	shl    $0x4,%rdx
   e58c6:	mov    (%rcx,%rdx,1),%rdx
   e58ca:	movq   $0x1,0x40(%rsp)
   e58d3:	mov    %r13,0x48(%rsp)
   e58d8:	mov    %rdx,0x50(%rsp)
   e58dd:	mov    0x18(%rsp),%rax
   e58e2:	lea    -0x1(%rax),%rcx
   e58e6:	mov    0x38(%rsp),%rax
   e58eb:	mov    (%rax),%rdi
   e58ee:	mov    0x10(%rax),%rsi
   e58f2:	cmp    $0x1,%rdi
   e58f6:	jne    e5934 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xfe4>
   e58f8:	cmp    %rsi,%rcx
   e58fb:	jae    e66f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1da8>
   e5901:	mov    0x38(%rsp),%rdx
   e5906:	mov    0x8(%rdx),%rax
   e590a:	mov    0x18(%rsp),%rcx
   e590f:	mov    -0x4(%rax,%rcx,4),%ecx
   e5913:	cmp    $0xffffffff,%ecx
   e5916:	jne    e5962 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1012>
   e5918:	cmp    %rsi,%rbp
   e591b:	jae    e6702 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db2>
   e5921:	mov    (%rax,%rbp,4),%r9d
   e5925:	cmp    $0xffffffff,%r9d
   e5929:	jne    e5a77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1127>
   e592f:	jmp    e5d43 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e5934:	cmp    %rsi,%rcx
   e5937:	jae    e66f3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1da3>
   e593d:	mov    0x38(%rsp),%rax
   e5942:	mov    0x8(%rax),%rax
   e5946:	mov    0x18(%rsp),%rcx
   e594b:	movzwl -0x2(%rax,%rcx,2),%ecx
   e5950:	cmp    $0xffff,%cx
   e5954:	je     e5a5c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x110c>
   e595a:	movzwl %cx,%ecx
   e595d:	mov    0x38(%rsp),%rdx
   e5962:	mov    %ecx,%ecx
   e5964:	mov    0x50(%rdx),%rax
   e5968:	cmp    %rcx,%rax
   e596b:	jbe    e66da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d8a>
   e5971:	mov    0x48(%rdx),%rax
   e5975:	mov    %rcx,%r12
   e5978:	mov    (%rax,%rcx,8),%r9
   e597c:	mov    0x18(%rsp),%r8
   e5981:	sub    %r9,%r8
   e5984:	jne    e59fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x10ad>
   e5986:	mov    %rbp,%rax
   e5989:	sub    0x18(%rsp),%rax
   e598e:	mov    $0x0,%ecx
   e5993:	cmovae %rax,%rcx
   e5997:	add    %rcx,%r9
   e599a:	xor    %eax,%eax
   e599c:	cmp    0x120(%rsp),%r9
   e59a4:	setb   %al
   e59a7:	mov    %eax,(%rsp)
   e59aa:	lea    0x350(%rsp),%rdi
   e59b2:	xor    %esi,%esi
   e59b4:	mov    %r12d,%edx
   e59b7:	mov    %r12d,%ecx
   e59ba:	mov    %r13,%r9
   e59bd:	call   f49a0 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e59c2:	test   %rax,%rax
   e59c5:	jne    e5f96 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1646>
   e59cb:	mov    0x38(%rsp),%rax
   e59d0:	mov    (%rax),%rdi
   e59d3:	mov    0x10(%rax),%rsi
   e59d7:	test   %rdi,%rdi
   e59da:	je     e5a3b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x10eb>
   e59dc:	cmp    %rsi,%rbp
   e59df:	jae    e6702 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db2>
   e59e5:	mov    0x38(%rsp),%rax
   e59ea:	mov    0x8(%rax),%rax
   e59ee:	mov    (%rax,%rbp,4),%r9d
   e59f2:	cmp    $0xffffffff,%r9d
   e59f6:	jne    e5a77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1127>
   e59f8:	jmp    e5d43 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e59fd:	lea    -0x1(%r8),%rax
   e5a01:	cmpb   $0x0,0xf(%rsp)
   e5a06:	je     e5bdd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x128d>
   e5a0c:	cmp    0xd0(%rsp),%rax
   e5a14:	jae    e6717 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dc7>
   e5a1a:	mov    0x108(%rsp),%rax
   e5a22:	mov    -0x4(%rax,%r8,4),%r11d
   e5a27:	cmpq   $0x0,0x228(%rsp)
   e5a30:	jne    e5c19 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x12c9>
   e5a36:	jmp    e5986 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e5a3b:	cmp    %rsi,%rbp
   e5a3e:	jae    e66fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dad>
   e5a44:	mov    0x38(%rsp),%rax
   e5a49:	mov    0x8(%rax),%rax
   e5a4d:	movzwl (%rax,%rbp,2),%eax
   e5a51:	cmp    $0xffff,%ax
   e5a55:	jne    e5a73 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1123>
   e5a57:	jmp    e5d43 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e5a5c:	cmp    %rsi,%rbp
   e5a5f:	jae    e66fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dad>
   e5a65:	movzwl (%rax,%rbp,2),%eax
   e5a69:	cmp    $0xffff,%ax
   e5a6d:	je     e5d43 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e5a73:	movzwl %ax,%r9d
   e5a77:	mov    %r9d,%edi
   e5a7a:	cmp    %rdi,0x220(%rsp)
   e5a82:	jbe    e66e2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d92>
   e5a88:	mov    0x250(%rsp),%rax
   e5a90:	mov    (%rax,%rdi,8),%rax
   e5a94:	add    %rbp,%rax
   e5a97:	cmpb   $0x0,0xf(%rsp)
   e5a9c:	je     e5b8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x123f>
   e5aa2:	cmp    0xd0(%rsp),%rax
   e5aaa:	jae    e6717 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dc7>
   e5ab0:	mov    0x108(%rsp),%rcx
   e5ab8:	mov    (%rcx,%rax,4),%esi
   e5abb:	cmpq   $0x0,0x228(%rsp)
   e5ac4:	je     e5bc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1277>
   e5aca:	mov    0x218(%rsp),%rax
   e5ad2:	mov    %rdi,%r8
   e5ad5:	xor    %rdi,%rax
   e5ad8:	movabs $0x5851f42d4c957f2d,%rdi
   e5ae2:	mul    %rdi
   e5ae5:	xor    %rdx,%rax
   e5ae8:	mov    %esi,%ecx
   e5aea:	xor    %rcx,%rax
   e5aed:	mul    %rdi
   e5af0:	mov    %rax,%rcx
   e5af3:	xor    %rdx,%rcx
   e5af6:	mov    %rcx,%rax
   e5af9:	mulq   0x210(%rsp)
   e5b01:	xor    %rax,%rdx
   e5b04:	rol    %cl,%rdx
   e5b07:	mov    %rdx,%rax
   e5b0a:	shr    $0x39,%rax
   e5b0e:	movd   %eax,%xmm0
   e5b12:	punpcklbw %xmm0,%xmm0
   e5b16:	pshuflw $0x0,%xmm0,%xmm0
   e5b1b:	pshufd $0x44,%xmm0,%xmm0
   e5b20:	xor    %eax,%eax
   e5b22:	and    0x110(%rsp),%rdx
   e5b2a:	movdqu (%r14,%rdx,1),%xmm1
   e5b30:	movdqa %xmm1,%xmm2
   e5b34:	pcmpeqb %xmm0,%xmm2
   e5b38:	pmovmskb %xmm2,%ecx
   e5b3c:	test   %ecx,%ecx
   e5b3e:	je     e5b6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x121e>
   e5b40:	tzcnt  %ecx,%edi
   e5b44:	add    %rdx,%rdi
   e5b47:	and    0x110(%rsp),%rdi
   e5b4f:	neg    %rdi
   e5b52:	lea    (%rdi,%rdi,2),%rdi
   e5b56:	cmp    -0xc(%r14,%rdi,4),%r9d
   e5b5b:	jne    e5b64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1214>
   e5b5d:	cmp    -0x8(%r14,%rdi,4),%esi
   e5b62:	je     e5bcf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x127f>
   e5b64:	lea    -0x1(%rcx),%edi
   e5b67:	and    %cx,%di
   e5b6a:	mov    %edi,%ecx
   e5b6c:	jne    e5b40 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x11f0>
   e5b6e:	pcmpeqd %xmm2,%xmm2
   e5b72:	pcmpeqb %xmm2,%xmm1
   e5b76:	pmovmskb %xmm1,%ecx
   e5b7a:	test   %ecx,%ecx
   e5b7c:	jne    e5ce1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1391>
   e5b82:	add    %rax,%rdx
   e5b85:	add    $0x10,%rdx
   e5b89:	add    $0x10,%rax
   e5b8d:	jmp    e5b22 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x11d2>
   e5b8f:	cmp    0xd0(%rsp),%rax
   e5b97:	jae    e6707 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db7>
   e5b9d:	mov    0x108(%rsp),%rcx
   e5ba5:	movzwl (%rcx,%rax,2),%eax
   e5ba9:	cmp    $0xffff,%ax
   e5bad:	movzwl %ax,%esi
   e5bb0:	mov    $0xffffffff,%eax
   e5bb5:	cmove  %eax,%esi
   e5bb8:	cmpq   $0x0,0x228(%rsp)
   e5bc1:	jne    e5aca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x117a>
   e5bc7:	mov    %r9d,%ecx
   e5bca:	jmp    e5ce7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1397>
   e5bcf:	lea    (%r14,%rdi,4),%rax
   e5bd3:	mov    -0x4(%rax),%edi
   e5bd6:	mov    %edi,%ecx
   e5bd8:	jmp    e5ce7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1397>
   e5bdd:	cmp    0xd0(%rsp),%rax
   e5be5:	jae    e6707 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db7>
   e5beb:	mov    0x108(%rsp),%rax
   e5bf3:	movzwl -0x2(%rax,%r8,2),%eax
   e5bf9:	cmp    $0xffff,%ax
   e5bfd:	movzwl %ax,%r11d
   e5c01:	mov    $0xffffffff,%eax
   e5c06:	cmove  %eax,%r11d
   e5c0a:	cmpq   $0x0,0x228(%rsp)
   e5c13:	je     e5986 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e5c19:	mov    %r11d,%eax
   e5c1c:	xor    0x218(%rsp),%rax
   e5c24:	movabs $0x5851f42d4c957f2d,%rcx
   e5c2e:	mul    %rcx
   e5c31:	xor    %r12,%rdx
   e5c34:	xor    %rdx,%rax
   e5c37:	mul    %rcx
   e5c3a:	mov    %rax,%rcx
   e5c3d:	xor    %rdx,%rcx
   e5c40:	mov    %rcx,%rax
   e5c43:	mulq   0x210(%rsp)
   e5c4b:	xor    %rax,%rdx
   e5c4e:	rol    %cl,%rdx
   e5c51:	mov    %rdx,%rax
   e5c54:	shr    $0x39,%rax
   e5c58:	movd   %eax,%xmm0
   e5c5c:	punpcklbw %xmm0,%xmm0
   e5c60:	pshuflw $0x0,%xmm0,%xmm0
   e5c65:	pshufd $0x44,%xmm0,%xmm0
   e5c6a:	xor    %eax,%eax
   e5c6c:	and    0x110(%rsp),%rdx
   e5c74:	movdqu (%r14,%rdx,1),%xmm1
   e5c7a:	movdqa %xmm1,%xmm2
   e5c7e:	pcmpeqb %xmm0,%xmm2
   e5c82:	pmovmskb %xmm2,%ecx
   e5c86:	test   %ecx,%ecx
   e5c88:	je     e5cc0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1370>
   e5c8a:	tzcnt  %ecx,%r10d
   e5c8f:	add    %rdx,%r10
   e5c92:	and    0x110(%rsp),%r10
   e5c9a:	neg    %r10
   e5c9d:	lea    (%r10,%r10,2),%r10
   e5ca1:	cmp    -0xc(%r14,%r10,4),%r11d
   e5ca6:	jne    e5cb3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1363>
   e5ca8:	cmp    -0x8(%r14,%r10,4),%r12d
   e5cad:	je     e59d7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1087>
   e5cb3:	lea    -0x1(%rcx),%r10d
   e5cb7:	and    %cx,%r10w
   e5cbb:	mov    %r10d,%ecx
   e5cbe:	jne    e5c8a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x133a>
   e5cc0:	pcmpeqd %xmm2,%xmm2
   e5cc4:	pcmpeqb %xmm2,%xmm1
   e5cc8:	pmovmskb %xmm1,%ecx
   e5ccc:	test   %ecx,%ecx
   e5cce:	jne    e5986 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e5cd4:	add    %rax,%rdx
   e5cd7:	add    $0x10,%rdx
   e5cdb:	add    $0x10,%rax
   e5cdf:	jmp    e5c6c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x131c>
   e5ce1:	mov    %r9d,%ecx
   e5ce4:	mov    %r8,%rdi
   e5ce7:	mov    0x38(%rsp),%rdx
   e5cec:	mov    0x50(%rdx),%rsi
   e5cf0:	cmp    %rdi,%rsi
   e5cf3:	jbe    e66ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d9a>
   e5cf9:	mov    0x18(%rsp),%r8
   e5cfe:	sub    %r8,%rbp
   e5d01:	mov    $0x0,%eax
   e5d06:	cmovae %rbp,%rax
   e5d0a:	mov    0x48(%rdx),%rdx
   e5d0e:	add    (%rdx,%rdi,8),%rax
   e5d12:	xor    %edx,%edx
   e5d14:	cmp    0x120(%rsp),%rax
   e5d1c:	setb   %dl
   e5d1f:	mov    %edx,(%rsp)
   e5d22:	lea    0x350(%rsp),%rdi
   e5d2a:	mov    $0x1,%esi
   e5d2f:	mov    %r9d,%edx
   e5d32:	mov    %r13,%r9
   e5d35:	call   f49a0 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e5d3a:	test   %rax,%rax
   e5d3d:	jne    e5f96 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1646>
   e5d43:	mov    0xf0(%rsp),%rax
   e5d4b:	cmp    $0x3,%rax
   e5d4f:	jb     e5d8d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x143d>
   e5d51:	mov    0xe8(%rsp),%rcx
   e5d59:	cmp    %rax,%rcx
   e5d5c:	mov    0x20(%rsp),%r14
   e5d61:	mov    0x18(%rsp),%rdx
   e5d66:	je     e5dbf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x146f>
   e5d68:	mov    0xe0(%rsp),%rax
   e5d70:	mov    %rdx,(%rax,%rcx,8)
   e5d74:	inc    %rcx
   e5d77:	mov    %rcx,0xe8(%rsp)
   e5d7f:	test   %rbx,%rbx
   e5d82:	jne    e50b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e5d88:	jmp    e5e20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e5d8d:	lea    0xe0(%rsp),%rcx
   e5d95:	lea    0xf0(%rsp),%rdx
   e5d9d:	cmp    $0x2,%rax
   e5da1:	mov    0x20(%rsp),%r14
   e5da6:	mov    0x18(%rsp),%rsi
   e5dab:	je     e5de0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1490>
   e5dad:	mov    %rsi,(%rcx,%rax,8)
   e5db1:	incq   (%rdx)
   e5db4:	test   %rbx,%rbx
   e5db7:	jne    e50b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e5dbd:	jmp    e5e20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e5dbf:	lea    0xe0(%rsp),%rdi
   e5dc7:	call   f4e50 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e5dcc:	mov    0xe8(%rsp),%rcx
   e5dd4:	mov    0x20(%rsp),%r14
   e5dd9:	mov    0x18(%rsp),%rdx
   e5dde:	jmp    e5d68 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1418>
   e5de0:	lea    0xe0(%rsp),%rdi
   e5de8:	call   f4e50 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e5ded:	mov    0xe0(%rsp),%rcx
   e5df5:	mov    0xe8(%rsp),%rax
   e5dfd:	lea    0xe8(%rsp),%rdx
   e5e05:	mov    0x20(%rsp),%r14
   e5e0a:	mov    0x18(%rsp),%rsi
   e5e0f:	mov    %rsi,(%rcx,%rax,8)
   e5e13:	incq   (%rdx)
   e5e16:	test   %rbx,%rbx
   e5e19:	jne    e50b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e5e1f:	nop
   e5e20:	cmpq   $0x0,0xd8(%rsp)
   e5e29:	je     e5e39 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14e9>
   e5e2b:	mov    0x88(%rsp),%rdi
   e5e33:	call   *0x218ef(%rip)        # 107728 <free@GLIBC_2.2.5>
   e5e39:	mov    0xe0(%rsp),%r14
   e5e41:	movups 0xe8(%rsp),%xmm0
   e5e49:	movaps %xmm0,0x20(%rsp)
   e5e4e:	lea    0x40(%rsp),%rbp
   e5e53:	mov    %rbp,%rdi
   e5e56:	lea    0x350(%rsp),%rsi
   e5e5e:	lea    0x1c8(%rsp),%r15
   e5e66:	call   f5070 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::finish>
   e5e6b:	mov    0x40(%rsp),%rax
   e5e70:	cmp    $0xfffffffffffffffe,%rax
   e5e74:	je     e6025 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e5e7a:	movdqu 0x48(%rsp),%xmm0
   e5e80:	movdqa 0x20(%rsp),%xmm1
   e5e86:	mov    %rax,0x40(%rsp)
   e5e8b:	movdqu %xmm0,0x48(%rsp)
   e5e91:	mov    %r14,0x58(%rsp)
   e5e96:	movdqu %xmm1,0x60(%rsp)
   e5e9c:	mov    0x174(%rsp),%eax
   e5ea3:	mov    %eax,0x70(%rsp)
   e5ea7:	mov    0x238(%rsp),%rax
   e5eaf:	mov    (%rax),%rsi
   e5eb2:	lea    0x398(%rsp),%rdi
   e5eba:	mov    %rbp,%rdx
   e5ebd:	call   f54d0 <<&<core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0} as core::ops::function::FnMut<(core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>,)>>::call_mut>
   e5ec2:	mov    0x398(%rsp),%r15
   e5eca:	cmp    $0xfffffffffffffffe,%r15
   e5ece:	je     e6025 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e5ed4:	cmp    $0xffffffffffffffff,%r15
   e5ed8:	je     e6043 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16f3>
   e5ede:	mov    0x3a0(%rsp),%rbx
   e5ee6:	mov    0x3a8(%rsp),%r12
   e5eee:	mov    0x3b0(%rsp),%rdi
   e5ef6:	mov    0x3c0(%rsp),%rcx
   e5efe:	movdqu 0x3b8(%rsp),%xmm0
   e5f07:	mov    0x3c8(%rsp),%rax
   e5f0f:	mov    0x118(%rsp),%rdx
   e5f17:	movzbl (%rdx),%edx
   e5f1a:	test   %dl,%dl
   e5f1c:	jne    e6054 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1704>
   e5f22:	mov    %rbx,0x2c0(%rsp)
   e5f2a:	mov    %r12,0x2c8(%rsp)
   e5f32:	mov    %rdi,0x2d0(%rsp)
   e5f3a:	movdqu %xmm0,0x2d8(%rsp)
   e5f43:	mov    %rax,0x2e8(%rsp)
   e5f4b:	mov    %r15,0x2b8(%rsp)
   e5f53:	mov    0x248(%rsp),%rbx
   e5f5b:	cmp    0x188(%rsp),%rbx
   e5f63:	mov    0xf8(%rsp),%r12
   e5f6b:	jne    e4b60 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x210>
   e5f71:	mov    $0x1,%edx
   e5f76:	mov    $0x8,%ecx
   e5f7b:	mov    $0x38,%r8d
   e5f81:	lea    0x188(%rsp),%rdi
   e5f89:	mov    %rbx,%rsi
   e5f8c:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e5f91:	jmp    e4b50 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x200>
   e5f96:	mov    %rax,%rbx
   e5f99:	cmpq   $0x0,0xd8(%rsp)
   e5fa2:	lea    0x40(%rsp),%rbp
   e5fa7:	je     e5fb7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1667>
   e5fa9:	mov    0x88(%rsp),%rdi
   e5fb1:	call   *0x21771(%rip)        # 107728 <free@GLIBC_2.2.5>
   e5fb7:	cmpq   $0x3,0xf0(%rsp)
   e5fc0:	jb     e5fd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1680>
   e5fc2:	mov    0xe0(%rsp),%rdi
   e5fca:	call   *0x21758(%rip)        # 107728 <free@GLIBC_2.2.5>
   e5fd0:	lea    0x350(%rsp),%rdi
   e5fd8:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e5fdd:	lea    0x368(%rsp),%rdi
   e5fe5:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e5fea:	movq   %rbx,%xmm0
   e5fef:	punpcklqdq 0x3d0(%rsp),%xmm0
   e5ff8:	mov    $0xffffffffffffffff,%rax
   e5fff:	jmp    e5e86 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1536>
   e6004:	test   %dl,%dl
   e6006:	je     e6032 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16e2>
   e6008:	mov    %fs:0xffffffffffffffd0,%rax
   e6011:	test   %rax,%rax
   e6014:	je     e61de <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x188e>
   e601a:	add    $0x110,%rax
   e6020:	jmp    e61fb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x18ab>
   e6025:	mov    0x118(%rsp),%r14
   e602d:	jmp    e6101 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e6032:	test   %rcx,%rcx
   e6035:	je     e4a37 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe7>
   e603b:	shr    $1,%rcx
   e603e:	jmp    e620f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x18bf>
   e6043:	mov    0x118(%rsp),%r14
   e604b:	movb   $0x1,(%r14)
   e604f:	jmp    e6101 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e6054:	cmp    $0x3,%rcx
   e6058:	jb     e6060 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1710>
   e605a:	call   *0x216c8(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6060:	test   %r12,%r12
   e6063:	je     e60eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x179b>
   e6069:	xor    %r13d,%r13d
   e606c:	jmp    e6081 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1731>
   e606e:	mov    0x20(%rsp),%rdi
   e6073:	call   *0x216af(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6079:	inc    %r13
   e607c:	cmp    %r12,%r13
   e607f:	je     e60eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x179b>
   e6081:	lea    0x0(,%r13,8),%rcx
   e6089:	add    %r13,%rcx
   e608c:	lea    (%rbx,%rcx,8),%rax
   e6090:	cmpq   $0x0,(%rbx,%rcx,8)
   e6095:	je     e60b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1760>
   e6097:	cmpq   $0x11,0x18(%rax)
   e609c:	jb     e6079 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1729>
   e609e:	mov    0x8(%rax),%rdi
   e60a2:	jmp    e6073 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1723>
   e60a4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e60b0:	mov    0x10(%rax),%rbp
   e60b4:	test   %rbp,%rbp
   e60b7:	je     e6079 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1729>
   e60b9:	mov    0x8(%rax),%rax
   e60bd:	mov    %rax,0x20(%rsp)
   e60c2:	lea    0x18(%rax),%r14
   e60c6:	jmp    e60d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1789>
   e60c8:	nopl   0x0(%rax,%rax,1)
   e60d0:	add    $0x20,%r14
   e60d4:	dec    %rbp
   e60d7:	je     e606e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x171e>
   e60d9:	cmpq   $0x11,(%r14)
   e60dd:	jb     e60d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1780>
   e60df:	mov    -0x10(%r14),%rdi
   e60e3:	call   *0x2163f(%rip)        # 107728 <free@GLIBC_2.2.5>
   e60e9:	jmp    e60d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1780>
   e60eb:	test   %r15,%r15
   e60ee:	mov    0x118(%rsp),%r14
   e60f6:	je     e6101 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e60f8:	mov    %rbx,%rdi
   e60fb:	call   *0x21627(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6101:	mov    0x198(%rsp),%rax
   e6109:	mov    %rax,0x1b0(%rsp)
   e6111:	movups 0x188(%rsp),%xmm0
   e6119:	mov    %r14,0x1b8(%rsp)
   e6121:	mov    0x238(%rsp),%rax
   e6129:	mov    %rax,0x1c0(%rsp)
   e6131:	movaps %xmm0,0x40(%rsp)
   e6136:	movaps 0x1d0(%rsp),%xmm0
   e613e:	movaps 0x1e0(%rsp),%xmm1
   e6146:	movaps 0x1f0(%rsp),%xmm2
   e614e:	movaps %xmm0,0x70(%rsp)
   e6153:	movaps %xmm1,0x80(%rsp)
   e615b:	movaps %xmm2,0x90(%rsp)
   e6163:	mov    0x1b0(%rsp),%rax
   e616b:	mov    %rax,0x50(%rsp)
   e6170:	mov    0x1b8(%rsp),%rax
   e6178:	mov    %rax,0x58(%rsp)
   e617d:	mov    0x1c0(%rsp),%rax
   e6185:	mov    %rax,0x60(%rsp)
   e618a:	mov    0x1c8(%rsp),%rax
   e6192:	mov    %rax,0x68(%rsp)
   e6197:	lea    0x40(%rsp),%rsi
   e619c:	mov    0x208(%rsp),%rdi
   e61a4:	call   f3470 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e61a9:	cmpq   $0x0,0x68(%rsp)
   e61af:	je     e61bc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x186c>
   e61b1:	mov    0x70(%rsp),%rdi
   e61b6:	call   *0x2156c(%rip)        # 107728 <free@GLIBC_2.2.5>
   e61bc:	cmpq   $0x0,0x80(%rsp)
   e61c5:	je     e4a06 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e61cb:	mov    0x88(%rsp),%rdi
   e61d3:	call   *0x2154f(%rip)        # 107728 <free@GLIBC_2.2.5>
   e61d9:	jmp    e4a06 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e61de:	mov    %rbp,%r13
   e61e1:	mov    %rsi,%r15
   e61e4:	mov    %r12,%rbp
   e61e7:	mov    %rcx,%r12
   e61ea:	call   7f660 <rayon_core::registry::global_registry>
   e61ef:	mov    %r12,%rcx
   e61f2:	mov    %rbp,%r12
   e61f5:	mov    %r15,%rsi
   e61f8:	mov    %r13,%rbp
   e61fb:	mov    (%rax),%rax
   e61fe:	mov    0x208(%rax),%rax
   e6205:	shr    $1,%rcx
   e6208:	cmp    %rax,%rcx
   e620b:	cmovbe %rax,%rcx
   e620f:	mov    %rcx,0x178(%rsp)
   e6217:	mov    %rsi,0x1a0(%rsp)
   e621f:	sub    %rsi,%rbx
   e6222:	jb     e64dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b8d>
   e6228:	imul   $0x70,%rsi,%r13
   e622c:	add    %r12,%r13
   e622f:	mov    %r12,%rdi
   e6232:	mov    0x8(%r14),%r12
   e6236:	mov    0x10(%r14),%r15
   e623a:	mov    0x18(%r14),%r14
   e623e:	mov    %fs:0xffffffffffffffd0,%rdx
   e6247:	test   %rdx,%rdx
   e624a:	je     e62b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1969>
   e624c:	lea    0x138(%rsp),%rax
   e6254:	mov    %rax,0x40(%rsp)
   e6259:	lea    0x1a0(%rsp),%rax
   e6261:	mov    %rax,0x48(%rsp)
   e6266:	lea    0x178(%rsp),%rcx
   e626e:	mov    %rcx,0x50(%rsp)
   e6273:	mov    %r13,0x58(%rsp)
   e6278:	mov    %rbx,0x60(%rsp)
   e627d:	mov    %rbp,0x68(%rsp)
   e6282:	mov    %r12,0x70(%rsp)
   e6287:	mov    %r15,0x78(%rsp)
   e628c:	mov    %r14,0x80(%rsp)
   e6294:	mov    %rax,0x88(%rsp)
   e629c:	mov    %rcx,0x90(%rsp)
   e62a4:	mov    %rdi,0x98(%rsp)
   e62ac:	mov    %rsi,0xa0(%rsp)
   e62b4:	jmp    e6362 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a12>
   e62b9:	mov    %rdi,0xf8(%rsp)
   e62c1:	mov    %rsi,0x20(%rsp)
   e62c6:	call   7f660 <rayon_core::registry::global_registry>
   e62cb:	mov    (%rax),%rsi
   e62ce:	mov    %fs:0xffffffffffffffd0,%rdx
   e62d7:	test   %rdx,%rdx
   e62da:	je     e652e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1bde>
   e62e0:	cmp    %rsi,0x110(%rdx)
   e62e7:	mov    0x20(%rsp),%rax
   e62ec:	lea    0x138(%rsp),%rcx
   e62f4:	jne    e65de <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1c8e>
   e62fa:	mov    %rcx,0x40(%rsp)
   e62ff:	lea    0x1a0(%rsp),%rcx
   e6307:	mov    %rcx,0x48(%rsp)
   e630c:	lea    0x178(%rsp),%rsi
   e6314:	mov    %rsi,0x50(%rsp)
   e6319:	mov    %r13,0x58(%rsp)
   e631e:	mov    %rbx,0x60(%rsp)
   e6323:	mov    %rbp,0x68(%rsp)
   e6328:	mov    %r12,0x70(%rsp)
   e632d:	mov    %r15,0x78(%rsp)
   e6332:	mov    %r14,0x80(%rsp)
   e633a:	mov    %rcx,0x88(%rsp)
   e6342:	mov    %rsi,0x90(%rsp)
   e634a:	mov    0xf8(%rsp),%rcx
   e6352:	mov    %rcx,0x98(%rsp)
   e635a:	mov    %rax,0xa0(%rsp)
   e6362:	mov    %rbp,0xa8(%rsp)
   e636a:	mov    %r12,0xb0(%rsp)
   e6372:	mov    %r15,0xb8(%rsp)
   e637a:	mov    %r14,0xc0(%rsp)
   e6382:	lea    0x2f0(%rsp),%rdi
   e638a:	lea    0x40(%rsp),%rsi
   e638f:	xor    %ecx,%ecx
   e6391:	call   f5d90 <rayon_core::join::join_context::<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}>
   e6396:	mov    0x2f0(%rsp),%rdx
   e639e:	mov    0x2f8(%rsp),%rcx
   e63a6:	mov    0x300(%rsp),%rax
   e63ae:	mov    0x318(%rsp),%rsi
   e63b6:	mov    %rsi,0x50(%rsp)
   e63bb:	movups 0x308(%rsp),%xmm0
   e63c3:	movaps %xmm0,0x40(%rsp)
   e63c8:	test   %rcx,%rcx
   e63cb:	mov    0x208(%rsp),%r8
   e63d3:	je     e6404 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ab4>
   e63d5:	mov    0x40(%rsp),%rsi
   e63da:	movq   $0x0,0x40(%rsp)
   e63e3:	test   %rsi,%rsi
   e63e6:	je     e642c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1adc>
   e63e8:	mov    %rsi,0x18(%rcx)
   e63ec:	mov    %rcx,0x20(%rsi)
   e63f0:	mov    0x48(%rsp),%rcx
   e63f5:	xorps  %xmm0,%xmm0
   e63f8:	add    0x50(%rsp),%rax
   e63fd:	movups %xmm0,0x48(%rsp)
   e6402:	jmp    e642c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1adc>
   e6404:	mov    0x40(%rsp),%rsi
   e6409:	mov    0x48(%rsp),%rcx
   e640e:	mov    %rdx,0x40(%rsp)
   e6413:	movq   $0x0,0x48(%rsp)
   e641c:	mov    0x50(%rsp),%rdi
   e6421:	mov    %rax,0x50(%rsp)
   e6426:	mov    %rsi,%rdx
   e6429:	mov    %rdi,%rax
   e642c:	mov    %rdx,(%r8)
   e642f:	mov    %rcx,0x8(%r8)
   e6433:	mov    %rax,0x10(%r8)
   e6437:	lea    0x40(%rsp),%rdi
   e643c:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e6441:	jmp    e4a06 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e6446:	movq   $0x0,0x50(%rbp)
   e644e:	mov    %rax,%rdx
   e6451:	lea    0x20970(%rip),%rcx        # 106dc8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   e6458:	mov    %rax,%rsi
   e645b:	call   70c30 <core::slice::index::slice_index_fail>
   e6460:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6465:	lea    -0xc8178(%rip),%rdi        # 1e2f4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5a7>
   e646c:	lea    0x20ce5(%rip),%rdx        # 107158 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   e6473:	mov    $0x53,%esi
   e6478:	call   6ed60 <core::panicking::panic_fmt>
   e647d:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6482:	mov    %rdx,%rax
   e6485:	lea    0x20f6c(%rip),%rcx        # 1073f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4080>
   e648c:	jmp    e6458 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e648e:	lea    -0xc85cd(%rip),%rdi        # 1dec8 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x17b>
   e6495:	lea    0x20a94(%rip),%rdx        # 106f30 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3bb8>
   e649c:	mov    $0x11,%esi
   e64a1:	call   71e10 <core::option::expect_failed>
   e64a6:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e64ab:	mov    $0x8,%eax
   e64b0:	xor    %edi,%edi
   e64b2:	lea    0x20da7(%rip),%rcx        # 107260 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   e64b9:	jmp    e6458 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e64bb:	mov    %rsi,%rdi
   e64be:	jmp    e644e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1afe>
   e64c0:	mov    %rdx,%rax
   e64c3:	lea    0x20946(%rip),%rcx        # 106e10 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   e64ca:	jmp    e6458 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e64cc:	mov    $0x8,%edi
   e64d1:	lea    0x20950(%rip),%rcx        # 106e28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ab0>
   e64d8:	jmp    e6458 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e64dd:	lea    -0xc84a7(%rip),%rdi        # 1e03d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f0>
   e64e4:	lea    0x20e95(%rip),%rdx        # 107380 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4008>
   e64eb:	mov    $0x13,%esi
   e64f0:	call   6ed60 <core::panicking::panic_fmt>
   e64f5:	xor    %eax,%eax
   e64f7:	jmp    e64fb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1bab>
   e64f9:	mov    $0x1,%al
   e64fb:	mov    %al,0x280(%rsp)
   e6502:	lea    -0xc7f89(%rip),%rdi        # 1e580 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x833>
   e6509:	lea    0x205e0(%rip),%rcx        # 106af0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3778>
   e6510:	lea    0x20ec9(%rip),%r8        # 1073e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4068>
   e6517:	lea    0x280(%rsp),%rdx
   e651f:	mov    $0x13,%esi
   e6524:	call   71e50 <core::result::unwrap_failed>
   e6529:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e652e:	lea    0x138(%rsp),%rax
   e6536:	mov    %rax,0x40(%rsp)
   e653b:	lea    0x1a0(%rsp),%rax
   e6543:	mov    %rax,0x48(%rsp)
   e6548:	lea    0x178(%rsp),%rcx
   e6550:	mov    %rcx,0x50(%rsp)
   e6555:	mov    %r13,0x58(%rsp)
   e655a:	mov    %rbx,0x60(%rsp)
   e655f:	mov    %rbp,0x68(%rsp)
   e6564:	mov    %r12,0x70(%rsp)
   e6569:	mov    %r15,0x78(%rsp)
   e656e:	mov    %r14,0x80(%rsp)
   e6576:	mov    %rax,0x88(%rsp)
   e657e:	mov    %rcx,0x90(%rsp)
   e6586:	mov    0xf8(%rsp),%rax
   e658e:	mov    %rax,0x98(%rsp)
   e6596:	mov    0x20(%rsp),%rax
   e659b:	mov    %rax,0xa0(%rsp)
   e65a3:	mov    %rbp,0xa8(%rsp)
   e65ab:	mov    %r12,0xb0(%rsp)
   e65b3:	mov    %r15,0xb8(%rsp)
   e65bb:	mov    %r14,0xc0(%rsp)
   e65c3:	sub    $0xffffffffffffff80,%rsi
   e65c7:	lea    0x2f0(%rsp),%rdi
   e65cf:	lea    0x40(%rsp),%rdx
   e65d4:	call   f6250 <<rayon_core::registry::Registry>::in_worker_cold::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e65d9:	jmp    e6396 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a46>
   e65de:	mov    %rcx,0x40(%rsp)
   e65e3:	lea    0x1a0(%rsp),%rcx
   e65eb:	mov    %rcx,0x48(%rsp)
   e65f0:	lea    0x178(%rsp),%rdi
   e65f8:	mov    %rdi,0x50(%rsp)
   e65fd:	mov    %r13,0x58(%rsp)
   e6602:	mov    %rbx,0x60(%rsp)
   e6607:	mov    %rbp,0x68(%rsp)
   e660c:	mov    %r12,0x70(%rsp)
   e6611:	mov    %r15,0x78(%rsp)
   e6616:	mov    %r14,0x80(%rsp)
   e661e:	mov    %rcx,0x88(%rsp)
   e6626:	mov    %rdi,0x90(%rsp)
   e662e:	mov    0xf8(%rsp),%rcx
   e6636:	mov    %rcx,0x98(%rsp)
   e663e:	mov    %rax,0xa0(%rsp)
   e6646:	mov    %rbp,0xa8(%rsp)
   e664e:	mov    %r12,0xb0(%rsp)
   e6656:	mov    %r15,0xb8(%rsp)
   e665e:	mov    %r14,0xc0(%rsp)
   e6666:	sub    $0xffffffffffffff80,%rsi
   e666a:	lea    0x2f0(%rsp),%rdi
   e6672:	lea    0x40(%rsp),%rcx
   e6677:	call   f6350 <<rayon_core::registry::Registry>::in_worker_cross::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e667c:	jmp    e6396 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a46>
   e6681:	mov    %r9,%rdi
   e6684:	lea    0x20665(%rip),%rax        # 106cf0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3978>
   e668b:	jmp    e6697 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e668d:	mov    %r9,%rdi
   e6690:	lea    0x20671(%rip),%rax        # 106d08 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e6697:	mov    %rax,0x10(%rsp)
   e669c:	mov    0x10(%rsp),%rdx
   e66a1:	call   6fcfb <core::panicking::panic_bounds_check>
   e66a6:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e66a8:	shr    $0x7,%r9
   e66ac:	xor    %esi,%esi
   e66ae:	mov    %r9,%rdi
   e66b1:	jmp    e669c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d4c>
   e66b3:	mov    %rdx,%rdi
   e66b6:	jmp    e669c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d4c>
   e66b8:	mov    %r14,%rdi
   e66bb:	lea    0x1ff4e(%rip),%rdx        # 106610 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   e66c2:	call   6fcfb <core::panicking::panic_bounds_check>
   e66c7:	jmp    e66d8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e66c9:	mov    $0x8,%edi
   e66ce:	mov    $0x80,%esi
   e66d3:	call   690eb <alloc::raw_vec::handle_error>
   e66d8:	ud2
   e66da:	mov    %rcx,%rdi
   e66dd:	mov    %rax,%rsi
   e66e0:	jmp    e66ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d9a>
   e66e2:	mov    0x220(%rsp),%rsi
   e66ea:	lea    0x1ff1f(%rip),%rax        # 106610 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   e66f1:	jmp    e6697 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e66f3:	mov    %rcx,%rdi
   e66f6:	jmp    e6684 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e66f8:	mov    %rcx,%rdi
   e66fb:	jmp    e6690 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e66fd:	mov    %rbp,%rdi
   e6700:	jmp    e6684 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e6702:	mov    %rbp,%rdi
   e6705:	jmp    e6690 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e6707:	mov    %rax,%rdi
   e670a:	mov    0xd0(%rsp),%rsi
   e6712:	jmp    e6684 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e6717:	mov    %rax,%rdi
   e671a:	mov    0xd0(%rsp),%rsi
   e6722:	jmp    e6690 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e6727:	mov    $0xffffffffffffffff,%rdi
   e672e:	xor    %esi,%esi
   e6730:	lea    0x209d9(%rip),%rax        # 107110 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d98>
   e6737:	jmp    e6697 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e673c:	mov    %rax,%rbx
   e673f:	lea    0x68(%rsp),%r15
   e6744:	mov    %r15,%rdi
   e6747:	call   f3560 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e674c:	mov    %rbx,%rdi
   e674f:	call   101800 <_Unwind_Resume@plt>
   e6754:	jmp    e67e7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e97>
   e6759:	mov    %rax,%rbx
   e675c:	lea    0x2b8(%rsp),%rdi
   e6764:	call   f3310 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>
   e6769:	jmp    e679c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e4c>
   e676b:	mov    %rax,%rbx
   e676e:	mov    %r15,%rdi
   e6771:	call   f3560 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e6776:	mov    %rbx,%rdi
   e6779:	call   101800 <_Unwind_Resume@plt>
   e677e:	mov    %rax,%rbx
   e6781:	pshufd $0xee,0x20(%rsp),%xmm0
   e6788:	movq   %xmm0,%rax
   e678d:	cmp    $0x3,%rax
   e6791:	jb     e679c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e4c>
   e6793:	mov    %r14,%rdi
   e6796:	call   *0x20f8c(%rip)        # 107728 <free@GLIBC_2.2.5>
   e679c:	lea    0x1c8(%rsp),%r15
   e67a4:	jmp    e6884 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f34>
   e67a9:	mov    %rax,%rbx
   e67ac:	jmp    e6884 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f34>
   e67b1:	mov    %rax,%rbx
   e67b4:	jmp    e6803 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e67b6:	mov    %rax,%rbx
   e67b9:	mov    0x80(%rsp),%rax
   e67c1:	mov    %rax,0xd8(%rsp)
   e67c9:	jmp    e67ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e9a>
   e67cb:	jmp    e67cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e7f>
   e67cd:	jmp    e67cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e7f>
   e67cf:	mov    %rax,%rbx
   e67d2:	cmpq   $0x0,0x140(%rsp)
   e67db:	je     e6803 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e67dd:	mov    0x148(%rsp),%rdi
   e67e5:	jmp    e67fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ead>
   e67e7:	mov    %rax,%rbx
   e67ea:	cmpq   $0x0,0xd8(%rsp)
   e67f3:	je     e6803 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e67f5:	mov    0x88(%rsp),%rdi
   e67fd:	call   *0x20f25(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6803:	mov    %rbx,0x20(%rsp)
   e6808:	cmpq   $0x3,0xf0(%rsp)
   e6811:	jb     e6821 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ed1>
   e6813:	mov    0xe0(%rsp),%rdi
   e681b:	call   *0x20f07(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6821:	mov    0x358(%rsp),%r14
   e6829:	mov    0x360(%rsp),%r12
   e6831:	test   %r12,%r12
   e6834:	jne    e68a1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f51>
   e6836:	cmpq   $0x0,0x350(%rsp)
   e683f:	je     e684a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1efa>
   e6841:	mov    %r14,%rdi
   e6844:	call   *0x20ede(%rip)        # 107728 <free@GLIBC_2.2.5>
   e684a:	mov    0x370(%rsp),%r14
   e6852:	mov    0x378(%rsp),%r12
   e685a:	test   %r12,%r12
   e685d:	jne    e692b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fdb>
   e6863:	cmpq   $0x0,0x368(%rsp)
   e686c:	je     e6877 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f27>
   e686e:	mov    %r14,%rdi
   e6871:	call   *0x20eb1(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6877:	lea    0x1c8(%rsp),%r15
   e687f:	mov    0x20(%rsp),%rbx
   e6884:	lea    0x188(%rsp),%rdi
   e688c:	call   e90a0 <core::ptr::drop_glue::<rayon::vec::IntoIter<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e6891:	mov    %r15,%rdi
   e6894:	call   f3560 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e6899:	mov    %rbx,%rdi
   e689c:	call   101800 <_Unwind_Resume@plt>
   e68a1:	xor    %r13d,%r13d
   e68a4:	jmp    e68c5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f75>
   e68a6:	cs nopw 0x0(%rax,%rax,1)
   e68b0:	mov    %r15,%rdi
   e68b3:	call   *0x20e6f(%rip)        # 107728 <free@GLIBC_2.2.5>
   e68b9:	inc    %r13
   e68bc:	cmp    %r12,%r13
   e68bf:	je     e6836 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ee6>
   e68c5:	lea    0x0(,%r13,8),%rcx
   e68cd:	add    %r13,%rcx
   e68d0:	lea    (%r14,%rcx,8),%rax
   e68d4:	cmpq   $0x0,(%r14,%rcx,8)
   e68d9:	je     e68f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fa0>
   e68db:	cmpq   $0x11,0x18(%rax)
   e68e0:	jb     e68b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f69>
   e68e2:	mov    0x8(%rax),%r15
   e68e6:	jmp    e68b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f60>
   e68e8:	nopl   0x0(%rax,%rax,1)
   e68f0:	mov    0x10(%rax),%rbp
   e68f4:	test   %rbp,%rbp
   e68f7:	je     e68b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f69>
   e68f9:	mov    0x8(%rax),%r15
   e68fd:	lea    0x18(%r15),%rbx
   e6901:	jmp    e6919 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc9>
   e6903:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e6910:	add    $0x20,%rbx
   e6914:	dec    %rbp
   e6917:	je     e68b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f60>
   e6919:	cmpq   $0x11,(%rbx)
   e691d:	jb     e6910 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc0>
   e691f:	mov    -0x10(%rbx),%rdi
   e6923:	call   *0x20dff(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6929:	jmp    e6910 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc0>
   e692b:	xor    %r13d,%r13d
   e692e:	jmp    e6945 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ff5>
   e6930:	mov    %r15,%rdi
   e6933:	call   *0x20def(%rip)        # 107728 <free@GLIBC_2.2.5>
   e6939:	inc    %r13
   e693c:	cmp    %r12,%r13
   e693f:	je     e6863 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f13>
   e6945:	lea    0x0(,%r13,8),%rcx
   e694d:	add    %r13,%rcx
   e6950:	lea    (%r14,%rcx,8),%rax
   e6954:	cmpq   $0x0,(%r14,%rcx,8)
   e6959:	je     e6970 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2020>
   e695b:	cmpq   $0x11,0x18(%rax)
   e6960:	jb     e6939 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe9>
   e6962:	mov    0x8(%rax),%r15
   e6966:	jmp    e6930 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe0>
   e6968:	nopl   0x0(%rax,%rax,1)
   e6970:	mov    0x10(%rax),%rbp
   e6974:	test   %rbp,%rbp
   e6977:	je     e6939 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe9>
   e6979:	mov    0x8(%rax),%r15
   e697d:	lea    0x18(%r15),%rbx
   e6981:	jmp    e6999 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2049>
   e6983:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e6990:	add    $0x20,%rbx
   e6994:	dec    %rbp
   e6997:	je     e6930 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe0>
   e6999:	cmpq   $0x11,(%rbx)
   e699d:	jb     e6990 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2040>
   e699f:	mov    -0x10(%rbx),%rdi
   e69a3:	call   *0x20d7f(%rip)        # 107728 <free@GLIBC_2.2.5>
   e69a9:	jmp    e6990 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2040>
