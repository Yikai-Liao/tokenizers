
/tmp/bpe-corpus-binaries-20261013/final-candidate-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000e2940 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>>:
   e2940:	push   %rbp
   e2941:	push   %r15
   e2943:	push   %r14
   e2945:	push   %r13
   e2947:	push   %r12
   e2949:	push   %rbx
   e294a:	sub    $0x2c8,%rsp
   e2951:	mov    0x308(%rsp),%rbx
   e2959:	mov    %rsi,0x120(%rsp)
   e2961:	mov    %rcx,0x138(%rsp)
   e2969:	mov    %r8,0x140(%rsp)
   e2971:	mov    (%rbx),%r13
   e2974:	movzbl 0x0(%r13),%eax
   e2979:	test   %al,%al
   e297b:	je     e2a0f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xcf>
   e2981:	mov    0x8(%rbx),%rax
   e2985:	xorps  %xmm0,%xmm0
   e2988:	movups %xmm0,0x88(%rsp)
   e2990:	mov    0x18(%rbx),%rcx
   e2994:	movq   $0x0,0x50(%rsp)
   e299d:	movq   $0x8,0x58(%rsp)
   e29a6:	movq   $0x0,0x60(%rsp)
   e29af:	mov    %r13,0x68(%rsp)
   e29b4:	mov    %rax,0x70(%rsp)
   e29b9:	lea    0x78(%rsp),%rbp
   e29be:	movq   $0x0,0x78(%rsp)
   e29c7:	movq   $0x4,0x80(%rsp)
   e29d3:	movq   $0x4,0x98(%rsp)
   e29df:	movq   $0x0,0xa0(%rsp)
   e29eb:	mov    %rcx,0xa8(%rsp)
   e29f3:	lea    0x50(%rsp),%rsi
   e29f8:	call   f3620 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e29fd:	add    $0x2c8,%rsp
   e2a04:	pop    %rbx
   e2a05:	pop    %r12
   e2a07:	pop    %r13
   e2a09:	pop    %r14
   e2a0b:	pop    %r15
   e2a0d:	pop    %rbp
   e2a0e:	ret
   e2a0f:	mov    %r9,%r14
   e2a12:	mov    0x300(%rsp),%r15
   e2a1a:	shr    $1,%rsi
   e2a1d:	cmp    %r8,%rsi
   e2a20:	mov    %r13,0x40(%rsp)
   e2a25:	mov    %rdi,0x1d0(%rsp)
   e2a2d:	jae    e3caf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x136f>
   e2a33:	mov    0x8(%rbx),%rax
   e2a37:	xorps  %xmm0,%xmm0
   e2a3a:	movups %xmm0,0x88(%rsp)
   e2a42:	mov    0x18(%rbx),%rcx
   e2a46:	movq   $0x0,0x50(%rsp)
   e2a4f:	movq   $0x8,0x58(%rsp)
   e2a58:	movq   $0x0,0x60(%rsp)
   e2a61:	mov    %r13,0x68(%rsp)
   e2a66:	mov    %rax,0x1d8(%rsp)
   e2a6e:	mov    %rax,0x70(%rsp)
   e2a73:	lea    0x78(%rsp),%rbp
   e2a78:	movq   $0x0,0x78(%rsp)
   e2a81:	movq   $0x4,0x80(%rsp)
   e2a8d:	movq   $0x4,0x98(%rsp)
   e2a99:	movq   $0x0,0xa0(%rsp)
   e2aa5:	mov    %rcx,0x200(%rsp)
   e2aad:	mov    %rcx,0xa8(%rsp)
   e2ab5:	shl    $0x5,%r15
   e2ab9:	add    %r14,%r15
   e2abc:	mov    %r15,0x208(%rsp)
   e2ac4:	movq   $0x0,0x148(%rsp)
   e2ad0:	movq   $0x8,0x150(%rsp)
   e2adc:	movq   $0x0,0x158(%rsp)
   e2ae8:	mov    $0x8,%eax
   e2aed:	mov    %rax,0x1e0(%rsp)
   e2af5:	lea    0x23d14(%rip),%rax        # 106810 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3318>
   e2afc:	movq   %rax,%xmm0
   e2b01:	movdqa %xmm0,0x2b0(%rsp)
   e2b0a:	lea    0x2477f(%rip),%rax        # 107290 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d98>
   e2b11:	mov    %rax,0x38(%rsp)
   e2b16:	movq   $0x0,0x1e8(%rsp)
   e2b22:	jmp    e2b88 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x248>
   e2b24:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e2b30:	add    $0x20,%r14
   e2b34:	imul   $0x38,%rbx,%rax
   e2b38:	mov    0x298(%rsp),%rcx
   e2b40:	mov    0x1e0(%rsp),%rdx
   e2b48:	mov    %rcx,0x30(%rdx,%rax,1)
   e2b4d:	movdqu 0x268(%rsp),%xmm0
   e2b56:	movups 0x278(%rsp),%xmm1
   e2b5e:	movups 0x288(%rsp),%xmm2
   e2b66:	movups %xmm2,0x20(%rdx,%rax,1)
   e2b6b:	movups %xmm1,0x10(%rdx,%rax,1)
   e2b70:	movdqu %xmm0,(%rdx,%rax,1)
   e2b75:	inc    %rbx
   e2b78:	mov    %rbx,0x1e8(%rsp)
   e2b80:	mov    %rbx,0x158(%rsp)
   e2b88:	cmp    0x208(%rsp),%r14
   e2b90:	je     e3cd4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1394>
   e2b96:	mov    0x10(%r14),%r12
   e2b9a:	test   %r12,%r12
   e2b9d:	je     e3cd4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1394>
   e2ba3:	mov    (%r14),%ebx
   e2ba6:	mov    0x4(%r14),%r15d
   e2baa:	mov    0x8(%r14),%ecx
   e2bae:	mov    %r14,0x48(%rsp)
   e2bb3:	mov    0x18(%r14),%rax
   e2bb7:	mov    %rax,0x18(%rsp)
   e2bbc:	mov    0x200(%rsp),%rax
   e2bc4:	mov    (%rax),%r13
   e2bc7:	mov    0x8(%rax),%rax
   e2bcb:	mov    %rax,0x10(%rsp)
   e2bd0:	mov    0x0(%r13),%r14
   e2bd4:	mov    0x50(%r14),%r8
   e2bd8:	lea    0x160(%rsp),%rdi
   e2be0:	mov    %ebx,%esi
   e2be2:	mov    %r15d,%edx
   e2be5:	mov    %ecx,0x104(%rsp)
   e2bec:	mov    %rbp,%r9
   e2bef:	call   f47f0 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::new>
   e2bf4:	movq   $0x0,0xf8(%rsp)
   e2c00:	mov    0x50(%r14),%rsi
   e2c04:	cmp    %rbx,%rsi
   e2c07:	mov    %rbx,0x130(%rsp)
   e2c0f:	jbe    e431d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19dd>
   e2c15:	cmp    %r15,%rsi
   e2c18:	jbe    e4315 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19d5>
   e2c1e:	mov    0x48(%r14),%rax
   e2c22:	mov    (%rax,%rbx,8),%rcx
   e2c26:	mov    %rcx,0x1f0(%rsp)
   e2c2e:	mov    (%rax,%r15,8),%rbx
   e2c32:	mov    $0x80,%edi
   e2c37:	call   *0x24c83(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   e2c3d:	mov    %rax,0x20(%rsp)
   e2c42:	test   %rax,%rax
   e2c45:	je     e4304 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19c4>
   e2c4b:	mov    0x10(%r14),%rdx
   e2c4f:	mov    (%r14),%rsi
   e2c52:	mov    0x8(%r14),%rax
   e2c56:	cmp    0x18(%rsp),%r12
   e2c5b:	mov    %r15,0x210(%rsp)
   e2c63:	je     e2ca0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x360>
   e2c65:	mov    (%r12),%rdi
   e2c69:	cmp    %rdx,%rdi
   e2c6c:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2c72:	mov    %esi,%ecx
   e2c74:	and    $0x1,%cl
   e2c77:	inc    %cl
   e2c79:	mov    %rdi,%r8
   e2c7c:	shl    %cl,%r8
   e2c7f:	add    $0x8,%r12
   e2c83:	mov    $0x1,%ecx
   e2c88:	add    %rax,%r8
   e2c8b:	prefetcht0 (%r8)
   e2c8f:	cmp    $0x1,%rcx
   e2c93:	je     e2cac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x36c>
   e2c95:	jmp    e3aa0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1160>
   e2c9a:	nopw   0x0(%rax,%rax,1)
   e2ca0:	xor    %ecx,%ecx
   e2ca2:	cmp    $0x1,%rcx
   e2ca6:	jne    e3aa0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1160>
   e2cac:	mov    0x20(%rsp),%rcx
   e2cb1:	mov    %rdi,(%rcx)
   e2cb4:	test   $0x1,%sil
   e2cb8:	je     e2cf0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x3b0>
   e2cba:	cmp    0x18(%rsp),%r12
   e2cbf:	je     e2d25 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x3e5>
   e2cc1:	mov    (%r12),%rcx
   e2cc5:	cmp    %rdx,%rcx
   e2cc8:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2cce:	add    $0x8,%r12
   e2cd2:	mov    $0x1,%esi
   e2cd7:	lea    (%rax,%rcx,4),%rdi
   e2cdb:	prefetcht0 (%rdi)
   e2cde:	mov    $0x1,%ebp
   e2ce3:	cmp    $0x1,%rsi
   e2ce7:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2ced:	jmp    e2d36 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x3f6>
   e2cef:	nop
   e2cf0:	cmp    0x18(%rsp),%r12
   e2cf5:	je     e2d74 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x434>
   e2cf7:	mov    (%r12),%rcx
   e2cfb:	cmp    %rdx,%rcx
   e2cfe:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2d04:	add    $0x8,%r12
   e2d08:	mov    $0x1,%esi
   e2d0d:	lea    (%rax,%rcx,2),%rdi
   e2d11:	prefetcht0 (%rdi)
   e2d14:	mov    $0x1,%ebp
   e2d19:	cmp    $0x1,%rsi
   e2d1d:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2d23:	jmp    e2d85 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x445>
   e2d25:	xor    %esi,%esi
   e2d27:	mov    $0x1,%ebp
   e2d2c:	cmp    $0x1,%rsi
   e2d30:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2d36:	mov    0x20(%rsp),%rsi
   e2d3b:	mov    %rcx,0x8(%rsi)
   e2d3f:	cmp    0x18(%rsp),%r12
   e2d44:	je     e2dc3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x483>
   e2d46:	mov    (%r12),%rcx
   e2d4a:	cmp    %rdx,%rcx
   e2d4d:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2d53:	add    $0x8,%r12
   e2d57:	mov    $0x1,%esi
   e2d5c:	lea    (%rax,%rcx,4),%rdi
   e2d60:	prefetcht0 (%rdi)
   e2d63:	mov    $0x2,%ebp
   e2d68:	cmp    $0x1,%rsi
   e2d6c:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2d72:	jmp    e2dd4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x494>
   e2d74:	xor    %esi,%esi
   e2d76:	mov    $0x1,%ebp
   e2d7b:	cmp    $0x1,%rsi
   e2d7f:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2d85:	mov    0x20(%rsp),%rsi
   e2d8a:	mov    %rcx,0x8(%rsi)
   e2d8e:	cmp    0x18(%rsp),%r12
   e2d93:	je     e2e12 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x4d2>
   e2d95:	mov    (%r12),%rcx
   e2d99:	cmp    %rdx,%rcx
   e2d9c:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2da2:	add    $0x8,%r12
   e2da6:	mov    $0x1,%esi
   e2dab:	lea    (%rax,%rcx,2),%rdi
   e2daf:	prefetcht0 (%rdi)
   e2db2:	mov    $0x2,%ebp
   e2db7:	cmp    $0x1,%rsi
   e2dbb:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2dc1:	jmp    e2e23 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x4e3>
   e2dc3:	xor    %esi,%esi
   e2dc5:	mov    $0x2,%ebp
   e2dca:	cmp    $0x1,%rsi
   e2dce:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2dd4:	mov    0x20(%rsp),%rsi
   e2dd9:	mov    %rcx,0x10(%rsi)
   e2ddd:	cmp    0x18(%rsp),%r12
   e2de2:	je     e2e61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x521>
   e2de4:	mov    (%r12),%rcx
   e2de8:	cmp    %rdx,%rcx
   e2deb:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2df1:	add    $0x8,%r12
   e2df5:	mov    $0x1,%esi
   e2dfa:	lea    (%rax,%rcx,4),%rdi
   e2dfe:	prefetcht0 (%rdi)
   e2e01:	mov    $0x3,%ebp
   e2e06:	cmp    $0x1,%rsi
   e2e0a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2e10:	jmp    e2e72 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x532>
   e2e12:	xor    %esi,%esi
   e2e14:	mov    $0x2,%ebp
   e2e19:	cmp    $0x1,%rsi
   e2e1d:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2e23:	mov    0x20(%rsp),%rsi
   e2e28:	mov    %rcx,0x10(%rsi)
   e2e2c:	cmp    0x18(%rsp),%r12
   e2e31:	je     e2eb0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x570>
   e2e33:	mov    (%r12),%rcx
   e2e37:	cmp    %rdx,%rcx
   e2e3a:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2e40:	add    $0x8,%r12
   e2e44:	mov    $0x1,%esi
   e2e49:	lea    (%rax,%rcx,2),%rdi
   e2e4d:	prefetcht0 (%rdi)
   e2e50:	mov    $0x3,%ebp
   e2e55:	cmp    $0x1,%rsi
   e2e59:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2e5f:	jmp    e2ec1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x581>
   e2e61:	xor    %esi,%esi
   e2e63:	mov    $0x3,%ebp
   e2e68:	cmp    $0x1,%rsi
   e2e6c:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2e72:	mov    0x20(%rsp),%rsi
   e2e77:	mov    %rcx,0x18(%rsi)
   e2e7b:	cmp    0x18(%rsp),%r12
   e2e80:	je     e2eff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x5bf>
   e2e82:	mov    (%r12),%rcx
   e2e86:	cmp    %rdx,%rcx
   e2e89:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2e8f:	add    $0x8,%r12
   e2e93:	mov    $0x1,%esi
   e2e98:	lea    (%rax,%rcx,4),%rdi
   e2e9c:	prefetcht0 (%rdi)
   e2e9f:	mov    $0x4,%ebp
   e2ea4:	cmp    $0x1,%rsi
   e2ea8:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2eae:	jmp    e2f10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x5d0>
   e2eb0:	xor    %esi,%esi
   e2eb2:	mov    $0x3,%ebp
   e2eb7:	cmp    $0x1,%rsi
   e2ebb:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2ec1:	mov    0x20(%rsp),%rsi
   e2ec6:	mov    %rcx,0x18(%rsi)
   e2eca:	cmp    0x18(%rsp),%r12
   e2ecf:	je     e2f3f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x5ff>
   e2ed1:	mov    (%r12),%rcx
   e2ed5:	cmp    %rdx,%rcx
   e2ed8:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2ede:	add    $0x8,%r12
   e2ee2:	mov    $0x1,%esi
   e2ee7:	lea    (%rax,%rcx,2),%rdi
   e2eeb:	prefetcht0 (%rdi)
   e2eee:	mov    $0x4,%ebp
   e2ef3:	cmp    $0x1,%rsi
   e2ef7:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2efd:	jmp    e2f50 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x610>
   e2eff:	xor    %esi,%esi
   e2f01:	mov    $0x4,%ebp
   e2f06:	cmp    $0x1,%rsi
   e2f0a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2f10:	mov    0x20(%rsp),%rsi
   e2f15:	mov    %rcx,0x20(%rsi)
   e2f19:	cmp    0x18(%rsp),%r12
   e2f1e:	je     e2f7f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x63f>
   e2f20:	mov    (%r12),%rcx
   e2f24:	cmp    %rdx,%rcx
   e2f27:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2f2d:	add    $0x8,%r12
   e2f31:	mov    $0x1,%esi
   e2f36:	lea    (%rax,%rcx,4),%rdi
   e2f3a:	prefetcht0 (%rdi)
   e2f3d:	jmp    e2f81 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x641>
   e2f3f:	xor    %esi,%esi
   e2f41:	mov    $0x4,%ebp
   e2f46:	cmp    $0x1,%rsi
   e2f4a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2f50:	mov    0x20(%rsp),%rsi
   e2f55:	mov    %rcx,0x20(%rsi)
   e2f59:	cmp    0x18(%rsp),%r12
   e2f5e:	je     e2fbf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x67f>
   e2f60:	mov    (%r12),%rcx
   e2f64:	cmp    %rdx,%rcx
   e2f67:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2f6d:	add    $0x8,%r12
   e2f71:	mov    $0x1,%esi
   e2f76:	lea    (%rax,%rcx,2),%rdi
   e2f7a:	prefetcht0 (%rdi)
   e2f7d:	jmp    e2fc1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x681>
   e2f7f:	xor    %esi,%esi
   e2f81:	mov    $0x5,%ebp
   e2f86:	cmp    $0x1,%rsi
   e2f8a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2f90:	mov    0x20(%rsp),%rsi
   e2f95:	mov    %rcx,0x28(%rsi)
   e2f99:	cmp    0x18(%rsp),%r12
   e2f9e:	je     e2fff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x6bf>
   e2fa0:	mov    (%r12),%rcx
   e2fa4:	cmp    %rdx,%rcx
   e2fa7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2fad:	add    $0x8,%r12
   e2fb1:	mov    $0x1,%esi
   e2fb6:	lea    (%rax,%rcx,4),%rdi
   e2fba:	prefetcht0 (%rdi)
   e2fbd:	jmp    e3001 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x6c1>
   e2fbf:	xor    %esi,%esi
   e2fc1:	mov    $0x5,%ebp
   e2fc6:	cmp    $0x1,%rsi
   e2fca:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e2fd0:	mov    0x20(%rsp),%rsi
   e2fd5:	mov    %rcx,0x28(%rsi)
   e2fd9:	cmp    0x18(%rsp),%r12
   e2fde:	je     e303f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x6ff>
   e2fe0:	mov    (%r12),%rcx
   e2fe4:	cmp    %rdx,%rcx
   e2fe7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e2fed:	add    $0x8,%r12
   e2ff1:	mov    $0x1,%esi
   e2ff6:	lea    (%rax,%rcx,2),%rdi
   e2ffa:	prefetcht0 (%rdi)
   e2ffd:	jmp    e3041 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x701>
   e2fff:	xor    %esi,%esi
   e3001:	mov    $0x6,%ebp
   e3006:	cmp    $0x1,%rsi
   e300a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3010:	mov    0x20(%rsp),%rsi
   e3015:	mov    %rcx,0x30(%rsi)
   e3019:	cmp    0x18(%rsp),%r12
   e301e:	je     e307f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x73f>
   e3020:	mov    (%r12),%rcx
   e3024:	cmp    %rdx,%rcx
   e3027:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e302d:	add    $0x8,%r12
   e3031:	mov    $0x1,%esi
   e3036:	lea    (%rax,%rcx,4),%rdi
   e303a:	prefetcht0 (%rdi)
   e303d:	jmp    e3081 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x741>
   e303f:	xor    %esi,%esi
   e3041:	mov    $0x6,%ebp
   e3046:	cmp    $0x1,%rsi
   e304a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3050:	mov    0x20(%rsp),%rsi
   e3055:	mov    %rcx,0x30(%rsi)
   e3059:	cmp    0x18(%rsp),%r12
   e305e:	je     e30bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x77f>
   e3060:	mov    (%r12),%rcx
   e3064:	cmp    %rdx,%rcx
   e3067:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e306d:	add    $0x8,%r12
   e3071:	mov    $0x1,%esi
   e3076:	lea    (%rax,%rcx,2),%rdi
   e307a:	prefetcht0 (%rdi)
   e307d:	jmp    e30c1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x781>
   e307f:	xor    %esi,%esi
   e3081:	mov    $0x7,%ebp
   e3086:	cmp    $0x1,%rsi
   e308a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3090:	mov    0x20(%rsp),%rsi
   e3095:	mov    %rcx,0x38(%rsi)
   e3099:	cmp    0x18(%rsp),%r12
   e309e:	je     e30ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x7bf>
   e30a0:	mov    (%r12),%rcx
   e30a4:	cmp    %rdx,%rcx
   e30a7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e30ad:	add    $0x8,%r12
   e30b1:	mov    $0x1,%esi
   e30b6:	lea    (%rax,%rcx,4),%rdi
   e30ba:	prefetcht0 (%rdi)
   e30bd:	jmp    e3101 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x7c1>
   e30bf:	xor    %esi,%esi
   e30c1:	mov    $0x7,%ebp
   e30c6:	cmp    $0x1,%rsi
   e30ca:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e30d0:	mov    0x20(%rsp),%rsi
   e30d5:	mov    %rcx,0x38(%rsi)
   e30d9:	cmp    0x18(%rsp),%r12
   e30de:	je     e313f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x7ff>
   e30e0:	mov    (%r12),%rcx
   e30e4:	cmp    %rdx,%rcx
   e30e7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e30ed:	add    $0x8,%r12
   e30f1:	mov    $0x1,%esi
   e30f6:	lea    (%rax,%rcx,2),%rdi
   e30fa:	prefetcht0 (%rdi)
   e30fd:	jmp    e3141 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x801>
   e30ff:	xor    %esi,%esi
   e3101:	mov    $0x8,%ebp
   e3106:	cmp    $0x1,%rsi
   e310a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3110:	mov    0x20(%rsp),%rsi
   e3115:	mov    %rcx,0x40(%rsi)
   e3119:	cmp    0x18(%rsp),%r12
   e311e:	je     e317f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x83f>
   e3120:	mov    (%r12),%rcx
   e3124:	cmp    %rdx,%rcx
   e3127:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e312d:	add    $0x8,%r12
   e3131:	mov    $0x1,%esi
   e3136:	lea    (%rax,%rcx,4),%rdi
   e313a:	prefetcht0 (%rdi)
   e313d:	jmp    e3181 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x841>
   e313f:	xor    %esi,%esi
   e3141:	mov    $0x8,%ebp
   e3146:	cmp    $0x1,%rsi
   e314a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3150:	mov    0x20(%rsp),%rsi
   e3155:	mov    %rcx,0x40(%rsi)
   e3159:	cmp    0x18(%rsp),%r12
   e315e:	je     e31bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x87f>
   e3160:	mov    (%r12),%rcx
   e3164:	cmp    %rdx,%rcx
   e3167:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e316d:	add    $0x8,%r12
   e3171:	mov    $0x1,%esi
   e3176:	lea    (%rax,%rcx,2),%rdi
   e317a:	prefetcht0 (%rdi)
   e317d:	jmp    e31c1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x881>
   e317f:	xor    %esi,%esi
   e3181:	mov    $0x9,%ebp
   e3186:	cmp    $0x1,%rsi
   e318a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3190:	mov    0x20(%rsp),%rsi
   e3195:	mov    %rcx,0x48(%rsi)
   e3199:	cmp    0x18(%rsp),%r12
   e319e:	je     e31ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x8bf>
   e31a0:	mov    (%r12),%rcx
   e31a4:	cmp    %rdx,%rcx
   e31a7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e31ad:	add    $0x8,%r12
   e31b1:	mov    $0x1,%esi
   e31b6:	lea    (%rax,%rcx,4),%rdi
   e31ba:	prefetcht0 (%rdi)
   e31bd:	jmp    e3201 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x8c1>
   e31bf:	xor    %esi,%esi
   e31c1:	mov    $0x9,%ebp
   e31c6:	cmp    $0x1,%rsi
   e31ca:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e31d0:	mov    0x20(%rsp),%rsi
   e31d5:	mov    %rcx,0x48(%rsi)
   e31d9:	cmp    0x18(%rsp),%r12
   e31de:	je     e323f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x8ff>
   e31e0:	mov    (%r12),%rcx
   e31e4:	cmp    %rdx,%rcx
   e31e7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e31ed:	add    $0x8,%r12
   e31f1:	mov    $0x1,%esi
   e31f6:	lea    (%rax,%rcx,2),%rdi
   e31fa:	prefetcht0 (%rdi)
   e31fd:	jmp    e3241 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x901>
   e31ff:	xor    %esi,%esi
   e3201:	mov    $0xa,%ebp
   e3206:	cmp    $0x1,%rsi
   e320a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3210:	mov    0x20(%rsp),%rsi
   e3215:	mov    %rcx,0x50(%rsi)
   e3219:	cmp    0x18(%rsp),%r12
   e321e:	je     e327f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x93f>
   e3220:	mov    (%r12),%rcx
   e3224:	cmp    %rdx,%rcx
   e3227:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e322d:	add    $0x8,%r12
   e3231:	mov    $0x1,%esi
   e3236:	lea    (%rax,%rcx,4),%rdi
   e323a:	prefetcht0 (%rdi)
   e323d:	jmp    e3281 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x941>
   e323f:	xor    %esi,%esi
   e3241:	mov    $0xa,%ebp
   e3246:	cmp    $0x1,%rsi
   e324a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3250:	mov    0x20(%rsp),%rsi
   e3255:	mov    %rcx,0x50(%rsi)
   e3259:	cmp    0x18(%rsp),%r12
   e325e:	je     e32bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x97f>
   e3260:	mov    (%r12),%rcx
   e3264:	cmp    %rdx,%rcx
   e3267:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e326d:	add    $0x8,%r12
   e3271:	mov    $0x1,%esi
   e3276:	lea    (%rax,%rcx,2),%rdi
   e327a:	prefetcht0 (%rdi)
   e327d:	jmp    e32c1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x981>
   e327f:	xor    %esi,%esi
   e3281:	mov    $0xb,%ebp
   e3286:	cmp    $0x1,%rsi
   e328a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3290:	mov    0x20(%rsp),%rsi
   e3295:	mov    %rcx,0x58(%rsi)
   e3299:	cmp    0x18(%rsp),%r12
   e329e:	je     e32ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x9bf>
   e32a0:	mov    (%r12),%rcx
   e32a4:	cmp    %rdx,%rcx
   e32a7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e32ad:	add    $0x8,%r12
   e32b1:	mov    $0x1,%esi
   e32b6:	lea    (%rax,%rcx,4),%rdi
   e32ba:	prefetcht0 (%rdi)
   e32bd:	jmp    e3301 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x9c1>
   e32bf:	xor    %esi,%esi
   e32c1:	mov    $0xb,%ebp
   e32c6:	cmp    $0x1,%rsi
   e32ca:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e32d0:	mov    0x20(%rsp),%rsi
   e32d5:	mov    %rcx,0x58(%rsi)
   e32d9:	cmp    0x18(%rsp),%r12
   e32de:	je     e333f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x9ff>
   e32e0:	mov    (%r12),%rcx
   e32e4:	cmp    %rdx,%rcx
   e32e7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e32ed:	add    $0x8,%r12
   e32f1:	mov    $0x1,%esi
   e32f6:	lea    (%rax,%rcx,2),%rdi
   e32fa:	prefetcht0 (%rdi)
   e32fd:	jmp    e3341 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xa01>
   e32ff:	xor    %esi,%esi
   e3301:	mov    $0xc,%ebp
   e3306:	cmp    $0x1,%rsi
   e330a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3310:	mov    0x20(%rsp),%rsi
   e3315:	mov    %rcx,0x60(%rsi)
   e3319:	cmp    0x18(%rsp),%r12
   e331e:	je     e337f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xa3f>
   e3320:	mov    (%r12),%rcx
   e3324:	cmp    %rdx,%rcx
   e3327:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e332d:	add    $0x8,%r12
   e3331:	mov    $0x1,%esi
   e3336:	lea    (%rax,%rcx,4),%rdi
   e333a:	prefetcht0 (%rdi)
   e333d:	jmp    e3381 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xa41>
   e333f:	xor    %esi,%esi
   e3341:	mov    $0xc,%ebp
   e3346:	cmp    $0x1,%rsi
   e334a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3350:	mov    0x20(%rsp),%rsi
   e3355:	mov    %rcx,0x60(%rsi)
   e3359:	cmp    0x18(%rsp),%r12
   e335e:	je     e33bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xa7f>
   e3360:	mov    (%r12),%rcx
   e3364:	cmp    %rdx,%rcx
   e3367:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e336d:	add    $0x8,%r12
   e3371:	mov    $0x1,%esi
   e3376:	lea    (%rax,%rcx,2),%rdi
   e337a:	prefetcht0 (%rdi)
   e337d:	jmp    e33c1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xa81>
   e337f:	xor    %esi,%esi
   e3381:	mov    $0xd,%ebp
   e3386:	cmp    $0x1,%rsi
   e338a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3390:	mov    0x20(%rsp),%rsi
   e3395:	mov    %rcx,0x68(%rsi)
   e3399:	cmp    0x18(%rsp),%r12
   e339e:	je     e33ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xabf>
   e33a0:	mov    (%r12),%rcx
   e33a4:	cmp    %rdx,%rcx
   e33a7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e33ad:	add    $0x8,%r12
   e33b1:	mov    $0x1,%esi
   e33b6:	lea    (%rax,%rcx,4),%rdi
   e33ba:	prefetcht0 (%rdi)
   e33bd:	jmp    e3401 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xac1>
   e33bf:	xor    %esi,%esi
   e33c1:	mov    $0xd,%ebp
   e33c6:	cmp    $0x1,%rsi
   e33ca:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e33d0:	mov    0x20(%rsp),%rsi
   e33d5:	mov    %rcx,0x68(%rsi)
   e33d9:	cmp    0x18(%rsp),%r12
   e33de:	je     e3446 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb06>
   e33e0:	mov    (%r12),%rcx
   e33e4:	cmp    %rdx,%rcx
   e33e7:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e33ed:	add    $0x8,%r12
   e33f1:	mov    $0x1,%esi
   e33f6:	lea    (%rax,%rcx,2),%rdi
   e33fa:	prefetcht0 (%rdi)
   e33fd:	jmp    e3448 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb08>
   e33ff:	xor    %esi,%esi
   e3401:	mov    $0xe,%ebp
   e3406:	cmp    $0x1,%rsi
   e340a:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3410:	mov    0x20(%rsp),%rsi
   e3415:	mov    %rcx,0x70(%rsi)
   e3419:	cmp    0x18(%rsp),%r12
   e341e:	je     e3c87 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1347>
   e3424:	mov    (%r12),%rcx
   e3428:	cmp    %rdx,%rcx
   e342b:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e3431:	add    $0x8,%r12
   e3435:	mov    $0x1,%edx
   e343a:	lea    (%rax,%rcx,4),%rax
   e343e:	prefetcht0 (%rax)
   e3441:	jmp    e3c89 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1349>
   e3446:	xor    %esi,%esi
   e3448:	mov    $0xe,%ebp
   e344d:	cmp    $0x1,%rsi
   e3451:	jne    e3490 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb50>
   e3453:	mov    0x20(%rsp),%rsi
   e3458:	mov    %rcx,0x70(%rsi)
   e345c:	cmp    0x18(%rsp),%r12
   e3461:	je     e3c87 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1347>
   e3467:	mov    (%r12),%rcx
   e346b:	cmp    %rdx,%rcx
   e346e:	jae    e4124 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17e4>
   e3474:	add    $0x8,%r12
   e3478:	mov    $0x1,%edx
   e347d:	lea    (%rax,%rcx,2),%rax
   e3481:	prefetcht0 (%rax)
   e3484:	jmp    e3c89 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1349>
   e3489:	nopl   0x0(%rax)
   e3490:	mov    0x10(%rsp),%rdx
   e3495:	mov    0x1f0(%rsp),%rax
   e349d:	lea    (%rbx,%rax,1),%rcx
   e34a1:	mov    %rcx,0x128(%rsp)
   e34a9:	mov    0x8(%r13),%rcx
   e34ad:	mov    %rcx,0x1c8(%rsp)
   e34b5:	mov    0x8(%rdx),%rcx
   e34b9:	mov    %rcx,0x110(%rsp)
   e34c1:	mov    0x10(%rdx),%rcx
   e34c5:	mov    %rcx,0x118(%rsp)
   e34cd:	test   %rax,%rax
   e34d0:	sete   %al
   e34d3:	test   %rbx,%rbx
   e34d6:	sete   %cl
   e34d9:	or     %al,%cl
   e34db:	mov    %cl,0x10(%rsp)
   e34df:	xor    %edx,%edx
   e34e1:	xor    %r15d,%r15d
   e34e4:	jmp    e3506 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbc6>
   e34e6:	cs nopw 0x0(%rax,%rax,1)
   e34f0:	cmp    0x210(%rsp),%edi
   e34f7:	je     e3696 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xd56>
   e34fd:	test   %rbp,%rbp
   e3500:	je     e3aa0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1160>
   e3506:	lea    0x1(%r15),%rax
   e350a:	xor    %esi,%esi
   e350c:	cmp    $0x10,%rax
   e3510:	setb   %dil
   e3514:	mov    0x20(%rsp),%rax
   e3519:	mov    (%rax,%r15,8),%r13
   e351d:	cmp    0x18(%rsp),%r12
   e3522:	je     e3560 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc20>
   e3524:	mov    (%r12),%rax
   e3528:	cmp    0x10(%r14),%rax
   e352c:	jae    e4107 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17c7>
   e3532:	add    $0x8,%r12
   e3536:	movzbl (%r14),%ecx
   e353a:	inc    %cl
   e353c:	mov    %rax,%r8
   e353f:	shl    %cl,%r8
   e3542:	add    0x8(%r14),%r8
   e3546:	prefetcht0 (%r8)
   e354a:	mov    $0x1,%ecx
   e354f:	jmp    e3562 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc22>
   e3551:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e3560:	xor    %ecx,%ecx
   e3562:	mov    %dil,%sil
   e3565:	shl    $0x4,%esi
   e3568:	add    %rsi,%r15
   e356b:	add    $0xfffffffffffffff1,%r15
   e356f:	mov    %rbp,%rdi
   e3572:	dec    %rdi
   e3575:	lea    (%r15,%rdi,1),%rsi
   e3579:	xor    %r8d,%r8d
   e357c:	cmp    $0x10,%rsi
   e3580:	setb   %r8b
   e3584:	shl    $0x4,%r8d
   e3588:	mov    $0x20,%r10d
   e358e:	sub    %r8,%r10
   e3591:	sub    %rsi,%r10
   e3594:	add    %r8,%rsi
   e3597:	add    $0xfffffffffffffff0,%rsi
   e359b:	cmp    %rcx,%r10
   e359e:	jae    e35b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc70>
   e35a0:	cmp    $0x10,%rsi
   e35a4:	jne    e35b6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc76>
   e35a6:	mov    0x20(%rsp),%rcx
   e35ab:	mov    %rax,(%rcx)
   e35ae:	jmp    e35bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc7f>
   e35b0:	cmp    $0x1,%rcx
   e35b4:	jne    e35f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xcb0>
   e35b6:	mov    0x20(%rsp),%rcx
   e35bb:	mov    %rax,(%rcx,%rsi,8)
   e35bf:	mov    $0x1,%ebp
   e35c4:	mov    (%r14),%rax
   e35c7:	mov    0x10(%r14),%rsi
   e35cb:	cmp    $0x1,%rax
   e35cf:	jne    e35ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xcbf>
   e35d1:	cmp    %rsi,%r13
   e35d4:	jae    e42d1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1991>
   e35da:	mov    0x8(%r14),%rcx
   e35de:	mov    (%rcx,%r13,4),%r8d
   e35e2:	jmp    e3624 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xce4>
   e35e4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e35f0:	xor    %ebp,%ebp
   e35f2:	mov    (%r14),%rax
   e35f5:	mov    0x10(%r14),%rsi
   e35f9:	cmp    $0x1,%rax
   e35fd:	je     e35d1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xc91>
   e35ff:	cmp    %rsi,%r13
   e3602:	jae    e42e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19a9>
   e3608:	mov    0x8(%r14),%rcx
   e360c:	movzwl (%rcx,%r13,2),%r8d
   e3611:	cmp    $0xffff,%r8w
   e3616:	movzwl %r8w,%r8d
   e361a:	mov    $0xffffffff,%r9d
   e3620:	cmove  %r9d,%r8d
   e3624:	add    %rdi,%rbp
   e3627:	cmp    0x130(%rsp),%r8d
   e362f:	setne  %dil
   e3633:	or     0x10(%rsp),%dil
   e3638:	jne    e34fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbbd>
   e363e:	mov    0x128(%rsp),%rdi
   e3646:	lea    (%rdi,%r13,1),%rbx
   e364a:	cmp    %rsi,%rbx
   e364d:	jae    e34fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbbd>
   e3653:	mov    0x1f0(%rsp),%rdi
   e365b:	add    %r13,%rdi
   e365e:	test   %al,%al
   e3660:	je     e3673 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xd33>
   e3662:	cmp    %rsi,%rdi
   e3665:	jae    e42da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x199a>
   e366b:	mov    (%rcx,%rdi,4),%edi
   e366e:	jmp    e34f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbb0>
   e3673:	cmp    %rsi,%rdi
   e3676:	jae    e42e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19a6>
   e367c:	movzwl (%rcx,%rdi,2),%edi
   e3680:	cmp    $0xffff,%di
   e3684:	movzwl %di,%edi
   e3687:	mov    $0xffffffff,%r8d
   e368d:	cmove  %r8d,%edi
   e3691:	jmp    e34f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbb0>
   e3696:	cmp    0xe0(%rsp),%r13
   e369e:	setb   %dil
   e36a2:	test   %dil,%dl
   e36a5:	jne    e3792 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xe52>
   e36ab:	mov    0x68(%r14),%rdx
   e36af:	mov    0x70(%r14),%r9
   e36b3:	cmp    $0x1,%r9
   e36b7:	jne    e36d2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xd92>
   e36b9:	mov    0x8(%rdx),%rdx
   e36bd:	mov    %rdx,0x108(%rsp)
   e36c5:	mov    %rsi,0xe0(%rsp)
   e36cd:	jmp    e3792 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xe52>
   e36d2:	cmpl   $0x1,0x18(%r14)
   e36d7:	jne    e3701 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xdc1>
   e36d9:	cmp    0x20(%r14),%r13
   e36dd:	jb     e3701 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xdc1>
   e36df:	mov    0x28(%r14),%r8
   e36e3:	mov    $0x1,%edi
   e36e8:	mov    %rdi,0x108(%rsp)
   e36f0:	mov    %r8,0xe0(%rsp)
   e36f8:	cmp    %r8,%r13
   e36fb:	jb     e3792 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xe52>
   e3701:	test   %r9,%r9
   e3704:	je     e4371 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a31>
   e370a:	mov    %r9,0xe0(%rsp)
   e3712:	xor    %r10d,%r10d
   e3715:	data16 cs nopw 0x0(%rax,%rax,1)
   e3720:	mov    %r10,%rdi
   e3723:	mov    %r9,%r11
   e3726:	shr    $1,%r11
   e3729:	add    %r11,%r10
   e372c:	mov    %r10,%r8
   e372f:	shl    $0x4,%r8
   e3733:	cmp    %r13,(%rdx,%r8,1)
   e3737:	cmova  %rdi,%r10
   e373b:	sub    %r11,%r9
   e373e:	cmp    $0x1,%r9
   e3742:	ja     e3720 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xde0>
   e3744:	mov    %r10,%rdi
   e3747:	shl    $0x4,%rdi
   e374b:	cmp    (%rdx,%rdi,1),%r13
   e374f:	sbb    $0xffffffffffffffff,%r10
   e3753:	lea    -0x1(%r10),%r9
   e3757:	mov    0xe0(%rsp),%r8
   e375f:	cmp    %r8,%r9
   e3762:	jae    e4369 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a29>
   e3768:	shl    $0x4,%r9
   e376c:	mov    0x8(%rdx,%r9,1),%rdi
   e3771:	mov    %rdi,0x108(%rsp)
   e3779:	cmp    %r8,%r10
   e377c:	jae    e36c5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xd85>
   e3782:	shl    $0x4,%r10
   e3786:	mov    (%rdx,%r10,1),%rdx
   e378a:	mov    %rdx,0xe0(%rsp)
   e3792:	lea    -0x1(%r13),%rdx
   e3796:	test   %al,%al
   e3798:	je     e37be <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xe7e>
   e379a:	cmp    %rsi,%rdx
   e379d:	jae    e434c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a0c>
   e37a3:	mov    -0x4(%rcx,%r13,4),%edx
   e37a8:	cmp    $0xffffffff,%edx
   e37ab:	jne    e37da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xe9a>
   e37ad:	mov    (%rcx,%rbx,4),%edx
   e37b0:	cmp    $0xffffffff,%edx
   e37b3:	jne    e3949 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1009>
   e37b9:	jmp    e3a05 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x10c5>
   e37be:	cmp    %rsi,%rdx
   e37c1:	jae    e4347 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a07>
   e37c7:	movzwl -0x2(%rcx,%r13,2),%edx
   e37cd:	cmp    $0xffff,%dx
   e37d1:	je     e3938 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xff8>
   e37d7:	movzwl %dx,%edx
   e37da:	mov    %edx,%ecx
   e37dc:	mov    0x50(%r14),%rdx
   e37e0:	cmp    %rcx,%rdx
   e37e3:	jbe    e4333 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f3>
   e37e9:	mov    0x48(%r14),%rdx
   e37ed:	mov    (%rdx,%rcx,8),%rdx
   e37f1:	cmp    %rbx,%r13
   e37f4:	jae    e3851 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf11>
   e37f6:	cmp    0x128(%rsp),%r13
   e37fe:	jb     e38bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf7b>
   e3804:	mov    %rbx,%rdi
   e3807:	sub    %r13,%rdi
   e380a:	mov    $0x0,%r9d
   e3810:	cmovae %rdi,%r9
   e3814:	movq   $0x1,0x2a0(%rsp)
   e3820:	mov    0x128(%rsp),%rdi
   e3828:	mov    %rdi,0x2a8(%rsp)
   e3830:	mov    %r9,0x220(%rsp)
   e3838:	movq   $0x1,0x218(%rsp)
   e3844:	mov    %rdi,%r8
   e3847:	cmp    %r9,%rdi
   e384a:	je     e3854 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf14>
   e384c:	jmp    e4159 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1819>
   e3851:	xor    %r8d,%r8d
   e3854:	cmpq   $0x0,0x118(%rsp)
   e385d:	je     e38bb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf7b>
   e385f:	mov    %rdx,0x1f8(%rsp)
   e3867:	mov    %r13,%rdi
   e386a:	sub    %r8,%rdi
   e386d:	mov    0x118(%rsp),%r8
   e3875:	xor    %r9d,%r9d
   e3878:	cmp    $0x1,%r8
   e387c:	jne    e3888 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf48>
   e387e:	mov    0x110(%rsp),%rdx
   e3886:	jmp    e38ad <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf6d>
   e3888:	mov    0x110(%rsp),%rdx
   e3890:	mov    %r9,%r10
   e3893:	mov    %r8,%r11
   e3896:	shr    $1,%r11
   e3899:	add    %r11,%r9
   e389c:	cmp    %rdi,(%rdx,%r9,8)
   e38a0:	cmova  %r10,%r9
   e38a4:	sub    %r11,%r8
   e38a7:	cmp    $0x1,%r8
   e38ab:	ja     e3890 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf50>
   e38ad:	cmp    %rdi,(%rdx,%r9,8)
   e38b1:	mov    0x1f8(%rsp),%rdx
   e38b9:	je     e390c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xfcc>
   e38bb:	mov    %r13,%r8
   e38be:	sub    %rdx,%r8
   e38c1:	mov    %rbx,%rax
   e38c4:	sub    %r13,%rax
   e38c7:	mov    $0x0,%esi
   e38cc:	cmovae %rax,%rsi
   e38d0:	add    %rdx,%rsi
   e38d3:	xor    %eax,%eax
   e38d5:	cmp    0x1c8(%rsp),%rsi
   e38dd:	setb   %al
   e38e0:	mov    %eax,(%rsp)
   e38e3:	lea    0x160(%rsp),%rdi
   e38eb:	xor    %esi,%esi
   e38ed:	mov    %ecx,%edx
   e38ef:	mov    0x108(%rsp),%r9
   e38f7:	call   f4b50 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e38fc:	test   %rax,%rax
   e38ff:	jne    e3c27 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x12e7>
   e3905:	mov    (%r14),%rax
   e3908:	mov    0x10(%r14),%rsi
   e390c:	test   %rax,%rax
   e390f:	je     e392b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xfeb>
   e3911:	cmp    %rsi,%rbx
   e3914:	jae    e435d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a1d>
   e391a:	mov    0x8(%r14),%rcx
   e391e:	mov    (%rcx,%rbx,4),%edx
   e3921:	cmp    $0xffffffff,%edx
   e3924:	jne    e3949 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1009>
   e3926:	jmp    e3a05 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x10c5>
   e392b:	cmp    %rsi,%rbx
   e392e:	jae    e4358 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1a18>
   e3934:	mov    0x8(%r14),%rcx
   e3938:	movzwl (%rcx,%rbx,2),%eax
   e393c:	cmp    $0xffff,%ax
   e3940:	je     e3a05 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x10c5>
   e3946:	movzwl %ax,%edx
   e3949:	mov    %edx,%eax
   e394b:	cmpq   $0x0,0x118(%rsp)
   e3954:	je     e39ab <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x106b>
   e3956:	mov    0x118(%rsp),%rax
   e395e:	xor    %ecx,%ecx
   e3960:	cmp    $0x1,%rax
   e3964:	jne    e3970 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1030>
   e3966:	mov    0x110(%rsp),%r8
   e396e:	jmp    e399d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x105d>
   e3970:	mov    0x110(%rsp),%r8
   e3978:	nopl   0x0(%rax,%rax,1)
   e3980:	mov    %rcx,%rsi
   e3983:	mov    %rax,%rdi
   e3986:	shr    $1,%rdi
   e3989:	add    %rdi,%rcx
   e398c:	cmp    %rbx,(%r8,%rcx,8)
   e3990:	cmova  %rsi,%rcx
   e3994:	sub    %rdi,%rax
   e3997:	cmp    $0x1,%rax
   e399b:	ja     e3980 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1040>
   e399d:	cmp    %rbx,(%r8,%rcx,8)
   e39a1:	mov    %edx,%eax
   e39a3:	cmove  0x104(%rsp),%eax
   e39ab:	mov    %eax,%ecx
   e39ad:	mov    0x50(%r14),%rsi
   e39b1:	cmp    %rcx,%rsi
   e39b4:	jbe    e433b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19fb>
   e39ba:	sub    %r13,%rbx
   e39bd:	mov    $0x0,%eax
   e39c2:	cmovae %rbx,%rax
   e39c6:	mov    0x48(%r14),%rsi
   e39ca:	add    (%rsi,%rcx,8),%rax
   e39ce:	xor    %esi,%esi
   e39d0:	cmp    0x1c8(%rsp),%rax
   e39d8:	setb   %sil
   e39dc:	mov    %esi,(%rsp)
   e39df:	lea    0x160(%rsp),%rdi
   e39e7:	mov    $0x1,%esi
   e39ec:	mov    %r13,%r8
   e39ef:	mov    0x108(%rsp),%r9
   e39f7:	call   f4b50 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e39fc:	test   %rax,%rax
   e39ff:	jne    e3c27 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x12e7>
   e3a05:	mov    0xf8(%rsp),%rax
   e3a0d:	cmp    $0x3,%rax
   e3a11:	jb     e3a3e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x10fe>
   e3a13:	mov    0xf0(%rsp),%rcx
   e3a1b:	cmp    %rax,%rcx
   e3a1e:	je     e3a62 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1122>
   e3a20:	mov    0xe8(%rsp),%rax
   e3a28:	mov    %r13,(%rax,%rcx,8)
   e3a2c:	inc    %rcx
   e3a2f:	mov    %rcx,0xf0(%rsp)
   e3a37:	mov    $0x1,%dl
   e3a39:	jmp    e34fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbbd>
   e3a3e:	lea    0xe8(%rsp),%rcx
   e3a46:	lea    0xf8(%rsp),%rdx
   e3a4e:	cmp    $0x2,%rax
   e3a52:	je     e3a79 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1139>
   e3a54:	mov    %r13,(%rcx,%rax,8)
   e3a58:	incq   (%rdx)
   e3a5b:	mov    $0x1,%dl
   e3a5d:	jmp    e34fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbbd>
   e3a62:	lea    0xe8(%rsp),%rdi
   e3a6a:	call   f5000 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e3a6f:	mov    0xf0(%rsp),%rcx
   e3a77:	jmp    e3a20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x10e0>
   e3a79:	lea    0xe8(%rsp),%rdi
   e3a81:	call   f5000 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e3a86:	mov    0xe8(%rsp),%rcx
   e3a8e:	mov    0xf0(%rsp),%rax
   e3a96:	lea    0xf0(%rsp),%rdx
   e3a9e:	jmp    e3a54 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1114>
   e3aa0:	mov    0x20(%rsp),%rdi
   e3aa5:	call   *0x23dfd(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e3aab:	mov    0xe8(%rsp),%r15
   e3ab3:	movups 0xf0(%rsp),%xmm0
   e3abb:	movaps %xmm0,0x20(%rsp)
   e3ac0:	lea    0x218(%rsp),%rdi
   e3ac8:	lea    0x160(%rsp),%rsi
   e3ad0:	call   f5220 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::finish>
   e3ad5:	mov    0x218(%rsp),%rax
   e3add:	cmp    $0xfffffffffffffffe,%rax
   e3ae1:	lea    0x78(%rsp),%rbp
   e3ae6:	je     e3cd4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1394>
   e3aec:	movdqu 0x220(%rsp),%xmm0
   e3af5:	movaps 0x20(%rsp),%xmm1
   e3afa:	mov    %rax,0x160(%rsp)
   e3b02:	movdqu %xmm0,0x168(%rsp)
   e3b0b:	mov    %r15,0x178(%rsp)
   e3b13:	movups %xmm1,0x180(%rsp)
   e3b1b:	mov    0x104(%rsp),%eax
   e3b22:	mov    %eax,0x190(%rsp)
   e3b29:	mov    0x1d8(%rsp),%rax
   e3b31:	mov    (%rax),%rsi
   e3b34:	lea    0x230(%rsp),%rdi
   e3b3c:	lea    0x160(%rsp),%rdx
   e3b44:	call   f5680 <<&<core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0} as core::ops::function::FnMut<(core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>,)>>::call_mut>
   e3b49:	mov    0x230(%rsp),%r15
   e3b51:	cmp    $0xfffffffffffffffe,%r15
   e3b55:	mov    0x48(%rsp),%r14
   e3b5a:	je     e3cd4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1394>
   e3b60:	cmp    $0xffffffffffffffff,%r15
   e3b64:	je     e4043 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1703>
   e3b6a:	mov    0x238(%rsp),%rbx
   e3b72:	mov    0x240(%rsp),%r12
   e3b7a:	mov    0x248(%rsp),%rdi
   e3b82:	mov    0x258(%rsp),%rcx
   e3b8a:	movdqu 0x250(%rsp),%xmm0
   e3b93:	mov    0x260(%rsp),%rax
   e3b9b:	mov    0x40(%rsp),%rdx
   e3ba0:	movzbl (%rdx),%edx
   e3ba3:	test   %dl,%dl
   e3ba5:	jne    e4051 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1711>
   e3bab:	mov    %rbx,0x270(%rsp)
   e3bb3:	mov    %r12,0x278(%rsp)
   e3bbb:	mov    %rdi,0x280(%rsp)
   e3bc3:	movdqu %xmm0,0x288(%rsp)
   e3bcc:	mov    %rax,0x298(%rsp)
   e3bd4:	mov    %r15,0x268(%rsp)
   e3bdc:	mov    0x1e8(%rsp),%rbx
   e3be4:	cmp    0x148(%rsp),%rbx
   e3bec:	jne    e2b30 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1f0>
   e3bf2:	mov    $0x1,%edx
   e3bf7:	mov    $0x8,%ecx
   e3bfc:	mov    $0x38,%r8d
   e3c02:	lea    0x148(%rsp),%rdi
   e3c0a:	mov    %rbx,%rsi
   e3c0d:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e3c12:	mov    0x150(%rsp),%rax
   e3c1a:	mov    %rax,0x1e0(%rsp)
   e3c22:	jmp    e2b30 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1f0>
   e3c27:	mov    %rax,%rbx
   e3c2a:	mov    0x20(%rsp),%rdi
   e3c2f:	call   *0x23c73(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e3c35:	cmpq   $0x3,0xf8(%rsp)
   e3c3e:	lea    0x78(%rsp),%rbp
   e3c43:	jb     e3c53 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1313>
   e3c45:	mov    0xe8(%rsp),%rdi
   e3c4d:	call   *0x23c55(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e3c53:	lea    0x160(%rsp),%rdi
   e3c5b:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e3c60:	lea    0x178(%rsp),%rdi
   e3c68:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e3c6d:	movq   %rbx,%xmm0
   e3c72:	punpcklqdq 0x2b0(%rsp),%xmm0
   e3c7b:	mov    $0xffffffffffffffff,%rax
   e3c82:	jmp    e3afa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x11ba>
   e3c87:	xor    %edx,%edx
   e3c89:	mov    $0xf,%ebp
   e3c8e:	test   $0x1,%dl
   e3c91:	mov    0x10(%rsp),%rdx
   e3c96:	je     e3495 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb55>
   e3c9c:	mov    0x20(%rsp),%rax
   e3ca1:	mov    %rcx,0x78(%rax)
   e3ca5:	mov    $0x10,%ebp
   e3caa:	jmp    e3495 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xb55>
   e3caf:	test   %dl,%dl
   e3cb1:	je     e3dbc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x147c>
   e3cb7:	mov    %fs:0xffffffffffffffd0,%rax
   e3cc0:	test   %rax,%rax
   e3cc3:	je     e3dca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x148a>
   e3cc9:	add    $0x110,%rax
   e3ccf:	jmp    e3de5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x14a5>
   e3cd4:	mov    0x40(%rsp),%r14
   e3cd9:	mov    0x158(%rsp),%rax
   e3ce1:	mov    %rax,0x60(%rsp)
   e3ce6:	movups 0x148(%rsp),%xmm0
   e3cee:	mov    %r14,0x68(%rsp)
   e3cf3:	mov    0x1d8(%rsp),%rax
   e3cfb:	mov    %rax,0x70(%rsp)
   e3d00:	movaps %xmm0,0x160(%rsp)
   e3d08:	movaps 0x80(%rsp),%xmm0
   e3d10:	movaps 0x90(%rsp),%xmm1
   e3d18:	movaps 0xa0(%rsp),%xmm2
   e3d20:	movaps %xmm0,0x190(%rsp)
   e3d28:	movaps %xmm1,0x1a0(%rsp)
   e3d30:	movaps %xmm2,0x1b0(%rsp)
   e3d38:	mov    0x60(%rsp),%rax
   e3d3d:	mov    %rax,0x170(%rsp)
   e3d45:	mov    0x68(%rsp),%rax
   e3d4a:	mov    %rax,0x178(%rsp)
   e3d52:	mov    0x70(%rsp),%rax
   e3d57:	mov    %rax,0x180(%rsp)
   e3d5f:	mov    0x78(%rsp),%rax
   e3d64:	mov    %rax,0x188(%rsp)
   e3d6c:	lea    0x160(%rsp),%rsi
   e3d74:	mov    0x1d0(%rsp),%rdi
   e3d7c:	call   f3620 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e3d81:	cmpq   $0x0,0x188(%rsp)
   e3d8a:	je     e3d9a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x145a>
   e3d8c:	mov    0x190(%rsp),%rdi
   e3d94:	call   *0x23b0e(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e3d9a:	cmpq   $0x0,0x1a0(%rsp)
   e3da3:	je     e29fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbd>
   e3da9:	mov    0x1a8(%rsp),%rdi
   e3db1:	call   *0x23af1(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e3db7:	jmp    e29fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbd>
   e3dbc:	test   %rcx,%rcx
   e3dbf:	je     e2a33 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xf3>
   e3dc5:	shr    $1,%rcx
   e3dc8:	jmp    e3df9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x14b9>
   e3dca:	mov    %r14,0x48(%rsp)
   e3dcf:	mov    %rsi,%r14
   e3dd2:	mov    %rcx,%rbp
   e3dd5:	call   7f660 <rayon_core::registry::global_registry>
   e3dda:	mov    %rbp,%rcx
   e3ddd:	mov    %r14,%rsi
   e3de0:	mov    0x48(%rsp),%r14
   e3de5:	mov    (%rax),%rax
   e3de8:	mov    0x208(%rax),%rax
   e3def:	shr    $1,%rcx
   e3df2:	cmp    %rax,%rcx
   e3df5:	cmovbe %rax,%rcx
   e3df9:	mov    %rcx,0x138(%rsp)
   e3e01:	mov    %rsi,0x230(%rsp)
   e3e09:	sub    %rsi,%r15
   e3e0c:	jb     e4141 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1801>
   e3e12:	mov    %rsi,%r12
   e3e15:	shl    $0x5,%r12
   e3e19:	add    %r14,%r12
   e3e1c:	mov    0x8(%rbx),%rbp
   e3e20:	mov    %r14,%rdi
   e3e23:	mov    0x10(%rbx),%r14
   e3e27:	mov    0x18(%rbx),%rbx
   e3e2b:	mov    %fs:0xffffffffffffffd0,%rdx
   e3e34:	test   %rdx,%rdx
   e3e37:	je     e3eb4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1574>
   e3e39:	lea    0x120(%rsp),%rax
   e3e41:	mov    %rax,0x50(%rsp)
   e3e46:	lea    0x230(%rsp),%rax
   e3e4e:	mov    %rax,0x58(%rsp)
   e3e53:	lea    0x138(%rsp),%rcx
   e3e5b:	mov    %rcx,0x60(%rsp)
   e3e60:	mov    %r12,0x68(%rsp)
   e3e65:	mov    %r15,0x70(%rsp)
   e3e6a:	mov    %r13,0x78(%rsp)
   e3e6f:	mov    %rbp,0x80(%rsp)
   e3e77:	mov    %r14,0x88(%rsp)
   e3e7f:	mov    %rbx,0x90(%rsp)
   e3e87:	mov    %rax,0x98(%rsp)
   e3e8f:	mov    %rcx,0xa0(%rsp)
   e3e97:	mov    %rdi,0xa8(%rsp)
   e3e9f:	mov    %rsi,0xb0(%rsp)
   e3ea7:	mov    %r13,0xb8(%rsp)
   e3eaf:	jmp    e3f66 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1626>
   e3eb4:	mov    %rdi,0x48(%rsp)
   e3eb9:	mov    %rsi,%r13
   e3ebc:	call   7f660 <rayon_core::registry::global_registry>
   e3ec1:	mov    (%rax),%rsi
   e3ec4:	mov    %fs:0xffffffffffffffd0,%rdx
   e3ecd:	test   %rdx,%rdx
   e3ed0:	je     e4173 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1833>
   e3ed6:	cmp    %rsi,0x110(%rdx)
   e3edd:	mov    %r13,%rax
   e3ee0:	lea    0x120(%rsp),%rcx
   e3ee8:	jne    e4226 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x18e6>
   e3eee:	mov    %rcx,0x50(%rsp)
   e3ef3:	lea    0x230(%rsp),%rcx
   e3efb:	mov    %rcx,0x58(%rsp)
   e3f00:	lea    0x138(%rsp),%rsi
   e3f08:	mov    %rsi,0x60(%rsp)
   e3f0d:	mov    %r12,0x68(%rsp)
   e3f12:	mov    %r15,0x70(%rsp)
   e3f17:	mov    0x40(%rsp),%rdi
   e3f1c:	mov    %rdi,0x78(%rsp)
   e3f21:	mov    %rbp,0x80(%rsp)
   e3f29:	mov    %r14,0x88(%rsp)
   e3f31:	mov    %rbx,0x90(%rsp)
   e3f39:	mov    %rcx,0x98(%rsp)
   e3f41:	mov    %rsi,0xa0(%rsp)
   e3f49:	mov    0x48(%rsp),%rcx
   e3f4e:	mov    %rcx,0xa8(%rsp)
   e3f56:	mov    %rax,0xb0(%rsp)
   e3f5e:	mov    %rdi,0xb8(%rsp)
   e3f66:	mov    %rbp,0xc0(%rsp)
   e3f6e:	mov    %r14,0xc8(%rsp)
   e3f76:	mov    %rbx,0xd0(%rsp)
   e3f7e:	lea    0x160(%rsp),%rdi
   e3f86:	lea    0x50(%rsp),%rsi
   e3f8b:	xor    %ecx,%ecx
   e3f8d:	call   f7100 <rayon_core::join::join_context::<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}>
   e3f92:	mov    0x160(%rsp),%rdx
   e3f9a:	mov    0x168(%rsp),%rcx
   e3fa2:	mov    0x170(%rsp),%rax
   e3faa:	mov    0x188(%rsp),%rsi
   e3fb2:	mov    %rsi,0x60(%rsp)
   e3fb7:	movups 0x178(%rsp),%xmm0
   e3fbf:	movaps %xmm0,0x50(%rsp)
   e3fc4:	test   %rcx,%rcx
   e3fc7:	mov    0x1d0(%rsp),%rbp
   e3fcf:	je     e4000 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x16c0>
   e3fd1:	mov    0x50(%rsp),%rsi
   e3fd6:	movq   $0x0,0x50(%rsp)
   e3fdf:	test   %rsi,%rsi
   e3fe2:	je     e4028 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x16e8>
   e3fe4:	mov    %rsi,0x18(%rcx)
   e3fe8:	mov    %rcx,0x20(%rsi)
   e3fec:	mov    0x58(%rsp),%rcx
   e3ff1:	xorps  %xmm0,%xmm0
   e3ff4:	add    0x60(%rsp),%rax
   e3ff9:	movups %xmm0,0x58(%rsp)
   e3ffe:	jmp    e4028 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x16e8>
   e4000:	mov    0x50(%rsp),%rsi
   e4005:	mov    0x58(%rsp),%rcx
   e400a:	mov    %rdx,0x50(%rsp)
   e400f:	movq   $0x0,0x58(%rsp)
   e4018:	mov    0x60(%rsp),%rdi
   e401d:	mov    %rax,0x60(%rsp)
   e4022:	mov    %rsi,%rdx
   e4025:	mov    %rdi,%rax
   e4028:	mov    %rdx,0x0(%rbp)
   e402c:	mov    %rcx,0x8(%rbp)
   e4030:	mov    %rax,0x10(%rbp)
   e4034:	lea    0x50(%rsp),%rdi
   e4039:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e403e:	jmp    e29fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0xbd>
   e4043:	mov    0x40(%rsp),%r14
   e4048:	movb   $0x1,(%r14)
   e404c:	jmp    e3cd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1399>
   e4051:	cmp    $0x3,%rcx
   e4055:	jb     e405d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x171d>
   e4057:	call   *0x2384b(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e405d:	test   %r12,%r12
   e4060:	je     e40eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17ab>
   e4066:	xor    %r13d,%r13d
   e4069:	jmp    e407e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x173e>
   e406b:	mov    0x20(%rsp),%rdi
   e4070:	call   *0x23832(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e4076:	inc    %r13
   e4079:	cmp    %r12,%r13
   e407c:	je     e40eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x17ab>
   e407e:	lea    0x0(,%r13,8),%rcx
   e4086:	add    %r13,%rcx
   e4089:	lea    (%rbx,%rcx,8),%rax
   e408d:	cmpq   $0x0,(%rbx,%rcx,8)
   e4092:	je     e40b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1770>
   e4094:	cmpq   $0x11,0x18(%rax)
   e4099:	jb     e4076 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1736>
   e409b:	mov    0x8(%rax),%rdi
   e409f:	jmp    e4070 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1730>
   e40a1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e40b0:	mov    0x10(%rax),%rbp
   e40b4:	test   %rbp,%rbp
   e40b7:	je     e4076 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1736>
   e40b9:	mov    0x8(%rax),%rax
   e40bd:	mov    %rax,0x20(%rsp)
   e40c2:	lea    0x18(%rax),%r14
   e40c6:	jmp    e40d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1799>
   e40c8:	nopl   0x0(%rax,%rax,1)
   e40d0:	add    $0x20,%r14
   e40d4:	dec    %rbp
   e40d7:	je     e406b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x172b>
   e40d9:	cmpq   $0x11,(%r14)
   e40dd:	jb     e40d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1790>
   e40df:	mov    -0x10(%r14),%rdi
   e40e3:	call   *0x237bf(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e40e9:	jmp    e40d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1790>
   e40eb:	test   %r15,%r15
   e40ee:	mov    0x40(%rsp),%r14
   e40f3:	je     e3cd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1399>
   e40f9:	mov    %rbx,%rdi
   e40fc:	call   *0x237a6(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e4102:	jmp    e3cd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1399>
   e4107:	lea    -0xc5e1a(%rip),%rdi        # 1e2f4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5a7>
   e410e:	lea    0x231c3(%rip),%rdx        # 1072d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   e4115:	mov    $0x53,%esi
   e411a:	call   6ed60 <core::panicking::panic_fmt>
   e411f:	jmp    e4331 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f1>
   e4124:	lea    -0xc5e37(%rip),%rdi        # 1e2f4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5a7>
   e412b:	lea    0x231a6(%rip),%rdx        # 1072d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   e4132:	mov    $0x53,%esi
   e4137:	call   6ed60 <core::panicking::panic_fmt>
   e413c:	jmp    e4331 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f1>
   e4141:	lea    -0xc610b(%rip),%rdi        # 1e03d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f0>
   e4148:	lea    0x233b1(%rip),%rdx        # 107500 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4008>
   e414f:	mov    $0x13,%esi
   e4154:	call   6ed60 <core::panicking::panic_fmt>
   e4159:	lea    0x2a0(%rsp),%rdi
   e4161:	lea    0x218(%rsp),%rsi
   e4169:	call   6ebe3 <core::panicking::assert_failed::<core::option::Option<usize>, core::option::Option<usize>>>
   e416e:	jmp    e4331 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f1>
   e4173:	lea    0x120(%rsp),%rax
   e417b:	mov    %rax,0x50(%rsp)
   e4180:	lea    0x230(%rsp),%rax
   e4188:	mov    %rax,0x58(%rsp)
   e418d:	lea    0x138(%rsp),%rcx
   e4195:	mov    %rcx,0x60(%rsp)
   e419a:	mov    %r12,0x68(%rsp)
   e419f:	mov    %r15,0x70(%rsp)
   e41a4:	mov    0x40(%rsp),%rdx
   e41a9:	mov    %rdx,0x78(%rsp)
   e41ae:	mov    %rbp,0x80(%rsp)
   e41b6:	mov    %r14,0x88(%rsp)
   e41be:	mov    %rbx,0x90(%rsp)
   e41c6:	mov    %rax,0x98(%rsp)
   e41ce:	mov    %rcx,0xa0(%rsp)
   e41d6:	mov    0x48(%rsp),%rax
   e41db:	mov    %rax,0xa8(%rsp)
   e41e3:	mov    %r13,0xb0(%rsp)
   e41eb:	mov    %rdx,0xb8(%rsp)
   e41f3:	mov    %rbp,0xc0(%rsp)
   e41fb:	mov    %r14,0xc8(%rsp)
   e4203:	mov    %rbx,0xd0(%rsp)
   e420b:	sub    $0xffffffffffffff80,%rsi
   e420f:	lea    0x160(%rsp),%rdi
   e4217:	lea    0x50(%rsp),%rdx
   e421c:	call   f75c0 <<rayon_core::registry::Registry>::in_worker_cold::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e4221:	jmp    e3f92 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1652>
   e4226:	mov    %rcx,0x50(%rsp)
   e422b:	lea    0x230(%rsp),%rcx
   e4233:	mov    %rcx,0x58(%rsp)
   e4238:	lea    0x138(%rsp),%rdi
   e4240:	mov    %rdi,0x60(%rsp)
   e4245:	mov    %r12,0x68(%rsp)
   e424a:	mov    %r15,0x70(%rsp)
   e424f:	mov    0x40(%rsp),%r8
   e4254:	mov    %r8,0x78(%rsp)
   e4259:	mov    %rbp,0x80(%rsp)
   e4261:	mov    %r14,0x88(%rsp)
   e4269:	mov    %rbx,0x90(%rsp)
   e4271:	mov    %rcx,0x98(%rsp)
   e4279:	mov    %rdi,0xa0(%rsp)
   e4281:	mov    0x48(%rsp),%rcx
   e4286:	mov    %rcx,0xa8(%rsp)
   e428e:	mov    %rax,0xb0(%rsp)
   e4296:	mov    %r8,0xb8(%rsp)
   e429e:	mov    %rbp,0xc0(%rsp)
   e42a6:	mov    %r14,0xc8(%rsp)
   e42ae:	mov    %rbx,0xd0(%rsp)
   e42b6:	sub    $0xffffffffffffff80,%rsi
   e42ba:	lea    0x160(%rsp),%rdi
   e42c2:	lea    0x50(%rsp),%rcx
   e42c7:	call   f76c0 <<rayon_core::registry::Registry>::in_worker_cross::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e42cc:	jmp    e3f92 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1652>
   e42d1:	lea    0x22bb0(%rip),%rax        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e42d8:	jmp    e42f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b0>
   e42da:	mov    %rdi,%r13
   e42dd:	lea    0x22ba4(%rip),%rax        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e42e4:	jmp    e42f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b0>
   e42e6:	mov    %rdi,%r13
   e42e9:	lea    0x22b80(%rip),%rax        # 106e70 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3978>
   e42f0:	mov    %rax,0x38(%rsp)
   e42f5:	mov    %r13,%rdi
   e42f8:	mov    0x38(%rsp),%rdx
   e42fd:	call   6fcfb <core::panicking::panic_bounds_check>
   e4302:	jmp    e4331 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f1>
   e4304:	mov    $0x8,%edi
   e4309:	mov    $0x80,%esi
   e430e:	call   690eb <alloc::raw_vec::handle_error>
   e4313:	jmp    e4331 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19f1>
   e4315:	mov    %r15,0x130(%rsp)
   e431d:	lea    0x2246c(%rip),%rdx        # 106790 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   e4324:	mov    0x130(%rsp),%rdi
   e432c:	call   6fcfb <core::panicking::panic_bounds_check>
   e4331:	ud2
   e4333:	mov    %rcx,%r13
   e4336:	mov    %rdx,%rsi
   e4339:	jmp    e433e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19fe>
   e433b:	mov    %rcx,%r13
   e433e:	lea    0x2244b(%rip),%rax        # 106790 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   e4345:	jmp    e42f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b0>
   e4347:	mov    %rdx,%r13
   e434a:	jmp    e42e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19a9>
   e434c:	mov    %rdx,%r13
   e434f:	lea    0x22b32(%rip),%rax        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e4356:	jmp    e42f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b0>
   e4358:	mov    %rbx,%r13
   e435b:	jmp    e42e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19a9>
   e435d:	mov    %rbx,%r13
   e4360:	lea    0x22b21(%rip),%rax        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e4367:	jmp    e42f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b0>
   e4369:	mov    %r9,%r13
   e436c:	mov    %r8,%rsi
   e436f:	jmp    e42f5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b5>
   e4371:	mov    $0xffffffffffffffff,%r13
   e4378:	xor    %esi,%esi
   e437a:	jmp    e42f5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x19b5>
   e437f:	jmp    e440b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1acb>
   e4384:	mov    %rax,%rbx
   e4387:	lea    0x188(%rsp),%rbp
   e438f:	mov    %rbp,%rdi
   e4392:	call   f3710 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e4397:	mov    %rbx,%rdi
   e439a:	call   1019a0 <_Unwind_Resume@plt>
   e439f:	mov    %rax,%rbx
   e43a2:	lea    0x268(%rsp),%rdi
   e43aa:	call   f34c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>
   e43af:	jmp    e4451 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1b11>
   e43b4:	mov    %rax,%rbx
   e43b7:	mov    %rbp,%rdi
   e43ba:	call   f3710 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e43bf:	mov    %rbx,%rdi
   e43c2:	call   1019a0 <_Unwind_Resume@plt>
   e43c7:	mov    %rax,%rbx
   e43ca:	pshufd $0xee,0x20(%rsp),%xmm0
   e43d1:	movq   %xmm0,%rax
   e43d6:	cmp    $0x3,%rax
   e43da:	jb     e43e5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1aa5>
   e43dc:	mov    %r15,%rdi
   e43df:	call   *0x234c3(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e43e5:	lea    0x78(%rsp),%rbp
   e43ea:	jmp    e4451 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1b11>
   e43ec:	mov    %rax,%rbx
   e43ef:	jmp    e4451 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1b11>
   e43f1:	mov    %rax,%rbx
   e43f4:	jmp    e4404 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1ac4>
   e43f6:	mov    %rax,%rbx
   e43f9:	mov    0x20(%rsp),%rdi
   e43fe:	call   *0x234a4(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e4404:	lea    0x78(%rsp),%rbp
   e4409:	jmp    e4437 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1af7>
   e440b:	mov    %rax,%rbx
   e440e:	lea    0x78(%rsp),%rbp
   e4413:	mov    0x20(%rsp),%rdi
   e4418:	call   *0x2348a(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e441e:	cmpq   $0x3,0xf8(%rsp)
   e4427:	jb     e4437 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>+0x1af7>
   e4429:	mov    0xe8(%rsp),%rdi
   e4431:	call   *0x23471(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e4437:	lea    0x160(%rsp),%rdi
   e443f:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e4444:	lea    0x178(%rsp),%rdi
   e444c:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e4451:	lea    0x148(%rsp),%rdi
   e4459:	call   e9250 <core::ptr::drop_glue::<rayon::vec::IntoIter<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e445e:	mov    %rbp,%rdi
   e4461:	call   f3710 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e4466:	mov    %rbx,%rdi
   e4469:	call   1019a0 <_Unwind_Resume@plt>
