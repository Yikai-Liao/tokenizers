
/tmp/bpe-corpus-binaries-20261013/codegen-borrow-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000e5130 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>>:
   e5130:	push   %rbp
   e5131:	push   %r15
   e5133:	push   %r14
   e5135:	push   %r13
   e5137:	push   %r12
   e5139:	push   %rbx
   e513a:	sub    $0x3e8,%rsp
   e5141:	mov    0x428(%rsp),%r14
   e5149:	mov    %rsi,0x138(%rsp)
   e5151:	mov    %rcx,0x178(%rsp)
   e5159:	mov    %r8,0x180(%rsp)
   e5161:	mov    (%r14),%rbp
   e5164:	movzbl 0x0(%rbp),%eax
   e5168:	test   %al,%al
   e516a:	je     e51f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc8>
   e5170:	mov    0x8(%r14),%rax
   e5174:	xorps  %xmm0,%xmm0
   e5177:	movups %xmm0,0x78(%rsp)
   e517c:	mov    0x18(%r14),%rcx
   e5180:	movq   $0x0,0x40(%rsp)
   e5189:	movq   $0x8,0x48(%rsp)
   e5192:	movq   $0x0,0x50(%rsp)
   e519b:	mov    %rbp,0x58(%rsp)
   e51a0:	mov    %rax,0x60(%rsp)
   e51a5:	lea    0x68(%rsp),%r15
   e51aa:	movq   $0x0,0x68(%rsp)
   e51b3:	movq   $0x4,0x70(%rsp)
   e51bc:	movq   $0x4,0x88(%rsp)
   e51c8:	movq   $0x0,0x90(%rsp)
   e51d4:	mov    %rcx,0x98(%rsp)
   e51dc:	lea    0x40(%rsp),%rsi
   e51e1:	call   f3440 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e51e6:	add    $0x3e8,%rsp
   e51ed:	pop    %rbx
   e51ee:	pop    %r12
   e51f0:	pop    %r13
   e51f2:	pop    %r14
   e51f4:	pop    %r15
   e51f6:	pop    %rbp
   e51f7:	ret
   e51f8:	mov    %r9,%r12
   e51fb:	mov    %rdi,0x208(%rsp)
   e5203:	mov    0x420(%rsp),%rbx
   e520b:	shr    $1,%rsi
   e520e:	cmp    %r8,%rsi
   e5211:	jae    e67e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16b4>
   e5217:	mov    0x8(%r14),%rax
   e521b:	xorps  %xmm0,%xmm0
   e521e:	movups %xmm0,0x1d8(%rsp)
   e5226:	mov    0x18(%r14),%rcx
   e522a:	movq   $0x0,0x1a0(%rsp)
   e5236:	movq   $0x8,0x1a8(%rsp)
   e5242:	movq   $0x0,0x1b0(%rsp)
   e524e:	mov    %rbp,0x118(%rsp)
   e5256:	mov    %rbp,0x1b8(%rsp)
   e525e:	mov    %rax,0x238(%rsp)
   e5266:	mov    %rax,0x1c0(%rsp)
   e526e:	lea    0x1c8(%rsp),%r15
   e5276:	movq   $0x0,0x1c8(%rsp)
   e5282:	movq   $0x4,0x1d0(%rsp)
   e528e:	movq   $0x4,0x1e8(%rsp)
   e529a:	movq   $0x0,0x1f0(%rsp)
   e52a6:	mov    %rcx,0x258(%rsp)
   e52ae:	mov    %rcx,0x1f8(%rsp)
   e52b6:	imul   $0x70,%rbx,%rax
   e52ba:	add    %r12,%rax
   e52bd:	mov    %rax,0x260(%rsp)
   e52c5:	movq   $0x0,0x188(%rsp)
   e52d1:	movq   $0x8,0x190(%rsp)
   e52dd:	movq   $0x0,0x198(%rsp)
   e52e9:	mov    $0x8,%eax
   e52ee:	mov    %rax,0x240(%rsp)
   e52f6:	lea    0x21213(%rip),%rax        # 106510 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3318>
   e52fd:	movq   %rax,%xmm0
   e5302:	movdqa %xmm0,0x3d0(%rsp)
   e530b:	lea    0x40(%rsp),%rbp
   e5310:	lea    0x21961(%rip),%rax        # 106c78 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a80>
   e5317:	mov    %rax,0x10(%rsp)
   e531c:	movq   $0x0,0x248(%rsp)
   e5328:	jmp    e53a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x270>
   e532a:	nopw   0x0(%rax,%rax,1)
   e5330:	mov    0x190(%rsp),%rax
   e5338:	mov    %rax,0x240(%rsp)
   e5340:	imul   $0x38,%rbx,%rax
   e5344:	mov    0x2e8(%rsp),%rcx
   e534c:	mov    0x240(%rsp),%rdx
   e5354:	mov    %rcx,0x30(%rdx,%rax,1)
   e5359:	movdqu 0x2b8(%rsp),%xmm0
   e5362:	movdqu 0x2c8(%rsp),%xmm1
   e536b:	movdqu 0x2d8(%rsp),%xmm2
   e5374:	movdqu %xmm2,0x20(%rdx,%rax,1)
   e537a:	movdqu %xmm1,0x10(%rdx,%rax,1)
   e5380:	movdqu %xmm0,(%rdx,%rax,1)
   e5385:	inc    %rbx
   e5388:	mov    %rbx,0x248(%rsp)
   e5390:	mov    %rbx,0x198(%rsp)
   e5398:	lea    0x1c8(%rsp),%r15
   e53a0:	cmp    0x260(%rsp),%r12
   e53a8:	je     e6805 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e53ae:	mov    (%r12),%r14d
   e53b2:	mov    0x4(%r12),%edx
   e53b7:	mov    0x8(%r12),%ecx
   e53bc:	movups 0x60(%r12),%xmm0
   e53c2:	movaps %xmm0,0x340(%rsp)
   e53ca:	movups 0x50(%r12),%xmm0
   e53d0:	movaps %xmm0,0x330(%rsp)
   e53d8:	movdqu 0x10(%r12),%xmm0
   e53df:	movdqu 0x20(%r12),%xmm1
   e53e6:	movdqu 0x30(%r12),%xmm2
   e53ed:	movups 0x40(%r12),%xmm3
   e53f3:	movaps %xmm3,0x320(%rsp)
   e53fb:	movdqa %xmm2,0x310(%rsp)
   e5404:	movdqa %xmm1,0x300(%rsp)
   e540d:	movdqa %xmm0,0x2f0(%rsp)
   e5416:	mov    0x258(%rsp),%rax
   e541e:	mov    (%rax),%rdi
   e5421:	mov    0x8(%rax),%rsi
   e5425:	mov    %rsi,0x160(%rsp)
   e542d:	mov    0x10(%rax),%rax
   e5431:	mov    %rax,0x100(%rsp)
   e5439:	mov    %rdi,0x120(%rsp)
   e5441:	mov    (%rdi),%rbx
   e5444:	mov    0x50(%rbx),%r8
   e5448:	lea    0x350(%rsp),%rdi
   e5450:	mov    %r14d,%esi
   e5453:	mov    %rdx,0x20(%rsp)
   e5458:	mov    %ecx,0x174(%rsp)
   e545f:	mov    %r15,%r9
   e5462:	call   f4610 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::new>
   e5467:	movq   $0x0,0xf0(%rsp)
   e5473:	mov    0x50(%rbx),%rsi
   e5477:	cmp    %r14,%rsi
   e547a:	jbe    e6e98 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d68>
   e5480:	mov    0x20(%rsp),%rdi
   e5485:	cmp    %rdi,%rsi
   e5488:	jbe    e6e9b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d6b>
   e548e:	mov    0x48(%rbx),%rax
   e5492:	mov    %r14,0x130(%rsp)
   e549a:	mov    (%rax,%r14,8),%rcx
   e549e:	mov    %rcx,0x128(%rsp)
   e54a6:	mov    (%rax,%rdi,8),%rax
   e54aa:	mov    %rax,0x168(%rsp)
   e54b2:	lea    0x2f0(%rsp),%rax
   e54ba:	mov    %rax,0x268(%rsp)
   e54c2:	mov    %rbx,0x270(%rsp)
   e54ca:	mov    %rbx,0x38(%rsp)
   e54cf:	mov    %rbx,0x278(%rsp)
   e54d7:	mov    $0x80,%edi
   e54dc:	call   *0x220de(%rip)        # 1075c0 <malloc@GLIBC_2.2.5>
   e54e2:	test   %rax,%rax
   e54e5:	je     e6ea9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d79>
   e54eb:	mov    %rax,%rbx
   e54ee:	add    $0x70,%r12
   e54f2:	mov    %r12,0xf8(%rsp)
   e54fa:	pxor   %xmm0,%xmm0
   e54fe:	lea    0x150(%rsp),%rax
   e5506:	movdqu %xmm0,(%rax)
   e550a:	movq   $0x10,0x140(%rsp)
   e5516:	mov    %rbx,0x148(%rsp)
   e551e:	lea    0x268(%rsp),%rax
   e5526:	mov    %rax,0x40(%rsp)
   e552b:	movq   $0x10,0x48(%rsp)
   e5534:	mov    $0x10,%r12d
   e553a:	xor    %r14d,%r14d
   e553d:	xor    %r15d,%r15d
   e5540:	mov    %rbp,%rdi
   e5543:	call   f6df0 <<core::iter::adapters::take::Take<&mut core::iter::adapters::inspect::Inspect<core::iter::adapters::map::Map<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#1}>> as core::iter::traits::iterator::Iterator>::next>
   e5548:	cmp    $0x1,%rax
   e554c:	jne    e56f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x5c8>
   e5552:	cmp    $0xffffffffffffffff,%r15
   e5556:	je     e6c6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b3e>
   e555c:	mov    %rdx,%rbp
   e555f:	mov    %r12,%rax
   e5562:	sub    %r15,%rax
   e5565:	jbe    e5570 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x440>
   e5567:	mov    %r15,%r13
   e556a:	jmp    e565e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e556f:	nop
   e5570:	je     e55f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x4c8>
   e5576:	mov    %r12,%rsi
   e5579:	mov    %r15,%r13
   e557c:	mov    %r14,%rcx
   e557f:	cmp    %rax,%r14
   e5582:	jbe    e5650 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x520>
   e5588:	mov    %r12,%rdx
   e558b:	sub    %rcx,%rdx
   e558e:	mov    %r13,%rax
   e5591:	sub    %rdx,%rax
   e5594:	cmp    %rax,%rdx
   e5597:	jbe    e55c8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x498>
   e5599:	mov    %rsi,%rcx
   e559c:	sub    %r12,%rcx
   e559f:	cmp    %rax,%rcx
   e55a2:	jb     e55c8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x498>
   e55a4:	mov    0x148(%rsp),%rbx
   e55ac:	lea    (%rbx,%r12,8),%rdi
   e55b0:	shl    $0x3,%rax
   e55b4:	mov    %rsi,%r12
   e55b7:	mov    %rbx,%rsi
   e55ba:	mov    %rax,%rdx
   e55bd:	call   *0x21fed(%rip)        # 1075b0 <memcpy@GLIBC_2.14>
   e55c3:	jmp    e565e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e55c8:	mov    %rsi,%r15
   e55cb:	sub    %rdx,%r15
   e55ce:	mov    0x148(%rsp),%rbx
   e55d6:	mov    %rsi,%r12
   e55d9:	lea    (%rbx,%r14,8),%rsi
   e55dd:	lea    (%rbx,%r15,8),%rdi
   e55e1:	shl    $0x3,%rdx
   e55e5:	call   *0x21fcd(%rip)        # 1075b8 <memmove@GLIBC_2.2.5>
   e55eb:	mov    %r15,0x150(%rsp)
   e55f3:	mov    %r15,%r14
   e55f6:	jmp    e565e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x52e>
   e55f8:	mov    $0x1,%edx
   e55fd:	mov    $0x8,%ecx
   e5602:	mov    $0x8,%r8d
   e5608:	lea    0x140(%rsp),%rdi
   e5610:	mov    %r15,%rsi
   e5613:	call   cc100 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e5618:	mov    0x140(%rsp),%rsi
   e5620:	mov    0x150(%rsp),%r14
   e5628:	mov    0x158(%rsp),%r13
   e5630:	mov    %r12,%rax
   e5633:	sub    %r13,%rax
   e5636:	mov    %r14,%rcx
   e5639:	cmp    %rax,%r14
   e563c:	ja     e5588 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x458>
   e5642:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5650:	mov    0x148(%rsp),%rbx
   e5658:	mov    %rsi,%r12
   e565b:	mov    %rcx,%r14
   e565e:	lea    (%r14,%r13,1),%rax
   e5662:	cmp    %r12,%rax
   e5665:	mov    $0x0,%ecx
   e566a:	cmovae %r12,%rcx
   e566e:	sub    %rcx,%rax
   e5671:	mov    %rbp,(%rbx,%rax,8)
   e5675:	lea    0x1(%r13),%r15
   e5679:	mov    %r15,0x158(%rsp)
   e5681:	cmp    %r12,%r15
   e5684:	lea    0x40(%rsp),%rbp
   e5689:	jae    e5540 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x410>
   e568f:	mov    %r14,0x18(%rsp)
   e5694:	add    %r13,%r14
   e5697:	inc    %r14
   e569a:	nopw   0x0(%rax,%rax,1)
   e56a0:	mov    %rbp,%rdi
   e56a3:	call   f6df0 <<core::iter::adapters::take::Take<&mut core::iter::adapters::inspect::Inspect<core::iter::adapters::map::Map<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::fresh_matches<&mut core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>::{closure#1}>> as core::iter::traits::iterator::Iterator>::next>
   e56a8:	cmp    $0x1,%rax
   e56ac:	jne    e56f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x5c0>
   e56ae:	cmp    %r12,%r14
   e56b1:	mov    $0x0,%eax
   e56b6:	cmovae %r12,%rax
   e56ba:	shl    $0x3,%rax
   e56be:	mov    %rbx,%rcx
   e56c1:	sub    %rax,%rcx
   e56c4:	mov    %rdx,(%rcx,%r14,8)
   e56c8:	inc    %r15
   e56cb:	inc    %r14
   e56ce:	cmp    %r15,%r12
   e56d1:	jne    e56a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x570>
   e56d3:	mov    %r12,0x158(%rsp)
   e56db:	mov    %r12,%r15
   e56de:	mov    0x18(%rsp),%r14
   e56e3:	jmp    e5540 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x410>
   e56e8:	nopl   0x0(%rax,%rax,1)
   e56f0:	mov    %r15,0x158(%rsp)
   e56f8:	mov    0x128(%rsp),%rsi
   e5700:	mov    0x168(%rsp),%rdi
   e5708:	add    %rsi,%rdi
   e570b:	movups 0x140(%rsp),%xmm0
   e5713:	mov    0x150(%rsp),%rax
   e571b:	mov    0x158(%rsp),%rcx
   e5723:	movaps %xmm0,0x280(%rsp)
   e572b:	mov    %rax,0x290(%rsp)
   e5733:	mov    %rcx,0x298(%rsp)
   e573b:	mov    0x278(%rsp),%rax
   e5743:	lea    0x2a0(%rsp),%rcx
   e574b:	mov    %rax,0x10(%rcx)
   e574f:	movups 0x268(%rsp),%xmm0
   e5757:	movups %xmm0,(%rcx)
   e575a:	mov    0x2b0(%rsp),%rax
   e5762:	lea    0x80(%rsp),%rdx
   e576a:	mov    %rax,0x30(%rdx)
   e576e:	movdqa 0x280(%rsp),%xmm0
   e5777:	mov    0x290(%rsp),%rax
   e577f:	mov    0x298(%rsp),%rcx
   e5787:	movdqa 0x2a0(%rsp),%xmm1
   e5790:	movdqu %xmm1,0x20(%rdx)
   e5795:	mov    %rax,0x10(%rdx)
   e5799:	mov    %rcx,0x18(%rdx)
   e579d:	movdqu %xmm0,(%rdx)
   e57a1:	movq   $0x0,0x40(%rsp)
   e57aa:	mov    0x38(%rsp),%rax
   e57af:	mov    %rax,0x58(%rsp)
   e57b4:	mov    %rsi,0x60(%rsp)
   e57b9:	mov    %rdi,0x68(%rsp)
   e57be:	mov    0x130(%rsp),%rcx
   e57c6:	mov    %ecx,0x70(%rsp)
   e57ca:	mov    0x20(%rsp),%rcx
   e57cf:	mov    %ecx,0x74(%rsp)
   e57d3:	mov    %rax,0x78(%rsp)
   e57d8:	mov    0x80(%rsp),%rax
   e57e0:	mov    %rax,0xd8(%rsp)
   e57e8:	mov    0x98(%rsp),%rbx
   e57f0:	test   %rbx,%rbx
   e57f3:	je     e6600 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e57f9:	mov    0x120(%rsp),%rax
   e5801:	mov    0x8(%rax),%rax
   e5805:	mov    %rax,0x120(%rsp)
   e580d:	mov    0x100(%rsp),%rcx
   e5815:	movzbl (%rcx),%eax
   e5818:	mov    %al,0xf(%rsp)
   e581c:	mov    0x10(%rcx),%rax
   e5820:	mov    %rax,0xd0(%rsp)
   e5828:	mov    0x8(%rcx),%rax
   e582c:	mov    %rax,0x108(%rsp)
   e5834:	mov    0x160(%rsp),%rdx
   e583c:	mov    0x18(%rdx),%rax
   e5840:	mov    %rax,0x228(%rsp)
   e5848:	mov    0x28(%rdx),%rax
   e584c:	mov    %rax,0x218(%rsp)
   e5854:	mov    0x20(%rdx),%rax
   e5858:	mov    %rax,0x210(%rsp)
   e5860:	mov    (%rdx),%rax
   e5863:	mov    %rax,0x230(%rsp)
   e586b:	mov    0x8(%rdx),%rax
   e586f:	mov    %rax,0x110(%rsp)
   e5877:	mov    0x50(%rcx),%rax
   e587b:	mov    %rax,0x220(%rsp)
   e5883:	mov    0x48(%rcx),%rax
   e5887:	mov    %rax,0x250(%rsp)
   e588f:	mov    0x90(%rsp),%r14
   e5897:	mov    0x58(%rsp),%r12
   e589c:	mov    0x60(%rsp),%rax
   e58a1:	mov    %rax,0x128(%rsp)
   e58a9:	mov    0x70(%rsp),%eax
   e58ad:	mov    %eax,0x168(%rsp)
   e58b4:	mov    0x68(%rsp),%rax
   e58b9:	mov    %rax,0x130(%rsp)
   e58c1:	mov    0x74(%rsp),%eax
   e58c5:	mov    %eax,0x160(%rsp)
   e58cc:	mov    0xd8(%rsp),%r10
   e58d4:	mov    %r12,0x100(%rsp)
   e58dc:	jmp    e58f6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7c6>
   e58de:	xchg   %ax,%ax
   e58e0:	cmp    0x160(%rsp),%eax
   e58e7:	je     e5faf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe7f>
   e58ed:	test   %rbx,%rbx
   e58f0:	je     e6600 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e58f6:	mov    %r14,%rax
   e58f9:	lea    0x1(%r14),%rcx
   e58fd:	cmp    %r10,%rcx
   e5900:	mov    %r10,0x20(%rsp)
   e5905:	mov    %r10,%rcx
   e5908:	mov    $0x0,%edx
   e590d:	cmovb  %rdx,%rcx
   e5911:	neg    %rcx
   e5914:	add    %rcx,%r14
   e5917:	inc    %r14
   e591a:	mov    %r14,0x90(%rsp)
   e5922:	mov    %rbx,%r13
   e5925:	dec    %r13
   e5928:	mov    %r13,0x98(%rsp)
   e5930:	mov    0x88(%rsp),%rcx
   e5938:	mov    0xa0(%rsp),%rbp
   e5940:	mov    (%rcx,%rax,8),%rax
   e5944:	mov    %rax,0x18(%rsp)
   e5949:	mov    0x58(%rbp),%rax
   e594d:	mov    $0x0,%r15d
   e5953:	test   %rax,%rax
   e5956:	je     e5cf6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc6>
   e595c:	dec    %rax
   e595f:	mov    %rax,0x58(%rbp)
   e5963:	lea    0x8(%rbp),%rax
   e5967:	lea    0x10(%rbp),%r8
   e596b:	mov    0x0(%rbp),%rsi
   e596f:	test   %rsi,%rsi
   e5972:	je     e59af <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x87f>
   e5974:	mov    (%rax),%rdx
   e5977:	test   %rdx,%rdx
   e597a:	je     e59a7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x877>
   e597c:	mov    0x18(%rbp),%r9
   e5980:	test   $0x7f,%r9b
   e5984:	je     e5b9a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa6a>
   e598a:	movsbq (%rsi),%rbx
   e598e:	test   %rbx,%rbx
   e5991:	js     e5a65 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x935>
   e5997:	add    (%r8),%rbx
   e599a:	mov    %rbx,(%r8)
   e599d:	mov    $0x1,%edi
   e59a2:	jmp    e5acf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x99f>
   e59a7:	movq   $0x0,0x0(%rbp)
   e59af:	mov    0x40(%rbp),%r15
   e59b3:	test   %r15,%r15
   e59b6:	je     e5da3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc73>
   e59bc:	mov    0x48(%rbp),%r11
   e59c0:	cmp    %r11,%r15
   e59c3:	je     e5da3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc73>
   e59c9:	lea    0x20(%r15),%rbx
   e59cd:	mov    %rbx,0x40(%rbp)
   e59d1:	mov    0x50(%rbp),%r9
   e59d5:	mov    (%r15),%rcx
   e59d8:	dec    %rcx
   e59db:	cmp    %rcx,%r9
   e59de:	jne    e59f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x8c9>
   e59e0:	movq   $0x1,0x0(%rbp)
   e59e8:	pxor   %xmm0,%xmm0
   e59ec:	movdqu %xmm0,(%rax)
   e59f0:	mov    %r9,0x18(%rbp)
   e59f4:	jmp    e5c34 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb04>
   e59f9:	mov    %rcx,%r10
   e59fc:	shr    $0x7,%r10
   e5a00:	test   %r9,%r9
   e5a03:	je     e5ad7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9a7>
   e5a09:	mov    %ecx,%eax
   e5a0b:	and    $0x7f,%eax
   e5a0e:	cmp    $0x1,%rax
   e5a12:	mov    %r10,%rsi
   e5a15:	sbb    $0xffffffffffffffff,%rsi
   e5a19:	cmp    $0x2,%rsi
   e5a1d:	jb     e6e88 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d58>
   e5a23:	mov    0x18(%r15),%rdi
   e5a27:	mov    %rdi,%rdx
   e5a2a:	cmp    $0x11,%rdi
   e5a2e:	jb     e5a34 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x904>
   e5a30:	mov    0x10(%r15),%rdx
   e5a34:	lea    0x8(,%rsi,8),%rax
   e5a3c:	cmp    %rdx,%rax
   e5a3f:	ja     e6cac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b7c>
   e5a45:	mov    %r9,%rdx
   e5a48:	shr    $0x7,%rdx
   e5a4c:	cmp    %rsi,%rdx
   e5a4f:	jae    e6e93 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d63>
   e5a55:	cmp    $0x11,%rdi
   e5a59:	jb     e5adb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9ab>
   e5a5f:	mov    0x8(%r15),%rax
   e5a63:	jmp    e5adf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9af>
   e5a65:	cmp    $0x1,%rdx
   e5a69:	je     e6cd5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba5>
   e5a6f:	and    $0x7f,%ebx
   e5a72:	lea    -0x1(%rdx),%r10
   e5a76:	mov    $0x7,%ecx
   e5a7b:	xor    %edi,%edi
   e5a7d:	mov    %rbx,%rax
   e5a80:	movzbl 0x1(%rsi,%rdi,1),%r11d
   e5a86:	cmp    $0x3f,%ecx
   e5a89:	jne    e5a95 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x965>
   e5a8b:	cmp    $0x1,%r11b
   e5a8f:	ja     e6cd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba9>
   e5a95:	movzbl %r11b,%ebx
   e5a99:	test   %bl,%bl
   e5a9b:	jns    e5ab6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x986>
   e5a9d:	and    $0x7f,%ebx
   e5aa0:	shl    %cl,%rbx
   e5aa3:	or     %rbx,%rax
   e5aa6:	add    $0x7,%ecx
   e5aa9:	inc    %rdi
   e5aac:	cmp    %rdi,%r10
   e5aaf:	jne    e5a80 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x950>
   e5ab1:	jmp    e6cd5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ba5>
   e5ab6:	shl    %cl,%rbx
   e5ab9:	or     %rax,%rbx
   e5abc:	add    $0x2,%rdi
   e5ac0:	add    (%r8),%rbx
   e5ac3:	mov    %rbx,(%r8)
   e5ac6:	cmp    %rdx,%rdi
   e5ac9:	ja     e6c62 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b32>
   e5acf:	sub    %rdi,%rdx
   e5ad2:	jmp    e5bb3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa83>
   e5ad7:	xor    %edi,%edi
   e5ad9:	jmp    e5ae4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9b4>
   e5adb:	lea    0x8(%r15),%rax
   e5adf:	mov    0x8(%rax,%rdx,8),%rdi
   e5ae4:	and    $0x7f,%ecx
   e5ae7:	cmp    $0x1,%rcx
   e5aeb:	sbb    $0xffffffffffffffff,%r10
   e5aef:	mov    $0x8,%esi
   e5af4:	cmp    $0x2,%r10
   e5af8:	jae    e5b18 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9e8>
   e5afa:	mov    0x18(%r15),%rcx
   e5afe:	mov    %rcx,%rax
   e5b01:	cmp    $0x11,%rcx
   e5b05:	jb     e5b40 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa10>
   e5b07:	mov    0x10(%r15),%rax
   e5b0b:	mov    %rax,%rdx
   e5b0e:	sub    %rsi,%rdx
   e5b11:	jae    e5b4c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa1c>
   e5b13:	jmp    e6c9b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b6b>
   e5b18:	dec    %r10
   e5b1b:	bsr    %r10,%rcx
   e5b1f:	not    %ecx
   e5b21:	mov    $0xffffffffffffffff,%rax
   e5b28:	shr    %cl,%rax
   e5b2b:	lea    0x10(,%rax,8),%rsi
   e5b33:	mov    0x18(%r15),%rcx
   e5b37:	mov    %rcx,%rax
   e5b3a:	cmp    $0x11,%rcx
   e5b3e:	jae    e5b07 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x9d7>
   e5b40:	mov    %rax,%rdx
   e5b43:	sub    %rsi,%rdx
   e5b46:	jb     e6c9b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b6b>
   e5b4c:	cmp    %rdi,%rdx
   e5b4f:	jb     e6ca0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b70>
   e5b55:	cmp    $0x11,%rcx
   e5b59:	jb     e5b61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa31>
   e5b5b:	mov    0x8(%r15),%r15
   e5b5f:	jmp    e5b65 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa35>
   e5b61:	add    $0x8,%r15
   e5b65:	add    %rdi,%rsi
   e5b68:	add    %r15,%rsi
   e5b6b:	sub    %rdi,%rdx
   e5b6e:	mov    %rsi,0x0(%rbp)
   e5b72:	mov    %rdx,0x8(%rbp)
   e5b76:	movq   $0x0,0x10(%rbp)
   e5b7e:	mov    %r9,0x18(%rbp)
   e5b82:	je     e5c34 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb04>
   e5b88:	movq   $0x0,0x50(%rbp)
   e5b90:	test   $0x7f,%r9b
   e5b94:	jne    e598a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x85a>
   e5b9a:	cmp    $0x7,%rdx
   e5b9e:	jbe    e6c8b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b5b>
   e5ba4:	mov    (%rsi),%rbx
   e5ba7:	mov    %rbx,(%r8)
   e5baa:	add    $0xfffffffffffffff8,%rdx
   e5bae:	mov    $0x8,%edi
   e5bb3:	add    %rdi,%rsi
   e5bb6:	mov    %rsi,0x0(%rbp)
   e5bba:	mov    %rdx,0x8(%rbp)
   e5bbe:	inc    %r9
   e5bc1:	mov    %r9,0x18(%rbp)
   e5bc5:	mov    0xa8(%rsp),%rax
   e5bcd:	cmp    0x10(%rax),%rbx
   e5bd1:	jae    e6c45 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b15>
   e5bd7:	mov    0xb0(%rsp),%rax
   e5bdf:	mov    0x10(%rax),%rcx
   e5be3:	cmpl   $0x1,(%rax)
   e5be6:	jne    e5ce0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbb0>
   e5bec:	cmp    %rcx,%rbx
   e5bef:	jae    e5cf0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc0>
   e5bf5:	lea    0x0(,%rbx,4),%rcx
   e5bfd:	jmp    e5ce9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbb9>
   e5c02:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5c10:	mov    $0x1,%esi
   e5c15:	xor    %edx,%edx
   e5c17:	mov    %rsi,0x0(%rbp)
   e5c1b:	mov    %rdx,0x8(%rbp)
   e5c1f:	pxor   %xmm0,%xmm0
   e5c23:	movdqu %xmm0,(%r8)
   e5c28:	mov    %r9,%rbx
   e5c2b:	test   %rdx,%rdx
   e5c2e:	jne    e5de5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xcb5>
   e5c34:	movq   $0x0,0x0(%rbp)
   e5c3c:	cmp    %r11,%rbx
   e5c3f:	je     e5d9b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc6b>
   e5c45:	lea    0x20(%rbx),%r9
   e5c49:	mov    %r9,0x40(%rbp)
   e5c4d:	mov    (%rbx),%rax
   e5c50:	dec    %rax
   e5c53:	je     e5c10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xae0>
   e5c55:	mov    %eax,%ecx
   e5c57:	and    $0x7f,%ecx
   e5c5a:	shr    $0x7,%rax
   e5c5e:	cmp    $0x1,%rcx
   e5c62:	sbb    $0xffffffffffffffff,%rax
   e5c66:	mov    $0x8,%edi
   e5c6b:	cmp    $0x2,%rax
   e5c6f:	jae    e5c90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb60>
   e5c71:	mov    0x18(%rbx),%rcx
   e5c75:	mov    %rcx,%rax
   e5c78:	cmp    $0x11,%rcx
   e5c7c:	jb     e5cb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb88>
   e5c7e:	mov    0x10(%rbx),%rax
   e5c82:	mov    %rax,%rdx
   e5c85:	sub    %rdi,%rdx
   e5c88:	jae    e5cc4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb94>
   e5c8a:	jmp    e6c26 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1af6>
   e5c8f:	nop
   e5c90:	dec    %rax
   e5c93:	bsr    %rax,%rcx
   e5c97:	not    %ecx
   e5c99:	mov    $0xffffffffffffffff,%rax
   e5ca0:	shr    %cl,%rax
   e5ca3:	lea    0x10(,%rax,8),%rdi
   e5cab:	mov    0x18(%rbx),%rcx
   e5caf:	mov    %rcx,%rax
   e5cb2:	cmp    $0x11,%rcx
   e5cb6:	jae    e5c7e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb4e>
   e5cb8:	mov    %rax,%rdx
   e5cbb:	sub    %rdi,%rdx
   e5cbe:	jb     e6c26 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1af6>
   e5cc4:	cmp    $0x11,%rcx
   e5cc8:	jb     e5cd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xba0>
   e5cca:	mov    0x8(%rbx),%rsi
   e5cce:	jmp    e5cd7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xba7>
   e5cd0:	add    $0x8,%rbx
   e5cd4:	mov    %rbx,%rsi
   e5cd7:	add    %rdi,%rsi
   e5cda:	jmp    e5c17 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xae7>
   e5cdf:	nop
   e5ce0:	cmp    %rcx,%rbx
   e5ce3:	jae    e5cf0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbc0>
   e5ce5:	lea    (%rbx,%rbx,1),%rcx
   e5ce9:	add    0x8(%rax),%rcx
   e5ced:	prefetcht0 (%rcx)
   e5cf0:	mov    $0x1,%r15d
   e5cf6:	lea    (%r15,%r13,1),%rax
   e5cfa:	mov    0x20(%rsp),%r10
   e5cff:	cmp    %r10,%rax
   e5d02:	jbe    e5e6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd3e>
   e5d08:	mov    %r10,%rax
   e5d0b:	sub    %r13,%rax
   e5d0e:	cmp    %rax,%r15
   e5d11:	ja     e5dff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xccf>
   e5d17:	mov    %r10,%r12
   e5d1a:	mov    %r14,%rcx
   e5d1d:	cmp    %rax,%r14
   e5d20:	jbe    e5e60 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd30>
   e5d26:	mov    0x20(%rsp),%rdi
   e5d2b:	mov    %rdi,%rdx
   e5d2e:	sub    %rcx,%rdx
   e5d31:	mov    %r13,%rax
   e5d34:	sub    %rdx,%rax
   e5d37:	cmp    %rax,%rdx
   e5d3a:	jbe    e5d68 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc38>
   e5d3c:	mov    %r12,%rcx
   e5d3f:	sub    %rdi,%rcx
   e5d42:	cmp    %rax,%rcx
   e5d45:	jb     e5d68 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc38>
   e5d47:	mov    0x88(%rsp),%rsi
   e5d4f:	lea    (%rsi,%rdi,8),%rdi
   e5d53:	shl    $0x3,%rax
   e5d57:	mov    %rax,%rdx
   e5d5a:	call   *0x21850(%rip)        # 1075b0 <memcpy@GLIBC_2.14>
   e5d60:	mov    %r12,%r10
   e5d63:	jmp    e5e66 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd36>
   e5d68:	mov    %r12,%rbp
   e5d6b:	sub    %rdx,%rbp
   e5d6e:	mov    0x88(%rsp),%rax
   e5d76:	lea    (%rax,%r14,8),%rsi
   e5d7a:	lea    (%rax,%rbp,8),%rdi
   e5d7e:	shl    $0x3,%rdx
   e5d82:	call   *0x21830(%rip)        # 1075b8 <memmove@GLIBC_2.2.5>
   e5d88:	mov    %rbp,0x90(%rsp)
   e5d90:	mov    %r12,%r10
   e5d93:	mov    %rbp,%r14
   e5d96:	jmp    e5e66 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd36>
   e5d9b:	movq   $0x0,0x50(%rbp)
   e5da3:	cmpq   $0x0,0x20(%rbp)
   e5da8:	je     e5dcb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xc9b>
   e5daa:	add    $0x20,%rbp
   e5dae:	mov    %rbp,%rdi
   e5db1:	call   e2f50 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   e5db6:	mov    %rdx,%rbx
   e5db9:	cmp    $0x1,%rax
   e5dbd:	je     e5bc5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa95>
   e5dc3:	movq   $0x0,0x0(%rbp)
   e5dcb:	xor    %r15d,%r15d
   e5dce:	lea    (%r15,%r13,1),%rax
   e5dd2:	mov    0x20(%rsp),%r10
   e5dd7:	cmp    %r10,%rax
   e5dda:	jbe    e5e6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd3e>
   e5de0:	jmp    e5d08 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbd8>
   e5de5:	xor    %r9d,%r9d
   e5de8:	movq   $0x0,0x50(%rbp)
   e5df0:	test   $0x7f,%r9b
   e5df4:	jne    e598a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x85a>
   e5dfa:	jmp    e5b9a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xa6a>
   e5dff:	mov    $0x1,%edx
   e5e04:	mov    $0x8,%ecx
   e5e09:	mov    $0x8,%r8d
   e5e0f:	lea    0x80(%rsp),%rdi
   e5e17:	mov    %r13,%rsi
   e5e1a:	call   cc100 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e5e1f:	mov    0x80(%rsp),%r12
   e5e27:	mov    0x90(%rsp),%r14
   e5e2f:	mov    0x98(%rsp),%r13
   e5e37:	mov    0x20(%rsp),%rax
   e5e3c:	sub    %r13,%rax
   e5e3f:	mov    %r12,0xd8(%rsp)
   e5e47:	mov    %r14,%rcx
   e5e4a:	cmp    %rax,%r14
   e5e4d:	ja     e5d26 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xbf6>
   e5e53:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e5e60:	mov    %r12,%r10
   e5e63:	mov    %rcx,%r14
   e5e66:	mov    0x100(%rsp),%r12
   e5e6e:	lea    (%r14,%r13,1),%rax
   e5e72:	cmp    %r10,%rax
   e5e75:	mov    $0x0,%ecx
   e5e7a:	cmovae %r10,%rcx
   e5e7e:	sub    %rcx,%rax
   e5e81:	mov    %r10,%rcx
   e5e84:	sub    %rax,%rcx
   e5e87:	cmp    %r15,%rcx
   e5e8a:	jae    e5ea0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd70>
   e5e8c:	mov    0x88(%rsp),%rcx
   e5e94:	cmp    %rax,%r10
   e5e97:	jne    e5eae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd7e>
   e5e99:	mov    %rbx,(%rcx)
   e5e9c:	jmp    e5eb2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd82>
   e5e9e:	xchg   %ax,%ax
   e5ea0:	cmp    $0x1,%r15
   e5ea4:	jne    e5ec0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd90>
   e5ea6:	mov    0x88(%rsp),%rcx
   e5eae:	mov    %rbx,(%rcx,%rax,8)
   e5eb2:	mov    $0x1,%ebx
   e5eb7:	jmp    e5ec2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xd92>
   e5eb9:	nopl   0x0(%rax)
   e5ec0:	xor    %ebx,%ebx
   e5ec2:	add    %r13,%rbx
   e5ec5:	mov    %rbx,0x98(%rsp)
   e5ecd:	mov    (%r12),%rax
   e5ed1:	mov    0x10(%r12),%rsi
   e5ed6:	cmp    $0x1,%rax
   e5eda:	jne    e5f00 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xdd0>
   e5edc:	mov    0x18(%rsp),%r9
   e5ee1:	cmp    %rsi,%r9
   e5ee4:	jae    e6e6d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d3d>
   e5eea:	mov    0x8(%r12),%rcx
   e5eef:	mov    (%rcx,%r9,4),%edx
   e5ef3:	jmp    e5f27 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xdf7>
   e5ef5:	data16 cs nopw 0x0(%rax,%rax,1)
   e5f00:	mov    0x18(%rsp),%r9
   e5f05:	cmp    %rsi,%r9
   e5f08:	jae    e6e61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d31>
   e5f0e:	mov    0x8(%r12),%rcx
   e5f13:	movzwl (%rcx,%r9,2),%edx
   e5f18:	cmp    $0xffff,%dx
   e5f1c:	movzwl %dx,%edx
   e5f1f:	mov    $0xffffffff,%edi
   e5f24:	cmove  %edi,%edx
   e5f27:	mov    0x128(%rsp),%r8
   e5f2f:	cmp    %r8,0x130(%rsp)
   e5f37:	sete   %dil
   e5f3b:	test   %r8,%r8
   e5f3e:	sete   %r8b
   e5f42:	cmp    0x168(%rsp),%edx
   e5f49:	setne  %dl
   e5f4c:	or     %dil,%r8b
   e5f4f:	or     %dl,%r8b
   e5f52:	jne    e58ed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7bd>
   e5f58:	mov    0x130(%rsp),%rdx
   e5f60:	lea    (%r9,%rdx,1),%rbp
   e5f64:	cmp    %rsi,%rbp
   e5f67:	jae    e58ed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7bd>
   e5f6d:	mov    0x128(%rsp),%rdx
   e5f75:	lea    (%r9,%rdx,1),%rdi
   e5f79:	test   %al,%al
   e5f7b:	je     e5f8e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe5e>
   e5f7d:	cmp    %rsi,%rdi
   e5f80:	jae    e6e70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e5f86:	mov    (%rcx,%rdi,4),%eax
   e5f89:	jmp    e58e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7b0>
   e5f8e:	cmp    %rsi,%rdi
   e5f91:	jae    e6e64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e5f97:	movzwl (%rcx,%rdi,2),%eax
   e5f9b:	cmp    $0xffff,%ax
   e5f9f:	movzwl %ax,%eax
   e5fa2:	mov    $0xffffffff,%ecx
   e5fa7:	cmove  %ecx,%eax
   e5faa:	jmp    e58e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x7b0>
   e5faf:	mov    %r14,0x20(%rsp)
   e5fb4:	cmpl   $0x1,0x40(%rsp)
   e5fb9:	jne    e5fdf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xeaf>
   e5fbb:	mov    0x50(%rsp),%rdx
   e5fc0:	cmp    %rdx,%r9
   e5fc3:	jae    e5fdf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xeaf>
   e5fc5:	mov    0x48(%rsp),%r13
   e5fca:	lea    0x1c8(%rsp),%r15
   e5fd2:	mov    0x230(%rsp),%r14
   e5fda:	jmp    e60aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e5fdf:	mov    0x78(%rsp),%rax
   e5fe4:	mov    0x68(%rax),%rcx
   e5fe8:	mov    0x70(%rax),%rsi
   e5fec:	cmp    $0x1,%rsi
   e5ff0:	lea    0x1c8(%rsp),%r15
   e5ff8:	mov    0x230(%rsp),%r14
   e6000:	jne    e600f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xedf>
   e6002:	mov    0x8(%rcx),%r13
   e6006:	mov    0x10(%rax),%rdx
   e600a:	jmp    e60aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e600f:	cmpl   $0x1,0x18(%rax)
   e6013:	jne    e6031 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf01>
   e6015:	mov    0x18(%rsp),%rdx
   e601a:	cmp    0x20(%rax),%rdx
   e601e:	jb     e6031 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf01>
   e6020:	mov    0x28(%rax),%rdx
   e6024:	mov    $0x1,%r13d
   e602a:	cmp    %rdx,0x18(%rsp)
   e602f:	jb     e60aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf7a>
   e6031:	test   %rsi,%rsi
   e6034:	je     e6f07 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dd7>
   e603a:	mov    %rsi,%rdi
   e603d:	xor    %edx,%edx
   e603f:	mov    0x18(%rsp),%r11
   e6044:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e6050:	mov    %rdx,%r8
   e6053:	mov    %rdi,%r9
   e6056:	shr    $1,%r9
   e6059:	add    %r9,%rdx
   e605c:	mov    %rdx,%r10
   e605f:	shl    $0x4,%r10
   e6063:	cmp    %r11,(%rcx,%r10,1)
   e6067:	cmova  %r8,%rdx
   e606b:	sub    %r9,%rdi
   e606e:	cmp    $0x1,%rdi
   e6072:	ja     e6050 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xf20>
   e6074:	mov    %rdx,%rdi
   e6077:	shl    $0x4,%rdi
   e607b:	cmp    (%rcx,%rdi,1),%r11
   e607f:	sbb    $0xffffffffffffffff,%rdx
   e6083:	lea    -0x1(%rdx),%rdi
   e6087:	cmp    %rsi,%rdi
   e608a:	jae    e6f10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1de0>
   e6090:	shl    $0x4,%rdi
   e6094:	mov    0x8(%rcx,%rdi,1),%r13
   e6099:	cmp    %rsi,%rdx
   e609c:	jae    e6006 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xed6>
   e60a2:	shl    $0x4,%rdx
   e60a6:	mov    (%rcx,%rdx,1),%rdx
   e60aa:	movq   $0x1,0x40(%rsp)
   e60b3:	mov    %r13,0x48(%rsp)
   e60b8:	mov    %rdx,0x50(%rsp)
   e60bd:	mov    0x18(%rsp),%rax
   e60c2:	lea    -0x1(%rax),%rcx
   e60c6:	mov    0x38(%rsp),%rax
   e60cb:	mov    (%rax),%rdi
   e60ce:	mov    0x10(%rax),%rsi
   e60d2:	cmp    $0x1,%rdi
   e60d6:	jne    e6114 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xfe4>
   e60d8:	cmp    %rsi,%rcx
   e60db:	jae    e6ed8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1da8>
   e60e1:	mov    0x38(%rsp),%rdx
   e60e6:	mov    0x8(%rdx),%rax
   e60ea:	mov    0x18(%rsp),%rcx
   e60ef:	mov    -0x4(%rax,%rcx,4),%ecx
   e60f3:	cmp    $0xffffffff,%ecx
   e60f6:	jne    e6142 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1012>
   e60f8:	cmp    %rsi,%rbp
   e60fb:	jae    e6ee2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db2>
   e6101:	mov    (%rax,%rbp,4),%r9d
   e6105:	cmp    $0xffffffff,%r9d
   e6109:	jne    e6257 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1127>
   e610f:	jmp    e6523 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e6114:	cmp    %rsi,%rcx
   e6117:	jae    e6ed3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1da3>
   e611d:	mov    0x38(%rsp),%rax
   e6122:	mov    0x8(%rax),%rax
   e6126:	mov    0x18(%rsp),%rcx
   e612b:	movzwl -0x2(%rax,%rcx,2),%ecx
   e6130:	cmp    $0xffff,%cx
   e6134:	je     e623c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x110c>
   e613a:	movzwl %cx,%ecx
   e613d:	mov    0x38(%rsp),%rdx
   e6142:	mov    %ecx,%ecx
   e6144:	mov    0x50(%rdx),%rax
   e6148:	cmp    %rcx,%rax
   e614b:	jbe    e6eba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d8a>
   e6151:	mov    0x48(%rdx),%rax
   e6155:	mov    %rcx,%r12
   e6158:	mov    (%rax,%rcx,8),%r9
   e615c:	mov    0x18(%rsp),%r8
   e6161:	sub    %r9,%r8
   e6164:	jne    e61dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x10ad>
   e6166:	mov    %rbp,%rax
   e6169:	sub    0x18(%rsp),%rax
   e616e:	mov    $0x0,%ecx
   e6173:	cmovae %rax,%rcx
   e6177:	add    %rcx,%r9
   e617a:	xor    %eax,%eax
   e617c:	cmp    0x120(%rsp),%r9
   e6184:	setb   %al
   e6187:	mov    %eax,(%rsp)
   e618a:	lea    0x350(%rsp),%rdi
   e6192:	xor    %esi,%esi
   e6194:	mov    %r12d,%edx
   e6197:	mov    %r12d,%ecx
   e619a:	mov    %r13,%r9
   e619d:	call   f4970 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e61a2:	test   %rax,%rax
   e61a5:	jne    e6776 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1646>
   e61ab:	mov    0x38(%rsp),%rax
   e61b0:	mov    (%rax),%rdi
   e61b3:	mov    0x10(%rax),%rsi
   e61b7:	test   %rdi,%rdi
   e61ba:	je     e621b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x10eb>
   e61bc:	cmp    %rsi,%rbp
   e61bf:	jae    e6ee2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db2>
   e61c5:	mov    0x38(%rsp),%rax
   e61ca:	mov    0x8(%rax),%rax
   e61ce:	mov    (%rax,%rbp,4),%r9d
   e61d2:	cmp    $0xffffffff,%r9d
   e61d6:	jne    e6257 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1127>
   e61d8:	jmp    e6523 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e61dd:	lea    -0x1(%r8),%rax
   e61e1:	cmpb   $0x0,0xf(%rsp)
   e61e6:	je     e63bd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x128d>
   e61ec:	cmp    0xd0(%rsp),%rax
   e61f4:	jae    e6ef7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dc7>
   e61fa:	mov    0x108(%rsp),%rax
   e6202:	mov    -0x4(%rax,%r8,4),%r11d
   e6207:	cmpq   $0x0,0x228(%rsp)
   e6210:	jne    e63f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x12c9>
   e6216:	jmp    e6166 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e621b:	cmp    %rsi,%rbp
   e621e:	jae    e6edd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dad>
   e6224:	mov    0x38(%rsp),%rax
   e6229:	mov    0x8(%rax),%rax
   e622d:	movzwl (%rax,%rbp,2),%eax
   e6231:	cmp    $0xffff,%ax
   e6235:	jne    e6253 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1123>
   e6237:	jmp    e6523 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e623c:	cmp    %rsi,%rbp
   e623f:	jae    e6edd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dad>
   e6245:	movzwl (%rax,%rbp,2),%eax
   e6249:	cmp    $0xffff,%ax
   e624d:	je     e6523 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x13f3>
   e6253:	movzwl %ax,%r9d
   e6257:	mov    %r9d,%edi
   e625a:	cmp    %rdi,0x220(%rsp)
   e6262:	jbe    e6ec2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d92>
   e6268:	mov    0x250(%rsp),%rax
   e6270:	mov    (%rax,%rdi,8),%rax
   e6274:	add    %rbp,%rax
   e6277:	cmpb   $0x0,0xf(%rsp)
   e627c:	je     e636f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x123f>
   e6282:	cmp    0xd0(%rsp),%rax
   e628a:	jae    e6ef7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1dc7>
   e6290:	mov    0x108(%rsp),%rcx
   e6298:	mov    (%rcx,%rax,4),%esi
   e629b:	cmpq   $0x0,0x228(%rsp)
   e62a4:	je     e63a7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1277>
   e62aa:	mov    0x218(%rsp),%rax
   e62b2:	mov    %rdi,%r8
   e62b5:	xor    %rdi,%rax
   e62b8:	movabs $0x5851f42d4c957f2d,%rdi
   e62c2:	mul    %rdi
   e62c5:	xor    %rdx,%rax
   e62c8:	mov    %esi,%ecx
   e62ca:	xor    %rcx,%rax
   e62cd:	mul    %rdi
   e62d0:	mov    %rax,%rcx
   e62d3:	xor    %rdx,%rcx
   e62d6:	mov    %rcx,%rax
   e62d9:	mulq   0x210(%rsp)
   e62e1:	xor    %rax,%rdx
   e62e4:	rol    %cl,%rdx
   e62e7:	mov    %rdx,%rax
   e62ea:	shr    $0x39,%rax
   e62ee:	movd   %eax,%xmm0
   e62f2:	punpcklbw %xmm0,%xmm0
   e62f6:	pshuflw $0x0,%xmm0,%xmm0
   e62fb:	pshufd $0x44,%xmm0,%xmm0
   e6300:	xor    %eax,%eax
   e6302:	and    0x110(%rsp),%rdx
   e630a:	movdqu (%r14,%rdx,1),%xmm1
   e6310:	movdqa %xmm1,%xmm2
   e6314:	pcmpeqb %xmm0,%xmm2
   e6318:	pmovmskb %xmm2,%ecx
   e631c:	test   %ecx,%ecx
   e631e:	je     e634e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x121e>
   e6320:	tzcnt  %ecx,%edi
   e6324:	add    %rdx,%rdi
   e6327:	and    0x110(%rsp),%rdi
   e632f:	neg    %rdi
   e6332:	lea    (%rdi,%rdi,2),%rdi
   e6336:	cmp    -0xc(%r14,%rdi,4),%r9d
   e633b:	jne    e6344 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1214>
   e633d:	cmp    -0x8(%r14,%rdi,4),%esi
   e6342:	je     e63af <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x127f>
   e6344:	lea    -0x1(%rcx),%edi
   e6347:	and    %cx,%di
   e634a:	mov    %edi,%ecx
   e634c:	jne    e6320 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x11f0>
   e634e:	pcmpeqd %xmm2,%xmm2
   e6352:	pcmpeqb %xmm2,%xmm1
   e6356:	pmovmskb %xmm1,%ecx
   e635a:	test   %ecx,%ecx
   e635c:	jne    e64c1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1391>
   e6362:	add    %rax,%rdx
   e6365:	add    $0x10,%rdx
   e6369:	add    $0x10,%rax
   e636d:	jmp    e6302 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x11d2>
   e636f:	cmp    0xd0(%rsp),%rax
   e6377:	jae    e6ee7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db7>
   e637d:	mov    0x108(%rsp),%rcx
   e6385:	movzwl (%rcx,%rax,2),%eax
   e6389:	cmp    $0xffff,%ax
   e638d:	movzwl %ax,%esi
   e6390:	mov    $0xffffffff,%eax
   e6395:	cmove  %eax,%esi
   e6398:	cmpq   $0x0,0x228(%rsp)
   e63a1:	jne    e62aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x117a>
   e63a7:	mov    %r9d,%ecx
   e63aa:	jmp    e64c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1397>
   e63af:	lea    (%r14,%rdi,4),%rax
   e63b3:	mov    -0x4(%rax),%edi
   e63b6:	mov    %edi,%ecx
   e63b8:	jmp    e64c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1397>
   e63bd:	cmp    0xd0(%rsp),%rax
   e63c5:	jae    e6ee7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1db7>
   e63cb:	mov    0x108(%rsp),%rax
   e63d3:	movzwl -0x2(%rax,%r8,2),%eax
   e63d9:	cmp    $0xffff,%ax
   e63dd:	movzwl %ax,%r11d
   e63e1:	mov    $0xffffffff,%eax
   e63e6:	cmove  %eax,%r11d
   e63ea:	cmpq   $0x0,0x228(%rsp)
   e63f3:	je     e6166 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e63f9:	mov    %r11d,%eax
   e63fc:	xor    0x218(%rsp),%rax
   e6404:	movabs $0x5851f42d4c957f2d,%rcx
   e640e:	mul    %rcx
   e6411:	xor    %r12,%rdx
   e6414:	xor    %rdx,%rax
   e6417:	mul    %rcx
   e641a:	mov    %rax,%rcx
   e641d:	xor    %rdx,%rcx
   e6420:	mov    %rcx,%rax
   e6423:	mulq   0x210(%rsp)
   e642b:	xor    %rax,%rdx
   e642e:	rol    %cl,%rdx
   e6431:	mov    %rdx,%rax
   e6434:	shr    $0x39,%rax
   e6438:	movd   %eax,%xmm0
   e643c:	punpcklbw %xmm0,%xmm0
   e6440:	pshuflw $0x0,%xmm0,%xmm0
   e6445:	pshufd $0x44,%xmm0,%xmm0
   e644a:	xor    %eax,%eax
   e644c:	and    0x110(%rsp),%rdx
   e6454:	movdqu (%r14,%rdx,1),%xmm1
   e645a:	movdqa %xmm1,%xmm2
   e645e:	pcmpeqb %xmm0,%xmm2
   e6462:	pmovmskb %xmm2,%ecx
   e6466:	test   %ecx,%ecx
   e6468:	je     e64a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1370>
   e646a:	tzcnt  %ecx,%r10d
   e646f:	add    %rdx,%r10
   e6472:	and    0x110(%rsp),%r10
   e647a:	neg    %r10
   e647d:	lea    (%r10,%r10,2),%r10
   e6481:	cmp    -0xc(%r14,%r10,4),%r11d
   e6486:	jne    e6493 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1363>
   e6488:	cmp    -0x8(%r14,%r10,4),%r12d
   e648d:	je     e61b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1087>
   e6493:	lea    -0x1(%rcx),%r10d
   e6497:	and    %cx,%r10w
   e649b:	mov    %r10d,%ecx
   e649e:	jne    e646a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x133a>
   e64a0:	pcmpeqd %xmm2,%xmm2
   e64a4:	pcmpeqb %xmm2,%xmm1
   e64a8:	pmovmskb %xmm1,%ecx
   e64ac:	test   %ecx,%ecx
   e64ae:	jne    e6166 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1036>
   e64b4:	add    %rax,%rdx
   e64b7:	add    $0x10,%rdx
   e64bb:	add    $0x10,%rax
   e64bf:	jmp    e644c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x131c>
   e64c1:	mov    %r9d,%ecx
   e64c4:	mov    %r8,%rdi
   e64c7:	mov    0x38(%rsp),%rdx
   e64cc:	mov    0x50(%rdx),%rsi
   e64d0:	cmp    %rdi,%rsi
   e64d3:	jbe    e6eca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d9a>
   e64d9:	mov    0x18(%rsp),%r8
   e64de:	sub    %r8,%rbp
   e64e1:	mov    $0x0,%eax
   e64e6:	cmovae %rbp,%rax
   e64ea:	mov    0x48(%rdx),%rdx
   e64ee:	add    (%rdx,%rdi,8),%rax
   e64f2:	xor    %edx,%edx
   e64f4:	cmp    0x120(%rsp),%rax
   e64fc:	setb   %dl
   e64ff:	mov    %edx,(%rsp)
   e6502:	lea    0x350(%rsp),%rdi
   e650a:	mov    $0x1,%esi
   e650f:	mov    %r9d,%edx
   e6512:	mov    %r13,%r9
   e6515:	call   f4970 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::record>
   e651a:	test   %rax,%rax
   e651d:	jne    e6776 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1646>
   e6523:	mov    0xf0(%rsp),%rax
   e652b:	cmp    $0x3,%rax
   e652f:	jb     e656d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x143d>
   e6531:	mov    0xe8(%rsp),%rcx
   e6539:	cmp    %rax,%rcx
   e653c:	mov    0x20(%rsp),%r14
   e6541:	mov    0x18(%rsp),%rdx
   e6546:	je     e659f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x146f>
   e6548:	mov    0xe0(%rsp),%rax
   e6550:	mov    %rdx,(%rax,%rcx,8)
   e6554:	inc    %rcx
   e6557:	mov    %rcx,0xe8(%rsp)
   e655f:	test   %rbx,%rbx
   e6562:	jne    e5897 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e6568:	jmp    e6600 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e656d:	lea    0xe0(%rsp),%rcx
   e6575:	lea    0xf0(%rsp),%rdx
   e657d:	cmp    $0x2,%rax
   e6581:	mov    0x20(%rsp),%r14
   e6586:	mov    0x18(%rsp),%rsi
   e658b:	je     e65c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1490>
   e658d:	mov    %rsi,(%rcx,%rax,8)
   e6591:	incq   (%rdx)
   e6594:	test   %rbx,%rbx
   e6597:	jne    e5897 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e659d:	jmp    e6600 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14d0>
   e659f:	lea    0xe0(%rsp),%rdi
   e65a7:	call   f4e20 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e65ac:	mov    0xe8(%rsp),%rcx
   e65b4:	mov    0x20(%rsp),%r14
   e65b9:	mov    0x18(%rsp),%rdx
   e65be:	jmp    e6548 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1418>
   e65c0:	lea    0xe0(%rsp),%rdi
   e65c8:	call   f4e20 <<smallvec::SmallVec<[usize; 2]>>::reserve_one_unchecked>
   e65cd:	mov    0xe0(%rsp),%rcx
   e65d5:	mov    0xe8(%rsp),%rax
   e65dd:	lea    0xe8(%rsp),%rdx
   e65e5:	mov    0x20(%rsp),%r14
   e65ea:	mov    0x18(%rsp),%rsi
   e65ef:	mov    %rsi,(%rcx,%rax,8)
   e65f3:	incq   (%rdx)
   e65f6:	test   %rbx,%rbx
   e65f9:	jne    e5897 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x767>
   e65ff:	nop
   e6600:	cmpq   $0x0,0xd8(%rsp)
   e6609:	je     e6619 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x14e9>
   e660b:	mov    0x88(%rsp),%rdi
   e6613:	call   *0x20f8f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6619:	mov    0xe0(%rsp),%r14
   e6621:	movups 0xe8(%rsp),%xmm0
   e6629:	movaps %xmm0,0x20(%rsp)
   e662e:	lea    0x40(%rsp),%rbp
   e6633:	mov    %rbp,%rdi
   e6636:	lea    0x350(%rsp),%rsi
   e663e:	lea    0x1c8(%rsp),%r15
   e6646:	call   f5040 <<tk_train::trainers::bpe::corpus::merging::NeighborChanges>::finish>
   e664b:	mov    0x40(%rsp),%rax
   e6650:	cmp    $0xfffffffffffffffe,%rax
   e6654:	je     e6805 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e665a:	movdqu 0x48(%rsp),%xmm0
   e6660:	movdqa 0x20(%rsp),%xmm1
   e6666:	mov    %rax,0x40(%rsp)
   e666b:	movdqu %xmm0,0x48(%rsp)
   e6671:	mov    %r14,0x58(%rsp)
   e6676:	movdqu %xmm1,0x60(%rsp)
   e667c:	mov    0x174(%rsp),%eax
   e6683:	mov    %eax,0x70(%rsp)
   e6687:	mov    0x238(%rsp),%rax
   e668f:	mov    (%rax),%rsi
   e6692:	lea    0x398(%rsp),%rdi
   e669a:	mov    %rbp,%rdx
   e669d:	call   f54a0 <<&<core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0} as core::ops::function::FnMut<(core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>,)>>::call_mut>
   e66a2:	mov    0x398(%rsp),%r15
   e66aa:	cmp    $0xfffffffffffffffe,%r15
   e66ae:	je     e6805 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16d5>
   e66b4:	cmp    $0xffffffffffffffff,%r15
   e66b8:	je     e6823 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16f3>
   e66be:	mov    0x3a0(%rsp),%rbx
   e66c6:	mov    0x3a8(%rsp),%r12
   e66ce:	mov    0x3b0(%rsp),%rdi
   e66d6:	mov    0x3c0(%rsp),%rcx
   e66de:	movdqu 0x3b8(%rsp),%xmm0
   e66e7:	mov    0x3c8(%rsp),%rax
   e66ef:	mov    0x118(%rsp),%rdx
   e66f7:	movzbl (%rdx),%edx
   e66fa:	test   %dl,%dl
   e66fc:	jne    e6834 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1704>
   e6702:	mov    %rbx,0x2c0(%rsp)
   e670a:	mov    %r12,0x2c8(%rsp)
   e6712:	mov    %rdi,0x2d0(%rsp)
   e671a:	movdqu %xmm0,0x2d8(%rsp)
   e6723:	mov    %rax,0x2e8(%rsp)
   e672b:	mov    %r15,0x2b8(%rsp)
   e6733:	mov    0x248(%rsp),%rbx
   e673b:	cmp    0x188(%rsp),%rbx
   e6743:	mov    0xf8(%rsp),%r12
   e674b:	jne    e5340 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x210>
   e6751:	mov    $0x1,%edx
   e6756:	mov    $0x8,%ecx
   e675b:	mov    $0x38,%r8d
   e6761:	lea    0x188(%rsp),%rdi
   e6769:	mov    %rbx,%rsi
   e676c:	call   cc100 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e6771:	jmp    e5330 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x200>
   e6776:	mov    %rax,%rbx
   e6779:	cmpq   $0x0,0xd8(%rsp)
   e6782:	lea    0x40(%rsp),%rbp
   e6787:	je     e6797 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1667>
   e6789:	mov    0x88(%rsp),%rdi
   e6791:	call   *0x20e11(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6797:	cmpq   $0x3,0xf0(%rsp)
   e67a0:	jb     e67b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1680>
   e67a2:	mov    0xe0(%rsp),%rdi
   e67aa:	call   *0x20df8(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e67b0:	lea    0x350(%rsp),%rdi
   e67b8:	call   e4eb0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e67bd:	lea    0x368(%rsp),%rdi
   e67c5:	call   e4eb0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e67ca:	movq   %rbx,%xmm0
   e67cf:	punpcklqdq 0x3d0(%rsp),%xmm0
   e67d8:	mov    $0xffffffffffffffff,%rax
   e67df:	jmp    e6666 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1536>
   e67e4:	test   %dl,%dl
   e67e6:	je     e6812 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x16e2>
   e67e8:	mov    %fs:0xffffffffffffffd0,%rax
   e67f1:	test   %rax,%rax
   e67f4:	je     e69be <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x188e>
   e67fa:	add    $0x110,%rax
   e6800:	jmp    e69db <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x18ab>
   e6805:	mov    0x118(%rsp),%r14
   e680d:	jmp    e68e1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e6812:	test   %rcx,%rcx
   e6815:	je     e5217 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xe7>
   e681b:	shr    $1,%rcx
   e681e:	jmp    e69ef <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x18bf>
   e6823:	mov    0x118(%rsp),%r14
   e682b:	movb   $0x1,(%r14)
   e682f:	jmp    e68e1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e6834:	cmp    $0x3,%rcx
   e6838:	jb     e6840 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1710>
   e683a:	call   *0x20d68(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6840:	test   %r12,%r12
   e6843:	je     e68cb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x179b>
   e6849:	xor    %r13d,%r13d
   e684c:	jmp    e6861 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1731>
   e684e:	mov    0x20(%rsp),%rdi
   e6853:	call   *0x20d4f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6859:	inc    %r13
   e685c:	cmp    %r12,%r13
   e685f:	je     e68cb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x179b>
   e6861:	lea    0x0(,%r13,8),%rcx
   e6869:	add    %r13,%rcx
   e686c:	lea    (%rbx,%rcx,8),%rax
   e6870:	cmpq   $0x0,(%rbx,%rcx,8)
   e6875:	je     e6890 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1760>
   e6877:	cmpq   $0x11,0x18(%rax)
   e687c:	jb     e6859 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1729>
   e687e:	mov    0x8(%rax),%rdi
   e6882:	jmp    e6853 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1723>
   e6884:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e6890:	mov    0x10(%rax),%rbp
   e6894:	test   %rbp,%rbp
   e6897:	je     e6859 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1729>
   e6899:	mov    0x8(%rax),%rax
   e689d:	mov    %rax,0x20(%rsp)
   e68a2:	lea    0x18(%rax),%r14
   e68a6:	jmp    e68b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1789>
   e68a8:	nopl   0x0(%rax,%rax,1)
   e68b0:	add    $0x20,%r14
   e68b4:	dec    %rbp
   e68b7:	je     e684e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x171e>
   e68b9:	cmpq   $0x11,(%r14)
   e68bd:	jb     e68b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1780>
   e68bf:	mov    -0x10(%r14),%rdi
   e68c3:	call   *0x20cdf(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e68c9:	jmp    e68b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1780>
   e68cb:	test   %r15,%r15
   e68ce:	mov    0x118(%rsp),%r14
   e68d6:	je     e68e1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x17b1>
   e68d8:	mov    %rbx,%rdi
   e68db:	call   *0x20cc7(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e68e1:	mov    0x198(%rsp),%rax
   e68e9:	mov    %rax,0x1b0(%rsp)
   e68f1:	movups 0x188(%rsp),%xmm0
   e68f9:	mov    %r14,0x1b8(%rsp)
   e6901:	mov    0x238(%rsp),%rax
   e6909:	mov    %rax,0x1c0(%rsp)
   e6911:	movaps %xmm0,0x40(%rsp)
   e6916:	movaps 0x1d0(%rsp),%xmm0
   e691e:	movaps 0x1e0(%rsp),%xmm1
   e6926:	movaps 0x1f0(%rsp),%xmm2
   e692e:	movaps %xmm0,0x70(%rsp)
   e6933:	movaps %xmm1,0x80(%rsp)
   e693b:	movaps %xmm2,0x90(%rsp)
   e6943:	mov    0x1b0(%rsp),%rax
   e694b:	mov    %rax,0x50(%rsp)
   e6950:	mov    0x1b8(%rsp),%rax
   e6958:	mov    %rax,0x58(%rsp)
   e695d:	mov    0x1c0(%rsp),%rax
   e6965:	mov    %rax,0x60(%rsp)
   e696a:	mov    0x1c8(%rsp),%rax
   e6972:	mov    %rax,0x68(%rsp)
   e6977:	lea    0x40(%rsp),%rsi
   e697c:	mov    0x208(%rsp),%rdi
   e6984:	call   f3440 <<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}> as rayon::iter::plumbing::Folder<core::result::Result<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>::complete>
   e6989:	cmpq   $0x0,0x68(%rsp)
   e698f:	je     e699c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x186c>
   e6991:	mov    0x70(%rsp),%rdi
   e6996:	call   *0x20c0c(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e699c:	cmpq   $0x0,0x80(%rsp)
   e69a5:	je     e51e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e69ab:	mov    0x88(%rsp),%rdi
   e69b3:	call   *0x20bef(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e69b9:	jmp    e51e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e69be:	mov    %rbp,%r13
   e69c1:	mov    %rsi,%r15
   e69c4:	mov    %r12,%rbp
   e69c7:	mov    %rcx,%r12
   e69ca:	call   7ff10 <rayon_core::registry::global_registry>
   e69cf:	mov    %r12,%rcx
   e69d2:	mov    %rbp,%r12
   e69d5:	mov    %r15,%rsi
   e69d8:	mov    %r13,%rbp
   e69db:	mov    (%rax),%rax
   e69de:	mov    0x208(%rax),%rax
   e69e5:	shr    $1,%rcx
   e69e8:	cmp    %rax,%rcx
   e69eb:	cmovbe %rax,%rcx
   e69ef:	mov    %rcx,0x178(%rsp)
   e69f7:	mov    %rsi,0x1a0(%rsp)
   e69ff:	sub    %rsi,%rbx
   e6a02:	jb     e6cbd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b8d>
   e6a08:	imul   $0x70,%rsi,%r13
   e6a0c:	add    %r12,%r13
   e6a0f:	mov    %r12,%rdi
   e6a12:	mov    0x8(%r14),%r12
   e6a16:	mov    0x10(%r14),%r15
   e6a1a:	mov    0x18(%r14),%r14
   e6a1e:	mov    %fs:0xffffffffffffffd0,%rdx
   e6a27:	test   %rdx,%rdx
   e6a2a:	je     e6a99 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1969>
   e6a2c:	lea    0x138(%rsp),%rax
   e6a34:	mov    %rax,0x40(%rsp)
   e6a39:	lea    0x1a0(%rsp),%rax
   e6a41:	mov    %rax,0x48(%rsp)
   e6a46:	lea    0x178(%rsp),%rcx
   e6a4e:	mov    %rcx,0x50(%rsp)
   e6a53:	mov    %r13,0x58(%rsp)
   e6a58:	mov    %rbx,0x60(%rsp)
   e6a5d:	mov    %rbp,0x68(%rsp)
   e6a62:	mov    %r12,0x70(%rsp)
   e6a67:	mov    %r15,0x78(%rsp)
   e6a6c:	mov    %r14,0x80(%rsp)
   e6a74:	mov    %rax,0x88(%rsp)
   e6a7c:	mov    %rcx,0x90(%rsp)
   e6a84:	mov    %rdi,0x98(%rsp)
   e6a8c:	mov    %rsi,0xa0(%rsp)
   e6a94:	jmp    e6b42 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a12>
   e6a99:	mov    %rdi,0xf8(%rsp)
   e6aa1:	mov    %rsi,0x20(%rsp)
   e6aa6:	call   7ff10 <rayon_core::registry::global_registry>
   e6aab:	mov    (%rax),%rsi
   e6aae:	mov    %fs:0xffffffffffffffd0,%rdx
   e6ab7:	test   %rdx,%rdx
   e6aba:	je     e6d0e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1bde>
   e6ac0:	cmp    %rsi,0x110(%rdx)
   e6ac7:	mov    0x20(%rsp),%rax
   e6acc:	lea    0x138(%rsp),%rcx
   e6ad4:	jne    e6dbe <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1c8e>
   e6ada:	mov    %rcx,0x40(%rsp)
   e6adf:	lea    0x1a0(%rsp),%rcx
   e6ae7:	mov    %rcx,0x48(%rsp)
   e6aec:	lea    0x178(%rsp),%rsi
   e6af4:	mov    %rsi,0x50(%rsp)
   e6af9:	mov    %r13,0x58(%rsp)
   e6afe:	mov    %rbx,0x60(%rsp)
   e6b03:	mov    %rbp,0x68(%rsp)
   e6b08:	mov    %r12,0x70(%rsp)
   e6b0d:	mov    %r15,0x78(%rsp)
   e6b12:	mov    %r14,0x80(%rsp)
   e6b1a:	mov    %rcx,0x88(%rsp)
   e6b22:	mov    %rsi,0x90(%rsp)
   e6b2a:	mov    0xf8(%rsp),%rcx
   e6b32:	mov    %rcx,0x98(%rsp)
   e6b3a:	mov    %rax,0xa0(%rsp)
   e6b42:	mov    %rbp,0xa8(%rsp)
   e6b4a:	mov    %r12,0xb0(%rsp)
   e6b52:	mov    %r15,0xb8(%rsp)
   e6b5a:	mov    %r14,0xc0(%rsp)
   e6b62:	lea    0x2f0(%rsp),%rdi
   e6b6a:	lea    0x40(%rsp),%rsi
   e6b6f:	xor    %ecx,%ecx
   e6b71:	call   f5d60 <rayon_core::join::join_context::<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}>
   e6b76:	mov    0x2f0(%rsp),%rdx
   e6b7e:	mov    0x2f8(%rsp),%rcx
   e6b86:	mov    0x300(%rsp),%rax
   e6b8e:	mov    0x318(%rsp),%rsi
   e6b96:	mov    %rsi,0x50(%rsp)
   e6b9b:	movups 0x308(%rsp),%xmm0
   e6ba3:	movaps %xmm0,0x40(%rsp)
   e6ba8:	test   %rcx,%rcx
   e6bab:	mov    0x208(%rsp),%r8
   e6bb3:	je     e6be4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ab4>
   e6bb5:	mov    0x40(%rsp),%rsi
   e6bba:	movq   $0x0,0x40(%rsp)
   e6bc3:	test   %rsi,%rsi
   e6bc6:	je     e6c0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1adc>
   e6bc8:	mov    %rsi,0x18(%rcx)
   e6bcc:	mov    %rcx,0x20(%rsi)
   e6bd0:	mov    0x48(%rsp),%rcx
   e6bd5:	xorps  %xmm0,%xmm0
   e6bd8:	add    0x50(%rsp),%rax
   e6bdd:	movups %xmm0,0x48(%rsp)
   e6be2:	jmp    e6c0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1adc>
   e6be4:	mov    0x40(%rsp),%rsi
   e6be9:	mov    0x48(%rsp),%rcx
   e6bee:	mov    %rdx,0x40(%rsp)
   e6bf3:	movq   $0x0,0x48(%rsp)
   e6bfc:	mov    0x50(%rsp),%rdi
   e6c01:	mov    %rax,0x50(%rsp)
   e6c06:	mov    %rsi,%rdx
   e6c09:	mov    %rdi,%rax
   e6c0c:	mov    %rdx,(%r8)
   e6c0f:	mov    %rcx,0x8(%r8)
   e6c13:	mov    %rax,0x10(%r8)
   e6c17:	lea    0x40(%rsp),%rdi
   e6c1c:	call   e4cd0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e6c21:	jmp    e51e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0xb6>
   e6c26:	movq   $0x0,0x50(%rbp)
   e6c2e:	mov    %rax,%rdx
   e6c31:	lea    0x20010(%rip),%rcx        # 106c48 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   e6c38:	mov    %rax,%rsi
   e6c3b:	call   714e0 <core::slice::index::slice_index_fail>
   e6c40:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6c45:	lea    -0xc8b28(%rip),%rdi        # 1e124 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5a7>
   e6c4c:	lea    0x20385(%rip),%rdx        # 106fd8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   e6c53:	mov    $0x53,%esi
   e6c58:	call   6f610 <core::panicking::panic_fmt>
   e6c5d:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6c62:	mov    %rdx,%rax
   e6c65:	lea    0x2060c(%rip),%rcx        # 107278 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4080>
   e6c6c:	jmp    e6c38 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e6c6e:	lea    -0xc8f7d(%rip),%rdi        # 1dcf8 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x17b>
   e6c75:	lea    0x20134(%rip),%rdx        # 106db0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3bb8>
   e6c7c:	mov    $0x11,%esi
   e6c81:	call   726c0 <core::option::expect_failed>
   e6c86:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6c8b:	mov    $0x8,%eax
   e6c90:	xor    %edi,%edi
   e6c92:	lea    0x20447(%rip),%rcx        # 1070e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   e6c99:	jmp    e6c38 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e6c9b:	mov    %rsi,%rdi
   e6c9e:	jmp    e6c2e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1afe>
   e6ca0:	mov    %rdx,%rax
   e6ca3:	lea    0x1ffe6(%rip),%rcx        # 106c90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   e6caa:	jmp    e6c38 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e6cac:	mov    $0x8,%edi
   e6cb1:	lea    0x1fff0(%rip),%rcx        # 106ca8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ab0>
   e6cb8:	jmp    e6c38 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1b08>
   e6cbd:	lea    -0xc8e57(%rip),%rdi        # 1de6d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f0>
   e6cc4:	lea    0x20535(%rip),%rdx        # 107200 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4008>
   e6ccb:	mov    $0x13,%esi
   e6cd0:	call   6f610 <core::panicking::panic_fmt>
   e6cd5:	xor    %eax,%eax
   e6cd7:	jmp    e6cdb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1bab>
   e6cd9:	mov    $0x1,%al
   e6cdb:	mov    %al,0x280(%rsp)
   e6ce2:	lea    -0xc8939(%rip),%rdi        # 1e3b0 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x833>
   e6ce9:	lea    0x1fc80(%rip),%rcx        # 106970 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3778>
   e6cf0:	lea    0x20569(%rip),%r8        # 107260 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4068>
   e6cf7:	lea    0x280(%rsp),%rdx
   e6cff:	mov    $0x13,%esi
   e6d04:	call   72700 <core::result::unwrap_failed>
   e6d09:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6d0e:	lea    0x138(%rsp),%rax
   e6d16:	mov    %rax,0x40(%rsp)
   e6d1b:	lea    0x1a0(%rsp),%rax
   e6d23:	mov    %rax,0x48(%rsp)
   e6d28:	lea    0x178(%rsp),%rcx
   e6d30:	mov    %rcx,0x50(%rsp)
   e6d35:	mov    %r13,0x58(%rsp)
   e6d3a:	mov    %rbx,0x60(%rsp)
   e6d3f:	mov    %rbp,0x68(%rsp)
   e6d44:	mov    %r12,0x70(%rsp)
   e6d49:	mov    %r15,0x78(%rsp)
   e6d4e:	mov    %r14,0x80(%rsp)
   e6d56:	mov    %rax,0x88(%rsp)
   e6d5e:	mov    %rcx,0x90(%rsp)
   e6d66:	mov    0xf8(%rsp),%rax
   e6d6e:	mov    %rax,0x98(%rsp)
   e6d76:	mov    0x20(%rsp),%rax
   e6d7b:	mov    %rax,0xa0(%rsp)
   e6d83:	mov    %rbp,0xa8(%rsp)
   e6d8b:	mov    %r12,0xb0(%rsp)
   e6d93:	mov    %r15,0xb8(%rsp)
   e6d9b:	mov    %r14,0xc0(%rsp)
   e6da3:	sub    $0xffffffffffffff80,%rsi
   e6da7:	lea    0x2f0(%rsp),%rdi
   e6daf:	lea    0x40(%rsp),%rdx
   e6db4:	call   f6220 <<rayon_core::registry::Registry>::in_worker_cold::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e6db9:	jmp    e6b76 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a46>
   e6dbe:	mov    %rcx,0x40(%rsp)
   e6dc3:	lea    0x1a0(%rsp),%rcx
   e6dcb:	mov    %rcx,0x48(%rsp)
   e6dd0:	lea    0x178(%rsp),%rdi
   e6dd8:	mov    %rdi,0x50(%rsp)
   e6ddd:	mov    %r13,0x58(%rsp)
   e6de2:	mov    %rbx,0x60(%rsp)
   e6de7:	mov    %rbp,0x68(%rsp)
   e6dec:	mov    %r12,0x70(%rsp)
   e6df1:	mov    %r15,0x78(%rsp)
   e6df6:	mov    %r14,0x80(%rsp)
   e6dfe:	mov    %rcx,0x88(%rsp)
   e6e06:	mov    %rdi,0x90(%rsp)
   e6e0e:	mov    0xf8(%rsp),%rcx
   e6e16:	mov    %rcx,0x98(%rsp)
   e6e1e:	mov    %rax,0xa0(%rsp)
   e6e26:	mov    %rbp,0xa8(%rsp)
   e6e2e:	mov    %r12,0xb0(%rsp)
   e6e36:	mov    %r15,0xb8(%rsp)
   e6e3e:	mov    %r14,0xc0(%rsp)
   e6e46:	sub    $0xffffffffffffff80,%rsi
   e6e4a:	lea    0x2f0(%rsp),%rdi
   e6e52:	lea    0x40(%rsp),%rcx
   e6e57:	call   f6320 <<rayon_core::registry::Registry>::in_worker_cross::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>)>>
   e6e5c:	jmp    e6b76 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1a46>
   e6e61:	mov    %r9,%rdi
   e6e64:	lea    0x1fd05(%rip),%rax        # 106b70 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3978>
   e6e6b:	jmp    e6e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e6e6d:	mov    %r9,%rdi
   e6e70:	lea    0x1fd11(%rip),%rax        # 106b88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e6e77:	mov    %rax,0x10(%rsp)
   e6e7c:	mov    0x10(%rsp),%rdx
   e6e81:	call   705ab <core::panicking::panic_bounds_check>
   e6e86:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6e88:	shr    $0x7,%r9
   e6e8c:	xor    %esi,%esi
   e6e8e:	mov    %r9,%rdi
   e6e91:	jmp    e6e7c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d4c>
   e6e93:	mov    %rdx,%rdi
   e6e96:	jmp    e6e7c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d4c>
   e6e98:	mov    %r14,%rdi
   e6e9b:	lea    0x1f606(%rip),%rdx        # 1064a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x32b0>
   e6ea2:	call   705ab <core::panicking::panic_bounds_check>
   e6ea7:	jmp    e6eb8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d88>
   e6ea9:	mov    $0x8,%edi
   e6eae:	mov    $0x80,%esi
   e6eb3:	call   6999b <alloc::raw_vec::handle_error>
   e6eb8:	ud2
   e6eba:	mov    %rcx,%rdi
   e6ebd:	mov    %rax,%rsi
   e6ec0:	jmp    e6eca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d9a>
   e6ec2:	mov    0x220(%rsp),%rsi
   e6eca:	lea    0x1f5d7(%rip),%rax        # 1064a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x32b0>
   e6ed1:	jmp    e6e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e6ed3:	mov    %rcx,%rdi
   e6ed6:	jmp    e6e64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e6ed8:	mov    %rcx,%rdi
   e6edb:	jmp    e6e70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e6edd:	mov    %rbp,%rdi
   e6ee0:	jmp    e6e64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e6ee2:	mov    %rbp,%rdi
   e6ee5:	jmp    e6e70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e6ee7:	mov    %rax,%rdi
   e6eea:	mov    0xd0(%rsp),%rsi
   e6ef2:	jmp    e6e64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d34>
   e6ef7:	mov    %rax,%rdi
   e6efa:	mov    0xd0(%rsp),%rsi
   e6f02:	jmp    e6e70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d40>
   e6f07:	mov    $0xffffffffffffffff,%rdi
   e6f0e:	xor    %esi,%esi
   e6f10:	lea    0x20079(%rip),%rax        # 106f90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d98>
   e6f17:	jmp    e6e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1d47>
   e6f1c:	mov    %rax,%rbx
   e6f1f:	lea    0x68(%rsp),%r15
   e6f24:	mov    %r15,%rdi
   e6f27:	call   f3530 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e6f2c:	mov    %rbx,%rdi
   e6f2f:	call   101640 <_Unwind_Resume@plt>
   e6f34:	jmp    e6fc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e97>
   e6f39:	mov    %rax,%rbx
   e6f3c:	lea    0x2b8(%rsp),%rdi
   e6f44:	call   f32e0 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>
   e6f49:	jmp    e6f7c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e4c>
   e6f4b:	mov    %rax,%rbx
   e6f4e:	mov    %r15,%rdi
   e6f51:	call   f3530 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e6f56:	mov    %rbx,%rdi
   e6f59:	call   101640 <_Unwind_Resume@plt>
   e6f5e:	mov    %rax,%rbx
   e6f61:	pshufd $0xee,0x20(%rsp),%xmm0
   e6f68:	movq   %xmm0,%rax
   e6f6d:	cmp    $0x3,%rax
   e6f71:	jb     e6f7c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e4c>
   e6f73:	mov    %r14,%rdi
   e6f76:	call   *0x2062c(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6f7c:	lea    0x1c8(%rsp),%r15
   e6f84:	jmp    e7064 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f34>
   e6f89:	mov    %rax,%rbx
   e6f8c:	jmp    e7064 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f34>
   e6f91:	mov    %rax,%rbx
   e6f94:	jmp    e6fe3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e6f96:	mov    %rax,%rbx
   e6f99:	mov    0x80(%rsp),%rax
   e6fa1:	mov    %rax,0xd8(%rsp)
   e6fa9:	jmp    e6fca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e9a>
   e6fab:	jmp    e6faf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e7f>
   e6fad:	jmp    e6faf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1e7f>
   e6faf:	mov    %rax,%rbx
   e6fb2:	cmpq   $0x0,0x140(%rsp)
   e6fbb:	je     e6fe3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e6fbd:	mov    0x148(%rsp),%rdi
   e6fc5:	jmp    e6fdd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ead>
   e6fc7:	mov    %rax,%rbx
   e6fca:	cmpq   $0x0,0xd8(%rsp)
   e6fd3:	je     e6fe3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1eb3>
   e6fd5:	mov    0x88(%rsp),%rdi
   e6fdd:	call   *0x205c5(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e6fe3:	mov    %rbx,0x20(%rsp)
   e6fe8:	cmpq   $0x3,0xf0(%rsp)
   e6ff1:	jb     e7001 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ed1>
   e6ff3:	mov    0xe0(%rsp),%rdi
   e6ffb:	call   *0x205a7(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7001:	mov    0x358(%rsp),%r14
   e7009:	mov    0x360(%rsp),%r12
   e7011:	test   %r12,%r12
   e7014:	jne    e7081 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f51>
   e7016:	cmpq   $0x0,0x350(%rsp)
   e701f:	je     e702a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1efa>
   e7021:	mov    %r14,%rdi
   e7024:	call   *0x2057e(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e702a:	mov    0x370(%rsp),%r14
   e7032:	mov    0x378(%rsp),%r12
   e703a:	test   %r12,%r12
   e703d:	jne    e710b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fdb>
   e7043:	cmpq   $0x0,0x368(%rsp)
   e704c:	je     e7057 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f27>
   e704e:	mov    %r14,%rdi
   e7051:	call   *0x20551(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7057:	lea    0x1c8(%rsp),%r15
   e705f:	mov    0x20(%rsp),%rbx
   e7064:	lea    0x188(%rsp),%rdi
   e706c:	call   e9880 <core::ptr::drop_glue::<rayon::vec::IntoIter<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e7071:	mov    %r15,%rdi
   e7074:	call   f3530 <core::ptr::drop_glue::<tk_train::trainers::bpe::corpus::merging::NeighborScratch>>
   e7079:	mov    %rbx,%rdi
   e707c:	call   101640 <_Unwind_Resume@plt>
   e7081:	xor    %r13d,%r13d
   e7084:	jmp    e70a5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f75>
   e7086:	cs nopw 0x0(%rax,%rax,1)
   e7090:	mov    %r15,%rdi
   e7093:	call   *0x2050f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7099:	inc    %r13
   e709c:	cmp    %r12,%r13
   e709f:	je     e7016 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ee6>
   e70a5:	lea    0x0(,%r13,8),%rcx
   e70ad:	add    %r13,%rcx
   e70b0:	lea    (%r14,%rcx,8),%rax
   e70b4:	cmpq   $0x0,(%r14,%rcx,8)
   e70b9:	je     e70d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fa0>
   e70bb:	cmpq   $0x11,0x18(%rax)
   e70c0:	jb     e7099 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f69>
   e70c2:	mov    0x8(%rax),%r15
   e70c6:	jmp    e7090 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f60>
   e70c8:	nopl   0x0(%rax,%rax,1)
   e70d0:	mov    0x10(%rax),%rbp
   e70d4:	test   %rbp,%rbp
   e70d7:	je     e7099 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f69>
   e70d9:	mov    0x8(%rax),%r15
   e70dd:	lea    0x18(%r15),%rbx
   e70e1:	jmp    e70f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc9>
   e70e3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e70f0:	add    $0x20,%rbx
   e70f4:	dec    %rbp
   e70f7:	je     e7090 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f60>
   e70f9:	cmpq   $0x11,(%rbx)
   e70fd:	jb     e70f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc0>
   e70ff:	mov    -0x10(%rbx),%rdi
   e7103:	call   *0x2049f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7109:	jmp    e70f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fc0>
   e710b:	xor    %r13d,%r13d
   e710e:	jmp    e7125 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1ff5>
   e7110:	mov    %r15,%rdi
   e7113:	call   *0x2048f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7119:	inc    %r13
   e711c:	cmp    %r12,%r13
   e711f:	je     e7043 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1f13>
   e7125:	lea    0x0(,%r13,8),%rcx
   e712d:	add    %r13,%rcx
   e7130:	lea    (%r14,%rcx,8),%rax
   e7134:	cmpq   $0x0,(%r14,%rcx,8)
   e7139:	je     e7150 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2020>
   e713b:	cmpq   $0x11,0x18(%rax)
   e7140:	jb     e7119 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe9>
   e7142:	mov    0x8(%rax),%r15
   e7146:	jmp    e7110 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe0>
   e7148:	nopl   0x0(%rax,%rax,1)
   e7150:	mov    0x10(%rax),%rbp
   e7154:	test   %rbp,%rbp
   e7157:	je     e7119 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe9>
   e7159:	mov    0x8(%rax),%r15
   e715d:	lea    0x18(%r15),%rbx
   e7161:	jmp    e7179 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2049>
   e7163:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e7170:	add    $0x20,%rbx
   e7174:	dec    %rbp
   e7177:	je     e7110 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x1fe0>
   e7179:	cmpq   $0x11,(%rbx)
   e717d:	jb     e7170 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2040>
   e717f:	mov    -0x10(%rbx),%rdi
   e7183:	call   *0x2041f(%rip)        # 1075a8 <free@GLIBC_2.2.5>
   e7189:	jmp    e7170 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>+0x2040>
