
/tmp/bpe-corpus-binaries-20261013/final-candidate-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000ea4f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>>:
   ea4f0:	push   %rbp
   ea4f1:	push   %r15
   ea4f3:	push   %r14
   ea4f5:	push   %r13
   ea4f7:	push   %r12
   ea4f9:	push   %rbx
   ea4fa:	sub    $0x308,%rsp
   ea501:	mov    %rdi,%r15
   ea504:	mov    0x348(%rsp),%rbx
   ea50c:	mov    %rsi,0x258(%rsp)
   ea514:	mov    %rcx,0x260(%rsp)
   ea51c:	mov    %r8,0x268(%rsp)
   ea524:	mov    (%rbx),%r13
   ea527:	movzbl 0x0(%r13),%eax
   ea52c:	test   %al,%al
   ea52e:	je     ea581 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x91>
   ea530:	mov    0x18(%rbx),%rax
   ea534:	movq   $0x0,0x20(%rsp)
   ea53d:	movq   $0x8,0x28(%rsp)
   ea546:	movq   $0x0,0x30(%rsp)
   ea54f:	mov    %r13,0x38(%rsp)
   ea554:	movups 0x8(%rbx),%xmm0
   ea558:	movups %xmm0,0x40(%rsp)
   ea55d:	mov    %rax,0x50(%rsp)
   ea562:	lea    0x20(%rsp),%rsi
   ea567:	mov    %r15,%rdi
   ea56a:	call   edc30 <<rayon::iter::filter::FilterFolder<rayon::iter::map::MapFolder<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}> as rayon::iter::plumbing::Folder<&mut tk_train::trainers::bpe::index::Shard>>::complete>
   ea56f:	add    $0x308,%rsp
   ea576:	pop    %rbx
   ea577:	pop    %r12
   ea579:	pop    %r13
   ea57b:	pop    %r14
   ea57d:	pop    %r15
   ea57f:	pop    %rbp
   ea580:	ret
   ea581:	mov    %r9,%r12
   ea584:	mov    0x340(%rsp),%r14
   ea58c:	shr    $1,%rsi
   ea58f:	cmp    %r8,%rsi
   ea592:	jae    ecfb2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ac2>
   ea598:	mov    0x8(%rbx),%rdx
   ea59c:	mov    0x10(%rbx),%rcx
   ea5a0:	mov    0x18(%rbx),%rax
   ea5a4:	movq   $0x0,0x110(%rsp)
   ea5b0:	movq   $0x8,0x118(%rsp)
   ea5bc:	movq   $0x0,0x120(%rsp)
   ea5c8:	mov    %r13,0x128(%rsp)
   ea5d0:	mov    %rdx,0x150(%rsp)
   ea5d8:	mov    %rdx,0x130(%rsp)
   ea5e0:	mov    %rcx,0x170(%rsp)
   ea5e8:	mov    %rcx,0x138(%rsp)
   ea5f0:	mov    %rax,0x188(%rsp)
   ea5f8:	mov    %rax,0x140(%rsp)
   ea600:	imul   $0x70,%r14,%rcx
   ea604:	add    %r12,%rcx
   ea607:	movabs $0x155555555555556,%rax
   ea611:	dec    %rax
   ea614:	mov    %rax,0x248(%rsp)
   ea61c:	movq   $0x0,0x180(%rsp)
   ea628:	lea    0x1c289(%rip),%rax        # 1068b8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33c0>
   ea62f:	mov    %rax,0x178(%rsp)
   ea637:	lea    0x1cda2(%rip),%rax        # 1073e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   ea63e:	mov    %rax,0x208(%rsp)
   ea646:	mov    %r15,0x210(%rsp)
   ea64e:	cmp    %rcx,%r12
   ea651:	je     ed214 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d24>
   ea657:	mov    0x10(%r12),%r14
   ea65c:	test   %r14,%r14
   ea65f:	je     ea6ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fa>
   ea665:	mov    0x28(%r12),%r15
   ea66a:	movabs $0x249249249249249,%rax
   ea674:	cmp    %rax,%r15
   ea677:	ja     ecfcc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2adc>
   ea67d:	mov    %rcx,0x250(%rsp)
   ea685:	mov    0x170(%rsp),%rdx
   ea68d:	mov    (%rdx),%rax
   ea690:	mov    0x8(%rdx),%rcx
   ea694:	mov    0x8(%rax),%rsi
   ea698:	mov    %rsi,0xf0(%rsp)
   ea6a0:	mov    0x10(%rax),%rax
   ea6a4:	mov    %rax,0x10(%rsp)
   ea6a9:	movzbl (%rcx),%eax
   ea6ac:	mov    %al,0xf(%rsp)
   ea6b0:	mov    0x10(%rdx),%rax
   ea6b4:	mov    (%rax),%rax
   ea6b7:	mov    %rax,0x218(%rsp)
   ea6bf:	test   %r15,%r15
   ea6c2:	mov    %r13,0x148(%rsp)
   ea6ca:	je     ea709 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x219>
   ea6cc:	imul   $0x38,%r15,%rbx
   ea6d0:	mov    %rbx,%rdi
   ea6d3:	call   *0x1d1e7(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   ea6d9:	test   %rax,%rax
   ea6dc:	je     ed4b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fc0>
   ea6e2:	mov    %rax,%rsi
   ea6e5:	mov    %r15,%rax
   ea6e8:	jmp    ea710 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x220>
   ea6ea:	mov    0x120(%rsp),%rax
   ea6f2:	mov    %rax,0x30(%rsp)
   ea6f7:	movups 0x110(%rsp),%xmm0
   ea6ff:	movaps %xmm0,0x20(%rsp)
   ea704:	jmp    ecdef <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x28ff>
   ea709:	mov    $0x8,%esi
   ea70e:	xor    %eax,%eax
   ea710:	lea    0x18(%r12),%rcx
   ea715:	mov    %r15,0x190(%rsp)
   ea71d:	mov    %rsi,0x198(%rsp)
   ea725:	movq   $0x0,0x1a0(%rsp)
   ea731:	movq   $0x0,0x158(%rsp)
   ea73d:	movq   $0x8,0x160(%rsp)
   ea749:	movq   $0x0,0x168(%rsp)
   ea755:	movq   $0x0,0x28(%r12)
   ea75e:	mov    0x20(%r12),%rdx
   ea763:	mov    %rax,%rdi
   ea766:	shl    $0x5,%rdi
   ea76a:	add    %rdx,%rdi
   ea76d:	mov    %rax,0x1c0(%rsp)
   ea775:	movq   $0x0,0x1c8(%rsp)
   ea781:	mov    %rdx,0x1a8(%rsp)
   ea789:	mov    %rdi,0x18(%rsp)
   ea78e:	mov    %rdi,0x1b0(%rsp)
   ea796:	mov    %rcx,0x1b8(%rsp)
   ea79e:	movq   $0x0,0x10(%r12)
   ea7a7:	mov    0x8(%r12),%rax
   ea7ac:	mov    %r14,%r13
   ea7af:	shl    $0x4,%r13
   ea7b3:	add    %rax,%r13
   ea7b6:	mov    %r13,0xb8(%rsp)
   ea7be:	mov    %r12,0xc0(%rsp)
   ea7c6:	mov    %r14,0xc8(%rsp)
   ea7ce:	movq   $0x0,0xd0(%rsp)
   ea7da:	lea    0x30(%r12),%rcx
   ea7df:	mov    %rcx,0x240(%rsp)
   ea7e7:	xor    %ebx,%ebx
   ea7e9:	mov    %rsi,%r15
   ea7ec:	jmp    ea848 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x358>
   ea7ee:	xchg   %ax,%ax
   ea7f0:	mov    0x198(%rsp),%r15
   ea7f8:	imul   $0x38,%rbx,%rax
   ea7fc:	mov    0x50(%rsp),%rcx
   ea801:	mov    %rcx,0x30(%r15,%rax,1)
   ea806:	movdqa 0x20(%rsp),%xmm0
   ea80c:	movdqa 0x30(%rsp),%xmm1
   ea812:	movdqa 0x40(%rsp),%xmm2
   ea818:	movdqu %xmm2,0x20(%r15,%rax,1)
   ea81f:	movdqu %xmm1,0x10(%r15,%rax,1)
   ea826:	movdqu %xmm0,(%r15,%rax,1)
   ea82c:	inc    %rbx
   ea82f:	mov    %rbx,0x1a0(%rsp)
   ea837:	mov    0xb0(%rsp),%rax
   ea83f:	cmp    %r13,%rax
   ea842:	je     eac67 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x777>
   ea848:	lea    0x10(%rax),%rcx
   ea84c:	mov    %rcx,0xb0(%rsp)
   ea854:	mov    (%rax),%rdi
   ea857:	cmp    0x10(%rsp),%rdi
   ea85c:	jae    ed3c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ed3>
   ea862:	movzbl 0x9(%rax),%ebp
   ea866:	lea    (%rdi,%rdi,4),%rcx
   ea86a:	mov    0xf0(%rsp),%rdx
   ea872:	lea    (%rdx,%rcx,8),%r14
   ea876:	testb  $0x1,0x8(%rax)
   ea87a:	je     ea950 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x460>
   ea880:	mov    0x4(%r14),%r8d
   ea884:	cmpb   $0x0,0xf(%rsp)
   ea889:	je     ea970 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x480>
   ea88f:	mov    (%r14),%edx
   ea892:	lea    0x20(%rsp),%rdi
   ea897:	lea    0x30(%r12),%rsi
   ea89c:	mov    %r8d,%ecx
   ea89f:	call   efba0 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::rustc_entry>
   ea8a4:	mov    0x20(%rsp),%rax
   ea8a9:	test   %rax,%rax
   ea8ac:	je     eaa7f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x58f>
   ea8b2:	mov    0x28(%rsp),%rdx
   ea8b7:	mov    (%rax),%rcx
   ea8ba:	mov    0x8(%rax),%rsi
   ea8be:	mov    %rsi,%rdi
   ea8c1:	and    %rdx,%rdi
   ea8c4:	movdqu (%rcx,%rdi,1),%xmm0
   ea8c9:	pmovmskb %xmm0,%r8d
   ea8ce:	test   %r8d,%r8d
   ea8d1:	je     eabef <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x6ff>
   ea8d7:	tzcnt  %r8d,%r8d
   ea8dc:	add    %rdi,%r8
   ea8df:	and    %rsi,%r8
   ea8e2:	movzbl (%rcx,%r8,1),%r9d
   ea8e7:	test   %r9b,%r9b
   ea8ea:	jns    eac14 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x724>
   ea8f0:	mov    0x30(%rsp),%r10
   ea8f5:	shr    $0x39,%rdx
   ea8f9:	lea    -0x10(%r8),%rdi
   ea8fd:	and    %rsi,%rdi
   ea900:	mov    %dl,(%rcx,%r8,1)
   ea904:	mov    %dl,0x10(%rcx,%rdi,1)
   ea908:	shl    $0x4,%r8
   ea90c:	mov    %rcx,%rdi
   ea90f:	sub    %r8,%rdi
   ea912:	and    $0x1,%r9b
   ea916:	neg    %r8
   ea919:	mov    %r10,-0x10(%rcx,%r8,1)
   ea91e:	movq   $0x0,-0x8(%rdi)
   ea926:	movdqu 0x10(%rax),%xmm0
   ea92b:	movzbl %r9b,%ecx
   ea92f:	movd   %ecx,%xmm1
   ea933:	shufpd $0x2,-0xdefcc(%rip),%xmm1        # b970 <GCC_except_table1289+0xe8>
   ea93c:	psubq  %xmm1,%xmm0
   ea940:	movdqu %xmm0,0x10(%rax)
   ea945:	jmp    eaa84 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x594>
   ea94a:	nopw   0x0(%rax,%rax,1)
   ea950:	test   $0x1,%bpl
   ea954:	je     ea837 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x347>
   ea95a:	cmpb   $0x0,0xf(%rsp)
   ea95f:	jne    eaaa9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5b9>
   ea965:	jmp    eab80 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x690>
   ea96a:	nopw   0x0(%rax,%rax,1)
   ea970:	cmpq   $0x0,0x48(%r12)
   ea976:	je     eaa70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x580>
   ea97c:	mov    (%r14),%esi
   ea97f:	mov    0x58(%r12),%rax
   ea984:	xor    %rsi,%rax
   ea987:	movabs $0x5851f42d4c957f2d,%rcx
   ea991:	mul    %rcx
   ea994:	xor    %r8,%rdx
   ea997:	xor    %rdx,%rax
   ea99a:	mul    %rcx
   ea99d:	mov    %rax,%rcx
   ea9a0:	xor    %rdx,%rcx
   ea9a3:	mov    %rcx,%rax
   ea9a6:	mulq   0x50(%r12)
   ea9ab:	xor    %rax,%rdx
   ea9ae:	rol    %cl,%rdx
   ea9b1:	mov    %rdx,%rdi
   ea9b4:	shr    $0x39,%rdi
   ea9b8:	mov    0x30(%r12),%rax
   ea9bd:	mov    0x38(%r12),%rcx
   ea9c2:	movd   %edi,%xmm0
   ea9c6:	punpcklbw %xmm0,%xmm0
   ea9ca:	pshuflw $0x0,%xmm0,%xmm0
   ea9cf:	pshufd $0x44,%xmm0,%xmm0
   ea9d4:	xor    %edi,%edi
   ea9d6:	and    %rcx,%rdx
   ea9d9:	movdqu (%rax,%rdx,1),%xmm1
   ea9de:	movdqa %xmm1,%xmm2
   ea9e2:	pcmpeqb %xmm0,%xmm2
   ea9e6:	pmovmskb %xmm2,%r9d
   ea9eb:	test   %r9d,%r9d
   ea9ee:	je     eaa1e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x52e>
   ea9f0:	tzcnt  %r9d,%r11d
   ea9f5:	add    %rdx,%r11
   ea9f8:	and    %rcx,%r11
   ea9fb:	shl    $0x4,%r11
   ea9ff:	mov    %rax,%r10
   eaa02:	sub    %r11,%r10
   eaa05:	cmp    -0x10(%r10),%esi
   eaa09:	jne    eaa11 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x521>
   eaa0b:	cmp    -0xc(%r10),%r8d
   eaa0f:	je     eaa3d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x54d>
   eaa11:	lea    -0x1(%r9),%r10d
   eaa15:	and    %r9w,%r10w
   eaa19:	mov    %r10d,%r9d
   eaa1c:	jne    ea9f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x500>
   eaa1e:	pcmpeqd %xmm2,%xmm2
   eaa22:	pcmpeqb %xmm2,%xmm1
   eaa26:	pmovmskb %xmm1,%r9d
   eaa2b:	test   %r9d,%r9d
   eaa2e:	jne    eaa70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x580>
   eaa30:	add    %rdi,%rdx
   eaa33:	add    $0x10,%rdx
   eaa37:	add    $0x10,%rdi
   eaa3b:	jmp    ea9d6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x4e6>
   eaa3d:	mov    -0x8(%r10),%rax
   eaa41:	sub    0x10(%r14),%rax
   eaa45:	jb     ec746 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2256>
   eaa4b:	mov    %rax,-0x8(%r10)
   eaa4f:	cmp    0x218(%rsp),%rax
   eaa57:	jae    eaa70 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x580>
   eaa59:	lea    0x30(%r12),%rdi
   eaa5e:	mov    %r8d,%edx
   eaa61:	call   e2300 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::remove::<(u32, u32)>>
   eaa66:	cs nopw 0x0(%rax,%rax,1)
   eaa70:	test   $0x1,%bpl
   eaa74:	je     ea837 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x347>
   eaa7a:	jmp    eab80 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x690>
   eaa7f:	mov    0x28(%rsp),%rdi
   eaa84:	add    $0xfffffffffffffff8,%rdi
   eaa88:	mov    0x10(%r14),%rsi
   eaa8c:	mov    $0x1,%edx
   eaa91:	call   efcd0 <tk_train::trainers::bpe::index::adjust_signed>
   eaa96:	test   %rax,%rax
   eaa99:	jne    ec5cb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20db>
   eaa9f:	test   $0x1,%bpl
   eaaa3:	je     ea837 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x347>
   eaaa9:	mov    0x8(%r14),%edx
   eaaad:	mov    0xc(%r14),%ecx
   eaab1:	lea    0x20(%rsp),%rdi
   eaab6:	lea    0x30(%r12),%rsi
   eaabb:	call   efba0 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::rustc_entry>
   eaac0:	mov    0x20(%rsp),%rax
   eaac5:	test   %rax,%rax
   eaac8:	je     eab63 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x673>
   eaace:	mov    0x28(%rsp),%rdx
   eaad3:	mov    (%rax),%rcx
   eaad6:	mov    0x8(%rax),%rsi
   eaada:	mov    %rsi,%rdi
   eaadd:	and    %rdx,%rdi
   eaae0:	movdqu (%rcx,%rdi,1),%xmm0
   eaae5:	pmovmskb %xmm0,%r8d
   eaaea:	test   %r8d,%r8d
   eaaed:	je     eac2b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x73b>
   eaaf3:	tzcnt  %r8d,%r8d
   eaaf8:	add    %rdi,%r8
   eaafb:	and    %rsi,%r8
   eaafe:	movzbl (%rcx,%r8,1),%r9d
   eab03:	test   %r9b,%r9b
   eab06:	jns    eac50 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x760>
   eab0c:	mov    0x30(%rsp),%r10
   eab11:	shr    $0x39,%rdx
   eab15:	lea    -0x10(%r8),%rdi
   eab19:	and    %rsi,%rdi
   eab1c:	mov    %dl,(%rcx,%r8,1)
   eab20:	mov    %dl,0x10(%rcx,%rdi,1)
   eab24:	shl    $0x4,%r8
   eab28:	mov    %rcx,%rdi
   eab2b:	sub    %r8,%rdi
   eab2e:	and    $0x1,%r9b
   eab32:	neg    %r8
   eab35:	mov    %r10,-0x10(%rcx,%r8,1)
   eab3a:	movq   $0x0,-0x8(%rdi)
   eab42:	movdqu 0x10(%rax),%xmm0
   eab47:	movzbl %r9b,%ecx
   eab4b:	movd   %ecx,%xmm1
   eab4f:	shufpd $0x2,-0xdf1e8(%rip),%xmm1        # b970 <GCC_except_table1289+0xe8>
   eab58:	psubq  %xmm1,%xmm0
   eab5c:	movdqu %xmm0,0x10(%rax)
   eab61:	jmp    eab68 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x678>
   eab63:	mov    0x28(%rsp),%rdi
   eab68:	add    $0xfffffffffffffff8,%rdi
   eab6c:	mov    0x18(%r14),%rsi
   eab70:	xor    %edx,%edx
   eab72:	call   efcd0 <tk_train::trainers::bpe::index::adjust_signed>
   eab77:	test   %rax,%rax
   eab7a:	jne    ec5cb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20db>
   eab80:	mov    0x1a8(%rsp),%rax
   eab88:	cmp    0x18(%rsp),%rax
   eab8d:	je     ed28f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d9f>
   eab93:	lea    0x20(%rax),%rcx
   eab97:	mov    %rcx,0x1a8(%rsp)
   eab9f:	movdqu (%rax),%xmm0
   eaba3:	movdqu 0x10(%rax),%xmm1
   eaba8:	movdqa %xmm1,0x30(%rsp)
   eabae:	movdqa %xmm0,0x20(%rsp)
   eabb4:	mov    0x18(%r14),%rax
   eabb8:	mov    0x20(%r14),%rcx
   eabbc:	mov    %rcx,0x40(%rsp)
   eabc1:	mov    0x8(%r14),%rcx
   eabc5:	mov    %rcx,0x48(%rsp)
   eabca:	mov    %rax,0x50(%rsp)
   eabcf:	cmp    0x190(%rsp),%rbx
   eabd7:	jne    ea7f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308>
   eabdd:	lea    0x190(%rsp),%rdi
   eabe5:	call   efe00 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::BirthFragment>>::grow_one>
   eabea:	jmp    ea7f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x300>
   eabef:	mov    $0x10,%r9d
   eabf5:	add    %r9,%rdi
   eabf8:	and    %rsi,%rdi
   eabfb:	movdqu (%rcx,%rdi,1),%xmm0
   eac00:	pmovmskb %xmm0,%r8d
   eac05:	add    $0x10,%r9
   eac09:	test   %r8d,%r8d
   eac0c:	jne    ea8d7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3e7>
   eac12:	jmp    eabf5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x705>
   eac14:	movdqa (%rcx),%xmm0
   eac18:	pmovmskb %xmm0,%edi
   eac1c:	tzcnt  %edi,%r8d
   eac21:	movzbl (%rcx,%r8,1),%r9d
   eac26:	jmp    ea8f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x400>
   eac2b:	mov    $0x10,%r9d
   eac31:	add    %r9,%rdi
   eac34:	and    %rsi,%rdi
   eac37:	movdqu (%rcx,%rdi,1),%xmm0
   eac3c:	pmovmskb %xmm0,%r8d
   eac41:	add    $0x10,%r9
   eac45:	test   %r8d,%r8d
   eac48:	jne    eaaf3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x603>
   eac4e:	jmp    eac31 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x741>
   eac50:	movdqa (%rcx),%xmm0
   eac54:	pmovmskb %xmm0,%edi
   eac58:	tzcnt  %edi,%r8d
   eac5d:	movzbl (%rcx,%r8,1),%r9d
   eac62:	jmp    eab0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x61c>
   eac67:	cmp    $0x2,%rbx
   eac6b:	jae    ece21 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2931>
   eac71:	mov    %r12,0xe0(%rsp)
   eac79:	mov    0x190(%rsp),%rax
   eac81:	imul   $0x38,%rbx,%rbp
   eac85:	add    %r15,%rbp
   eac88:	mov    %r15,0x1d0(%rsp)
   eac90:	mov    %rax,0x1e0(%rsp)
   eac98:	mov    %r15,0x1d8(%rsp)
   eaca0:	mov    %rbp,0x1e8(%rsp)
   eaca8:	test   %rbx,%rbx
   eacab:	je     ec6a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21b4>
   eacb1:	mov    %rbp,0xe8(%rsp)
   eacb9:	mov    0x20(%r15),%rax
   eacbd:	mov    0x28(%r15),%r12d
   eacc1:	mov    0x2c(%r15),%ecx
   eacc5:	mov    %rcx,0x10(%rsp)
   eacca:	xor    %r14d,%r14d
   eaccd:	mov    %r15,%rcx
   eacd0:	xor    %r13d,%r13d
   eacd3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eace0:	cmp    %rax,0x20(%rcx)
   eace4:	jne    ead0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x81c>
   eace6:	mov    0x10(%rsp),%rdx
   eaceb:	cmp    %edx,0x2c(%rcx)
   eacee:	jne    ead0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x81c>
   eacf0:	cmp    %r12d,0x28(%rcx)
   eacf4:	jne    ead0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x81c>
   eacf6:	add    0x30(%rcx),%r13
   eacfa:	jb     eca25 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2535>
   ead00:	add    $0x38,%rcx
   ead04:	inc    %r14
   ead07:	cmp    %rbp,%rcx
   ead0a:	jne    eace0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7f0>
   ead0c:	cmpb   $0x0,0xf(%rsp)
   ead11:	mov    %r12,0xf0(%rsp)
   ead19:	je     eadfa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x90a>
   ead1f:	mov    0xe0(%rsp),%rdi
   ead27:	cmpq   $0x0,0x48(%rdi)
   ead2c:	je     ed1f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d07>
   ead32:	mov    0x58(%rdi),%rax
   ead36:	xor    %r12,%rax
   ead39:	movabs $0x5851f42d4c957f2d,%rsi
   ead43:	mul    %rsi
   ead46:	mov    0x10(%rsp),%rcx
   ead4b:	xor    %rdx,%rcx
   ead4e:	xor    %rcx,%rax
   ead51:	mul    %rsi
   ead54:	mov    %rax,%rcx
   ead57:	xor    %rdx,%rcx
   ead5a:	mov    %rcx,%rax
   ead5d:	mulq   0x50(%rdi)
   ead61:	xor    %rax,%rdx
   ead64:	rol    %cl,%rdx
   ead67:	mov    %rdx,%rsi
   ead6a:	shr    $0x39,%rsi
   ead6e:	mov    0x30(%rdi),%rax
   ead72:	mov    0x38(%rdi),%rcx
   ead76:	movd   %esi,%xmm0
   ead7a:	punpcklbw %xmm0,%xmm0
   ead7e:	pshuflw $0x0,%xmm0,%xmm0
   ead83:	pshufd $0x44,%xmm0,%xmm0
   ead88:	xor    %esi,%esi
   ead8a:	and    %rcx,%rdx
   ead8d:	movdqu (%rax,%rdx,1),%xmm1
   ead92:	movdqa %xmm1,%xmm2
   ead96:	pcmpeqb %xmm0,%xmm2
   ead9a:	pmovmskb %xmm2,%edi
   ead9e:	test   %edi,%edi
   eada0:	je     eadd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x8e9>
   eada2:	tzcnt  %edi,%r9d
   eada7:	add    %rdx,%r9
   eadaa:	and    %rcx,%r9
   eadad:	shl    $0x4,%r9
   eadb1:	mov    %rax,%r8
   eadb4:	sub    %r9,%r8
   eadb7:	cmp    -0x10(%r8),%r12d
   eadbb:	jne    eadcc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x8dc>
   eadbd:	mov    0x10(%rsp),%r9
   eadc2:	cmp    -0xc(%r8),%r9d
   eadc6:	je     eb0f2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc02>
   eadcc:	lea    -0x1(%rdi),%r8d
   eadd0:	and    %di,%r8w
   eadd4:	mov    %r8d,%edi
   eadd7:	jne    eada2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x8b2>
   eadd9:	pcmpeqd %xmm2,%xmm2
   eaddd:	pcmpeqb %xmm2,%xmm1
   eade1:	pmovmskb %xmm1,%edi
   eade5:	test   %edi,%edi
   eade7:	jne    ed1f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d07>
   eaded:	add    %rsi,%rdx
   eadf0:	add    $0x10,%rdx
   eadf4:	add    $0x10,%rsi
   eadf8:	jmp    ead8a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x89a>
   eadfa:	cmp    0x218(%rsp),%r13
   eae02:	jb     eb176 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc86>
   eae08:	movq   $0x0,0x40(%rsp)
   eae11:	test   %r14,%r14
   eae14:	je     eb2ac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdbc>
   eae1a:	cmp    %rbp,%r15
   eae1d:	je     eb2c5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdd5>
   eae23:	xor    %r12d,%r12d
   eae26:	xor    %ebx,%ebx
   eae28:	dec    %r14
   eae2b:	nopl   0x0(%rax,%rax,1)
   eae30:	movdqu (%r15),%xmm0
   eae35:	movdqu 0x10(%r15),%xmm1
   eae3b:	add    $0x38,%r15
   eae3f:	movdqa %xmm0,0x80(%rsp)
   eae48:	movdqa %xmm1,0x90(%rsp)
   eae51:	cmpq   $0x0,0x80(%rsp)
   eae5a:	je     ec7cd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x22dd>
   eae60:	movdqa 0x80(%rsp),%xmm0
   eae69:	movdqa 0x90(%rsp),%xmm1
   eae72:	movdqa %xmm0,0xb0(%rsp)
   eae7b:	movdqa %xmm1,0xc0(%rsp)
   eae84:	mov    0xb0(%rsp),%rbp
   eae8c:	cmp    $0x1,%rbp
   eae90:	jne    eaecb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x9db>
   eae92:	cmpq   $0x11,0xc8(%rsp)
   eae9b:	mov    0xe8(%rsp),%rbp
   eaea3:	jb     eaeb3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x9c3>
   eaea5:	mov    0xb8(%rsp),%rdi
   eaead:	call   *0x1c9f5(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eaeb3:	sub    $0x1,%r14
   eaeb7:	jb     eb2c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdd7>
   eaebd:	cmp    %rbp,%r15
   eaec0:	jne    eae30 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x940>
   eaec6:	jmp    eb2c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdd7>
   eaecb:	mov    %r15,0x1d8(%rsp)
   eaed3:	test   $0x1,%r12b
   eaed7:	je     eaf96 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xaa6>
   eaedd:	lea    -0x1(%rbp),%rcx
   eaee1:	mov    %rcx,%rax
   eaee4:	shr    $0x7,%rax
   eaee8:	and    $0x7f,%ecx
   eaeeb:	cmp    $0x1,%rcx
   eaeef:	sbb    $0xffffffffffffffff,%rax
   eaef3:	mov    $0x8,%edi
   eaef8:	cmp    $0x2,%rax
   eaefc:	jae    eaf2c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa3c>
   eaefe:	mov    0xc8(%rsp),%r12
   eaf06:	mov    %r12,%rsi
   eaf09:	cmp    $0x11,%r12
   eaf0d:	jb     eaf58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa68>
   eaf0f:	mov    0xc0(%rsp),%rsi
   eaf17:	mov    %rsi,%rdx
   eaf1a:	sub    %rdi,%rdx
   eaf1d:	mov    0x108(%rsp),%rcx
   eaf25:	jae    eaf6c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa7c>
   eaf27:	jmp    ed3d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ee9>
   eaf2c:	dec    %rax
   eaf2f:	bsr    %rax,%rcx
   eaf33:	not    %ecx
   eaf35:	mov    $0xffffffffffffffff,%rax
   eaf3c:	shr    %cl,%rax
   eaf3f:	lea    0x10(,%rax,8),%rdi
   eaf47:	mov    0xc8(%rsp),%r12
   eaf4f:	mov    %r12,%rsi
   eaf52:	cmp    $0x11,%r12
   eaf56:	jae    eaf0f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa1f>
   eaf58:	mov    %rsi,%rdx
   eaf5b:	sub    %rdi,%rdx
   eaf5e:	mov    0x108(%rsp),%rcx
   eaf66:	jb     ed3d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ee9>
   eaf6c:	cmp    $0x7,%rdx
   eaf70:	jbe    ed3a1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eb1>
   eaf76:	lea    0xb8(%rsp),%rax
   eaf7e:	cmp    $0x11,%r12
   eaf82:	jb     eaf8c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa9c>
   eaf84:	mov    0xb8(%rsp),%rax
   eaf8c:	cmp    (%rax,%rdi,1),%rcx
   eaf90:	ja     eca83 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2593>
   eaf96:	lea    (%rbx,%rbp,1),%rax
   eaf9a:	dec    %rax
   eaf9d:	mov    %rax,0x18(%rsp)
   eafa2:	cmp    %rbx,%rax
   eafa5:	jb     ecf35 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a45>
   eafab:	mov    0xc8(%rsp),%r12
   eafb3:	cmp    $0x11,%r12
   eafb7:	mov    0xc0(%rsp),%rbx
   eafbf:	mov    %r12,%rdx
   eafc2:	cmovae %rbx,%rdx
   eafc6:	cmp    $0x7,%rdx
   eafca:	jbe    ed3a1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eb1>
   eafd0:	mov    0xb8(%rsp),%rcx
   eafd8:	cmp    $0x11,%r12
   eafdc:	lea    0xb8(%rsp),%rax
   eafe4:	mov    %rcx,0xa0(%rsp)
   eafec:	cmovae %rcx,%rax
   eaff0:	mov    (%rax),%rax
   eaff3:	mov    %rax,0x108(%rsp)
   eaffb:	mov    0x40(%rsp),%rcx
   eb000:	cmp    $0x2,%rcx
   eb004:	jb     eb045 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb55>
   eb006:	mov    0x28(%rsp),%rax
   eb00b:	cmp    %rcx,%rax
   eb00e:	je     eb0a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xbb0>
   eb014:	mov    0x20(%rsp),%rcx
   eb019:	mov    %rax,%rdx
   eb01c:	shl    $0x5,%rdx
   eb020:	mov    %rbp,(%rcx,%rdx,1)
   eb024:	mov    0xa0(%rsp),%rsi
   eb02c:	mov    %rsi,0x8(%rcx,%rdx,1)
   eb031:	mov    %rbx,0x10(%rcx,%rdx,1)
   eb036:	mov    %r12,0x18(%rcx,%rdx,1)
   eb03b:	inc    %rax
   eb03e:	mov    %rax,0x28(%rsp)
   eb043:	jmp    eb079 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb89>
   eb045:	cmp    $0x1,%rcx
   eb049:	je     eb0c4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xbd4>
   eb04b:	xor    %eax,%eax
   eb04d:	lea    0x20(%rsp),%rcx
   eb052:	lea    0x40(%rsp),%rdx
   eb057:	shl    $0x5,%rax
   eb05b:	mov    %rbp,(%rcx,%rax,1)
   eb05f:	mov    0xa0(%rsp),%rsi
   eb067:	mov    %rsi,0x8(%rcx,%rax,1)
   eb06c:	mov    %rbx,0x10(%rcx,%rax,1)
   eb071:	mov    %r12,0x18(%rcx,%rax,1)
   eb076:	incq   (%rdx)
   eb079:	test   %r14,%r14
   eb07c:	mov    0xe8(%rsp),%rbp
   eb084:	mov    0x18(%rsp),%rbx
   eb089:	je     eb2cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xddf>
   eb08f:	mov    $0x1,%r12b
   eb092:	cmp    %rbp,%r15
   eb095:	jne    eae28 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x938>
   eb09b:	jmp    eb2c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdd7>
   eb0a0:	mov    %r15,0xa8(%rsp)
   eb0a8:	lea    0x20(%rsp),%rdi
   eb0ad:	call   f04c0 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::reserve_one_unchecked>
   eb0b2:	mov    0x28(%rsp),%rax
   eb0b7:	mov    0xa8(%rsp),%r15
   eb0bf:	jmp    eb014 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb24>
   eb0c4:	mov    %r15,0xa8(%rsp)
   eb0cc:	lea    0x20(%rsp),%rdi
   eb0d1:	call   f04c0 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::reserve_one_unchecked>
   eb0d6:	mov    0x20(%rsp),%rcx
   eb0db:	mov    0x28(%rsp),%rax
   eb0e0:	lea    0x28(%rsp),%rdx
   eb0e5:	mov    0xa8(%rsp),%r15
   eb0ed:	jmp    eb057 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb67>
   eb0f2:	mov    -0x8(%r8),%rax
   eb0f6:	mov    %rax,0xa0(%rsp)
   eb0fe:	test   %rax,%rax
   eb101:	jle    eb176 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc86>
   eb103:	movq   $0x0,0x40(%rsp)
   eb10c:	mov    $0x1,%edi
   eb111:	test   %r14,%r14
   eb114:	je     eb216 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd26>
   eb11a:	cmp    %rbx,%r14
   eb11d:	cmovb  %r14,%rbx
   eb121:	cmp    $0x2,%rbx
   eb125:	jb     eb216 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd26>
   eb12b:	dec    %rbx
   eb12e:	bsr    %rbx,%rcx
   eb132:	not    %ecx
   eb134:	mov    $0xffffffffffffffff,%rsi
   eb13b:	shr    %cl,%rsi
   eb13e:	inc    %rsi
   eb141:	lea    0x20(%rsp),%rdi
   eb146:	call   f07f0 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::try_grow>
   eb14b:	cmp    $0xffffffffffffffff,%rax
   eb14f:	mov    0x1c76a(%rip),%rbx        # 1078c0 <malloc@GLIBC_2.2.5>
   eb156:	jne    ed3aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eba>
   eb15c:	mov    0x20(%rsp),%rcx
   eb161:	mov    0x40(%rsp),%rsi
   eb166:	cmp    $0x1,%rsi
   eb16a:	mov    %rsi,%rdi
   eb16d:	adc    $0x0,%rdi
   eb171:	jmp    eb21f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd2f>
   eb176:	test   %r14,%r14
   eb179:	je     ebd4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x185f>
   eb17f:	nop
   eb180:	cmp    %rbp,%r15
   eb183:	je     eb2b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdc8>
   eb189:	mov    0x8(%r15),%rbx
   eb18d:	cmpq   $0x0,(%r15)
   eb191:	je     eb1b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcc0>
   eb193:	mov    %r15,%rbp
   eb196:	cmpq   $0x10,0x18(%r15)
   eb19b:	jbe    eb1f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd00>
   eb19d:	mov    %rbx,%rdi
   eb1a0:	call   *0x1c702(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eb1a6:	jmp    eb1f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd00>
   eb1a8:	nopl   0x0(%rax,%rax,1)
   eb1b0:	mov    %r15,%rbp
   eb1b3:	mov    0x10(%r15),%r15
   eb1b7:	test   %r15,%r15
   eb1ba:	je     eb1f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd00>
   eb1bc:	lea    0x18(%rbx),%r13
   eb1c0:	jmp    eb1d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xce9>
   eb1c2:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb1d0:	add    $0x20,%r13
   eb1d4:	dec    %r15
   eb1d7:	je     eb19d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcad>
   eb1d9:	cmpq   $0x11,0x0(%r13)
   eb1de:	jb     eb1d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xce0>
   eb1e0:	mov    -0x10(%r13),%rdi
   eb1e4:	call   *0x1c6be(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eb1ea:	jmp    eb1d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xce0>
   eb1ec:	nopl   0x0(%rax)
   eb1f0:	mov    %rbp,%r15
   eb1f3:	add    $0x38,%r15
   eb1f7:	dec    %r14
   eb1fa:	je     eb2b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdc0>
   eb200:	mov    0xe8(%rsp),%rbp
   eb208:	cmp    %rbp,%r15
   eb20b:	jne    eb189 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc99>
   eb211:	jmp    eb2b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdc8>
   eb216:	xor    %esi,%esi
   eb218:	mov    0x1c6a1(%rip),%rbx        # 1078c0 <malloc@GLIBC_2.2.5>
   eb21f:	cmp    $0x2,%rsi
   eb223:	lea    0x20(%rsp),%rax
   eb228:	cmovb  %rax,%rcx
   eb22c:	mov    $0x20,%eax
   eb231:	mov    $0x8,%edx
   eb236:	cmovae %rdx,%rax
   eb23a:	mov    0x20(%rsp,%rax,1),%rdx
   eb23f:	cmp    %rdi,%rdx
   eb242:	jae    eb2a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdb4>
   eb244:	test   %r14,%r14
   eb247:	je     eb292 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xda2>
   eb249:	cmp    %rbp,%r15
   eb24c:	je     eb292 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xda2>
   eb24e:	mov    %rdx,%rdi
   eb251:	shl    $0x5,%rdi
   eb255:	add    %rdi,%rcx
   eb258:	dec    %r14
   eb25b:	movdqu (%r15),%xmm0
   eb260:	movdqu 0x10(%r15),%xmm1
   eb266:	add    $0x38,%r15
   eb26a:	movdqu %xmm1,0x10(%rcx)
   eb26f:	movdqu %xmm0,(%rcx)
   eb273:	lea    0x1(%rdx),%rdi
   eb277:	cmp    %rdi,%rsi
   eb27a:	jbe    eb456 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf66>
   eb280:	inc    %rdx
   eb283:	sub    $0x1,%r14
   eb287:	jb     eb292 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xda2>
   eb289:	add    $0x20,%rcx
   eb28d:	cmp    %rbp,%r15
   eb290:	jne    eb25b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd6b>
   eb292:	mov    %r15,0x1d8(%rsp)
   eb29a:	mov    %rdx,0x20(%rsp,%rax,1)
   eb29f:	jmp    eb56e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x107e>
   eb2a4:	mov    %rdx,%rdi
   eb2a7:	jmp    eb45e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf6e>
   eb2ac:	xor    %ebx,%ebx
   eb2ae:	jmp    eb2cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xddf>
   eb2b0:	mov    0xe8(%rsp),%rbp
   eb2b8:	mov    %r15,0x1d8(%rsp)
   eb2c0:	jmp    ebd4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x185f>
   eb2c5:	xor    %ebx,%ebx
   eb2c7:	mov    %r15,0x1d8(%rsp)
   eb2cf:	mov    0x40(%rsp),%rax
   eb2d4:	mov    %rax,%r12
   eb2d7:	cmp    $0x2,%rax
   eb2db:	jb     eb2e2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdf2>
   eb2dd:	mov    0x28(%rsp),%r12
   eb2e2:	mov    0x20(%rsp),%r14
   eb2e7:	test   %r12,%r12
   eb2ea:	je     eb333 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe43>
   eb2ec:	cmp    $0x1,%r12
   eb2f0:	jne    eb3e3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xef3>
   eb2f6:	cmp    $0x2,%rax
   eb2fa:	lea    0x20(%rsp),%rax
   eb2ff:	cmovae %r14,%rax
   eb303:	lea    0x40(%rsp),%rcx
   eb308:	lea    0x28(%rsp),%rdx
   eb30d:	cmovae %rdx,%rcx
   eb311:	movq   $0x0,(%rcx)
   eb318:	mov    (%rax),%rcx
   eb31b:	mov    0x8(%rax),%rbx
   eb31f:	mov    0x10(%rax),%r12
   eb323:	mov    0x18(%rax),%rax
   eb327:	mov    %rax,0x18(%rsp)
   eb32c:	mov    0x40(%rsp),%rax
   eb331:	jmp    eb33f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe4f>
   eb333:	mov    $0x1,%ecx
   eb338:	mov    %r12,0x18(%rsp)
   eb33d:	xor    %ebx,%ebx
   eb33f:	cmp    $0x1,%rax
   eb343:	jbe    eb38c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe9c>
   eb345:	mov    %rcx,0xa0(%rsp)
   eb34d:	mov    %r15,0xa8(%rsp)
   eb355:	mov    0x28(%rsp),%r15
   eb35a:	test   %r15,%r15
   eb35d:	je     eb3ba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xeca>
   eb35f:	lea    0x18(%r14),%rbp
   eb363:	jmp    eb379 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe89>
   eb365:	data16 cs nopw 0x0(%rax,%rax,1)
   eb370:	add    $0x20,%rbp
   eb374:	dec    %r15
   eb377:	je     eb3ba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xeca>
   eb379:	cmpq   $0x11,0x0(%rbp)
   eb37e:	jb     eb370 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe80>
   eb380:	mov    -0x10(%rbp),%rdi
   eb384:	call   *0x1c51e(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eb38a:	jmp    eb370 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe80>
   eb38c:	test   %rax,%rax
   eb38f:	je     ebc7c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178c>
   eb395:	cmpq   $0x11,0x38(%rsp)
   eb39b:	mov    %r13,%r14
   eb39e:	jb     ebc7f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178f>
   eb3a4:	mov    0x28(%rsp),%rdi
   eb3a9:	mov    %rcx,%r13
   eb3ac:	call   *0x1c4f6(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eb3b2:	mov    %r13,%rcx
   eb3b5:	jmp    ebc7f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178f>
   eb3ba:	mov    %r14,%rdi
   eb3bd:	call   *0x1c4e5(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eb3c3:	mov    0xa8(%rsp),%r15
   eb3cb:	mov    0xe8(%rsp),%rbp
   eb3d3:	mov    %r13,%r14
   eb3d6:	mov    0xa0(%rsp),%rcx
   eb3de:	jmp    ebc7f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178f>
   eb3e3:	lea    0xb0(%rsp),%rdi
   eb3eb:	lea    0x20(%rsp),%rsi
   eb3f0:	call   f0180 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::into_vec>
   eb3f5:	mov    0xb0(%rsp),%rsi
   eb3fd:	mov    0xc0(%rsp),%r12
   eb405:	cmp    %r12,%rsi
   eb408:	mov    %rbx,0x18(%rsp)
   eb40d:	jbe    eb449 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf59>
   eb40f:	mov    0xb8(%rsp),%rdi
   eb417:	test   %r12,%r12
   eb41a:	je     ebc6f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x177f>
   eb420:	shl    $0x5,%rsi
   eb424:	mov    %r12,%r14
   eb427:	shl    $0x5,%r14
   eb42b:	mov    $0x8,%edx
   eb430:	mov    %r14,%rcx
   eb433:	call   68fe0 <__rustc::__rust_realloc>
   eb438:	mov    %rax,%rbx
   eb43b:	test   %rax,%rax
   eb43e:	jne    ebc7a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178a>
   eb444:	jmp    ed512 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3022>
   eb449:	mov    0xb8(%rsp),%rbx
   eb451:	jmp    ebc7a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x178a>
   eb456:	mov    %r15,0x1d8(%rsp)
   eb45e:	mov    %rdi,0x20(%rsp,%rax,1)
   eb463:	test   %r14,%r14
   eb466:	je     eb56e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x107e>
   eb46c:	cmp    %rbp,%r15
   eb46f:	je     eb566 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1076>
   eb475:	dec    %r14
   eb478:	nopl   0x0(%rax,%rax,1)
   eb480:	movdqu (%r15),%xmm0
   eb485:	movdqu 0x10(%r15),%xmm1
   eb48b:	movdqa %xmm1,0x90(%rsp)
   eb494:	movdqa %xmm0,0x80(%rsp)
   eb49d:	mov    0x40(%rsp),%rcx
   eb4a2:	cmp    $0x2,%rcx
   eb4a6:	jb     eb4e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xff6>
   eb4a8:	mov    0x28(%rsp),%rax
   eb4ad:	cmp    %rcx,%rax
   eb4b0:	je     eb537 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1047>
   eb4b6:	mov    0x20(%rsp),%rcx
   eb4bb:	mov    %rax,%rdx
   eb4be:	shl    $0x5,%rdx
   eb4c2:	movdqa 0x80(%rsp),%xmm0
   eb4cb:	movdqa 0x90(%rsp),%xmm1
   eb4d4:	movdqu %xmm1,0x10(%rcx,%rdx,1)
   eb4da:	movdqu %xmm0,(%rcx,%rdx,1)
   eb4df:	lea    0x28(%rsp),%rcx
   eb4e4:	jmp    eb51c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x102c>
   eb4e6:	cmp    $0x1,%rcx
   eb4ea:	je     eb54b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x105b>
   eb4ec:	xor    %eax,%eax
   eb4ee:	lea    0x20(%rsp),%rdx
   eb4f3:	lea    0x40(%rsp),%rcx
   eb4f8:	shl    $0x5,%rax
   eb4fc:	movdqa 0x80(%rsp),%xmm0
   eb505:	movdqa 0x90(%rsp),%xmm1
   eb50e:	movdqu %xmm1,0x10(%rdx,%rax,1)
   eb514:	movdqu %xmm0,(%rdx,%rax,1)
   eb519:	mov    (%rcx),%rax
   eb51c:	add    $0x38,%r15
   eb520:	inc    %rax
   eb523:	mov    %rax,(%rcx)
   eb526:	sub    $0x1,%r14
   eb52a:	jb     eb566 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1076>
   eb52c:	cmp    %rbp,%r15
   eb52f:	jne    eb480 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf90>
   eb535:	jmp    eb566 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1076>
   eb537:	lea    0x20(%rsp),%rdi
   eb53c:	call   f09b0 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::reserve_one_unchecked>
   eb541:	mov    0x28(%rsp),%rax
   eb546:	jmp    eb4b6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xfc6>
   eb54b:	lea    0x20(%rsp),%rdi
   eb550:	call   f09b0 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::reserve_one_unchecked>
   eb555:	mov    0x20(%rsp),%rdx
   eb55a:	mov    0x28(%rsp),%rax
   eb55f:	lea    0x28(%rsp),%rcx
   eb564:	jmp    eb4f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1008>
   eb566:	mov    %r15,0x1d8(%rsp)
   eb56e:	mov    0x40(%rsp),%rax
   eb573:	mov    %rax,0xd0(%rsp)
   eb57b:	movdqu 0x20(%rsp),%xmm0
   eb581:	movdqu 0x30(%rsp),%xmm1
   eb587:	movdqa %xmm1,0xc0(%rsp)
   eb590:	movdqa %xmm0,0xb0(%rsp)
   eb599:	cmp    $0x2,%rax
   eb59d:	mov    0xb8(%rsp),%rcx
   eb5a5:	mov    %rax,%r14
   eb5a8:	cmovae %rcx,%r14
   eb5ac:	cmp    $0x1,%r14
   eb5b0:	jne    eb690 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x11a0>
   eb5b6:	mov    0xb0(%rsp),%rdx
   eb5be:	lea    0x8(%rdx),%rsi
   eb5c2:	lea    0x10(%rdx),%r11
   eb5c6:	cmp    $0x2,%rax
   eb5ca:	lea    0xb8(%rsp),%r8
   eb5d2:	movups 0x8(%r8),%xmm0
   eb5d7:	lea    0x30(%rsp),%r10
   eb5dc:	movups %xmm0,(%r10)
   eb5e0:	mov    $0x0,%r9d
   eb5e6:	cmovae %r9,%rcx
   eb5ea:	mov    $0x0,%r8d
   eb5f0:	mov    %r8,0x180(%rsp)
   eb5f8:	cmovb  %r9,%rax
   eb5fc:	mov    %rdx,0x20(%rsp)
   eb601:	mov    %rcx,0x28(%rsp)
   eb606:	mov    %rax,0x40(%rsp)
   eb60b:	movq   $0x1,0x50(%rsp)
   eb614:	lea    0x20(%rsp),%rdi
   eb619:	cmovb  %rdi,%rdx
   eb61d:	lea    0x28(%rsp),%rax
   eb622:	cmovb  %rax,%rsi
   eb626:	movq   $0x1,0x48(%rsp)
   eb62f:	cmovb  %r10,%r11
   eb633:	movups (%r11),%xmm0
   eb637:	movq   (%rsi),%xmm1
   eb63b:	movq   (%rdx),%xmm2
   eb63f:	punpcklqdq %xmm1,%xmm2
   eb643:	movdqa %xmm2,0xf0(%rsp)
   eb64c:	movaps %xmm0,0x270(%rsp)
   eb654:	call   f0a40 <core::ptr::drop_glue::<smallvec::IntoIter<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   eb659:	mov    0xa0(%rsp),%r14
   eb661:	movaps 0x270(%rsp),%xmm0
   eb669:	lea    0x230(%rsp),%rax
   eb671:	movups %xmm0,(%rax)
   eb674:	movdqa 0xf0(%rsp),%xmm0
   eb67d:	movdqa %xmm0,0x220(%rsp)
   eb686:	mov    0x10(%rsp),%rbx
   eb68b:	jmp    ebcc6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d6>
   eb690:	lea    0xb0(%rsp),%r13
   eb698:	cmp    $0x2,%rax
   eb69c:	jb     eb6a6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x11b6>
   eb69e:	mov    0xb0(%rsp),%r13
   eb6a6:	movq   $0x1,0x80(%rsp)
   eb6b2:	mov    %r14,%rcx
   eb6b5:	shl    $0x5,%rcx
   eb6b9:	lea    0x88(%rsp),%rax
   eb6c1:	pxor   %xmm0,%xmm0
   eb6c5:	movdqu %xmm0,(%rax)
   eb6c9:	movq   $0x0,0x10(%rax)
   eb6d1:	mov    %rcx,0x100(%rsp)
   eb6d9:	lea    (%rcx,%rcx,2),%rsi
   eb6dd:	cmp    0x248(%rsp),%r14
   eb6e5:	ja     ed503 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3013>
   eb6eb:	test   %rsi,%rsi
   eb6ee:	mov    %r15,0xa8(%rsp)
   eb6f6:	je     eb719 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1229>
   eb6f8:	mov    %rsi,%r15
   eb6fb:	mov    %rsi,%rdi
   eb6fe:	call   *%rbx
   eb700:	test   %rax,%rax
   eb703:	je     ed4f3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3003>
   eb709:	mov    %rax,%rbx
   eb70c:	mov    %r14,%rax
   eb70f:	mov    0xa8(%rsp),%r15
   eb717:	jmp    eb720 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1230>
   eb719:	mov    $0x8,%ebx
   eb71e:	xor    %eax,%eax
   eb720:	mov    %rax,0x1f0(%rsp)
   eb728:	mov    %rbx,0x1f8(%rsp)
   eb730:	movq   $0x0,0x200(%rsp)
   eb73c:	test   %r14,%r14
   eb73f:	je     ebc5f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x176f>
   eb745:	mov    %r13,%rdx
   eb748:	add    %r13,0x100(%rsp)
   eb750:	xor    %r13d,%r13d
   eb753:	mov    0xa0(%rsp),%r14
   eb75b:	jmp    eb7e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x12f4>
   eb760:	add    $0xfffffffffffffff8,%r14
   eb764:	add    $0x8,%r13
   eb768:	lea    (%rsi,%rsi,2),%rax
   eb76c:	shl    $0x5,%rax
   eb770:	mov    %r12,(%rbx,%rax,1)
   eb774:	mov    %r13,0x8(%rbx,%rax,1)
   eb779:	mov    %r14,0x10(%rbx,%rax,1)
   eb77e:	mov    %r12,0x18(%rbx,%rax,1)
   eb783:	movq   $0x1,0x20(%rbx,%rax,1)
   eb78c:	movq   $0x0,0x28(%rbx,%rax,1)
   eb795:	mov    %r15,0x48(%rbx,%rax,1)
   eb79a:	mov    %rbp,0x50(%rbx,%rax,1)
   eb79f:	movq   $0x0,0x58(%rbx,%rax,1)
   eb7a8:	inc    %rsi
   eb7ab:	mov    %rsi,0x200(%rsp)
   eb7b3:	mov    0x108(%rsp),%rdx
   eb7bb:	cmp    0x100(%rsp),%rdx
   eb7c3:	mov    0xa8(%rsp),%r15
   eb7cb:	mov    0xf0(%rsp),%r12
   eb7d3:	mov    0xa0(%rsp),%r14
   eb7db:	mov    %rsi,%r13
   eb7de:	je     eba3a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x154a>
   eb7e4:	mov    %r13,0x18(%rsp)
   eb7e9:	jmp    eb804 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1314>
   eb7eb:	mov    0x108(%rsp),%rax
   eb7f3:	mov    %rax,%rdx
   eb7f6:	cmp    0x100(%rsp),%rax
   eb7fe:	je     eba3a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x154a>
   eb804:	mov    (%rdx),%r8
   eb807:	mov    0x8(%rdx),%rsi
   eb80b:	mov    0x10(%rdx),%rbp
   eb80f:	shl    $0x5,%rbp
   eb813:	test   %r8,%r8
   eb816:	mov    %rdx,%rax
   eb819:	cmove  %rsi,%rax
   eb81d:	lea    0x20(%rdx),%rcx
   eb821:	mov    %rcx,0x108(%rsp)
   eb829:	mov    $0x20,%ecx
   eb82e:	cmovne %rcx,%rbp
   eb832:	test   %rbp,%rbp
   eb835:	je     eb7eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x12fb>
   eb837:	add    %rax,%rbp
   eb83a:	mov    (%rax),%rdi
   eb83d:	dec    %rdi
   eb840:	je     eb8b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x13c7>
   eb842:	mov    %rdi,%rcx
   eb845:	shr    $0x7,%rcx
   eb849:	and    $0x7f,%edi
   eb84c:	cmp    $0x1,%rdi
   eb850:	sbb    $0xffffffffffffffff,%rcx
   eb854:	mov    $0x8,%edi
   eb859:	cmp    $0x2,%rcx
   eb85d:	jb     eb87a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x138a>
   eb85f:	dec    %rcx
   eb862:	bsr    %rcx,%rcx
   eb866:	not    %ecx
   eb868:	mov    $0xffffffffffffffff,%rdi
   eb86f:	shr    %cl,%rdi
   eb872:	lea    0x10(,%rdi,8),%rdi
   eb87a:	test   %r8,%r8
   eb87d:	mov    %rdx,%r9
   eb880:	cmove  %rsi,%r9
   eb884:	mov    0x18(%r9),%rcx
   eb888:	mov    %rcx,%r14
   eb88b:	cmp    $0x11,%rcx
   eb88f:	jb     eb895 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x13a5>
   eb891:	mov    0x10(%r9),%r14
   eb895:	cmp    %rdi,%r14
   eb898:	jb     ed30a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e1a>
   eb89e:	lea    0x20(%rsi),%r15
   eb8a2:	test   %r8,%r8
   eb8a5:	lea    0x20(%rdx),%r9
   eb8a9:	cmovne %r9,%r15
   eb8ad:	sub    %rdi,%r14
   eb8b0:	je     eb907 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1417>
   eb8b2:	jmp    eb9ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14da>
   eb8b7:	add    $0x20,%rsi
   eb8bb:	test   %r8,%r8
   eb8be:	cmovne 0x108(%rsp),%rsi
   eb8c7:	mov    %rsi,%r15
   eb8ca:	jmp    eb907 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1417>
   eb8cc:	nopl   0x0(%rax)
   eb8d0:	mov    $0x1,%r13d
   eb8d6:	xor    %r14d,%r14d
   eb8d9:	lea    0x20(%rdx),%r15
   eb8dd:	mov    %rdx,%rax
   eb8e0:	test   %r14,%r14
   eb8e3:	je     eb907 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1417>
   eb8e5:	jmp    eb9ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14fa>
   eb8ea:	nopw   0x0(%rax,%rax,1)
   eb8f0:	mov    0x28(%rax),%r13
   eb8f4:	add    %rdi,%r13
   eb8f7:	lea    0x20(%rdx),%r15
   eb8fb:	mov    %rdx,%rax
   eb8fe:	test   %r14,%r14
   eb901:	jne    eb9ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14fa>
   eb907:	mov    %r15,%rdx
   eb90a:	cmp    %rbp,%r15
   eb90d:	mov    0xa8(%rsp),%r15
   eb915:	mov    0xf0(%rsp),%r12
   eb91d:	mov    0xa0(%rsp),%r14
   eb925:	mov    0x18(%rsp),%r13
   eb92a:	je     eb7eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x12fb>
   eb930:	mov    (%rdx),%rsi
   eb933:	dec    %rsi
   eb936:	je     eb8d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x13e0>
   eb938:	mov    %rsi,%rcx
   eb93b:	shr    $0x7,%rcx
   eb93f:	and    $0x7f,%esi
   eb942:	cmp    $0x1,%rsi
   eb946:	sbb    $0xffffffffffffffff,%rcx
   eb94a:	mov    $0x8,%edi
   eb94f:	cmp    $0x2,%rcx
   eb953:	jae    eb980 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1490>
   eb955:	mov    0x38(%rax),%rcx
   eb959:	mov    %rcx,%rsi
   eb95c:	cmp    $0x11,%rcx
   eb960:	jb     eb9a8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14b8>
   eb962:	mov    0x30(%rax),%rsi
   eb966:	mov    %rsi,%r14
   eb969:	sub    %rdi,%r14
   eb96c:	jae    eb9b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14c4>
   eb96e:	jmp    ed2ac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dbc>
   eb973:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb980:	dec    %rcx
   eb983:	bsr    %rcx,%rcx
   eb987:	not    %ecx
   eb989:	mov    $0xffffffffffffffff,%rsi
   eb990:	shr    %cl,%rsi
   eb993:	lea    0x10(,%rsi,8),%rdi
   eb99b:	mov    0x38(%rax),%rcx
   eb99f:	mov    %rcx,%rsi
   eb9a2:	cmp    $0x11,%rcx
   eb9a6:	jae    eb962 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1472>
   eb9a8:	mov    %rsi,%r14
   eb9ab:	sub    %rdi,%r14
   eb9ae:	jb     ed2ac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dbc>
   eb9b4:	cmp    $0x11,%rcx
   eb9b8:	jae    eb8f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1400>
   eb9be:	add    $0x28,%rax
   eb9c2:	mov    %rax,%r13
   eb9c5:	jmp    eb8f4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1404>
   eb9ca:	lea    0x8(%rdx),%r13
   eb9ce:	test   %r8,%r8
   eb9d1:	cmove  %rsi,%rdx
   eb9d5:	lea    0x8(%rsi),%rax
   eb9d9:	cmove  %rax,%r13
   eb9dd:	cmp    $0x11,%rcx
   eb9e1:	jb     eb9e7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14f7>
   eb9e3:	mov    0x8(%rdx),%r13
   eb9e7:	add    %rdi,%r13
   eb9ea:	cmp    $0x7,%r14
   eb9ee:	jbe    ed350 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e60>
   eb9f4:	mov    0x0(%r13),%r12
   eb9f8:	mov    0x18(%rsp),%rsi
   eb9fd:	cmp    0x1f0(%rsp),%rsi
   eba05:	jne    eb760 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1270>
   eba0b:	mov    $0x1,%edx
   eba10:	mov    $0x8,%ecx
   eba15:	mov    $0x60,%r8d
   eba1b:	lea    0x1f0(%rsp),%rdi
   eba23:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   eba28:	mov    0x1f8(%rsp),%rbx
   eba30:	mov    0x18(%rsp),%rsi
   eba35:	jmp    eb760 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1270>
   eba3a:	mov    0x1f8(%rsp),%rbx
   eba42:	mov    %r13,%rax
   eba45:	shr    $1,%rax
   eba48:	je     ebd75 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1885>
   eba4e:	mov    0xe8(%rsp),%rbp
   eba56:	jmp    eba61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1571>
   eba58:	test   %rax,%rax
   eba5b:	je     ebd7d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x188d>
   eba61:	lea    -0x2(,%rax,2),%rcx
   eba69:	lea    -0x1(,%rax,2),%rdi
   eba71:	dec    %rax
   eba74:	add    $0x2,%rcx
   eba78:	mov    %rax,%rdx
   eba7b:	cmp    %r13,%rcx
   eba7e:	jae    ebb8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x169f>
   eba84:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   eba90:	lea    (%rcx,%rcx,2),%rcx
   eba94:	shl    $0x5,%rcx
   eba98:	lea    (%rdi,%rdi,2),%rsi
   eba9c:	shl    $0x5,%rsi
   ebaa0:	mov    (%rbx,%rcx,1),%rcx
   ebaa4:	cmp    (%rbx,%rsi,1),%rcx
   ebaa8:	adc    $0x0,%rdi
   ebaac:	cmp    %r13,%rdi
   ebaaf:	jae    ed442 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f52>
   ebab5:	lea    (%rdi,%rdi,2),%rcx
   ebab9:	shl    $0x5,%rcx
   ebabd:	lea    (%rdx,%rdx,2),%rdx
   ebac1:	shl    $0x5,%rdx
   ebac5:	mov    (%rbx,%rcx,1),%rsi
   ebac9:	cmp    (%rbx,%rdx,1),%rsi
   ebacd:	jae    eba58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1568>
   ebacf:	add    %rbx,%rcx
   ebad2:	add    %rbx,%rdx
   ebad5:	movups 0x50(%rdx),%xmm0
   ebad9:	movaps %xmm0,0x70(%rsp)
   ebade:	movups 0x40(%rdx),%xmm0
   ebae2:	movaps %xmm0,0x60(%rsp)
   ebae7:	movups (%rdx),%xmm0
   ebaea:	movups 0x10(%rdx),%xmm1
   ebaee:	movups 0x20(%rdx),%xmm2
   ebaf2:	movups 0x30(%rdx),%xmm3
   ebaf6:	movaps %xmm3,0x50(%rsp)
   ebafb:	movaps %xmm2,0x40(%rsp)
   ebb00:	movaps %xmm1,0x30(%rsp)
   ebb05:	movaps %xmm0,0x20(%rsp)
   ebb0a:	movups 0x40(%rcx),%xmm0
   ebb0e:	movups 0x50(%rcx),%xmm1
   ebb12:	movups (%rcx),%xmm2
   ebb15:	movups 0x10(%rcx),%xmm3
   ebb19:	movups 0x20(%rcx),%xmm4
   ebb1d:	movups 0x30(%rcx),%xmm5
   ebb21:	movups %xmm2,(%rdx)
   ebb24:	movups %xmm1,0x50(%rdx)
   ebb28:	movups %xmm0,0x40(%rdx)
   ebb2c:	movups %xmm5,0x30(%rdx)
   ebb30:	movups %xmm4,0x20(%rdx)
   ebb34:	movups %xmm3,0x10(%rdx)
   ebb38:	movaps 0x70(%rsp),%xmm0
   ebb3d:	movups %xmm0,0x50(%rcx)
   ebb41:	movaps 0x60(%rsp),%xmm0
   ebb46:	movups %xmm0,0x40(%rcx)
   ebb4a:	movdqa 0x20(%rsp),%xmm0
   ebb50:	movdqa 0x30(%rsp),%xmm1
   ebb56:	movdqa 0x40(%rsp),%xmm2
   ebb5c:	movaps 0x50(%rsp),%xmm3
   ebb61:	movups %xmm3,0x30(%rcx)
   ebb65:	movdqu %xmm2,0x20(%rcx)
   ebb6a:	movdqu %xmm1,0x10(%rcx)
   ebb6f:	movdqu %xmm0,(%rcx)
   ebb73:	lea    0x2(,%rdi,2),%rcx
   ebb7b:	mov    %rdi,%rdx
   ebb7e:	lea    0x1(,%rdi,2),%rdi
   ebb86:	cmp    %r13,%rcx
   ebb89:	jb     eba90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15a0>
   ebb8f:	cmp    %r13,%rcx
   ebb92:	jne    eba58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1568>
   ebb98:	lea    (%rdi,%rdi,2),%rcx
   ebb9c:	shl    $0x5,%rcx
   ebba0:	lea    (%rdx,%rdx,2),%rdx
   ebba4:	shl    $0x5,%rdx
   ebba8:	mov    (%rbx,%rcx,1),%rsi
   ebbac:	cmp    (%rbx,%rdx,1),%rsi
   ebbb0:	jae    eba58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1568>
   ebbb6:	add    %rbx,%rcx
   ebbb9:	add    %rbx,%rdx
   ebbbc:	movups 0x50(%rdx),%xmm0
   ebbc0:	movaps %xmm0,0x70(%rsp)
   ebbc5:	movups 0x40(%rdx),%xmm0
   ebbc9:	movaps %xmm0,0x60(%rsp)
   ebbce:	movups (%rdx),%xmm0
   ebbd1:	movups 0x10(%rdx),%xmm1
   ebbd5:	movups 0x20(%rdx),%xmm2
   ebbd9:	movups 0x30(%rdx),%xmm3
   ebbdd:	movaps %xmm3,0x50(%rsp)
   ebbe2:	movaps %xmm2,0x40(%rsp)
   ebbe7:	movaps %xmm1,0x30(%rsp)
   ebbec:	movaps %xmm0,0x20(%rsp)
   ebbf1:	movups 0x40(%rcx),%xmm0
   ebbf5:	movups 0x50(%rcx),%xmm1
   ebbf9:	movups (%rcx),%xmm2
   ebbfc:	movups 0x10(%rcx),%xmm3
   ebc00:	movups 0x20(%rcx),%xmm4
   ebc04:	movups 0x30(%rcx),%xmm5
   ebc08:	movups %xmm2,(%rdx)
   ebc0b:	movups %xmm1,0x50(%rdx)
   ebc0f:	movups %xmm0,0x40(%rdx)
   ebc13:	movups %xmm5,0x30(%rdx)
   ebc17:	movups %xmm4,0x20(%rdx)
   ebc1b:	movups %xmm3,0x10(%rdx)
   ebc1f:	movaps 0x70(%rsp),%xmm0
   ebc24:	movups %xmm0,0x50(%rcx)
   ebc28:	movaps 0x60(%rsp),%xmm0
   ebc2d:	movups %xmm0,0x40(%rcx)
   ebc31:	movdqa 0x20(%rsp),%xmm0
   ebc37:	movdqa 0x30(%rsp),%xmm1
   ebc3d:	movdqa 0x40(%rsp),%xmm2
   ebc43:	movaps 0x50(%rsp),%xmm3
   ebc48:	movups %xmm3,0x30(%rcx)
   ebc4c:	movdqu %xmm2,0x20(%rcx)
   ebc51:	movdqu %xmm1,0x10(%rcx)
   ebc56:	movdqu %xmm0,(%rcx)
   ebc5a:	jmp    eba58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1568>
   ebc5f:	xor    %r13d,%r13d
   ebc62:	mov    0xa0(%rsp),%r14
   ebc6a:	jmp    ebd7d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x188d>
   ebc6f:	call   *0x1bc33(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ebc75:	mov    $0x8,%ebx
   ebc7a:	xor    %ecx,%ecx
   ebc7c:	mov    %r13,%r14
   ebc7f:	mov    %rcx,0x220(%rsp)
   ebc87:	mov    %rbx,0x228(%rsp)
   ebc8f:	mov    %r12,0x230(%rsp)
   ebc97:	mov    0x18(%rsp),%rax
   ebc9c:	mov    %rax,0x238(%rsp)
   ebca4:	mov    0x240(%rsp),%rdi
   ebcac:	mov    0xf0(%rsp),%r12
   ebcb4:	mov    %r12d,%esi
   ebcb7:	mov    0x10(%rsp),%rbx
   ebcbc:	mov    %ebx,%edx
   ebcbe:	mov    %r14,%rcx
   ebcc1:	call   f11f0 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::insert>
   ebcc6:	movdqa 0x220(%rsp),%xmm0
   ebccf:	movdqa 0x230(%rsp),%xmm1
   ebcd8:	movdqa %xmm1,0x30(%rsp)
   ebcde:	movdqa %xmm0,0x20(%rsp)
   ebce4:	mov    %r14,0x40(%rsp)
   ebce9:	mov    %r12d,0x48(%rsp)
   ebcee:	mov    %ebx,0x4c(%rsp)
   ebcf2:	mov    0x168(%rsp),%rbx
   ebcfa:	cmp    0x158(%rsp),%rbx
   ebd02:	jne    ebd11 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1821>
   ebd04:	lea    0x158(%rsp),%rdi
   ebd0c:	call   e1700 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   ebd11:	mov    0x160(%rsp),%rax
   ebd19:	lea    (%rbx,%rbx,2),%rcx
   ebd1d:	shl    $0x4,%rcx
   ebd21:	movdqa 0x20(%rsp),%xmm0
   ebd27:	movdqa 0x30(%rsp),%xmm1
   ebd2d:	movdqa 0x40(%rsp),%xmm2
   ebd33:	movdqu %xmm2,0x20(%rax,%rcx,1)
   ebd39:	movdqu %xmm1,0x10(%rax,%rcx,1)
   ebd3f:	movdqu %xmm0,(%rax,%rcx,1)
   ebd44:	inc    %rbx
   ebd47:	mov    %rbx,0x168(%rsp)
   ebd4f:	mov    %rbp,%rbx
   ebd52:	sub    %r15,%rbx
   ebd55:	shr    $0x3,%rbx
   ebd59:	movabs $0x6db6db6db6db6db7,%rax
   ebd63:	imul   %rax,%rbx
   ebd67:	cmp    %r15,%rbp
   ebd6a:	jne    eacb9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7c9>
   ebd70:	jmp    ec6a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21b4>
   ebd75:	mov    0xe8(%rsp),%rbp
   ebd7d:	mov    0x1f0(%rsp),%rax
   ebd85:	mov    %rax,0x100(%rsp)
   ebd8d:	jmp    ebdb5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18c5>
   ebd8f:	mov    %rsi,(%rcx)
   ebd92:	mov    0x108(%rsp),%rax
   ebd9a:	mov    %rax,0x80(%rsp)
   ebda2:	xor    %eax,%eax
   ebda4:	mov    0xa0(%rsp),%r14
   ebdac:	test   %rax,%rax
   ebdaf:	jne    ec985 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2495>
   ebdb5:	test   %r13,%r13
   ebdb8:	je     ec582 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2092>
   ebdbe:	lea    0x8(%rbx),%r12
   ebdc2:	cmpq   $0x0,(%r12)
   ebdc7:	je     ebde3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18f3>
   ebdc9:	mov    %r12,%rdi
   ebdcc:	call   e2770 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   ebdd1:	cmp    $0x1,%rax
   ebdd5:	je     ec094 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ba4>
   ebddb:	movq   $0x0,(%r12)
   ebde3:	mov    0x48(%rbx),%r8
   ebde7:	test   %r8,%r8
   ebdea:	je     ec074 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b84>
   ebdf0:	cmp    0x50(%rbx),%r8
   ebdf4:	je     ec074 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b84>
   ebdfa:	lea    0x20(%r8),%rax
   ebdfe:	mov    %rax,0x48(%rbx)
   ebe02:	mov    0x58(%rbx),%rax
   ebe06:	mov    (%r8),%r9
   ebe09:	dec    %r9
   ebe0c:	cmp    %r9,%rax
   ebe0f:	jne    ebe1e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x192e>
   ebe11:	mov    $0x1,%edi
   ebe16:	xor    %r9d,%r9d
   ebe19:	jmp    ebf1b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a2b>
   ebe1e:	mov    %r9,%rcx
   ebe21:	shr    $0x7,%rcx
   ebe25:	test   %rax,%rax
   ebe28:	je     ebe83 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1993>
   ebe2a:	mov    %r9d,%edx
   ebe2d:	and    $0x7f,%edx
   ebe30:	cmp    $0x1,%rdx
   ebe34:	mov    %rcx,%rsi
   ebe37:	sbb    $0xffffffffffffffff,%rsi
   ebe3b:	cmp    $0x2,%rsi
   ebe3f:	jb     ed482 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f92>
   ebe45:	mov    0x18(%r8),%r10
   ebe49:	mov    %r10,%rdx
   ebe4c:	cmp    $0x11,%r10
   ebe50:	jb     ebe56 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1966>
   ebe52:	mov    0x10(%r8),%rdx
   ebe56:	lea    0x8(,%rsi,8),%r14
   ebe5e:	cmp    %rdx,%r14
   ebe61:	ja     ed2fc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e0c>
   ebe67:	mov    %rax,%rdi
   ebe6a:	shr    $0x7,%rdi
   ebe6e:	cmp    %rsi,%rdi
   ebe71:	jae    ed48b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f9b>
   ebe77:	cmp    $0x11,%r10
   ebe7b:	jb     ebe87 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1997>
   ebe7d:	mov    0x8(%r8),%rdx
   ebe81:	jmp    ebe8b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x199b>
   ebe83:	xor    %esi,%esi
   ebe85:	jmp    ebe90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19a0>
   ebe87:	lea    0x8(%r8),%rdx
   ebe8b:	mov    0x8(%rdx,%rdi,8),%rsi
   ebe90:	and    $0x7f,%r9d
   ebe94:	cmp    $0x1,%r9
   ebe98:	sbb    $0xffffffffffffffff,%rcx
   ebe9c:	mov    $0x8,%edi
   ebea1:	cmp    $0x2,%rcx
   ebea5:	jae    ebec5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19d5>
   ebea7:	mov    0x18(%r8),%rcx
   ebeab:	mov    %rcx,%r14
   ebeae:	cmp    $0x11,%rcx
   ebeb2:	jb     ebeed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19fd>
   ebeb4:	mov    0x10(%r8),%r14
   ebeb8:	mov    %r14,%rdx
   ebebb:	sub    %rdi,%rdx
   ebebe:	jae    ebef9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a09>
   ebec0:	jmp    ed2c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd3>
   ebec5:	dec    %rcx
   ebec8:	bsr    %rcx,%rcx
   ebecc:	not    %ecx
   ebece:	mov    $0xffffffffffffffff,%rdx
   ebed5:	shr    %cl,%rdx
   ebed8:	lea    0x10(,%rdx,8),%rdi
   ebee0:	mov    0x18(%r8),%rcx
   ebee4:	mov    %rcx,%r14
   ebee7:	cmp    $0x11,%rcx
   ebeeb:	jae    ebeb4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19c4>
   ebeed:	mov    %r14,%rdx
   ebef0:	sub    %rdi,%rdx
   ebef3:	jb     ed2c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd3>
   ebef9:	mov    %rdx,%r9
   ebefc:	sub    %rsi,%r9
   ebeff:	jb     ed2d2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2de2>
   ebf05:	cmp    $0x11,%rcx
   ebf09:	jb     ebf11 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a21>
   ebf0b:	mov    0x8(%r8),%r8
   ebf0f:	jmp    ebf15 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a25>
   ebf11:	add    $0x8,%r8
   ebf15:	add    %rsi,%rdi
   ebf18:	add    %r8,%rdi
   ebf1b:	movq   $0x0,0x58(%rbx)
   ebf23:	mov    %rdi,0x8(%rbx)
   ebf27:	mov    %r9,0x10(%rbx)
   ebf2b:	movq   $0x0,0x18(%rbx)
   ebf33:	mov    %rax,0x20(%rbx)
   ebf37:	mov    %r12,%rdi
   ebf3a:	call   e2770 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   ebf3f:	cmp    $0x1,%rax
   ebf43:	je     ec094 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ba4>
   ebf49:	movq   $0x0,0x8(%rbx)
   ebf51:	mov    0x48(%rbx),%r8
   ebf55:	test   %r8,%r8
   ebf58:	je     ec074 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b84>
   ebf5e:	cmp    0x50(%rbx),%r8
   ebf62:	je     ec074 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b84>
   ebf68:	lea    0x20(%r8),%rax
   ebf6c:	mov    %rax,0x48(%rbx)
   ebf70:	mov    0x58(%rbx),%rax
   ebf74:	mov    (%r8),%rcx
   ebf77:	dec    %rcx
   ebf7a:	mov    $0x1,%edi
   ebf7f:	mov    $0x0,%r9d
   ebf85:	cmp    %rcx,%rax
   ebf88:	je     ebf1b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a2b>
   ebf8a:	mov    %rcx,%r9
   ebf8d:	shr    $0x7,%r9
   ebf91:	test   %rax,%rax
   ebf94:	je     ebfee <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1afe>
   ebf96:	mov    %ecx,%edx
   ebf98:	and    $0x7f,%edx
   ebf9b:	cmp    $0x1,%rdx
   ebf9f:	mov    %r9,%rsi
   ebfa2:	sbb    $0xffffffffffffffff,%rsi
   ebfa6:	cmp    $0x1,%rsi
   ebfaa:	jbe    ed482 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f92>
   ebfb0:	mov    0x18(%r8),%r10
   ebfb4:	mov    %r10,%rdx
   ebfb7:	cmp    $0x11,%r10
   ebfbb:	jb     ebfc1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ad1>
   ebfbd:	mov    0x10(%r8),%rdx
   ebfc1:	lea    0x8(,%rsi,8),%r14
   ebfc9:	cmp    %rdx,%r14
   ebfcc:	ja     ed2fc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e0c>
   ebfd2:	mov    %rax,%rdi
   ebfd5:	shr    $0x7,%rdi
   ebfd9:	cmp    %rsi,%rdi
   ebfdc:	jae    ed48b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f9b>
   ebfe2:	cmp    $0x11,%r10
   ebfe6:	jb     ebff2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b02>
   ebfe8:	mov    0x8(%r8),%rdx
   ebfec:	jmp    ebff6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b06>
   ebfee:	xor    %esi,%esi
   ebff0:	jmp    ebffb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b0b>
   ebff2:	lea    0x8(%r8),%rdx
   ebff6:	mov    0x8(%rdx,%rdi,8),%rsi
   ebffb:	and    $0x7f,%ecx
   ebffe:	cmp    $0x1,%rcx
   ec002:	sbb    $0xffffffffffffffff,%r9
   ec006:	mov    $0x8,%edi
   ec00b:	cmp    $0x2,%r9
   ec00f:	jae    ec02f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b3f>
   ec011:	mov    0x18(%r8),%rcx
   ec015:	mov    %rcx,%r14
   ec018:	cmp    $0x11,%rcx
   ec01c:	jb     ec057 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b67>
   ec01e:	mov    0x10(%r8),%r14
   ec022:	mov    %r14,%rdx
   ec025:	sub    %rdi,%rdx
   ec028:	jae    ec063 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b73>
   ec02a:	jmp    ed2c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd3>
   ec02f:	dec    %r9
   ec032:	bsr    %r9,%rcx
   ec036:	not    %ecx
   ec038:	mov    $0xffffffffffffffff,%rdx
   ec03f:	shr    %cl,%rdx
   ec042:	lea    0x10(,%rdx,8),%rdi
   ec04a:	mov    0x18(%r8),%rcx
   ec04e:	mov    %rcx,%r14
   ec051:	cmp    $0x11,%rcx
   ec055:	jae    ec01e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b2e>
   ec057:	mov    %r14,%rdx
   ec05a:	sub    %rdi,%rdx
   ec05d:	jb     ed2c3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd3>
   ec063:	mov    %rdx,%r9
   ec066:	sub    %rsi,%r9
   ec069:	jae    ebf05 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a15>
   ec06f:	jmp    ed2d2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2de2>
   ec074:	lea    0x28(%rbx),%r14
   ec078:	cmpq   $0x0,(%r14)
   ec07c:	je     ec1c7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1cd7>
   ec082:	mov    %r14,%rdi
   ec085:	call   e2770 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   ec08a:	cmp    $0x1,%rax
   ec08e:	jne    ec1c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1cd0>
   ec094:	mov    (%rbx),%r12
   ec097:	mov    %rdx,(%rbx)
   ec09a:	cmp    $0x3,%r13
   ec09e:	jb     ec220 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d30>
   ec0a4:	mov    $0x1,%eax
   ec0a9:	mov    $0x2,%edx
   ec0ae:	xor    %ecx,%ecx
   ec0b0:	lea    (%rdx,%rdx,2),%rdx
   ec0b4:	shl    $0x5,%rdx
   ec0b8:	lea    (%rax,%rax,2),%rsi
   ec0bc:	shl    $0x5,%rsi
   ec0c0:	mov    (%rbx,%rdx,1),%rdx
   ec0c4:	cmp    (%rbx,%rsi,1),%rdx
   ec0c8:	mov    %rax,%rdi
   ec0cb:	adc    $0x0,%rdi
   ec0cf:	cmp    %r13,%rdi
   ec0d2:	jae    ed456 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f66>
   ec0d8:	cmp    %r13,%rcx
   ec0db:	jae    ed45b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f6b>
   ec0e1:	lea    (%rdi,%rdi,2),%rax
   ec0e5:	shl    $0x5,%rax
   ec0e9:	add    %rbx,%rax
   ec0ec:	lea    (%rcx,%rcx,2),%rcx
   ec0f0:	shl    $0x5,%rcx
   ec0f4:	mov    (%rax),%rdx
   ec0f7:	cmp    (%rbx,%rcx,1),%rdx
   ec0fb:	jae    ec2ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e0f>
   ec101:	add    %rbx,%rcx
   ec104:	movups 0x50(%rcx),%xmm0
   ec108:	movaps %xmm0,0x70(%rsp)
   ec10d:	movups 0x40(%rcx),%xmm0
   ec111:	movaps %xmm0,0x60(%rsp)
   ec116:	movups (%rcx),%xmm0
   ec119:	movups 0x10(%rcx),%xmm1
   ec11d:	movups 0x20(%rcx),%xmm2
   ec121:	movups 0x30(%rcx),%xmm3
   ec125:	movaps %xmm3,0x50(%rsp)
   ec12a:	movaps %xmm2,0x40(%rsp)
   ec12f:	movaps %xmm1,0x30(%rsp)
   ec134:	movaps %xmm0,0x20(%rsp)
   ec139:	movups 0x40(%rax),%xmm0
   ec13d:	movups 0x50(%rax),%xmm1
   ec141:	movups (%rax),%xmm2
   ec144:	movups 0x10(%rax),%xmm3
   ec148:	movups 0x20(%rax),%xmm4
   ec14c:	movups 0x30(%rax),%xmm5
   ec150:	movups %xmm2,(%rcx)
   ec153:	movups %xmm1,0x50(%rcx)
   ec157:	movups %xmm0,0x40(%rcx)
   ec15b:	movups %xmm5,0x30(%rcx)
   ec15f:	movups %xmm4,0x20(%rcx)
   ec163:	movups %xmm3,0x10(%rcx)
   ec167:	movaps 0x70(%rsp),%xmm0
   ec16c:	movups %xmm0,0x50(%rax)
   ec170:	movaps 0x60(%rsp),%xmm0
   ec175:	movups %xmm0,0x40(%rax)
   ec179:	movdqa 0x20(%rsp),%xmm0
   ec17f:	movdqa 0x30(%rsp),%xmm1
   ec185:	movdqa 0x40(%rsp),%xmm2
   ec18b:	movaps 0x50(%rsp),%xmm3
   ec190:	movups %xmm3,0x30(%rax)
   ec194:	movdqu %xmm2,0x20(%rax)
   ec199:	movdqu %xmm1,0x10(%rax)
   ec19e:	movdqu %xmm0,(%rax)
   ec1a2:	lea    0x1(,%rdi,2),%rax
   ec1aa:	lea    0x2(,%rdi,2),%rdx
   ec1b2:	mov    %rdi,%rcx
   ec1b5:	cmp    %r13,%rdx
   ec1b8:	jb     ec0b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1bc0>
   ec1be:	jmp    ec22c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d3c>
   ec1c0:	movq   $0x0,(%r14)
   ec1c7:	mov    (%rbx),%r12
   ec1ca:	dec    %r13
   ec1cd:	lea    0x0(,%r13,2),%rax
   ec1d5:	add    %r13,%rax
   ec1d8:	shl    $0x5,%rax
   ec1dc:	movdqu (%rbx,%rax,1),%xmm0
   ec1e1:	movdqu 0x10(%rbx,%rax,1),%xmm1
   ec1e7:	movdqu 0x20(%rbx,%rax,1),%xmm2
   ec1ed:	movups 0x30(%rbx,%rax,1),%xmm3
   ec1f2:	movups 0x40(%rbx,%rax,1),%xmm4
   ec1f7:	movups 0x50(%rbx,%rax,1),%xmm5
   ec1fc:	movups %xmm5,0x50(%rbx)
   ec200:	movups %xmm4,0x40(%rbx)
   ec204:	movups %xmm3,0x30(%rbx)
   ec208:	movdqu %xmm2,0x20(%rbx)
   ec20d:	movdqu %xmm1,0x10(%rbx)
   ec212:	movdqu %xmm0,(%rbx)
   ec216:	cmp    $0x3,%r13
   ec21a:	jae    ec0a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1bb4>
   ec220:	mov    $0x2,%edx
   ec225:	mov    $0x1,%eax
   ec22a:	xor    %edi,%edi
   ec22c:	cmp    %r13,%rdx
   ec22f:	jne    ec2ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e0f>
   ec235:	cmp    %r13,%rdi
   ec238:	jae    ed4c2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fd2>
   ec23e:	lea    (%rax,%rax,2),%rax
   ec242:	shl    $0x5,%rax
   ec246:	add    %rbx,%rax
   ec249:	lea    (%rdi,%rdi,2),%rcx
   ec24d:	shl    $0x5,%rcx
   ec251:	mov    (%rax),%rdx
   ec254:	cmp    (%rbx,%rcx,1),%rdx
   ec258:	jae    ec2ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e0f>
   ec25e:	add    %rbx,%rcx
   ec261:	movups 0x50(%rcx),%xmm0
   ec265:	movaps %xmm0,0x70(%rsp)
   ec26a:	movups 0x40(%rcx),%xmm0
   ec26e:	movaps %xmm0,0x60(%rsp)
   ec273:	movups (%rcx),%xmm0
   ec276:	movups 0x10(%rcx),%xmm1
   ec27a:	movups 0x20(%rcx),%xmm2
   ec27e:	movups 0x30(%rcx),%xmm3
   ec282:	movaps %xmm3,0x50(%rsp)
   ec287:	movaps %xmm2,0x40(%rsp)
   ec28c:	movaps %xmm1,0x30(%rsp)
   ec291:	movaps %xmm0,0x20(%rsp)
   ec296:	movups 0x40(%rax),%xmm0
   ec29a:	movups 0x50(%rax),%xmm1
   ec29e:	movups (%rax),%xmm2
   ec2a1:	movups 0x10(%rax),%xmm3
   ec2a5:	movups 0x20(%rax),%xmm4
   ec2a9:	movups 0x30(%rax),%xmm5
   ec2ad:	movups %xmm2,(%rcx)
   ec2b0:	movups %xmm1,0x50(%rcx)
   ec2b4:	movups %xmm0,0x40(%rcx)
   ec2b8:	movups %xmm5,0x30(%rcx)
   ec2bc:	movups %xmm4,0x20(%rcx)
   ec2c0:	movups %xmm3,0x10(%rcx)
   ec2c4:	movaps 0x70(%rsp),%xmm0
   ec2c9:	movups %xmm0,0x50(%rax)
   ec2cd:	movaps 0x60(%rsp),%xmm0
   ec2d2:	movups %xmm0,0x40(%rax)
   ec2d6:	movdqa 0x20(%rsp),%xmm0
   ec2dc:	movdqa 0x30(%rsp),%xmm1
   ec2e2:	movdqa 0x40(%rsp),%xmm2
   ec2e8:	movaps 0x50(%rsp),%xmm3
   ec2ed:	movups %xmm3,0x30(%rax)
   ec2f1:	movdqu %xmm2,0x20(%rax)
   ec2f6:	movdqu %xmm1,0x10(%rax)
   ec2fb:	movdqu %xmm0,(%rax)
   ec2ff:	mov    0x80(%rsp),%rdx
   ec307:	test   %rdx,%rdx
   ec30a:	je     ec8c8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23d8>
   ec310:	mov    %rdx,%rcx
   ec313:	inc    %rcx
   ec316:	je     ece3b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x294b>
   ec31c:	mov    %r13,0x18(%rsp)
   ec321:	mov    %edx,%eax
   ec323:	and    $0x7f,%eax
   ec326:	cmp    $0x1,%eax
   ec329:	jne    ec366 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e76>
   ec32b:	lea    0x80(%rsp),%rdi
   ec333:	mov    %r12,%rsi
   ec336:	call   f0b10 <<tk_train::trainers::bpe::positions::Leaf>::push_restart>
   ec33b:	mov    0xe8(%rsp),%rbp
   ec343:	mov    0xf0(%rsp),%r12
   ec34b:	mov    0xa0(%rsp),%r14
   ec353:	mov    0x18(%rsp),%r13
   ec358:	test   %rax,%rax
   ec35b:	je     ebdb5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18c5>
   ec361:	jmp    ec985 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2495>
   ec366:	mov    0x98(%rsp),%rax
   ec36e:	mov    %rax,%rdx
   ec371:	cmp    $0x11,%rax
   ec375:	jb     ec37f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e8f>
   ec377:	mov    0x90(%rsp),%rdx
   ec37f:	mov    $0x8,%r14d
   ec385:	cmp    $0x7,%rdx
   ec389:	jbe    ed363 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e73>
   ec38f:	mov    %rcx,0x108(%rsp)
   ec397:	lea    0x88(%rsp),%rcx
   ec39f:	cmp    $0x11,%rax
   ec3a3:	jb     ec3ad <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ebd>
   ec3a5:	mov    0x88(%rsp),%rcx
   ec3ad:	mov    %r12,%rbp
   ec3b0:	sub    (%rcx),%rbp
   ec3b3:	jb     ec922 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2432>
   ec3b9:	mov    $0x7f,%eax
   ec3be:	bsr    %rbp,%rax
   ec3c2:	xor    $0x3f,%eax
   ec3c5:	mov    $0x40,%cl
   ec3c7:	sub    %al,%cl
   ec3c9:	movzbl %cl,%eax
   ec3cc:	imul   $0x93,%eax,%r13d
   ec3d3:	shr    $0xa,%r13d
   ec3d7:	imul   $0xffffffb7,%eax,%eax
   ec3da:	xor    %ecx,%ecx
   ec3dc:	cmp    $0x25,%al
   ec3de:	setae  %cl
   ec3e1:	lea    (%rcx,%r13,1),%eax
   ec3e5:	cmp    $0x1,%eax
   ec3e8:	adc    %ecx,%r13d
   ec3eb:	lea    0x80(%rsp),%rdi
   ec3f3:	mov    %r13,%rsi
   ec3f6:	call   f1000 <<tk_train::trainers::bpe::positions::Leaf>::reserve>
   ec3fb:	test   %rax,%rax
   ec3fe:	jne    ec985 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2495>
   ec404:	mov    0x98(%rsp),%rax
   ec40c:	mov    %rax,%r15
   ec40f:	cmp    $0x11,%rax
   ec413:	jb     ec420 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f30>
   ec415:	mov    0x90(%rsp),%r15
   ec41d:	nopl   (%rax)
   ec420:	cmp    $0x11,%rax
   ec424:	jb     ec460 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f70>
   ec426:	mov    0x90(%rsp),%rcx
   ec42e:	cmp    %rax,%rcx
   ec431:	je     ec48c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f9c>
   ec433:	mov    0x88(%rsp),%rax
   ec43b:	movb   $0x0,(%rax,%rcx,1)
   ec43f:	inc    %rcx
   ec442:	mov    %rcx,0x90(%rsp)
   ec44a:	mov    0x98(%rsp),%rax
   ec452:	dec    %r13
   ec455:	jne    ec420 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f30>
   ec457:	jmp    ec4ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fda>
   ec459:	nopl   0x0(%rax)
   ec460:	lea    0x88(%rsp),%rcx
   ec468:	lea    0x98(%rsp),%rdx
   ec470:	cmp    $0x10,%rax
   ec474:	je     ec4a3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fb3>
   ec476:	movb   $0x0,(%rcx,%rax,1)
   ec47a:	incq   (%rdx)
   ec47d:	mov    0x98(%rsp),%rax
   ec485:	dec    %r13
   ec488:	jne    ec420 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f30>
   ec48a:	jmp    ec4ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fda>
   ec48c:	lea    0x88(%rsp),%rdi
   ec494:	call   f1160 <<smallvec::SmallVec<[u8; 16]>>::reserve_one_unchecked>
   ec499:	mov    0x90(%rsp),%rcx
   ec4a1:	jmp    ec433 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f43>
   ec4a3:	lea    0x88(%rsp),%rdi
   ec4ab:	call   f1160 <<smallvec::SmallVec<[u8; 16]>>::reserve_one_unchecked>
   ec4b0:	mov    0x88(%rsp),%rcx
   ec4b8:	mov    0x90(%rsp),%rax
   ec4c0:	lea    0x90(%rsp),%rdx
   ec4c8:	jmp    ec476 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f86>
   ec4ca:	mov    %r12,%rsi
   ec4cd:	lea    0x88(%rsp),%rcx
   ec4d5:	cmp    $0x11,%rax
   ec4d9:	jb     ec4eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ffb>
   ec4db:	mov    0x88(%rsp),%rcx
   ec4e3:	mov    0x90(%rsp),%rax
   ec4eb:	cmp    %r15,%rax
   ec4ee:	mov    0xf0(%rsp),%r12
   ec4f6:	mov    0x18(%rsp),%r13
   ec4fb:	jbe    ed411 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f21>
   ec501:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec510:	mov    %ebp,%edx
   ec512:	shr    $0x7,%rbp
   ec516:	je     ec52c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x203c>
   ec518:	or     $0x80,%dl
   ec51b:	mov    %dl,(%rcx,%r15,1)
   ec51f:	inc    %r15
   ec522:	cmp    %r15,%rax
   ec525:	jne    ec510 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2020>
   ec527:	jmp    ed411 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f21>
   ec52c:	mov    %dl,(%rcx,%r15,1)
   ec530:	mov    0x98(%rsp),%rax
   ec538:	mov    %rax,%rdx
   ec53b:	cmp    $0x11,%rax
   ec53f:	jb     ec549 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2059>
   ec541:	mov    0x90(%rsp),%rdx
   ec549:	cmp    $0x8,%rdx
   ec54d:	mov    0xa8(%rsp),%r15
   ec555:	mov    0xe8(%rsp),%rbp
   ec55d:	jb     ed345 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e55>
   ec563:	lea    0x88(%rsp),%rcx
   ec56b:	cmp    $0x11,%rax
   ec56f:	jb     ebd8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x189f>
   ec575:	mov    0x88(%rsp),%rcx
   ec57d:	jmp    ebd8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x189f>
   ec582:	cmpq   $0x0,0x100(%rsp)
   ec58b:	je     ec596 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20a6>
   ec58d:	mov    %rbx,%rdi
   ec590:	call   *0x1b312(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec596:	movaps 0x80(%rsp),%xmm0
   ec59e:	movaps %xmm0,0xf0(%rsp)
   ec5a6:	lea    0x90(%rsp),%rax
   ec5ae:	movups (%rax),%xmm0
   ec5b1:	movaps %xmm0,0x270(%rsp)
   ec5b9:	lea    0xb0(%rsp),%rdi
   ec5c1:	call   c3f10 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   ec5c6:	jmp    eb661 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1171>
   ec5cb:	mov    %rax,0x10(%rsp)
   ec5d0:	lea    0x1a8(%rsp),%rdi
   ec5d8:	call   ef9f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ec5dd:	cmpq   $0x0,0x158(%rsp)
   ec5e6:	je     ec5f3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2103>
   ec5e8:	mov    $0x8,%edi
   ec5ed:	call   *0x1b2b5(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec5f3:	test   %rbx,%rbx
   ec5f6:	je     ec68d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x219d>
   ec5fc:	xor    %r13d,%r13d
   ec5ff:	mov    %r15,0xa8(%rsp)
   ec607:	jmp    ec62c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x213c>
   ec609:	nopl   0x0(%rax)
   ec610:	mov    %r15,%r12
   ec613:	mov    0xa8(%rsp),%r15
   ec61b:	mov    %r14,%rdi
   ec61e:	call   *0x1b284(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec624:	inc    %r13
   ec627:	cmp    %rbx,%r13
   ec62a:	je     ec68d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x219d>
   ec62c:	imul   $0x38,%r13,%rcx
   ec630:	lea    (%r15,%rcx,1),%rax
   ec634:	cmpq   $0x0,(%r15,%rcx,1)
   ec639:	je     ec650 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2160>
   ec63b:	cmpq   $0x11,0x18(%rax)
   ec640:	jb     ec624 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2134>
   ec642:	mov    0x8(%rax),%r14
   ec646:	jmp    ec61b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x212b>
   ec648:	nopl   0x0(%rax,%rax,1)
   ec650:	mov    0x10(%rax),%rbp
   ec654:	test   %rbp,%rbp
   ec657:	je     ec624 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2134>
   ec659:	mov    %r12,%r15
   ec65c:	mov    0x8(%rax),%r14
   ec660:	lea    0x18(%r14),%r12
   ec664:	jmp    ec679 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2189>
   ec666:	cs nopw 0x0(%rax,%rax,1)
   ec670:	add    $0x20,%r12
   ec674:	dec    %rbp
   ec677:	je     ec610 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2120>
   ec679:	cmpq   $0x11,(%r12)
   ec67e:	jb     ec670 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2180>
   ec680:	mov    -0x10(%r12),%rdi
   ec685:	call   *0x1b21d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec68b:	jmp    ec670 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2180>
   ec68d:	cmpq   $0x0,0x190(%rsp)
   ec696:	je     ecc01 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2711>
   ec69c:	mov    %r15,%rdi
   ec69f:	jmp    ecbfb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x270b>
   ec6a4:	mov    0x158(%rsp),%rbx
   ec6ac:	mov    0x160(%rsp),%r14
   ec6b4:	mov    0x168(%rsp),%r13
   ec6bc:	lea    0x1d0(%rsp),%rdi
   ec6c4:	call   f0060 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::BirthFragment>>>
   ec6c9:	lea    0x1a8(%rsp),%rdi
   ec6d1:	call   ef9f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ec6d6:	cmp    $0xffffffffffffffff,%rbx
   ec6da:	je     ec729 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2239>
   ec6dc:	mov    %rbx,0x1a8(%rsp)
   ec6e4:	mov    %r14,0x1b0(%rsp)
   ec6ec:	mov    %r13,0x1b8(%rsp)
   ec6f4:	mov    0x120(%rsp),%rax
   ec6fc:	mov    %rax,0xc0(%rsp)
   ec704:	movups 0x110(%rsp),%xmm0
   ec70c:	movaps %xmm0,0xb0(%rsp)
   ec714:	mov    0x148(%rsp),%r13
   ec71c:	mov    0xe0(%rsp),%r12
   ec724:	jmp    eccce <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x27de>
   ec729:	mov    %r14,0x10(%rsp)
   ec72e:	mov    0x150(%rsp),%rax
   ec736:	mov    (%rax),%rbx
   ec739:	mov    0xe0(%rsp),%r12
   ec741:	jmp    ecc13 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2723>
   ec746:	mov    $0x2b,%edi
   ec74b:	mov    0x1b16e(%rip),%r14        # 1078c0 <malloc@GLIBC_2.2.5>
   ec752:	call   *%r14
   ec755:	test   %rax,%rax
   ec758:	je     ed4ce <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fde>
   ec75e:	mov    %rax,%r13
   ec761:	movups -0xce4b4(%rip),%xmm0        # 1e2b4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x567>
   ec768:	movups %xmm0,0x1b(%rax)
   ec76c:	movups -0xce4ca(%rip),%xmm0        # 1e2a9 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x55c>
   ec773:	movups %xmm0,0x10(%rax)
   ec777:	movdqu -0xce4e6(%rip),%xmm0        # 1e299 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x54c>
   ec77f:	movdqu %xmm0,(%rax)
   ec783:	mov    $0x18,%edi
   ec788:	call   *%r14
   ec78b:	test   %rax,%rax
   ec78e:	je     ed379 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e89>
   ec794:	movq   $0x2b,(%rax)
   ec79b:	mov    %r13,0x8(%rax)
   ec79f:	mov    %rax,0x10(%rsp)
   ec7a4:	movq   $0x2b,0x10(%rax)
   ec7ac:	lea    0x1a8(%rsp),%rdi
   ec7b4:	call   ef9f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ec7b9:	cmpq   $0x0,0x158(%rsp)
   ec7c2:	jne    ec5e8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20f8>
   ec7c8:	jmp    ec5f3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2103>
   ec7cd:	mov    0x88(%rsp),%rbx
   ec7d5:	mov    0x90(%rsp),%r14
   ec7dd:	mov    %r15,0x1d8(%rsp)
   ec7e5:	mov    $0x1a,%edi
   ec7ea:	mov    0x1b0cf(%rip),%r15        # 1078c0 <malloc@GLIBC_2.2.5>
   ec7f1:	call   *%r15
   ec7f4:	test   %rax,%rax
   ec7f7:	je     ed4e2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ff2>
   ec7fd:	mov    %rax,%r13
   ec800:	movups -0xce8d8(%rip),%xmm0        # 1df2f <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1e2>
   ec807:	movups %xmm0,0xa(%rax)
   ec80b:	movdqu -0xce8ee(%rip),%xmm0        # 1df25 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1d8>
   ec813:	movdqu %xmm0,(%rax)
   ec817:	mov    $0x18,%edi
   ec81c:	call   *%r15
   ec81f:	test   %rax,%rax
   ec822:	mov    0xe0(%rsp),%r12
   ec82a:	je     ed38d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e9d>
   ec830:	movq   $0x1a,(%rax)
   ec837:	mov    %r13,0x8(%rax)
   ec83b:	movq   $0x1a,0x10(%rax)
   ec843:	test   %r14,%r14
   ec846:	mov    %rax,0x10(%rsp)
   ec84b:	je     ec878 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2388>
   ec84d:	lea    0x18(%rbx),%r13
   ec851:	jmp    ec85c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x236c>
   ec853:	add    $0x20,%r13
   ec857:	dec    %r14
   ec85a:	je     ec86f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x237f>
   ec85c:	cmpq   $0x11,0x0(%r13)
   ec861:	jb     ec853 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2363>
   ec863:	mov    -0x10(%r13),%rdi
   ec867:	call   *0x1b03b(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec86d:	jmp    ec853 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2363>
   ec86f:	mov    %rbx,%rdi
   ec872:	call   *0x1b030(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec878:	mov    0x40(%rsp),%rax
   ec87d:	cmp    $0x1,%rax
   ec881:	jbe    ecb0f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x261f>
   ec887:	mov    0x20(%rsp),%rbx
   ec88c:	mov    0x28(%rsp),%r14
   ec891:	test   %r14,%r14
   ec894:	je     ecb21 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2631>
   ec89a:	mov    %r12,%r15
   ec89d:	lea    0x18(%rbx),%r12
   ec8a1:	jmp    ec8ac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23bc>
   ec8a3:	add    $0x20,%r12
   ec8a7:	dec    %r14
   ec8aa:	je     ec8c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23d0>
   ec8ac:	cmpq   $0x11,(%r12)
   ec8b1:	jb     ec8a3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23b3>
   ec8b3:	mov    -0x10(%r12),%rdi
   ec8b8:	call   *0x1afea(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec8be:	jmp    ec8a3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23b3>
   ec8c0:	mov    %r15,%r12
   ec8c3:	jmp    ecb21 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2631>
   ec8c8:	mov    $0x1f,%r14d
   ec8ce:	mov    $0x1f,%edi
   ec8d3:	mov    0x1afe6(%rip),%r15        # 1078c0 <malloc@GLIBC_2.2.5>
   ec8da:	call   *%r15
   ec8dd:	test   %rax,%rax
   ec8e0:	je     ed521 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3031>
   ec8e6:	mov    %rax,%r13
   ec8e9:	movups -0xce5c4(%rip),%xmm0        # 1e32c <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5df>
   ec8f0:	movups %xmm0,0xf(%rax)
   ec8f4:	movdqu -0xce5df(%rip),%xmm0        # 1e31d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5d0>
   ec8fc:	movdqu %xmm0,(%rax)
   ec900:	mov    $0x18,%edi
   ec905:	call   *%r15
   ec908:	test   %rax,%rax
   ec90b:	je     ed3fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f0d>
   ec911:	movq   $0x1f,(%rax)
   ec918:	mov    %rax,%rcx
   ec91b:	mov    $0x1f,%eax
   ec920:	jmp    ec97a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x248a>
   ec922:	mov    $0x1c,%r14d
   ec928:	mov    $0x1c,%edi
   ec92d:	mov    0x1af8c(%rip),%r15        # 1078c0 <malloc@GLIBC_2.2.5>
   ec934:	call   *%r15
   ec937:	test   %rax,%rax
   ec93a:	je     ed521 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3031>
   ec940:	mov    %rax,%r13
   ec943:	movups -0xcea35(%rip),%xmm0        # 1df15 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1c8>
   ec94a:	movups %xmm0,0xc(%rax)
   ec94e:	movdqu -0xcea4d(%rip),%xmm0        # 1df09 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1bc>
   ec956:	movdqu %xmm0,(%rax)
   ec95a:	mov    $0x18,%edi
   ec95f:	call   *%r15
   ec962:	test   %rax,%rax
   ec965:	je     ed42e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f3e>
   ec96b:	movq   $0x1c,(%rax)
   ec972:	mov    %rax,%rcx
   ec975:	mov    $0x1c,%eax
   ec97a:	mov    %r13,0x8(%rcx)
   ec97e:	mov    %rax,0x10(%rcx)
   ec982:	mov    %rcx,%rax
   ec985:	cmpq   $0x0,0x100(%rsp)
   ec98e:	je     ec99f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24af>
   ec990:	mov    %rbx,%rdi
   ec993:	mov    %rax,%rbx
   ec996:	call   *0x1af0c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ec99c:	mov    %rbx,%rax
   ec99f:	cmpq   $0x0,0x80(%rsp)
   ec9a8:	mov    0xe0(%rsp),%r12
   ec9b0:	mov    %rax,0x10(%rsp)
   ec9b5:	je     ec9cc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24dc>
   ec9b7:	cmpq   $0x11,0x98(%rsp)
   ec9c0:	jb     eca13 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2523>
   ec9c2:	mov    0x88(%rsp),%rbx
   ec9ca:	jmp    eca0a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x251a>
   ec9cc:	mov    0x90(%rsp),%r14
   ec9d4:	test   %r14,%r14
   ec9d7:	je     eca13 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2523>
   ec9d9:	mov    %r12,%r15
   ec9dc:	mov    0x88(%rsp),%rbx
   ec9e4:	lea    0x18(%rbx),%r12
   ec9e8:	jmp    ec9f3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2503>
   ec9ea:	add    $0x20,%r12
   ec9ee:	dec    %r14
   ec9f1:	je     eca07 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2517>
   ec9f3:	cmpq   $0x11,(%r12)
   ec9f8:	jb     ec9ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24fa>
   ec9fa:	mov    -0x10(%r12),%rdi
   ec9ff:	call   *0x1aea3(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eca05:	jmp    ec9ea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24fa>
   eca07:	mov    %r15,%r12
   eca0a:	mov    %rbx,%rdi
   eca0d:	call   *0x1ae95(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eca13:	lea    0xb0(%rsp),%rdi
   eca1b:	call   c3f10 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   eca20:	jmp    ecb2a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x263a>
   eca25:	mov    $0x22,%edi
   eca2a:	call   *0x1ae90(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   eca30:	test   %rax,%rax
   eca33:	je     ed530 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3040>
   eca39:	mov    %rax,%rbx
   eca3c:	movups -0xce6a5(%rip),%xmm0        # 1e39e <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x651>
   eca43:	movups %xmm0,0x10(%rax)
   eca47:	movdqu -0xce6c1(%rip),%xmm0        # 1e38e <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x641>
   eca4f:	movdqu %xmm0,(%rax)
   eca53:	movw   $0x3436,0x20(%rax)
   eca59:	call   f0150 <alloc::boxed::box_new_uninit>
   eca5e:	mov    0xe0(%rsp),%r12
   eca66:	movq   $0x22,(%rax)
   eca6d:	mov    %rbx,0x8(%rax)
   eca71:	mov    %rax,0x10(%rsp)
   eca76:	movq   $0x22,0x10(%rax)
   eca7e:	jmp    ecb2a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x263a>
   eca83:	mov    $0x1c,%r14d
   eca89:	mov    $0x1c,%edi
   eca8e:	mov    0x1ae2b(%rip),%r15        # 1078c0 <malloc@GLIBC_2.2.5>
   eca95:	call   *%r15
   eca98:	test   %rax,%rax
   eca9b:	je     ed541 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3051>
   ecaa1:	mov    %rax,%rbx
   ecaa4:	movups -0xceb96(%rip),%xmm0        # 1df15 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1c8>
   ecaab:	movups %xmm0,0xc(%rax)
   ecaaf:	movdqu -0xcebae(%rip),%xmm0        # 1df09 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1bc>
   ecab7:	movdqu %xmm0,(%rax)
   ecabb:	mov    $0x18,%edi
   ecac0:	call   *%r15
   ecac3:	test   %rax,%rax
   ecac6:	je     ed49c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fac>
   ecacc:	movq   $0x1c,(%rax)
   ecad3:	mov    %rbx,0x8(%rax)
   ecad7:	mov    %rax,0x10(%rsp)
   ecadc:	movq   $0x1c,0x10(%rax)
   ecae4:	cmp    $0x11,%r12
   ecae8:	jb     ecaf8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2608>
   ecaea:	mov    0xb8(%rsp),%rdi
   ecaf2:	call   *0x1adb0(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecaf8:	mov    0xe0(%rsp),%r12
   ecb00:	mov    0x40(%rsp),%rax
   ecb05:	cmp    $0x1,%rax
   ecb09:	ja     ec887 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2397>
   ecb0f:	test   %rax,%rax
   ecb12:	je     ecb2a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x263a>
   ecb14:	cmpq   $0x11,0x38(%rsp)
   ecb1a:	jb     ecb2a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x263a>
   ecb1c:	mov    0x28(%rsp),%rbx
   ecb21:	mov    %rbx,%rdi
   ecb24:	call   *0x1ad7e(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecb2a:	lea    0x1d0(%rsp),%rdi
   ecb32:	call   f0060 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::BirthFragment>>>
   ecb37:	lea    0x1a8(%rsp),%rdi
   ecb3f:	call   ef9f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ecb44:	mov    0x160(%rsp),%rbx
   ecb4c:	mov    0x168(%rsp),%r15
   ecb54:	test   %r15,%r15
   ecb57:	je     ecbed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26fd>
   ecb5d:	xor    %ebp,%ebp
   ecb5f:	mov    %r15,0xf0(%rsp)
   ecb67:	jmp    ecb85 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2695>
   ecb69:	mov    %r15,%r12
   ecb6c:	mov    0xf0(%rsp),%r15
   ecb74:	mov    %r14,%rdi
   ecb77:	call   *0x1ad2b(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecb7d:	inc    %rbp
   ecb80:	cmp    %r15,%rbp
   ecb83:	je     ecbed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26fd>
   ecb85:	lea    0x0(,%rbp,2),%rcx
   ecb8d:	add    %rbp,%rcx
   ecb90:	shl    $0x4,%rcx
   ecb94:	lea    (%rbx,%rcx,1),%rax
   ecb98:	cmpq   $0x0,(%rbx,%rcx,1)
   ecb9d:	je     ecbac <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26bc>
   ecb9f:	cmpq   $0x11,0x18(%rax)
   ecba4:	jb     ecb7d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x268d>
   ecba6:	mov    0x8(%rax),%r14
   ecbaa:	jmp    ecb74 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2684>
   ecbac:	mov    0x10(%rax),%r13
   ecbb0:	test   %r13,%r13
   ecbb3:	je     ecb7d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x268d>
   ecbb5:	mov    %r12,%r15
   ecbb8:	mov    0x8(%rax),%r14
   ecbbc:	lea    0x18(%r14),%r12
   ecbc0:	jmp    ecbd9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26e9>
   ecbc2:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ecbd0:	add    $0x20,%r12
   ecbd4:	dec    %r13
   ecbd7:	je     ecb69 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2679>
   ecbd9:	cmpq   $0x11,(%r12)
   ecbde:	jb     ecbd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26e0>
   ecbe0:	mov    -0x10(%r12),%rdi
   ecbe5:	call   *0x1acbd(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecbeb:	jmp    ecbd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26e0>
   ecbed:	cmpq   $0x0,0x158(%rsp)
   ecbf6:	je     ecc01 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2711>
   ecbf8:	mov    %rbx,%rdi
   ecbfb:	call   *0x1aca7(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecc01:	mov    0x150(%rsp),%rax
   ecc09:	mov    (%rax),%rbx
   ecc0c:	lea    0x19bfd(%rip),%r13        # 106810 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3318>
   ecc13:	xor    %eax,%eax
   ecc15:	mov    $0x1,%ecx
   ecc1a:	lock cmpxchg %ecx,(%rbx)
   ecc1e:	jne    ecc66 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2776>
   ecc20:	mov    0x1d019(%rip),%rax        # 109c40 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   ecc27:	shl    $1,%rax
   ecc2a:	test   %rax,%rax
   ecc2d:	jne    ece88 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2998>
   ecc33:	movzbl 0x4(%rbx),%eax
   ecc37:	test   %al,%al
   ecc39:	je     ecd40 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2850>
   ecc3f:	mov    0x1cffa(%rip),%rax        # 109c40 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   ecc46:	movabs $0x7fffffffffffffff,%rcx
   ecc50:	test   %rcx,%rax
   ecc53:	jne    ecec1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29d1>
   ecc59:	xor    %eax,%eax
   ecc5b:	xchg   %eax,(%rbx)
   ecc5d:	cmp    $0x2,%eax
   ecc60:	je     ecea2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29b2>
   ecc66:	movq   $0xffffffffffffffff,0x1a8(%rsp)
   ecc72:	mov    0x0(%r13),%rax
   ecc76:	test   %rax,%rax
   ecc79:	je     ecc82 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2792>
   ecc7b:	mov    0x10(%rsp),%rdi
   ecc80:	call   *%rax
   ecc82:	cmpq   $0x0,0x8(%r13)
   ecc87:	je     ecc94 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x27a4>
   ecc89:	mov    0x10(%rsp),%rdi
   ecc8e:	call   *0x1ac14(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ecc94:	mov    0x1a8(%rsp),%rax
   ecc9c:	mov    0x120(%rsp),%rcx
   ecca4:	mov    %rcx,0xc0(%rsp)
   eccac:	movups 0x110(%rsp),%xmm0
   eccb4:	movaps %xmm0,0xb0(%rsp)
   eccbc:	cmp    $0xffffffffffffffff,%rax
   eccc0:	mov    0x148(%rsp),%r13
   eccc8:	je     ecdc0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x28d0>
   eccce:	movups 0x110(%rsp),%xmm0
   eccd6:	movaps %xmm0,0x20(%rsp)
   eccdb:	mov    0x120(%rsp),%rbx
   ecce3:	mov    %rbx,0x30(%rsp)
   ecce8:	cmp    0x20(%rsp),%rbx
   ecced:	mov    0x210(%rsp),%r15
   eccf5:	jne    ecd01 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2811>
   eccf7:	lea    0x20(%rsp),%rdi
   eccfc:	call   edd80 <<alloc::raw_vec::RawVec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>::grow_one>
   ecd01:	mov    0x28(%rsp),%rax
   ecd06:	lea    (%rbx,%rbx,2),%rcx
   ecd0a:	mov    0x1b8(%rsp),%rdx
   ecd12:	mov    %rdx,0x10(%rax,%rcx,8)
   ecd17:	movups 0x1a8(%rsp),%xmm0
   ecd1f:	movups %xmm0,(%rax,%rcx,8)
   ecd23:	inc    %rbx
   ecd26:	mov    %rbx,0xc0(%rsp)
   ecd2e:	movaps 0x20(%rsp),%xmm0
   ecd33:	movaps %xmm0,0xb0(%rsp)
   ecd3b:	jmp    ecdcd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x28dd>
   ecd40:	xor    %eax,%eax
   ecd42:	mov    0x8(%rbx),%r14
   ecd46:	test   %r14,%r14
   ecd49:	jne    ecd58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2868>
   ecd4b:	mov    0x10(%rsp),%rcx
   ecd50:	mov    %rcx,0x8(%rbx)
   ecd54:	mov    %r13,0x10(%rbx)
   ecd58:	test   %al,%al
   ecd5a:	jne    ecd76 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2886>
   ecd5c:	mov    0x1cedd(%rip),%rax        # 109c40 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   ecd63:	movabs $0x7fffffffffffffff,%rcx
   ecd6d:	test   %rcx,%rax
   ecd70:	jne    ecf1f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a2f>
   ecd76:	xor    %eax,%eax
   ecd78:	xchg   %eax,(%rbx)
   ecd7a:	cmp    $0x2,%eax
   ecd7d:	je     eced7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29e7>
   ecd83:	movq   $0xffffffffffffffff,0x1a8(%rsp)
   ecd8f:	test   %r14,%r14
   ecd92:	jne    ecc72 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2782>
   ecd98:	mov    0x120(%rsp),%rax
   ecda0:	mov    %rax,0xc0(%rsp)
   ecda8:	movups 0x110(%rsp),%xmm0
   ecdb0:	movaps %xmm0,0xb0(%rsp)
   ecdb8:	mov    0x148(%rsp),%r13
   ecdc0:	movb   $0x1,0x0(%r13)
   ecdc5:	mov    0x210(%rsp),%r15
   ecdcd:	mov    0xc0(%rsp),%rax
   ecdd5:	mov    %rax,0x30(%rsp)
   ecdda:	movaps 0xb0(%rsp),%xmm0
   ecde2:	movaps %xmm0,0x20(%rsp)
   ecde7:	mov    0x250(%rsp),%rcx
   ecdef:	add    $0x70,%r12
   ecdf3:	mov    0x30(%rsp),%rax
   ecdf8:	mov    %rax,0x120(%rsp)
   ece00:	movdqa 0x20(%rsp),%xmm0
   ece06:	movdqa %xmm0,0x110(%rsp)
   ece0f:	movzbl 0x0(%r13),%eax
   ece14:	test   %al,%al
   ece16:	je     ea64e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15e>
   ece1c:	jmp    ed214 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d24>
   ece21:	cmp    $0x15,%rbx
   ece25:	jae    ecfa2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ab2>
   ece2b:	mov    %r15,%rdi
   ece2e:	mov    %rbx,%rsi
   ece31:	call   efef0 <core::slice::sort::shared::smallsort::insertion_sort_shift_left::<tk_train::trainers::bpe::index::BirthFragment, <[tk_train::trainers::bpe::index::BirthFragment]>::sort_by_key<(usize, (u32, u32)), <tk_train::trainers::bpe::index::Shard>::commit::{closure#0}>::{closure#0}>>
   ece36:	jmp    eac71 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x781>
   ece3b:	mov    $0x1e,%r14d
   ece41:	mov    $0x1e,%edi
   ece46:	call   *0x1aa74(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   ece4c:	test   %rax,%rax
   ece4f:	je     ed521 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3031>
   ece55:	mov    %rax,%r13
   ece58:	movups -0xcec83(%rip),%xmm0        # 1e1dc <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x48f>
   ece5f:	movups %xmm0,0xe(%rax)
   ece63:	movdqu -0xcec9d(%rip),%xmm0        # 1e1ce <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x481>
   ece6b:	movdqu %xmm0,(%rax)
   ece6f:	call   f0150 <alloc::boxed::box_new_uninit>
   ece74:	mov    %rax,%rcx
   ece77:	movq   $0x1e,(%rax)
   ece7e:	mov    $0x1e,%eax
   ece83:	jmp    ec97a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x248a>
   ece88:	call   8da10 <std::panicking::panic_count::is_zero_slow_path>
   ece8d:	movzbl 0x4(%rbx),%ecx
   ece91:	test   %cl,%cl
   ece93:	je     ecf0b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a1b>
   ece95:	test   %al,%al
   ece97:	jne    ecc3f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x274f>
   ece9d:	jmp    ecc59 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2769>
   ecea2:	mov    $0xca,%edi
   ecea7:	mov    %rbx,%rsi
   eceaa:	mov    $0x81,%edx
   eceaf:	mov    $0x1,%ecx
   eceb4:	xor    %eax,%eax
   eceb6:	call   *0x1aa34(%rip)        # 1078f0 <syscall@GLIBC_2.2.5>
   ecebc:	jmp    ecc66 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2776>
   ecec1:	call   8da10 <std::panicking::panic_count::is_zero_slow_path>
   ecec6:	test   %al,%al
   ecec8:	jne    ecc59 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2769>
   ecece:	movb   $0x1,0x4(%rbx)
   eced2:	jmp    ecc59 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2769>
   eced7:	mov    $0xca,%edi
   ecedc:	mov    %rbx,%rsi
   ecedf:	mov    $0x81,%edx
   ecee4:	mov    $0x1,%ecx
   ecee9:	xor    %eax,%eax
   eceeb:	call   *0x1a9ff(%rip)        # 1078f0 <syscall@GLIBC_2.2.5>
   ecef1:	movq   $0xffffffffffffffff,0x1a8(%rsp)
   ecefd:	test   %r14,%r14
   ecf00:	jne    ecc72 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2782>
   ecf06:	jmp    ecd98 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x28a8>
   ecf0b:	xor    $0x1,%al
   ecf0d:	mov    0x8(%rbx),%r14
   ecf11:	test   %r14,%r14
   ecf14:	jne    ecd58 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2868>
   ecf1a:	jmp    ecd4b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x285b>
   ecf1f:	call   8da10 <std::panicking::panic_count::is_zero_slow_path>
   ecf24:	test   %al,%al
   ecf26:	jne    ecd76 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2886>
   ecf2c:	movb   $0x1,0x4(%rbx)
   ecf30:	jmp    ecd76 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2886>
   ecf35:	mov    $0x20,%r14d
   ecf3b:	mov    $0x20,%edi
   ecf40:	call   *0x1a97a(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   ecf46:	test   %rax,%rax
   ecf49:	je     ed541 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3051>
   ecf4f:	mov    %rax,%rbx
   ecf52:	movups -0xdae79(%rip),%xmm0        # 120e0 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x6160>
   ecf59:	movups %xmm0,0x10(%rax)
   ecf5d:	movdqu -0xdae95(%rip),%xmm0        # 120d0 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x6150>
   ecf65:	movdqu %xmm0,(%rax)
   ecf69:	call   f0150 <alloc::boxed::box_new_uninit>
   ecf6e:	mov    %rax,0x10(%rsp)
   ecf73:	mov    0x10(%rsp),%rax
   ecf78:	movq   $0x20,(%rax)
   ecf7f:	mov    %rbx,0x8(%rax)
   ecf83:	movq   $0x20,0x10(%rax)
   ecf8b:	mov    0xc8(%rsp),%r12
   ecf93:	cmp    $0x11,%r12
   ecf97:	jae    ecaea <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25fa>
   ecf9d:	jmp    ecaf8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2608>
   ecfa2:	mov    %r15,%rdi
   ecfa5:	mov    %rbx,%rsi
   ecfa8:	call   c55c0 <core::slice::sort::stable::driftsort_main::<tk_train::trainers::bpe::index::BirthFragment, <[tk_train::trainers::bpe::index::BirthFragment]>::sort_by_key<(usize, (u32, u32)), <tk_train::trainers::bpe::index::Shard>::commit::{closure#0}>::{closure#0}, alloc::vec::Vec<tk_train::trainers::bpe::index::BirthFragment>>>
   ecfad:	jmp    eac71 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x781>
   ecfb2:	test   %dl,%dl
   ecfb4:	je     ecfd6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ae6>
   ecfb6:	mov    %fs:0xffffffffffffffd0,%rax
   ecfbf:	test   %rax,%rax
   ecfc2:	je     ecfe4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2af4>
   ecfc4:	add    $0x110,%rax
   ecfca:	jmp    ed00b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2b1b>
   ecfcc:	call   69210 <alloc::raw_vec::capacity_overflow>
   ecfd1:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ecfd6:	test   %rcx,%rcx
   ecfd9:	je     ea598 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa8>
   ecfdf:	shr    $1,%rcx
   ecfe2:	jmp    ed01f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2b2f>
   ecfe4:	mov    %r15,%rbp
   ecfe7:	mov    %rsi,%r15
   ecfea:	mov    %r12,0xe0(%rsp)
   ecff2:	mov    %rcx,%r12
   ecff5:	call   7f660 <rayon_core::registry::global_registry>
   ecffa:	mov    %r12,%rcx
   ecffd:	mov    0xe0(%rsp),%r12
   ed005:	mov    %r15,%rsi
   ed008:	mov    %rbp,%r15
   ed00b:	mov    (%rax),%rax
   ed00e:	mov    0x208(%rax),%rax
   ed015:	shr    $1,%rcx
   ed018:	cmp    %rax,%rcx
   ed01b:	cmovbe %rax,%rcx
   ed01f:	mov    %rcx,0x260(%rsp)
   ed027:	mov    %rsi,0xb0(%rsp)
   ed02f:	sub    %rsi,%r14
   ed032:	jb     ed2e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2df4>
   ed038:	imul   $0x70,%rsi,%rax
   ed03c:	add    %r12,%rax
   ed03f:	mov    0x8(%rbx),%rcx
   ed043:	mov    0x10(%rbx),%rdx
   ed047:	mov    0x18(%rbx),%rdi
   ed04b:	lea    0x258(%rsp),%r8
   ed053:	mov    %r8,0x280(%rsp)
   ed05b:	lea    0xb0(%rsp),%r8
   ed063:	mov    %r8,0x288(%rsp)
   ed06b:	lea    0x260(%rsp),%r9
   ed073:	mov    %r9,0x290(%rsp)
   ed07b:	mov    %rax,0x298(%rsp)
   ed083:	mov    %r14,0x2a0(%rsp)
   ed08b:	mov    %r13,0x2a8(%rsp)
   ed093:	mov    %rcx,0x2b0(%rsp)
   ed09b:	mov    %rdx,0x2b8(%rsp)
   ed0a3:	mov    %rdi,0x2c0(%rsp)
   ed0ab:	mov    %r8,0x2c8(%rsp)
   ed0b3:	mov    %r9,0x2d0(%rsp)
   ed0bb:	mov    %r12,0x2d8(%rsp)
   ed0c3:	mov    %rsi,0x2e0(%rsp)
   ed0cb:	mov    %r13,0x2e8(%rsp)
   ed0d3:	mov    %rcx,0x2f0(%rsp)
   ed0db:	mov    %rdx,0x2f8(%rsp)
   ed0e3:	mov    %rdi,0x300(%rsp)
   ed0eb:	mov    %fs:0xffffffffffffffd0,%rdx
   ed0f4:	test   %rdx,%rdx
   ed0f7:	jne    ed120 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c30>
   ed0f9:	call   7f660 <rayon_core::registry::global_registry>
   ed0fe:	mov    (%rax),%rsi
   ed101:	mov    %fs:0xffffffffffffffd0,%rdx
   ed10a:	test   %rdx,%rdx
   ed10d:	je     ed30f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e1f>
   ed113:	cmp    %rsi,0x110(%rdx)
   ed11a:	jne    ed32a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e3a>
   ed120:	lea    0x20(%rsp),%rdi
   ed125:	lea    0x280(%rsp),%rsi
   ed12d:	xor    %ecx,%ecx
   ed12f:	call   edfa0 <rayon_core::join::join_context::<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}>
   ed134:	mov    0x20(%rsp),%rdx
   ed139:	mov    0x28(%rsp),%rcx
   ed13e:	mov    0x30(%rsp),%rax
   ed143:	mov    0x48(%rsp),%rsi
   ed148:	mov    %rsi,0x120(%rsp)
   ed150:	movups 0x38(%rsp),%xmm0
   ed155:	movaps %xmm0,0x110(%rsp)
   ed15d:	test   %rcx,%rcx
   ed160:	je     ed1a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2cb0>
   ed162:	mov    0x110(%rsp),%rsi
   ed16a:	movq   $0x0,0x110(%rsp)
   ed176:	test   %rsi,%rsi
   ed179:	je     ed1da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2cea>
   ed17b:	mov    %rsi,0x18(%rcx)
   ed17f:	mov    %rcx,0x20(%rsi)
   ed183:	mov    0x118(%rsp),%rcx
   ed18b:	xorps  %xmm0,%xmm0
   ed18e:	add    0x120(%rsp),%rax
   ed196:	movups %xmm0,0x118(%rsp)
   ed19e:	jmp    ed1da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2cea>
   ed1a0:	mov    0x110(%rsp),%rsi
   ed1a8:	mov    0x118(%rsp),%rcx
   ed1b0:	mov    %rdx,0x110(%rsp)
   ed1b8:	movq   $0x0,0x118(%rsp)
   ed1c4:	mov    0x120(%rsp),%rdi
   ed1cc:	mov    %rax,0x120(%rsp)
   ed1d4:	mov    %rsi,%rdx
   ed1d7:	mov    %rdi,%rax
   ed1da:	mov    %rdx,(%r15)
   ed1dd:	mov    %rcx,0x8(%r15)
   ed1e1:	mov    %rax,0x10(%r15)
   ed1e5:	lea    0x110(%rsp),%rdi
   ed1ed:	call   eda60 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   ed1f2:	jmp    ea56f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7f>
   ed1f7:	lea    -0xcef20(%rip),%rdi        # 1e2de <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x591>
   ed1fe:	lea    0x19f9b(%rip),%rdx        # 1071a0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ca8>
   ed205:	mov    $0x16,%esi
   ed20a:	call   71e10 <core::option::expect_failed>
   ed20f:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed214:	mov    %r13,0x128(%rsp)
   ed21c:	mov    0x150(%rsp),%rax
   ed224:	mov    %rax,0x130(%rsp)
   ed22c:	mov    0x170(%rsp),%rax
   ed234:	mov    %rax,0x138(%rsp)
   ed23c:	movaps 0x110(%rsp),%xmm0
   ed244:	movaps %xmm0,0x20(%rsp)
   ed249:	mov    0x188(%rsp),%rax
   ed251:	mov    %rax,0x50(%rsp)
   ed256:	mov    0x120(%rsp),%rax
   ed25e:	mov    %rax,0x30(%rsp)
   ed263:	mov    0x128(%rsp),%rax
   ed26b:	mov    %rax,0x38(%rsp)
   ed270:	mov    0x130(%rsp),%rax
   ed278:	mov    %rax,0x40(%rsp)
   ed27d:	mov    0x138(%rsp),%rax
   ed285:	mov    %rax,0x48(%rsp)
   ed28a:	jmp    ea562 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x72>
   ed28f:	lea    -0xcefd2(%rip),%rdi        # 1e2c4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x577>
   ed296:	lea    0x19eeb(%rip),%rdx        # 107188 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c90>
   ed29d:	mov    $0x1a,%esi
   ed2a2:	call   71e10 <core::option::expect_failed>
   ed2a7:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed2ac:	mov    %rsi,%r14
   ed2af:	lea    0x19c92(%rip),%rcx        # 106f48 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   ed2b6:	mov    %r14,%rdx
   ed2b9:	call   70c30 <core::slice::index::slice_index_fail>
   ed2be:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed2c3:	mov    %r14,%rdx
   ed2c6:	lea    0x19c7b(%rip),%rcx        # 106f48 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   ed2cd:	jmp    ed36c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7c>
   ed2d2:	mov    %rsi,%rdi
   ed2d5:	mov    %rdx,%r14
   ed2d8:	lea    0x19cb1(%rip),%rcx        # 106f90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   ed2df:	jmp    ed36c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7c>
   ed2e4:	lea    -0xcf2ae(%rip),%rdi        # 1e03d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f0>
   ed2eb:	lea    0x1a29e(%rip),%rdx        # 107590 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4098>
   ed2f2:	mov    $0x13,%esi
   ed2f7:	call   6ed60 <core::panicking::panic_fmt>
   ed2fc:	mov    $0x8,%edi
   ed301:	lea    0x19ca0(%rip),%rcx        # 106fa8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ab0>
   ed308:	jmp    ed36c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7c>
   ed30a:	mov    %r14,%rsi
   ed30d:	jmp    ed2af <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dbf>
   ed30f:	sub    $0xffffffffffffff80,%rsi
   ed313:	lea    0x20(%rsp),%rdi
   ed318:	lea    0x280(%rsp),%rdx
   ed320:	call   edeb0 <<rayon_core::registry::Registry>::in_worker_cold::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>)>>
   ed325:	jmp    ed134 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c44>
   ed32a:	sub    $0xffffffffffffff80,%rsi
   ed32e:	lea    0x20(%rsp),%rdi
   ed333:	lea    0x280(%rsp),%rcx
   ed33b:	call   ee460 <<rayon_core::registry::Registry>::in_worker_cross::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>)>>
   ed340:	jmp    ed134 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c44>
   ed345:	xor    %edi,%edi
   ed347:	lea    0x19c12(%rip),%rcx        # 106f60 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a68>
   ed34e:	jmp    ed36c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7c>
   ed350:	mov    $0x8,%esi
   ed355:	xor    %edi,%edi
   ed357:	lea    0x1a082(%rip),%rcx        # 1073e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   ed35e:	jmp    ed2b6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dc6>
   ed363:	xor    %edi,%edi
   ed365:	lea    0x1a074(%rip),%rcx        # 1073e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   ed36c:	mov    %r14,%rsi
   ed36f:	call   70c30 <core::slice::index::slice_index_fail>
   ed374:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed379:	mov    $0x8,%edi
   ed37e:	mov    $0x18,%esi
   ed383:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed388:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed38d:	mov    $0x8,%edi
   ed392:	mov    $0x18,%esi
   ed397:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed39c:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed3a1:	mov    $0x8,%esi
   ed3a6:	xor    %edi,%edi
   ed3a8:	jmp    ed3eb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2efb>
   ed3aa:	test   %rax,%rax
   ed3ad:	je     ed550 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3060>
   ed3b3:	mov    %rax,%rdi
   ed3b6:	mov    %rdx,%rsi
   ed3b9:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed3be:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed3c3:	lea    0x19da6(%rip),%rdx        # 107170 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c78>
   ed3ca:	mov    0x10(%rsp),%rsi
   ed3cf:	call   6fcfb <core::panicking::panic_bounds_check>
   ed3d4:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed3d9:	mov    %rsi,%rdx
   ed3dc:	lea    0x19b65(%rip),%rax        # 106f48 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   ed3e3:	mov    %rax,0x208(%rsp)
   ed3eb:	mov    0x208(%rsp),%rcx
   ed3f3:	call   70c30 <core::slice::index::slice_index_fail>
   ed3f8:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed3fd:	mov    $0x8,%edi
   ed402:	mov    $0x18,%esi
   ed407:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed40c:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed411:	lea    -0xcf010(%rip),%rdi        # 1e408 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x6bb>
   ed418:	lea    0x19fa9(%rip),%rdx        # 1073c8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ed0>
   ed41f:	mov    $0x1c,%esi
   ed424:	call   71e10 <core::option::expect_failed>
   ed429:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed42e:	mov    $0x8,%edi
   ed433:	mov    $0x18,%esi
   ed438:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed43d:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed442:	lea    0x1946f(%rip),%rdx        # 1068b8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33c0>
   ed449:	mov    %r13,%rsi
   ed44c:	call   6fcfb <core::panicking::panic_bounds_check>
   ed451:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed456:	mov    %r13,%rsi
   ed459:	jmp    ed470 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f80>
   ed45b:	mov    %r13,%rsi
   ed45e:	mov    %rcx,%rdi
   ed461:	lea    0x19468(%rip),%rax        # 1068d0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33d8>
   ed468:	mov    %rax,0x178(%rsp)
   ed470:	mov    0x178(%rsp),%rdx
   ed478:	call   6fcfb <core::panicking::panic_bounds_check>
   ed47d:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed482:	shr    $0x7,%rax
   ed486:	xor    %esi,%esi
   ed488:	mov    %rax,%rdi
   ed48b:	lea    0x19ae6(%rip),%rdx        # 106f78 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a80>
   ed492:	call   6fcfb <core::panicking::panic_bounds_check>
   ed497:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed49c:	mov    $0x8,%edi
   ed4a1:	mov    $0x18,%esi
   ed4a6:	call   691f2 <alloc::alloc::handle_alloc_error>
   ed4ab:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed4b0:	mov    $0x8,%edi
   ed4b5:	mov    %rbx,%rsi
   ed4b8:	call   690eb <alloc::raw_vec::handle_error>
   ed4bd:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed4c2:	mov    %r13,%rsi
   ed4c5:	lea    0x193d4(%rip),%rax        # 1068a0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33a8>
   ed4cc:	jmp    ed468 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f78>
   ed4ce:	mov    $0x1,%edi
   ed4d3:	mov    $0x2b,%esi
   ed4d8:	call   690eb <alloc::raw_vec::handle_error>
   ed4dd:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed4e2:	mov    $0x1,%edi
   ed4e7:	mov    $0x1a,%esi
   ed4ec:	call   690eb <alloc::raw_vec::handle_error>
   ed4f1:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed4f3:	mov    $0x8,%eax
   ed4f8:	mov    %rax,0x180(%rsp)
   ed500:	mov    %r15,%rsi
   ed503:	mov    0x180(%rsp),%rdi
   ed50b:	call   690eb <alloc::raw_vec::handle_error>
   ed510:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed512:	mov    $0x8,%edi
   ed517:	mov    %r14,%rsi
   ed51a:	call   690eb <alloc::raw_vec::handle_error>
   ed51f:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed521:	mov    $0x1,%edi
   ed526:	mov    %r14,%rsi
   ed529:	call   690eb <alloc::raw_vec::handle_error>
   ed52e:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed530:	mov    $0x1,%edi
   ed535:	mov    $0x22,%esi
   ed53a:	call   690eb <alloc::raw_vec::handle_error>
   ed53f:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed541:	mov    $0x1,%edi
   ed546:	mov    %r14,%rsi
   ed549:	call   690eb <alloc::raw_vec::handle_error>
   ed54e:	jmp    ed568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3078>
   ed550:	lea    -0xcf68f(%rip),%rdi        # 1dec8 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x17b>
   ed557:	lea    0x19202(%rip),%rdx        # 106760 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3268>
   ed55e:	mov    $0x11,%esi
   ed563:	call   70d00 <core::panicking::panic>
   ed568:	ud2
   ed56a:	jmp    ed6e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x31f4>
   ed56f:	jmp    ed571 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3081>
   ed571:	mov    %rax,%r12
   ed574:	add    $0x38,%r15
   ed578:	mov    %r15,0x1d8(%rsp)
   ed580:	lea    0x80(%rsp),%rdi
   ed588:	call   e25c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   ed58d:	jmp    ed6fa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x320a>
   ed592:	mov    %rax,0x10(%rsp)
   ed597:	jmp    ed7a7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32b7>
   ed59c:	jmp    ed6ba <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x31ca>
   ed5a1:	mov    %rax,%r14
   ed5a4:	cmp    $0x11,%r12
   ed5a8:	jb     ed74c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x325c>
   ed5ae:	mov    0xa0(%rsp),%rdi
   ed5b6:	jmp    ed746 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3256>
   ed5bb:	jmp    ed75e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x326e>
   ed5c0:	mov    %rax,%r12
   ed5c3:	mov    %rbx,%rdi
   ed5c6:	call   *0x1a2dc(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed5cc:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed5d1:	mov    %rax,%r12
   ed5d4:	lea    0xb0(%rsp),%rdi
   ed5dc:	call   f0450 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::positions::Leaf>>>
   ed5e1:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed5e6:	mov    %rax,%r14
   ed5e9:	jmp    ed710 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3220>
   ed5ee:	jmp    ed6f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3207>
   ed5f3:	jmp    ed83c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334c>
   ed5f8:	jmp    ed777 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3287>
   ed5fd:	mov    %rax,%r12
   ed600:	lea    0x20(%rsp),%rdi
   ed605:	call   db020 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   ed60a:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed60f:	mov    %rax,%r12
   ed612:	lea    0x220(%rsp),%rdi
   ed61a:	call   e25c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   ed61f:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed624:	mov    %rax,%rbx
   ed627:	lea    0x1a8(%rsp),%rdi
   ed62f:	call   dae20 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   ed634:	lea    0x20(%rsp),%rdi
   ed639:	call   edd20 <core::ptr::drop_glue::<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   ed63e:	mov    %rbx,%rdi
   ed641:	call   1019a0 <_Unwind_Resume@plt>
   ed646:	jmp    ed75e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x326e>
   ed64b:	mov    %rax,%rbx
   ed64e:	cmpq   $0x0,0x8(%r13)
   ed653:	je     ed660 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3170>
   ed655:	mov    0x10(%rsp),%rdi
   ed65a:	call   *0x1a248(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed660:	mov    0x148(%rsp),%rax
   ed668:	mov    %rax,0x128(%rsp)
   ed670:	mov    0x150(%rsp),%rax
   ed678:	mov    %rax,0x130(%rsp)
   ed680:	mov    0x170(%rsp),%rax
   ed688:	mov    %rax,0x138(%rsp)
   ed690:	mov    0x188(%rsp),%rax
   ed698:	mov    %rax,0x140(%rsp)
   ed6a0:	lea    0x110(%rsp),%rdi
   ed6a8:	call   eddf0 <core::ptr::drop_glue::<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   ed6ad:	mov    %rbx,%rdi
   ed6b0:	call   1019a0 <_Unwind_Resume@plt>
   ed6b5:	jmp    ed75e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x326e>
   ed6ba:	mov    %rax,%r14
   ed6bd:	mov    %rbx,%rdi
   ed6c0:	call   *0x1a1e2(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed6c6:	jmp    ed733 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3243>
   ed6c8:	jmp    ed75e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x326e>
   ed6cd:	mov    %rax,%rbx
   ed6d0:	lea    0x20(%rsp),%rdi
   ed6d5:	call   efe70 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::BirthFragment>>
   ed6da:	mov    %rbx,%rax
   ed6dd:	jmp    ed795 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32a5>
   ed6e2:	jmp    ed6e4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x31f4>
   ed6e4:	mov    %rax,%r12
   ed6e7:	mov    %r13,%rdi
   ed6ea:	call   *0x1a1b8(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed6f0:	jmp    ed761 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3271>
   ed6f2:	jmp    ed795 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32a5>
   ed6f7:	mov    %rax,%r12
   ed6fa:	lea    0x20(%rsp),%rbx
   ed6ff:	jmp    ed832 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3342>
   ed704:	mov    %rax,%r14
   ed707:	mov    %r13,%rdi
   ed70a:	call   *0x1a198(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed710:	lea    0x80(%rsp),%rdi
   ed718:	call   f06f0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Repr>>
   ed71d:	jmp    ed74c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x325c>
   ed71f:	mov    %rax,%rbx
   ed722:	mov    %r13,%rdi
   ed725:	call   *0x1a17d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed72b:	mov    %rbx,%rax
   ed72e:	jmp    ed795 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32a5>
   ed730:	mov    %rax,%r14
   ed733:	cmpq   $0x11,0xc8(%rsp)
   ed73c:	jb     ed74c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x325c>
   ed73e:	mov    0xb8(%rsp),%rdi
   ed746:	call   *0x1a15c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed74c:	lea    0x20(%rsp),%rdi
   ed751:	call   f0770 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>>
   ed756:	mov    %r14,%r12
   ed759:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed75e:	mov    %rax,%r12
   ed761:	cmpq   $0x0,0x100(%rsp)
   ed76a:	je     ed7b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c8>
   ed76c:	mov    %rbx,%rdi
   ed76f:	call   *0x1a133(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed775:	jmp    ed7b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c8>
   ed777:	mov    %rax,%r12
   ed77a:	cmpq   $0x0,0x1f0(%rsp)
   ed783:	je     ed7b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c8>
   ed785:	mov    0x1f8(%rsp),%rdi
   ed78d:	call   *0x1a115(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed793:	jmp    ed7b8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c8>
   ed795:	mov    %rax,0x10(%rsp)
   ed79a:	lea    0xb0(%rsp),%rdi
   ed7a2:	call   efb30 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::index::PendingAction>>>
   ed7a7:	mov    $0x1,%al
   ed7a9:	mov    %eax,0xf0(%rsp)
   ed7b0:	jmp    ed85c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x336c>
   ed7b5:	mov    %rax,%r12
   ed7b8:	cmpq   $0x0,0x80(%rsp)
   ed7c1:	jne    ed80c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x331c>
   ed7c3:	mov    0x90(%rsp),%r14
   ed7cb:	lea    0xb0(%rsp),%rbx
   ed7d3:	test   %r14,%r14
   ed7d6:	je     ed832 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3342>
   ed7d8:	mov    0x88(%rsp),%r15
   ed7e0:	lea    0x18(%r15),%r13
   ed7e4:	jmp    ed7f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3309>
   ed7e6:	cs nopw 0x0(%rax,%rax,1)
   ed7f0:	add    $0x20,%r13
   ed7f4:	dec    %r14
   ed7f7:	je     ed829 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3339>
   ed7f9:	cmpq   $0x11,0x0(%r13)
   ed7fe:	jb     ed7f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3300>
   ed800:	mov    -0x10(%r13),%rdi
   ed804:	call   *0x1a09e(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed80a:	jmp    ed7f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3300>
   ed80c:	lea    0xb0(%rsp),%rbx
   ed814:	cmpq   $0x11,0x98(%rsp)
   ed81d:	jb     ed832 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3342>
   ed81f:	mov    0x88(%rsp),%rdi
   ed827:	jmp    ed82c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x333c>
   ed829:	mov    %r15,%rdi
   ed82c:	call   *0x1a076(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed832:	mov    %rbx,%rdi
   ed835:	call   c3f10 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   ed83a:	jmp    ed83f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x334f>
   ed83c:	mov    %rax,%r12
   ed83f:	mov    %r12,0x10(%rsp)
   ed844:	lea    0x1d0(%rsp),%rdi
   ed84c:	call   f0060 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::BirthFragment>>>
   ed851:	movl   $0x0,0xf0(%rsp)
   ed85c:	lea    0x1a8(%rsp),%rdi
   ed864:	call   ef9f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ed869:	mov    0x160(%rsp),%rbx
   ed871:	mov    0x168(%rsp),%r13
   ed879:	test   %r13,%r13
   ed87c:	jne    ed8da <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33ea>
   ed87e:	cmpq   $0x0,0x158(%rsp)
   ed887:	je     ed892 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33a2>
   ed889:	mov    %rbx,%rdi
   ed88c:	call   *0x1a016(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed892:	cmpb   $0x0,0xf0(%rsp)
   ed89a:	je     ed95d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x346d>
   ed8a0:	mov    0x198(%rsp),%rbx
   ed8a8:	mov    0x1a0(%rsp),%r13
   ed8b0:	test   %r13,%r13
   ed8b3:	jne    ed96f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x347f>
   ed8b9:	cmpq   $0x0,0x190(%rsp)
   ed8c2:	mov    0x148(%rsp),%r13
   ed8ca:	jne    ed9fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x350d>
   ed8d0:	mov    0x10(%rsp),%rbx
   ed8d5:	jmp    eda10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3520>
   ed8da:	xor    %r14d,%r14d
   ed8dd:	jmp    ed8f1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3401>
   ed8df:	nop
   ed8e0:	mov    %r15,%rdi
   ed8e3:	call   *0x19fbf(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed8e9:	inc    %r14
   ed8ec:	cmp    %r13,%r14
   ed8ef:	je     ed87e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x338e>
   ed8f1:	lea    (%r14,%r14,2),%rcx
   ed8f5:	shl    $0x4,%rcx
   ed8f9:	lea    (%rbx,%rcx,1),%rax
   ed8fd:	cmpq   $0x0,(%rbx,%rcx,1)
   ed902:	je     ed920 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3430>
   ed904:	cmpq   $0x11,0x18(%rax)
   ed909:	jb     ed8e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33f9>
   ed90b:	mov    0x8(%rax),%r15
   ed90f:	jmp    ed8e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33f0>
   ed911:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed920:	mov    0x10(%rax),%rbp
   ed924:	test   %rbp,%rbp
   ed927:	je     ed8e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33f9>
   ed929:	mov    0x8(%rax),%r15
   ed92d:	lea    0x18(%r15),%r12
   ed931:	jmp    ed949 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3459>
   ed933:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed940:	add    $0x20,%r12
   ed944:	dec    %rbp
   ed947:	je     ed8e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33f0>
   ed949:	cmpq   $0x11,(%r12)
   ed94e:	jb     ed940 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3450>
   ed950:	mov    -0x10(%r12),%rdi
   ed955:	call   *0x19f4d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed95b:	jmp    ed940 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3450>
   ed95d:	mov    0x148(%rsp),%r13
   ed965:	mov    0x10(%rsp),%rbx
   ed96a:	jmp    eda10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3520>
   ed96f:	xor    %r14d,%r14d
   ed972:	jmp    ed995 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34a5>
   ed974:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed980:	mov    %r15,%rdi
   ed983:	call   *0x19f1f(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed989:	inc    %r14
   ed98c:	cmp    %r13,%r14
   ed98f:	je     ed8b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33c9>
   ed995:	imul   $0x38,%r14,%rcx
   ed999:	lea    (%rbx,%rcx,1),%rax
   ed99d:	cmpq   $0x0,(%rbx,%rcx,1)
   ed9a2:	je     ed9c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34d0>
   ed9a4:	cmpq   $0x11,0x18(%rax)
   ed9a9:	jb     ed989 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3499>
   ed9ab:	mov    0x8(%rax),%r15
   ed9af:	jmp    ed980 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3490>
   ed9b1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed9c0:	mov    0x10(%rax),%rbp
   ed9c4:	test   %rbp,%rbp
   ed9c7:	je     ed989 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3499>
   ed9c9:	mov    0x8(%rax),%r15
   ed9cd:	lea    0x18(%r15),%r12
   ed9d1:	jmp    ed9e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34f9>
   ed9d3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed9e0:	add    $0x20,%r12
   ed9e4:	dec    %rbp
   ed9e7:	je     ed980 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3490>
   ed9e9:	cmpq   $0x11,(%r12)
   ed9ee:	jb     ed9e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34f0>
   ed9f0:	mov    -0x10(%r12),%rdi
   ed9f5:	call   *0x19ead(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ed9fb:	jmp    ed9e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34f0>
   ed9fd:	mov    %rbx,%rdi
   eda00:	call   *0x19ea2(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   eda06:	mov    0x10(%rsp),%rbx
   eda0b:	jmp    eda10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3520>
   eda0d:	mov    %rax,%rbx
   eda10:	mov    %r13,0x128(%rsp)
   eda18:	mov    0x150(%rsp),%rax
   eda20:	mov    %rax,0x130(%rsp)
   eda28:	mov    0x170(%rsp),%rax
   eda30:	mov    %rax,0x138(%rsp)
   eda38:	mov    0x188(%rsp),%rax
   eda40:	mov    %rax,0x140(%rsp)
   eda48:	lea    0x110(%rsp),%rdi
   eda50:	call   ede50 <core::ptr::drop_glue::<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>>>
   eda55:	mov    %rbx,%rdi
   eda58:	call   1019a0 <_Unwind_Resume@plt>
