
/tmp/bpe-corpus-binaries-20261013/final-baseline-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000e83f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>>:
   e83f0:	push   %rbp
   e83f1:	push   %r15
   e83f3:	push   %r14
   e83f5:	push   %r13
   e83f7:	push   %r12
   e83f9:	push   %rbx
   e83fa:	sub    $0x318,%rsp
   e8401:	mov    %rdi,%r15
   e8404:	mov    0x358(%rsp),%rbx
   e840c:	mov    %rsi,0x268(%rsp)
   e8414:	mov    %rcx,0x270(%rsp)
   e841c:	mov    %r8,0x278(%rsp)
   e8424:	mov    (%rbx),%rbp
   e8427:	movzbl 0x0(%rbp),%eax
   e842b:	test   %al,%al
   e842d:	je     e8480 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x90>
   e842f:	mov    0x18(%rbx),%rax
   e8433:	movq   $0x0,0x20(%rsp)
   e843c:	movq   $0x8,0x28(%rsp)
   e8445:	movq   $0x0,0x30(%rsp)
   e844e:	mov    %rbp,0x38(%rsp)
   e8453:	movups 0x8(%rbx),%xmm0
   e8457:	movups %xmm0,0x40(%rsp)
   e845c:	mov    %rax,0x50(%rsp)
   e8461:	lea    0x20(%rsp),%rsi
   e8466:	mov    %r15,%rdi
   e8469:	call   ebb40 <<rayon::iter::filter::FilterFolder<rayon::iter::map::MapFolder<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}> as rayon::iter::plumbing::Folder<&mut tk_train::trainers::bpe::index::Shard>>::complete>
   e846e:	add    $0x318,%rsp
   e8475:	pop    %rbx
   e8476:	pop    %r12
   e8478:	pop    %r13
   e847a:	pop    %r14
   e847c:	pop    %r15
   e847e:	pop    %rbp
   e847f:	ret
   e8480:	mov    %r9,%r13
   e8483:	mov    0x350(%rsp),%r14
   e848b:	shr    $1,%rsi
   e848e:	cmp    %r8,%rsi
   e8491:	jae    eaeb5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ac5>
   e8497:	mov    0x8(%rbx),%rdx
   e849b:	mov    0x10(%rbx),%rcx
   e849f:	mov    0x18(%rbx),%rax
   e84a3:	movq   $0x0,0x120(%rsp)
   e84af:	movq   $0x8,0x128(%rsp)
   e84bb:	movq   $0x0,0x130(%rsp)
   e84c7:	mov    %rbp,0x138(%rsp)
   e84cf:	mov    %rdx,0x160(%rsp)
   e84d7:	mov    %rdx,0x140(%rsp)
   e84df:	mov    %rcx,0x180(%rsp)
   e84e7:	mov    %rcx,0x148(%rsp)
   e84ef:	mov    %rax,0x1c0(%rsp)
   e84f7:	mov    %rax,0x150(%rsp)
   e84ff:	imul   $0x70,%r14,%rcx
   e8503:	mov    %r13,%rbx
   e8506:	add    %r13,%rcx
   e8509:	movabs $0x155555555555556,%rax
   e8513:	dec    %rax
   e8516:	mov    %rax,0x258(%rsp)
   e851e:	movq   $0x0,0x1b8(%rsp)
   e852a:	lea    0x1a987(%rip),%rax        # 102eb8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33c0>
   e8531:	mov    %rax,0x1b0(%rsp)
   e8539:	lea    0x1b4e8(%rip),%rax        # 103a28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f30>
   e8540:	mov    %rax,0x220(%rsp)
   e8548:	mov    %r15,0x158(%rsp)
   e8550:	cmp    %rcx,%rbx
   e8553:	je     eb117 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d27>
   e8559:	mov    0x10(%rbx),%r15
   e855d:	test   %r15,%r15
   e8560:	je     e85ee <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fe>
   e8566:	mov    0x28(%rbx),%r14
   e856a:	movabs $0x249249249249249,%rax
   e8574:	cmp    %rax,%r14
   e8577:	ja     eaecf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2adf>
   e857d:	mov    %rcx,0x260(%rsp)
   e8585:	mov    0x180(%rsp),%rdx
   e858d:	mov    (%rdx),%rax
   e8590:	mov    0x8(%rdx),%rcx
   e8594:	mov    0x8(%rax),%rsi
   e8598:	mov    %rsi,0xb0(%rsp)
   e85a0:	mov    0x10(%rax),%rax
   e85a4:	mov    %rax,(%rsp)
   e85a8:	movzbl (%rcx),%r12d
   e85ac:	mov    0x10(%rdx),%rax
   e85b0:	mov    (%rax),%rax
   e85b3:	mov    %rax,0x228(%rsp)
   e85bb:	test   %r14,%r14
   e85be:	mov    %rbx,0x118(%rsp)
   e85c6:	je     e8615 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x225>
   e85c8:	imul   $0x38,%r14,%rbx
   e85cc:	mov    %rbx,%rdi
   e85cf:	call   *0x1b933(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   e85d5:	test   %rax,%rax
   e85d8:	je     eb3c4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fd4>
   e85de:	mov    %rax,%r13
   e85e1:	mov    %r14,%rax
   e85e4:	mov    0x118(%rsp),%rbx
   e85ec:	jmp    e861d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x22d>
   e85ee:	mov    0x130(%rsp),%rax
   e85f6:	mov    %rax,0x30(%rsp)
   e85fb:	movups 0x120(%rsp),%xmm0
   e8603:	movaps %xmm0,0x20(%rsp)
   e8608:	mov    0x158(%rsp),%r15
   e8610:	jmp    eac07 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2817>
   e8615:	mov    $0x8,%r13d
   e861b:	xor    %eax,%eax
   e861d:	lea    0x18(%rbx),%rcx
   e8621:	mov    %r14,0x1c8(%rsp)
   e8629:	mov    %r13,0x1d0(%rsp)
   e8631:	movq   $0x0,0x1d8(%rsp)
   e863d:	movq   $0x0,0x168(%rsp)
   e8649:	movq   $0x8,0x170(%rsp)
   e8655:	movq   $0x0,0x178(%rsp)
   e8661:	movq   $0x0,0x28(%rbx)
   e8669:	mov    0x20(%rbx),%rdx
   e866d:	mov    %rax,%rsi
   e8670:	shl    $0x5,%rsi
   e8674:	add    %rdx,%rsi
   e8677:	mov    %rax,0x1f8(%rsp)
   e867f:	movq   $0x0,0x200(%rsp)
   e868b:	mov    %rdx,0x1e0(%rsp)
   e8693:	mov    %rsi,0xb8(%rsp)
   e869b:	mov    %rsi,0x1e8(%rsp)
   e86a3:	mov    %rcx,0x1f0(%rsp)
   e86ab:	movq   $0x0,0x10(%rbx)
   e86b3:	mov    0x8(%rbx),%rax
   e86b7:	mov    %r15,%r14
   e86ba:	shl    $0x4,%r14
   e86be:	add    %rax,%r14
   e86c1:	mov    %r14,0xc8(%rsp)
   e86c9:	mov    %rbx,0xd0(%rsp)
   e86d1:	mov    %r15,0xd8(%rsp)
   e86d9:	movq   $0x0,0xe0(%rsp)
   e86e5:	lea    0x30(%rbx),%rcx
   e86e9:	mov    %rcx,0x188(%rsp)
   e86f1:	xor    %ebx,%ebx
   e86f3:	mov    %rbp,0x10(%rsp)
   e86f8:	jmp    e8779 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x389>
   e86fd:	nopl   (%rax)
   e8700:	mov    0x10(%rsp),%rbp
   e8705:	mov    0xc0(%rsp),%rax
   e870d:	cmp    %r14,%rax
   e8710:	jne    e8779 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x389>
   e8712:	jmp    e8bb4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7c4>
   e8717:	nopw   0x0(%rax,%rax,1)
   e8720:	mov    0x1d0(%rsp),%r13
   e8728:	imul   $0x38,%rbx,%rax
   e872c:	mov    0x50(%rsp),%rcx
   e8731:	mov    %rcx,0x30(%r13,%rax,1)
   e8736:	movdqu 0x20(%rsp),%xmm0
   e873c:	movdqu 0x30(%rsp),%xmm1
   e8742:	movdqu 0x40(%rsp),%xmm2
   e8748:	movdqu %xmm2,0x20(%r13,%rax,1)
   e874f:	movdqu %xmm1,0x10(%r13,%rax,1)
   e8756:	movdqu %xmm0,0x0(%r13,%rax,1)
   e875d:	inc    %rbx
   e8760:	mov    %rbx,0x1d8(%rsp)
   e8768:	mov    0xc0(%rsp),%rax
   e8770:	cmp    %r14,%rax
   e8773:	je     e8bb4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7c4>
   e8779:	lea    0x10(%rax),%rcx
   e877d:	mov    %rcx,0xc0(%rsp)
   e8785:	mov    (%rax),%rdi
   e8788:	cmp    (%rsp),%rdi
   e878c:	jae    eb2c6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ed6>
   e8792:	movzbl 0x9(%rax),%ebp
   e8796:	lea    (%rdi,%rdi,4),%rcx
   e879a:	mov    0xb0(%rsp),%rdx
   e87a2:	lea    (%rdx,%rcx,8),%r15
   e87a6:	testb  $0x1,0x8(%rax)
   e87aa:	je     e8880 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x490>
   e87b0:	mov    0x4(%r15),%r8d
   e87b4:	test   %r12b,%r12b
   e87b7:	je     e88a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x4b0>
   e87bd:	mov    (%r15),%edx
   e87c0:	lea    0x20(%rsp),%rdi
   e87c5:	mov    0x188(%rsp),%rsi
   e87cd:	mov    %r8d,%ecx
   e87d0:	call   ed2a0 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::rustc_entry>
   e87d5:	mov    0x20(%rsp),%rax
   e87da:	test   %rax,%rax
   e87dd:	je     e89bf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5cf>
   e87e3:	mov    0x28(%rsp),%rdx
   e87e8:	mov    (%rax),%rcx
   e87eb:	mov    0x8(%rax),%rsi
   e87ef:	mov    %rsi,%rdi
   e87f2:	and    %rdx,%rdi
   e87f5:	movdqu (%rcx,%rdi,1),%xmm0
   e87fa:	pmovmskb %xmm0,%r8d
   e87ff:	test   %r8d,%r8d
   e8802:	je     e8b3c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x74c>
   e8808:	tzcnt  %r8d,%r8d
   e880d:	add    %rdi,%r8
   e8810:	and    %rsi,%r8
   e8813:	movzbl (%rcx,%r8,1),%r9d
   e8818:	test   %r9b,%r9b
   e881b:	jns    e8b61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x771>
   e8821:	mov    0x30(%rsp),%r10
   e8826:	shr    $0x39,%rdx
   e882a:	lea    -0x10(%r8),%rdi
   e882e:	and    %rsi,%rdi
   e8831:	mov    %dl,(%rcx,%r8,1)
   e8835:	mov    %dl,0x10(%rcx,%rdi,1)
   e8839:	shl    $0x4,%r8
   e883d:	mov    %rcx,%rdi
   e8840:	sub    %r8,%rdi
   e8843:	and    $0x1,%r9b
   e8847:	neg    %r8
   e884a:	mov    %r10,-0x10(%rcx,%r8,1)
   e884f:	movq   $0x0,-0x8(%rdi)
   e8857:	movdqu 0x10(%rax),%xmm0
   e885c:	movzbl %r9b,%ecx
   e8860:	movd   %ecx,%xmm1
   e8864:	shufpd $0x2,-0xdd0fd(%rip),%xmm1        # b770 <GCC_except_table1280+0xe8>
   e886d:	psubq  %xmm1,%xmm0
   e8871:	movdqu %xmm0,0x10(%rax)
   e8876:	jmp    e89c4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5d4>
   e887b:	nopl   0x0(%rax,%rax,1)
   e8880:	test   $0x1,%bpl
   e8884:	je     e8700 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x310>
   e888a:	test   %r12b,%r12b
   e888d:	jne    e89e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5f9>
   e8893:	jmp    e8ac3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x6d3>
   e8898:	nopl   0x0(%rax,%rax,1)
   e88a0:	mov    0x118(%rsp),%rax
   e88a8:	cmpq   $0x0,0x48(%rax)
   e88ad:	je     e89b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5c0>
   e88b3:	mov    (%r15),%esi
   e88b6:	mov    0x118(%rsp),%r9
   e88be:	mov    0x58(%r9),%rax
   e88c2:	xor    %rsi,%rax
   e88c5:	movabs $0x5851f42d4c957f2d,%rcx
   e88cf:	mul    %rcx
   e88d2:	xor    %r8,%rdx
   e88d5:	xor    %rdx,%rax
   e88d8:	mul    %rcx
   e88db:	mov    %rax,%rcx
   e88de:	xor    %rdx,%rcx
   e88e1:	mov    %rcx,%rax
   e88e4:	mulq   0x50(%r9)
   e88e8:	xor    %rax,%rdx
   e88eb:	rol    %cl,%rdx
   e88ee:	mov    %rdx,%rdi
   e88f1:	shr    $0x39,%rdi
   e88f5:	mov    0x30(%r9),%rax
   e88f9:	mov    0x38(%r9),%rcx
   e88fd:	movd   %edi,%xmm0
   e8901:	punpcklbw %xmm0,%xmm0
   e8905:	pshuflw $0x0,%xmm0,%xmm0
   e890a:	pshufd $0x44,%xmm0,%xmm0
   e890f:	xor    %edi,%edi
   e8911:	and    %rcx,%rdx
   e8914:	movdqu (%rax,%rdx,1),%xmm1
   e8919:	movdqa %xmm1,%xmm2
   e891d:	pcmpeqb %xmm0,%xmm2
   e8921:	pmovmskb %xmm2,%r9d
   e8926:	test   %r9d,%r9d
   e8929:	je     e8959 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x569>
   e892b:	tzcnt  %r9d,%r11d
   e8930:	add    %rdx,%r11
   e8933:	and    %rcx,%r11
   e8936:	shl    $0x4,%r11
   e893a:	mov    %rax,%r10
   e893d:	sub    %r11,%r10
   e8940:	cmp    -0x10(%r10),%esi
   e8944:	jne    e894c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x55c>
   e8946:	cmp    -0xc(%r10),%r8d
   e894a:	je     e8978 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x588>
   e894c:	lea    -0x1(%r9),%r10d
   e8950:	and    %r9w,%r10w
   e8954:	mov    %r10d,%r9d
   e8957:	jne    e892b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x53b>
   e8959:	pcmpeqd %xmm2,%xmm2
   e895d:	pcmpeqb %xmm2,%xmm1
   e8961:	pmovmskb %xmm1,%r9d
   e8966:	test   %r9d,%r9d
   e8969:	jne    e89b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5c0>
   e896b:	add    %rdi,%rdx
   e896e:	add    $0x10,%rdx
   e8972:	add    $0x10,%rdi
   e8976:	jmp    e8911 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x521>
   e8978:	mov    -0x8(%r10),%rax
   e897c:	sub    0x10(%r15),%rax
   e8980:	jb     ea685 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2295>
   e8986:	mov    %rax,-0x8(%r10)
   e898a:	cmp    0x228(%rsp),%rax
   e8992:	jae    e89b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x5c0>
   e8994:	mov    0x188(%rsp),%rdi
   e899c:	mov    %r8d,%edx
   e899f:	call   e2080 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::remove::<(u32, u32)>>
   e89a4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   e89b0:	test   $0x1,%bpl
   e89b4:	jne    e8ac3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x6d3>
   e89ba:	jmp    e8700 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x310>
   e89bf:	mov    0x28(%rsp),%rdi
   e89c4:	add    $0xfffffffffffffff8,%rdi
   e89c8:	mov    0x10(%r15),%rsi
   e89cc:	mov    $0x1,%edx
   e89d1:	call   ed3d0 <tk_train::trainers::bpe::index::adjust_signed>
   e89d6:	test   %rax,%rax
   e89d9:	jne    ea520 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2130>
   e89df:	test   $0x1,%bpl
   e89e3:	je     e8700 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x310>
   e89e9:	mov    0x8(%r15),%edx
   e89ed:	mov    0xc(%r15),%ecx
   e89f1:	lea    0x20(%rsp),%rdi
   e89f6:	mov    0x188(%rsp),%rsi
   e89fe:	call   ed2a0 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::rustc_entry>
   e8a03:	mov    0x20(%rsp),%rax
   e8a08:	test   %rax,%rax
   e8a0b:	je     e8aa6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x6b6>
   e8a11:	mov    0x28(%rsp),%rdx
   e8a16:	mov    (%rax),%rcx
   e8a19:	mov    0x8(%rax),%rsi
   e8a1d:	mov    %rsi,%rdi
   e8a20:	and    %rdx,%rdi
   e8a23:	movdqu (%rcx,%rdi,1),%xmm0
   e8a28:	pmovmskb %xmm0,%r8d
   e8a2d:	test   %r8d,%r8d
   e8a30:	je     e8b78 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x788>
   e8a36:	tzcnt  %r8d,%r8d
   e8a3b:	add    %rdi,%r8
   e8a3e:	and    %rsi,%r8
   e8a41:	movzbl (%rcx,%r8,1),%r9d
   e8a46:	test   %r9b,%r9b
   e8a49:	jns    e8b9d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7ad>
   e8a4f:	mov    0x30(%rsp),%r10
   e8a54:	shr    $0x39,%rdx
   e8a58:	lea    -0x10(%r8),%rdi
   e8a5c:	and    %rsi,%rdi
   e8a5f:	mov    %dl,(%rcx,%r8,1)
   e8a63:	mov    %dl,0x10(%rcx,%rdi,1)
   e8a67:	shl    $0x4,%r8
   e8a6b:	mov    %rcx,%rdi
   e8a6e:	sub    %r8,%rdi
   e8a71:	and    $0x1,%r9b
   e8a75:	neg    %r8
   e8a78:	mov    %r10,-0x10(%rcx,%r8,1)
   e8a7d:	movq   $0x0,-0x8(%rdi)
   e8a85:	movdqu 0x10(%rax),%xmm0
   e8a8a:	movzbl %r9b,%ecx
   e8a8e:	movd   %ecx,%xmm1
   e8a92:	shufpd $0x2,-0xdd32b(%rip),%xmm1        # b770 <GCC_except_table1280+0xe8>
   e8a9b:	psubq  %xmm1,%xmm0
   e8a9f:	movdqu %xmm0,0x10(%rax)
   e8aa4:	jmp    e8aab <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x6bb>
   e8aa6:	mov    0x28(%rsp),%rdi
   e8aab:	add    $0xfffffffffffffff8,%rdi
   e8aaf:	mov    0x18(%r15),%rsi
   e8ab3:	xor    %edx,%edx
   e8ab5:	call   ed3d0 <tk_train::trainers::bpe::index::adjust_signed>
   e8aba:	test   %rax,%rax
   e8abd:	jne    ea520 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2130>
   e8ac3:	mov    0x1e0(%rsp),%rax
   e8acb:	cmp    0xb8(%rsp),%rax
   e8ad3:	je     eb192 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2da2>
   e8ad9:	lea    0x20(%rax),%rcx
   e8add:	mov    %rcx,0x1e0(%rsp)
   e8ae5:	movdqu (%rax),%xmm0
   e8ae9:	movdqu 0x10(%rax),%xmm1
   e8aee:	lea    0x38(%rsp),%rax
   e8af3:	movdqu %xmm1,0x10(%rax)
   e8af8:	movdqu %xmm0,(%rax)
   e8afc:	mov    0x18(%r15),%rax
   e8b00:	mov    0x20(%r15),%rcx
   e8b04:	mov    %rcx,0x20(%rsp)
   e8b09:	mov    0x8(%r15),%rcx
   e8b0d:	mov    %rcx,0x28(%rsp)
   e8b12:	mov    %rax,0x30(%rsp)
   e8b17:	cmp    0x1c8(%rsp),%rbx
   e8b1f:	mov    0x10(%rsp),%rbp
   e8b24:	jne    e8728 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x338>
   e8b2a:	lea    0x1c8(%rsp),%rdi
   e8b32:	call   e21c0 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::merge::Rule>>::grow_one>
   e8b37:	jmp    e8720 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x330>
   e8b3c:	mov    $0x10,%r9d
   e8b42:	add    %r9,%rdi
   e8b45:	and    %rsi,%rdi
   e8b48:	movdqu (%rcx,%rdi,1),%xmm0
   e8b4d:	pmovmskb %xmm0,%r8d
   e8b52:	add    $0x10,%r9
   e8b56:	test   %r8d,%r8d
   e8b59:	jne    e8808 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x418>
   e8b5f:	jmp    e8b42 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x752>
   e8b61:	movdqa (%rcx),%xmm0
   e8b65:	pmovmskb %xmm0,%edi
   e8b69:	tzcnt  %edi,%r8d
   e8b6e:	movzbl (%rcx,%r8,1),%r9d
   e8b73:	jmp    e8821 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x431>
   e8b78:	mov    $0x10,%r9d
   e8b7e:	add    %r9,%rdi
   e8b81:	and    %rsi,%rdi
   e8b84:	movdqu (%rcx,%rdi,1),%xmm0
   e8b89:	pmovmskb %xmm0,%r8d
   e8b8e:	add    $0x10,%r9
   e8b92:	test   %r8d,%r8d
   e8b95:	jne    e8a36 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x646>
   e8b9b:	jmp    e8b7e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x78e>
   e8b9d:	movdqa (%rcx),%xmm0
   e8ba1:	pmovmskb %xmm0,%edi
   e8ba5:	tzcnt  %edi,%r8d
   e8baa:	movzbl (%rcx,%r8,1),%r9d
   e8baf:	jmp    e8a4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x65f>
   e8bb4:	cmp    $0x2,%rbx
   e8bb8:	jae    ead0c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x291c>
   e8bbe:	mov    0x1c8(%rsp),%rax
   e8bc6:	imul   $0x38,%rbx,%rcx
   e8bca:	add    %r13,%rcx
   e8bcd:	mov    %r13,0x190(%rsp)
   e8bd5:	mov    %rax,0x1a0(%rsp)
   e8bdd:	mov    %r13,0x198(%rsp)
   e8be5:	mov    %rcx,(%rsp)
   e8be9:	mov    %rcx,0x1a8(%rsp)
   e8bf1:	test   %rbx,%rbx
   e8bf4:	je     ea5f4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2204>
   e8bfa:	mov    %r12b,0x1f(%rsp)
   e8bff:	mov    0x0(%r13),%rax
   e8c03:	mov    0x8(%r13),%r15d
   e8c07:	mov    0xc(%r13),%ecx
   e8c0b:	mov    %rcx,0xb0(%rsp)
   e8c13:	xor    %r12d,%r12d
   e8c16:	xor    %r14d,%r14d
   e8c19:	mov    %r13,%rcx
   e8c1c:	nopl   0x0(%rax)
   e8c20:	cmp    %rax,(%rcx)
   e8c23:	jne    e8c4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x85f>
   e8c25:	cmp    %r15d,0x8(%rcx)
   e8c29:	jne    e8c4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x85f>
   e8c2b:	mov    0xb0(%rsp),%rdx
   e8c33:	cmp    %edx,0xc(%rcx)
   e8c36:	jne    e8c4f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x85f>
   e8c38:	add    0x10(%rcx),%r12
   e8c3c:	jb     eac38 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2848>
   e8c42:	inc    %r14
   e8c45:	add    $0x38,%rcx
   e8c49:	cmp    (%rsp),%rcx
   e8c4d:	jne    e8c20 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x830>
   e8c4f:	cmpb   $0x0,0x1f(%rsp)
   e8c54:	mov    %r15,0xb8(%rsp)
   e8c5c:	je     e8d43 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x953>
   e8c62:	mov    0x118(%rsp),%rdi
   e8c6a:	cmpq   $0x0,0x48(%rdi)
   e8c6f:	je     eb0fa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d0a>
   e8c75:	mov    0x58(%rdi),%rax
   e8c79:	xor    %r15,%rax
   e8c7c:	movabs $0x5851f42d4c957f2d,%rsi
   e8c86:	mul    %rsi
   e8c89:	mov    0xb0(%rsp),%rcx
   e8c91:	xor    %rdx,%rcx
   e8c94:	xor    %rcx,%rax
   e8c97:	mul    %rsi
   e8c9a:	mov    %rax,%rcx
   e8c9d:	xor    %rdx,%rcx
   e8ca0:	mov    %rcx,%rax
   e8ca3:	mulq   0x50(%rdi)
   e8ca7:	xor    %rax,%rdx
   e8caa:	rol    %cl,%rdx
   e8cad:	mov    %rdx,%rsi
   e8cb0:	shr    $0x39,%rsi
   e8cb4:	mov    0x30(%rdi),%rax
   e8cb8:	mov    0x38(%rdi),%rcx
   e8cbc:	movd   %esi,%xmm0
   e8cc0:	punpcklbw %xmm0,%xmm0
   e8cc4:	pshuflw $0x0,%xmm0,%xmm0
   e8cc9:	pshufd $0x44,%xmm0,%xmm0
   e8cce:	xor    %esi,%esi
   e8cd0:	and    %rcx,%rdx
   e8cd3:	movdqu (%rax,%rdx,1),%xmm1
   e8cd8:	movdqa %xmm1,%xmm2
   e8cdc:	pcmpeqb %xmm0,%xmm2
   e8ce0:	pmovmskb %xmm2,%edi
   e8ce4:	test   %edi,%edi
   e8ce6:	je     e8d22 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x932>
   e8ce8:	tzcnt  %edi,%r9d
   e8ced:	add    %rdx,%r9
   e8cf0:	and    %rcx,%r9
   e8cf3:	shl    $0x4,%r9
   e8cf7:	mov    %rax,%r8
   e8cfa:	sub    %r9,%r8
   e8cfd:	cmp    -0x10(%r8),%r15d
   e8d01:	jne    e8d15 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x925>
   e8d03:	mov    0xb0(%rsp),%r9
   e8d0b:	cmp    -0xc(%r8),%r9d
   e8d0f:	je     e9053 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc63>
   e8d15:	lea    -0x1(%rdi),%r8d
   e8d19:	and    %di,%r8w
   e8d1d:	mov    %r8d,%edi
   e8d20:	jne    e8ce8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x8f8>
   e8d22:	pcmpeqd %xmm2,%xmm2
   e8d26:	pcmpeqb %xmm2,%xmm1
   e8d2a:	pmovmskb %xmm1,%edi
   e8d2e:	test   %edi,%edi
   e8d30:	jne    eb0fa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d0a>
   e8d36:	add    %rsi,%rdx
   e8d39:	add    $0x10,%rdx
   e8d3d:	add    $0x10,%rsi
   e8d41:	jmp    e8cd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x8e0>
   e8d43:	cmp    0x228(%rsp),%r12
   e8d4b:	jb     e90d5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xce5>
   e8d51:	movq   $0x0,0x40(%rsp)
   e8d5a:	test   %r14,%r14
   e8d5d:	mov    %r13,0xf0(%rsp)
   e8d65:	je     e9200 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe10>
   e8d6b:	cmp    (%rsp),%r13
   e8d6f:	je     e9206 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe16>
   e8d75:	xor    %r13d,%r13d
   e8d78:	xor    %ebx,%ebx
   e8d7a:	dec    %r14
   e8d7d:	mov    0xf0(%rsp),%rax
   e8d85:	data16 cs nopw 0x0(%rax,%rax,1)
   e8d90:	movdqu 0x18(%rax),%xmm0
   e8d95:	movdqu 0x28(%rax),%xmm1
   e8d9a:	add    $0x38,%rax
   e8d9e:	mov    %rax,0xf0(%rsp)
   e8da6:	movdqa %xmm0,0x80(%rsp)
   e8daf:	movdqa %xmm1,0x90(%rsp)
   e8db8:	cmpq   $0x0,0x80(%rsp)
   e8dc1:	je     ea70b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x231b>
   e8dc7:	movdqa 0x80(%rsp),%xmm0
   e8dd0:	movdqa 0x90(%rsp),%xmm1
   e8dd9:	movdqa %xmm0,0xc0(%rsp)
   e8de2:	movdqa %xmm1,0xd0(%rsp)
   e8deb:	mov    0xc0(%rsp),%rbp
   e8df3:	cmp    $0x1,%rbp
   e8df7:	jne    e8e33 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa43>
   e8df9:	cmpq   $0x11,0xd8(%rsp)
   e8e02:	jb     e8e12 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa22>
   e8e04:	mov    0xc8(%rsp),%rdi
   e8e0c:	call   *0x1b0de(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e8e12:	sub    $0x1,%r14
   e8e16:	jb     e920a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe1a>
   e8e1c:	mov    0xf0(%rsp),%rax
   e8e24:	cmp    (%rsp),%rax
   e8e28:	jne    e8d90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x9a0>
   e8e2e:	jmp    e920a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe1a>
   e8e33:	test   $0x1,%r13b
   e8e37:	mov    0xf0(%rsp),%r8
   e8e3f:	je     e8efe <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb0e>
   e8e45:	lea    -0x1(%rbp),%rcx
   e8e49:	mov    %rcx,%rax
   e8e4c:	shr    $0x7,%rax
   e8e50:	and    $0x7f,%ecx
   e8e53:	cmp    $0x1,%rcx
   e8e57:	sbb    $0xffffffffffffffff,%rax
   e8e5b:	mov    $0x8,%edi
   e8e60:	cmp    $0x2,%rax
   e8e64:	jae    e8e94 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xaa4>
   e8e66:	mov    0xd8(%rsp),%r15
   e8e6e:	mov    %r15,%rsi
   e8e71:	cmp    $0x11,%r15
   e8e75:	jb     e8ec0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xad0>
   e8e77:	mov    0xd0(%rsp),%rsi
   e8e7f:	mov    %rsi,%rdx
   e8e82:	sub    %rdi,%rdx
   e8e85:	mov    0x110(%rsp),%rcx
   e8e8d:	jae    e8ed4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xae4>
   e8e8f:	jmp    eb2db <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eeb>
   e8e94:	dec    %rax
   e8e97:	bsr    %rax,%rcx
   e8e9b:	not    %ecx
   e8e9d:	mov    $0xffffffffffffffff,%rax
   e8ea4:	shr    %cl,%rax
   e8ea7:	lea    0x10(,%rax,8),%rdi
   e8eaf:	mov    0xd8(%rsp),%r15
   e8eb7:	mov    %r15,%rsi
   e8eba:	cmp    $0x11,%r15
   e8ebe:	jae    e8e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa87>
   e8ec0:	mov    %rsi,%rdx
   e8ec3:	sub    %rdi,%rdx
   e8ec6:	mov    0x110(%rsp),%rcx
   e8ece:	jb     eb2db <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eeb>
   e8ed4:	cmp    $0x7,%rdx
   e8ed8:	jbe    eb2a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eb4>
   e8ede:	lea    0xc8(%rsp),%rax
   e8ee6:	cmp    $0x11,%r15
   e8eea:	jb     e8ef4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb04>
   e8eec:	mov    0xc8(%rsp),%rax
   e8ef4:	cmp    (%rax,%rdi,1),%rcx
   e8ef8:	ja     eac8d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x289d>
   e8efe:	lea    (%rbx,%rbp,1),%rax
   e8f02:	dec    %rax
   e8f05:	mov    %rax,0xa8(%rsp)
   e8f0d:	cmp    %rbx,%rax
   e8f10:	jb     eae2a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a3a>
   e8f16:	mov    0xd8(%rsp),%r13
   e8f1e:	cmp    $0x11,%r13
   e8f22:	mov    0xd0(%rsp),%rbx
   e8f2a:	mov    %r13,%rdx
   e8f2d:	cmovae %rbx,%rdx
   e8f31:	cmp    $0x7,%rdx
   e8f35:	jbe    eb2a4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2eb4>
   e8f3b:	mov    0xc8(%rsp),%rcx
   e8f43:	cmp    $0x11,%r13
   e8f47:	lea    0xc8(%rsp),%rax
   e8f4f:	mov    %rcx,0x8(%rsp)
   e8f54:	cmovae %rcx,%rax
   e8f58:	mov    (%rax),%rax
   e8f5b:	mov    %rax,0x110(%rsp)
   e8f63:	mov    0x40(%rsp),%rcx
   e8f68:	cmp    $0x2,%rcx
   e8f6c:	mov    0xb8(%rsp),%r15
   e8f74:	jb     e8fb2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xbc2>
   e8f76:	mov    0x28(%rsp),%rax
   e8f7b:	cmp    %rcx,%rax
   e8f7e:	je     e9011 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc21>
   e8f84:	mov    0x20(%rsp),%rcx
   e8f89:	mov    %rax,%rdx
   e8f8c:	shl    $0x5,%rdx
   e8f90:	mov    %rbp,(%rcx,%rdx,1)
   e8f94:	mov    0x8(%rsp),%rsi
   e8f99:	mov    %rsi,0x8(%rcx,%rdx,1)
   e8f9e:	mov    %rbx,0x10(%rcx,%rdx,1)
   e8fa3:	mov    %r13,0x18(%rcx,%rdx,1)
   e8fa8:	inc    %rax
   e8fab:	mov    %rax,0x28(%rsp)
   e8fb0:	jmp    e8fe3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xbf3>
   e8fb2:	cmp    $0x1,%rcx
   e8fb6:	je     e902d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xc3d>
   e8fb8:	xor    %eax,%eax
   e8fba:	lea    0x20(%rsp),%rcx
   e8fbf:	lea    0x40(%rsp),%rdx
   e8fc4:	shl    $0x5,%rax
   e8fc8:	mov    %rbp,(%rcx,%rax,1)
   e8fcc:	mov    0x8(%rsp),%rsi
   e8fd1:	mov    %rsi,0x8(%rcx,%rax,1)
   e8fd6:	mov    %rbx,0x10(%rcx,%rax,1)
   e8fdb:	mov    %r13,0x18(%rcx,%rax,1)
   e8fe0:	incq   (%rdx)
   e8fe3:	test   %r14,%r14
   e8fe6:	mov    0xa8(%rsp),%rbp
   e8fee:	je     e920d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe1d>
   e8ff4:	mov    $0x1,%r13b
   e8ff7:	mov    %rbp,%rbx
   e8ffa:	mov    0xf0(%rsp),%rax
   e9002:	cmp    (%rsp),%rax
   e9006:	jne    e8d7a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x98a>
   e900c:	jmp    e920d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe1d>
   e9011:	lea    0x20(%rsp),%rdi
   e9016:	call   edb50 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::reserve_one_unchecked>
   e901b:	mov    0x28(%rsp),%rax
   e9020:	mov    0xb8(%rsp),%r15
   e9028:	jmp    e8f84 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xb94>
   e902d:	lea    0x20(%rsp),%rdi
   e9032:	call   edb50 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::reserve_one_unchecked>
   e9037:	mov    0x20(%rsp),%rcx
   e903c:	mov    0x28(%rsp),%rax
   e9041:	lea    0x28(%rsp),%rdx
   e9046:	mov    0xb8(%rsp),%r15
   e904e:	jmp    e8fc4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xbd4>
   e9053:	mov    -0x8(%r8),%rax
   e9057:	mov    %rax,0xa8(%rsp)
   e905f:	test   %rax,%rax
   e9062:	jle    e90d5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xce5>
   e9064:	movq   $0x0,0x40(%rsp)
   e906d:	mov    $0x1,%edi
   e9072:	test   %r14,%r14
   e9075:	je     e9169 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd79>
   e907b:	cmp    %rbx,%r14
   e907e:	cmovb  %r14,%rbx
   e9082:	cmp    $0x2,%rbx
   e9086:	jb     e9169 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd79>
   e908c:	dec    %rbx
   e908f:	bsr    %rbx,%rcx
   e9093:	not    %ecx
   e9095:	mov    $0xffffffffffffffff,%rsi
   e909c:	shr    %cl,%rsi
   e909f:	inc    %rsi
   e90a2:	lea    0x20(%rsp),%rdi
   e90a7:	call   ede80 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::try_grow>
   e90ac:	cmp    $0xffffffffffffffff,%rax
   e90b0:	lea    0x40(%rsp),%rbx
   e90b5:	jne    eb2ad <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ebd>
   e90bb:	mov    0x20(%rsp),%rcx
   e90c0:	mov    0x40(%rsp),%rsi
   e90c5:	cmp    $0x1,%rsi
   e90c9:	mov    %rsi,%rdi
   e90cc:	adc    $0x0,%rdi
   e90d0:	jmp    e9170 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd80>
   e90d5:	test   %r14,%r14
   e90d8:	jne    e90f5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd05>
   e90da:	jmp    e9ca2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18b2>
   e90df:	nop
   e90e0:	mov    %rbx,%rdi
   e90e3:	call   *0x1ae07(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e90e9:	mov    %r12,%r13
   e90ec:	add    $0x38,%r13
   e90f0:	dec    %r14
   e90f3:	je     e915c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd6c>
   e90f5:	cmp    (%rsp),%r13
   e90f9:	je     e915c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd6c>
   e90fb:	mov    0x20(%r13),%rbx
   e90ff:	cmpq   $0x0,0x18(%r13)
   e9104:	je     e9120 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd30>
   e9106:	mov    %r13,%r12
   e9109:	cmpq   $0x10,0x30(%r13)
   e910e:	ja     e90e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcf0>
   e9110:	jmp    e90e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcf9>
   e9112:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e9120:	mov    %r13,%r12
   e9123:	mov    0x28(%r13),%r15
   e9127:	test   %r15,%r15
   e912a:	je     e90e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcf9>
   e912c:	lea    0x18(%rbx),%r13
   e9130:	jmp    e9149 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd59>
   e9132:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e9140:	add    $0x20,%r13
   e9144:	dec    %r15
   e9147:	je     e90e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xcf0>
   e9149:	cmpq   $0x11,0x0(%r13)
   e914e:	jb     e9140 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd50>
   e9150:	mov    -0x10(%r13),%rdi
   e9154:	call   *0x1ad96(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e915a:	jmp    e9140 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xd50>
   e915c:	mov    %r13,0x198(%rsp)
   e9164:	jmp    e9ca2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18b2>
   e9169:	xor    %esi,%esi
   e916b:	lea    0x40(%rsp),%rbx
   e9170:	cmp    $0x2,%rsi
   e9174:	lea    0x20(%rsp),%rax
   e9179:	cmovb  %rax,%rcx
   e917d:	mov    $0x20,%eax
   e9182:	mov    $0x8,%edx
   e9187:	cmovae %rdx,%rax
   e918b:	mov    0x20(%rsp,%rax,1),%rdx
   e9190:	cmp    %rdi,%rdx
   e9193:	jae    e91f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe08>
   e9195:	test   %r14,%r14
   e9198:	je     e91e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdf6>
   e919a:	cmp    (%rsp),%r13
   e919e:	je     e91e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdf6>
   e91a0:	mov    %rdx,%rdi
   e91a3:	shl    $0x5,%rdi
   e91a7:	add    %rdi,%rcx
   e91aa:	dec    %r14
   e91ad:	movdqu 0x18(%r13),%xmm0
   e91b3:	movdqu 0x28(%r13),%xmm1
   e91b9:	add    $0x38,%r13
   e91bd:	movdqu %xmm1,0x10(%rcx)
   e91c2:	movdqu %xmm0,(%rcx)
   e91c6:	lea    0x1(%rdx),%rdi
   e91ca:	cmp    %rdi,%rsi
   e91cd:	jbe    e9397 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xfa7>
   e91d3:	inc    %rdx
   e91d6:	sub    $0x1,%r14
   e91da:	jb     e91e6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdf6>
   e91dc:	add    $0x20,%rcx
   e91e0:	cmp    (%rsp),%r13
   e91e4:	jne    e91ad <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xdbd>
   e91e6:	mov    %r13,0x198(%rsp)
   e91ee:	mov    %rdx,0x20(%rsp,%rax,1)
   e91f3:	jmp    e94ae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x10be>
   e91f8:	mov    %rdx,%rdi
   e91fb:	jmp    e939f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xfaf>
   e9200:	xor    %eax,%eax
   e9202:	xor    %ebp,%ebp
   e9204:	jmp    e9227 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe37>
   e9206:	xor    %ebp,%ebp
   e9208:	jmp    e920d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe1d>
   e920a:	mov    %rbx,%rbp
   e920d:	mov    0xf0(%rsp),%rax
   e9215:	mov    %rax,0x198(%rsp)
   e921d:	mov    0x20(%rsp),%rsi
   e9222:	mov    0x40(%rsp),%rax
   e9227:	mov    %rax,%r13
   e922a:	cmp    $0x2,%rax
   e922e:	jb     e9235 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe45>
   e9230:	mov    0x28(%rsp),%r13
   e9235:	test   %r13,%r13
   e9238:	je     e9284 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xe94>
   e923a:	cmp    $0x1,%r13
   e923e:	jne    e9321 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf31>
   e9244:	cmp    $0x2,%rax
   e9248:	lea    0x20(%rsp),%rax
   e924d:	cmovae %rsi,%rax
   e9251:	lea    0x40(%rsp),%rcx
   e9256:	lea    0x28(%rsp),%rdx
   e925b:	cmovae %rdx,%rcx
   e925f:	movq   $0x0,(%rcx)
   e9266:	mov    (%rax),%rbp
   e9269:	mov    0x8(%rax),%rbx
   e926d:	mov    0x10(%rax),%r13
   e9271:	mov    0x18(%rax),%rax
   e9275:	mov    %rax,0xa8(%rsp)
   e927d:	mov    0x40(%rsp),%rax
   e9282:	jmp    e9293 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xea3>
   e9284:	mov    $0x1,%ebp
   e9289:	mov    %r13,0xa8(%rsp)
   e9291:	xor    %ebx,%ebx
   e9293:	cmp    $0x1,%rax
   e9297:	jbe    e92db <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xeeb>
   e9299:	mov    %rbp,0x8(%rsp)
   e929e:	mov    %rsi,%rbp
   e92a1:	mov    0x28(%rsp),%r14
   e92a6:	test   %r14,%r14
   e92a9:	je     e9303 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf13>
   e92ab:	lea    0x18(%rbp),%r15
   e92af:	jmp    e92c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xed9>
   e92b1:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e92c0:	add    $0x20,%r15
   e92c4:	dec    %r14
   e92c7:	je     e9303 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf13>
   e92c9:	cmpq   $0x11,(%r15)
   e92cd:	jb     e92c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xed0>
   e92cf:	mov    -0x10(%r15),%rdi
   e92d3:	call   *0x1ac17(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e92d9:	jmp    e92c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xed0>
   e92db:	test   %rax,%rax
   e92de:	mov    %r12,%r14
   e92e1:	je     e9bc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d7>
   e92e7:	cmpq   $0x11,0x38(%rsp)
   e92ed:	jb     e9bc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d7>
   e92f3:	mov    0x28(%rsp),%rdi
   e92f8:	call   *0x1abf2(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e92fe:	jmp    e9bc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d7>
   e9303:	mov    %rbp,%rdi
   e9306:	call   *0x1abe4(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e930c:	mov    0xb8(%rsp),%r15
   e9314:	mov    %r12,%r14
   e9317:	mov    0x8(%rsp),%rbp
   e931c:	jmp    e9bc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d7>
   e9321:	lea    0xc0(%rsp),%rdi
   e9329:	lea    0x20(%rsp),%rsi
   e932e:	call   ed810 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>::into_vec>
   e9333:	mov    0xc0(%rsp),%rsi
   e933b:	mov    0xd0(%rsp),%r13
   e9343:	cmp    %r13,%rsi
   e9346:	mov    %rbp,0xa8(%rsp)
   e934e:	jbe    e938a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xf9a>
   e9350:	mov    0xc8(%rsp),%rdi
   e9358:	test   %r13,%r13
   e935b:	je     e9bb7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17c7>
   e9361:	shl    $0x5,%rsi
   e9365:	mov    %r13,%r14
   e9368:	shl    $0x5,%r14
   e936c:	mov    $0x8,%edx
   e9371:	mov    %r14,%rcx
   e9374:	call   68fa0 <__rustc::__rust_realloc>
   e9379:	mov    %rax,%rbx
   e937c:	test   %rax,%rax
   e937f:	jne    e9bc2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d2>
   e9385:	jmp    eb426 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3036>
   e938a:	mov    0xc8(%rsp),%rbx
   e9392:	jmp    e9bc2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17d2>
   e9397:	mov    %r13,0x198(%rsp)
   e939f:	mov    %rdi,0x20(%rsp,%rax,1)
   e93a4:	test   %r14,%r14
   e93a7:	je     e94ae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x10be>
   e93ad:	cmp    (%rsp),%r13
   e93b1:	je     e94a6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x10b6>
   e93b7:	dec    %r14
   e93ba:	nopw   0x0(%rax,%rax,1)
   e93c0:	movdqu 0x18(%r13),%xmm0
   e93c6:	movdqu 0x28(%r13),%xmm1
   e93cc:	movdqa %xmm1,0x90(%rsp)
   e93d5:	movdqa %xmm0,0x80(%rsp)
   e93de:	mov    0x40(%rsp),%rcx
   e93e3:	cmp    $0x2,%rcx
   e93e7:	jb     e9427 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1037>
   e93e9:	mov    0x28(%rsp),%rax
   e93ee:	cmp    %rcx,%rax
   e93f1:	je     e9477 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1087>
   e93f7:	mov    0x20(%rsp),%rcx
   e93fc:	mov    %rax,%rdx
   e93ff:	shl    $0x5,%rdx
   e9403:	movdqa 0x80(%rsp),%xmm0
   e940c:	movdqa 0x90(%rsp),%xmm1
   e9415:	movdqu %xmm1,0x10(%rcx,%rdx,1)
   e941b:	movdqu %xmm0,(%rcx,%rdx,1)
   e9420:	lea    0x28(%rsp),%rcx
   e9425:	jmp    e945b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x106b>
   e9427:	cmp    $0x1,%rcx
   e942b:	je     e948b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x109b>
   e942d:	xor    %eax,%eax
   e942f:	lea    0x20(%rsp),%rdx
   e9434:	mov    %rbx,%rcx
   e9437:	shl    $0x5,%rax
   e943b:	movdqa 0x80(%rsp),%xmm0
   e9444:	movdqa 0x90(%rsp),%xmm1
   e944d:	movdqu %xmm1,0x10(%rdx,%rax,1)
   e9453:	movdqu %xmm0,(%rdx,%rax,1)
   e9458:	mov    (%rcx),%rax
   e945b:	add    $0x38,%r13
   e945f:	inc    %rax
   e9462:	mov    %rax,(%rcx)
   e9465:	sub    $0x1,%r14
   e9469:	jb     e94a6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x10b6>
   e946b:	cmp    (%rsp),%r13
   e946f:	jne    e93c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xfd0>
   e9475:	jmp    e94a6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x10b6>
   e9477:	lea    0x20(%rsp),%rdi
   e947c:	call   ee040 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::reserve_one_unchecked>
   e9481:	mov    0x28(%rsp),%rax
   e9486:	jmp    e93f7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1007>
   e948b:	lea    0x20(%rsp),%rdi
   e9490:	call   ee040 <<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>::reserve_one_unchecked>
   e9495:	mov    0x20(%rsp),%rdx
   e949a:	mov    0x28(%rsp),%rax
   e949f:	lea    0x28(%rsp),%rcx
   e94a4:	jmp    e9437 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1047>
   e94a6:	mov    %r13,0x198(%rsp)
   e94ae:	mov    0x40(%rsp),%rax
   e94b3:	mov    %rax,0xe0(%rsp)
   e94bb:	movdqu 0x20(%rsp),%xmm0
   e94c1:	movdqu 0x30(%rsp),%xmm1
   e94c7:	movdqa %xmm1,0xd0(%rsp)
   e94d0:	movdqa %xmm0,0xc0(%rsp)
   e94d9:	cmp    $0x2,%rax
   e94dd:	mov    0xc8(%rsp),%rcx
   e94e5:	mov    %rax,%r14
   e94e8:	cmovae %rcx,%r14
   e94ec:	cmp    $0x1,%r14
   e94f0:	mov    0xb0(%rsp),%rbx
   e94f8:	jne    e95d3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x11e3>
   e94fe:	mov    0xc0(%rsp),%rdx
   e9506:	lea    0x8(%rdx),%rsi
   e950a:	lea    0x10(%rdx),%r11
   e950e:	cmp    $0x2,%rax
   e9512:	lea    0xc8(%rsp),%r8
   e951a:	movups 0x8(%r8),%xmm0
   e951f:	lea    0x30(%rsp),%r10
   e9524:	movups %xmm0,(%r10)
   e9528:	mov    $0x0,%r9d
   e952e:	cmovae %r9,%rcx
   e9532:	mov    $0x0,%r8d
   e9538:	mov    %r8,0x1b8(%rsp)
   e9540:	cmovb  %r9,%rax
   e9544:	mov    %rdx,0x20(%rsp)
   e9549:	mov    %rcx,0x28(%rsp)
   e954e:	mov    %rax,0x40(%rsp)
   e9553:	movq   $0x1,0x50(%rsp)
   e955c:	lea    0x20(%rsp),%rdi
   e9561:	cmovb  %rdi,%rdx
   e9565:	lea    0x28(%rsp),%rax
   e956a:	cmovb  %rax,%rsi
   e956e:	movq   $0x1,0x48(%rsp)
   e9577:	cmovb  %r10,%r11
   e957b:	movups (%r11),%xmm0
   e957f:	movq   (%rsi),%xmm1
   e9583:	movq   (%rdx),%xmm2
   e9587:	punpcklqdq %xmm1,%xmm2
   e958b:	movdqa %xmm2,0xf0(%rsp)
   e9594:	movaps %xmm0,0x280(%rsp)
   e959c:	call   ee0d0 <core::ptr::drop_glue::<smallvec::IntoIter<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   e95a1:	mov    0xa8(%rsp),%r14
   e95a9:	movaps 0x280(%rsp),%xmm0
   e95b1:	lea    0x240(%rsp),%rax
   e95b9:	movups %xmm0,(%rax)
   e95bc:	movdqa 0xf0(%rsp),%xmm0
   e95c5:	movdqa %xmm0,0x230(%rsp)
   e95ce:	jmp    e9c19 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1829>
   e95d3:	lea    0xc0(%rsp),%rdx
   e95db:	cmp    $0x2,%rax
   e95df:	mov    0x1a922(%rip),%rcx        # 103f08 <malloc@GLIBC_2.2.5>
   e95e6:	jb     e95f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1200>
   e95e8:	mov    0xc0(%rsp),%rdx
   e95f0:	mov    %rdx,%r12
   e95f3:	movq   $0x1,0x80(%rsp)
   e95ff:	mov    %r14,%rdx
   e9602:	shl    $0x5,%rdx
   e9606:	lea    0x88(%rsp),%rax
   e960e:	pxor   %xmm0,%xmm0
   e9612:	movdqu %xmm0,(%rax)
   e9616:	movq   $0x0,0x10(%rax)
   e961e:	mov    %rdx,0x108(%rsp)
   e9626:	lea    (%rdx,%rdx,2),%rsi
   e962a:	cmp    0x258(%rsp),%r14
   e9632:	ja     eb417 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3027>
   e9638:	test   %rsi,%rsi
   e963b:	je     e965e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x126e>
   e963d:	mov    %rsi,%r15
   e9640:	mov    %rsi,%rdi
   e9643:	call   *%rcx
   e9645:	test   %rax,%rax
   e9648:	je     eb407 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3017>
   e964e:	mov    %rax,%rbx
   e9651:	mov    %r14,%rax
   e9654:	mov    0xb8(%rsp),%r15
   e965c:	jmp    e9665 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1275>
   e965e:	mov    $0x8,%ebx
   e9663:	xor    %eax,%eax
   e9665:	mov    %rax,0x208(%rsp)
   e966d:	mov    %rbx,0x210(%rsp)
   e9675:	movq   $0x0,0x218(%rsp)
   e9681:	test   %r14,%r14
   e9684:	mov    %r13,0xf0(%rsp)
   e968c:	je     e9ba1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x17b1>
   e9692:	add    %r12,0x108(%rsp)
   e969a:	movq   $0x0,0x8(%rsp)
   e96a3:	mov    0xa8(%rsp),%r14
   e96ab:	jmp    e9736 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1346>
   e96b0:	add    $0xfffffffffffffff8,%r14
   e96b4:	add    $0x8,%r13
   e96b8:	lea    (%rsi,%rsi,2),%rax
   e96bc:	shl    $0x5,%rax
   e96c0:	mov    %r12,(%rbx,%rax,1)
   e96c4:	mov    %r13,0x8(%rbx,%rax,1)
   e96c9:	mov    %r14,0x10(%rbx,%rax,1)
   e96ce:	mov    %r12,0x18(%rbx,%rax,1)
   e96d3:	movq   $0x1,0x20(%rbx,%rax,1)
   e96dc:	movq   $0x0,0x28(%rbx,%rax,1)
   e96e5:	mov    %r15,0x48(%rbx,%rax,1)
   e96ea:	mov    %rbp,0x50(%rbx,%rax,1)
   e96ef:	movq   $0x0,0x58(%rbx,%rax,1)
   e96f8:	inc    %rsi
   e96fb:	mov    %rsi,0x8(%rsp)
   e9700:	mov    %rsi,0x218(%rsp)
   e9708:	mov    0x110(%rsp),%r12
   e9710:	cmp    0x108(%rsp),%r12
   e9718:	mov    0xf0(%rsp),%r13
   e9720:	mov    0xb8(%rsp),%r15
   e9728:	mov    0xa8(%rsp),%r14
   e9730:	je     e997a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x158a>
   e9736:	mov    %r12,%rdx
   e9739:	jmp    e9754 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1364>
   e973b:	mov    0x110(%rsp),%rax
   e9743:	mov    %rax,%rdx
   e9746:	cmp    0x108(%rsp),%rax
   e974e:	je     e997a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x158a>
   e9754:	mov    (%rdx),%r8
   e9757:	mov    0x8(%rdx),%rsi
   e975b:	mov    0x10(%rdx),%rbp
   e975f:	shl    $0x5,%rbp
   e9763:	test   %r8,%r8
   e9766:	mov    %rdx,%rax
   e9769:	cmove  %rsi,%rax
   e976d:	lea    0x20(%rdx),%rcx
   e9771:	mov    %rcx,0x110(%rsp)
   e9779:	mov    $0x20,%ecx
   e977e:	cmovne %rcx,%rbp
   e9782:	test   %rbp,%rbp
   e9785:	je     e973b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x134b>
   e9787:	add    %rax,%rbp
   e978a:	mov    (%rax),%rdi
   e978d:	dec    %rdi
   e9790:	je     e9807 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1417>
   e9792:	mov    %rdi,%rcx
   e9795:	shr    $0x7,%rcx
   e9799:	and    $0x7f,%edi
   e979c:	cmp    $0x1,%rdi
   e97a0:	sbb    $0xffffffffffffffff,%rcx
   e97a4:	mov    $0x8,%edi
   e97a9:	cmp    $0x2,%rcx
   e97ad:	jb     e97ca <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x13da>
   e97af:	dec    %rcx
   e97b2:	bsr    %rcx,%rcx
   e97b6:	not    %ecx
   e97b8:	mov    $0xffffffffffffffff,%rdi
   e97bf:	shr    %cl,%rdi
   e97c2:	lea    0x10(,%rdi,8),%rdi
   e97ca:	test   %r8,%r8
   e97cd:	mov    %rdx,%r9
   e97d0:	cmove  %rsi,%r9
   e97d4:	mov    0x18(%r9),%rcx
   e97d8:	mov    %rcx,%r14
   e97db:	cmp    $0x11,%rcx
   e97df:	jb     e97e5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x13f5>
   e97e1:	mov    0x10(%r9),%r14
   e97e5:	cmp    %rdi,%r14
   e97e8:	jb     eb20d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e1d>
   e97ee:	lea    0x20(%rsi),%r15
   e97f2:	test   %r8,%r8
   e97f5:	lea    0x20(%rdx),%r9
   e97f9:	cmovne %r9,%r15
   e97fd:	sub    %rdi,%r14
   e9800:	je     e9857 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1467>
   e9802:	jmp    e990a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x151a>
   e9807:	add    $0x20,%rsi
   e980b:	test   %r8,%r8
   e980e:	cmovne 0x110(%rsp),%rsi
   e9817:	mov    %rsi,%r15
   e981a:	jmp    e9857 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1467>
   e981c:	nopl   0x0(%rax)
   e9820:	mov    $0x1,%r13d
   e9826:	xor    %r14d,%r14d
   e9829:	lea    0x20(%rdx),%r15
   e982d:	mov    %rdx,%rax
   e9830:	test   %r14,%r14
   e9833:	je     e9857 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1467>
   e9835:	jmp    e992a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x153a>
   e983a:	nopw   0x0(%rax,%rax,1)
   e9840:	mov    0x28(%rax),%r13
   e9844:	add    %rdi,%r13
   e9847:	lea    0x20(%rdx),%r15
   e984b:	mov    %rdx,%rax
   e984e:	test   %r14,%r14
   e9851:	jne    e992a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x153a>
   e9857:	mov    %r15,%rdx
   e985a:	cmp    %rbp,%r15
   e985d:	mov    0xf0(%rsp),%r13
   e9865:	mov    0xb8(%rsp),%r15
   e986d:	mov    0xa8(%rsp),%r14
   e9875:	je     e973b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x134b>
   e987b:	mov    (%rdx),%rsi
   e987e:	dec    %rsi
   e9881:	je     e9820 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1430>
   e9883:	mov    %rsi,%rcx
   e9886:	shr    $0x7,%rcx
   e988a:	and    $0x7f,%esi
   e988d:	cmp    $0x1,%rsi
   e9891:	sbb    $0xffffffffffffffff,%rcx
   e9895:	mov    $0x8,%edi
   e989a:	cmp    $0x2,%rcx
   e989e:	jae    e98c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14d0>
   e98a0:	mov    0x38(%rax),%rcx
   e98a4:	mov    %rcx,%rsi
   e98a7:	cmp    $0x11,%rcx
   e98ab:	jb     e98e8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14f8>
   e98ad:	mov    0x30(%rax),%rsi
   e98b1:	mov    %rsi,%r14
   e98b4:	sub    %rdi,%r14
   e98b7:	jae    e98f4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1504>
   e98b9:	jmp    eb1af <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dbf>
   e98be:	xchg   %ax,%ax
   e98c0:	dec    %rcx
   e98c3:	bsr    %rcx,%rcx
   e98c7:	not    %ecx
   e98c9:	mov    $0xffffffffffffffff,%rsi
   e98d0:	shr    %cl,%rsi
   e98d3:	lea    0x10(,%rsi,8),%rdi
   e98db:	mov    0x38(%rax),%rcx
   e98df:	mov    %rcx,%rsi
   e98e2:	cmp    $0x11,%rcx
   e98e6:	jae    e98ad <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x14bd>
   e98e8:	mov    %rsi,%r14
   e98eb:	sub    %rdi,%r14
   e98ee:	jb     eb1af <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dbf>
   e98f4:	cmp    $0x11,%rcx
   e98f8:	jae    e9840 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1450>
   e98fe:	add    $0x28,%rax
   e9902:	mov    %rax,%r13
   e9905:	jmp    e9844 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1454>
   e990a:	lea    0x8(%rdx),%r13
   e990e:	test   %r8,%r8
   e9911:	cmove  %rsi,%rdx
   e9915:	lea    0x8(%rsi),%rax
   e9919:	cmove  %rax,%r13
   e991d:	cmp    $0x11,%rcx
   e9921:	jb     e9927 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1537>
   e9923:	mov    0x8(%rdx),%r13
   e9927:	add    %rdi,%r13
   e992a:	cmp    $0x7,%r14
   e992e:	jbe    eb253 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e63>
   e9934:	mov    0x0(%r13),%r12
   e9938:	mov    0x8(%rsp),%rsi
   e993d:	cmp    0x208(%rsp),%rsi
   e9945:	jne    e96b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x12c0>
   e994b:	mov    $0x1,%edx
   e9950:	mov    $0x8,%ecx
   e9955:	mov    $0x60,%r8d
   e995b:	lea    0x208(%rsp),%rdi
   e9963:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e9968:	mov    0x210(%rsp),%rbx
   e9970:	mov    0x8(%rsp),%rsi
   e9975:	jmp    e96b0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x12c0>
   e997a:	mov    0x210(%rsp),%rbx
   e9982:	mov    0x8(%rsp),%rax
   e9987:	shr    $1,%rax
   e998a:	je     e9ccc <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18dc>
   e9990:	mov    0x10(%rsp),%rbp
   e9995:	jmp    e99a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15b0>
   e9997:	test   %rax,%rax
   e999a:	je     e9cd1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18e1>
   e99a0:	lea    -0x2(,%rax,2),%rcx
   e99a8:	lea    -0x1(,%rax,2),%rdi
   e99b0:	dec    %rax
   e99b3:	add    $0x2,%rcx
   e99b7:	mov    %rax,%rdx
   e99ba:	mov    0x8(%rsp),%r8
   e99bf:	cmp    %r8,%rcx
   e99c2:	jae    e9ad1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x16e1>
   e99c8:	nopl   0x0(%rax,%rax,1)
   e99d0:	lea    (%rcx,%rcx,2),%rcx
   e99d4:	shl    $0x5,%rcx
   e99d8:	lea    (%rdi,%rdi,2),%rsi
   e99dc:	shl    $0x5,%rsi
   e99e0:	mov    (%rbx,%rcx,1),%rcx
   e99e4:	cmp    (%rbx,%rsi,1),%rcx
   e99e8:	adc    $0x0,%rdi
   e99ec:	cmp    %r8,%rdi
   e99ef:	jae    eb354 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f64>
   e99f5:	lea    (%rdi,%rdi,2),%rcx
   e99f9:	shl    $0x5,%rcx
   e99fd:	lea    (%rdx,%rdx,2),%rdx
   e9a01:	shl    $0x5,%rdx
   e9a05:	mov    (%rbx,%rcx,1),%rsi
   e9a09:	cmp    (%rbx,%rdx,1),%rsi
   e9a0d:	jae    e9997 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15a7>
   e9a0f:	add    %rbx,%rcx
   e9a12:	add    %rbx,%rdx
   e9a15:	movups 0x50(%rdx),%xmm0
   e9a19:	movaps %xmm0,0x70(%rsp)
   e9a1e:	movups 0x40(%rdx),%xmm0
   e9a22:	movaps %xmm0,0x60(%rsp)
   e9a27:	movups (%rdx),%xmm0
   e9a2a:	movups 0x10(%rdx),%xmm1
   e9a2e:	movups 0x20(%rdx),%xmm2
   e9a32:	movups 0x30(%rdx),%xmm3
   e9a36:	movaps %xmm3,0x50(%rsp)
   e9a3b:	movaps %xmm2,0x40(%rsp)
   e9a40:	movaps %xmm1,0x30(%rsp)
   e9a45:	movaps %xmm0,0x20(%rsp)
   e9a4a:	movups 0x40(%rcx),%xmm0
   e9a4e:	movups 0x50(%rcx),%xmm1
   e9a52:	movups (%rcx),%xmm2
   e9a55:	movups 0x10(%rcx),%xmm3
   e9a59:	movups 0x20(%rcx),%xmm4
   e9a5d:	movups 0x30(%rcx),%xmm5
   e9a61:	movups %xmm2,(%rdx)
   e9a64:	movups %xmm1,0x50(%rdx)
   e9a68:	movups %xmm0,0x40(%rdx)
   e9a6c:	movups %xmm5,0x30(%rdx)
   e9a70:	movups %xmm4,0x20(%rdx)
   e9a74:	movups %xmm3,0x10(%rdx)
   e9a78:	movaps 0x70(%rsp),%xmm0
   e9a7d:	movups %xmm0,0x50(%rcx)
   e9a81:	movaps 0x60(%rsp),%xmm0
   e9a86:	movups %xmm0,0x40(%rcx)
   e9a8a:	movdqa 0x20(%rsp),%xmm0
   e9a90:	movdqa 0x30(%rsp),%xmm1
   e9a96:	movdqa 0x40(%rsp),%xmm2
   e9a9c:	movaps 0x50(%rsp),%xmm3
   e9aa1:	movups %xmm3,0x30(%rcx)
   e9aa5:	movdqu %xmm2,0x20(%rcx)
   e9aaa:	movdqu %xmm1,0x10(%rcx)
   e9aaf:	movdqu %xmm0,(%rcx)
   e9ab3:	lea    0x2(%rdi,%rdi,1),%rcx
   e9ab8:	mov    %rdi,%rdx
   e9abb:	lea    0x1(,%rdi,2),%rdi
   e9ac3:	mov    0x8(%rsp),%r8
   e9ac8:	cmp    %r8,%rcx
   e9acb:	jb     e99d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15e0>
   e9ad1:	cmp    %r8,%rcx
   e9ad4:	jne    e9997 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15a7>
   e9ada:	lea    (%rdi,%rdi,2),%rcx
   e9ade:	shl    $0x5,%rcx
   e9ae2:	lea    (%rdx,%rdx,2),%rdx
   e9ae6:	shl    $0x5,%rdx
   e9aea:	mov    (%rbx,%rcx,1),%rsi
   e9aee:	cmp    (%rbx,%rdx,1),%rsi
   e9af2:	jae    e9997 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15a7>
   e9af8:	add    %rbx,%rcx
   e9afb:	add    %rbx,%rdx
   e9afe:	movups 0x50(%rdx),%xmm0
   e9b02:	movaps %xmm0,0x70(%rsp)
   e9b07:	movups 0x40(%rdx),%xmm0
   e9b0b:	movaps %xmm0,0x60(%rsp)
   e9b10:	movups (%rdx),%xmm0
   e9b13:	movups 0x10(%rdx),%xmm1
   e9b17:	movups 0x20(%rdx),%xmm2
   e9b1b:	movups 0x30(%rdx),%xmm3
   e9b1f:	movaps %xmm3,0x50(%rsp)
   e9b24:	movaps %xmm2,0x40(%rsp)
   e9b29:	movaps %xmm1,0x30(%rsp)
   e9b2e:	movaps %xmm0,0x20(%rsp)
   e9b33:	movups 0x40(%rcx),%xmm0
   e9b37:	movups 0x50(%rcx),%xmm1
   e9b3b:	movups (%rcx),%xmm2
   e9b3e:	movups 0x10(%rcx),%xmm3
   e9b42:	movups 0x20(%rcx),%xmm4
   e9b46:	movups 0x30(%rcx),%xmm5
   e9b4a:	movups %xmm2,(%rdx)
   e9b4d:	movups %xmm1,0x50(%rdx)
   e9b51:	movups %xmm0,0x40(%rdx)
   e9b55:	movups %xmm5,0x30(%rdx)
   e9b59:	movups %xmm4,0x20(%rdx)
   e9b5d:	movups %xmm3,0x10(%rdx)
   e9b61:	movaps 0x70(%rsp),%xmm0
   e9b66:	movups %xmm0,0x50(%rcx)
   e9b6a:	movaps 0x60(%rsp),%xmm0
   e9b6f:	movups %xmm0,0x40(%rcx)
   e9b73:	movdqa 0x20(%rsp),%xmm0
   e9b79:	movdqa 0x30(%rsp),%xmm1
   e9b7f:	movdqa 0x40(%rsp),%xmm2
   e9b85:	movaps 0x50(%rsp),%xmm3
   e9b8a:	movups %xmm3,0x30(%rcx)
   e9b8e:	movdqu %xmm2,0x20(%rcx)
   e9b93:	movdqu %xmm1,0x10(%rcx)
   e9b98:	movdqu %xmm0,(%rcx)
   e9b9c:	jmp    e9997 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x15a7>
   e9ba1:	movq   $0x0,0x8(%rsp)
   e9baa:	mov    0xa8(%rsp),%r14
   e9bb2:	jmp    e9cd1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18e1>
   e9bb7:	call   *0x1a333(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e9bbd:	mov    $0x8,%ebx
   e9bc2:	xor    %ebp,%ebp
   e9bc4:	mov    %r12,%r14
   e9bc7:	mov    %rbp,0x230(%rsp)
   e9bcf:	mov    %rbx,0x238(%rsp)
   e9bd7:	mov    %r13,0x240(%rsp)
   e9bdf:	mov    0xa8(%rsp),%rax
   e9be7:	mov    %rax,0x248(%rsp)
   e9bef:	mov    0x188(%rsp),%rdi
   e9bf7:	mov    %r15d,%esi
   e9bfa:	mov    0xb0(%rsp),%rbx
   e9c02:	mov    %ebx,%edx
   e9c04:	mov    %r14,%rcx
   e9c07:	mov    0x10(%rsp),%rbp
   e9c0c:	call   ee880 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::insert>
   e9c11:	mov    0xf0(%rsp),%r13
   e9c19:	movdqa 0x230(%rsp),%xmm0
   e9c22:	movdqa 0x240(%rsp),%xmm1
   e9c2b:	movdqa %xmm1,0x30(%rsp)
   e9c31:	movdqa %xmm0,0x20(%rsp)
   e9c37:	mov    %r14,0x40(%rsp)
   e9c3c:	mov    %r15d,0x48(%rsp)
   e9c41:	mov    %ebx,0x4c(%rsp)
   e9c45:	mov    0x178(%rsp),%rbx
   e9c4d:	cmp    0x168(%rsp),%rbx
   e9c55:	jne    e9c64 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1874>
   e9c57:	lea    0x168(%rsp),%rdi
   e9c5f:	call   e1480 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   e9c64:	mov    0x170(%rsp),%rax
   e9c6c:	lea    (%rbx,%rbx,2),%rcx
   e9c70:	shl    $0x4,%rcx
   e9c74:	movdqa 0x20(%rsp),%xmm0
   e9c7a:	movdqa 0x30(%rsp),%xmm1
   e9c80:	movdqa 0x40(%rsp),%xmm2
   e9c86:	movdqu %xmm2,0x20(%rax,%rcx,1)
   e9c8c:	movdqu %xmm1,0x10(%rax,%rcx,1)
   e9c92:	movdqu %xmm0,(%rax,%rcx,1)
   e9c97:	inc    %rbx
   e9c9a:	mov    %rbx,0x178(%rsp)
   e9ca2:	mov    (%rsp),%rcx
   e9ca6:	mov    %rcx,%rbx
   e9ca9:	sub    %r13,%rbx
   e9cac:	shr    $0x3,%rbx
   e9cb0:	movabs $0x6db6db6db6db6db7,%rax
   e9cba:	imul   %rax,%rbx
   e9cbe:	cmp    %r13,%rcx
   e9cc1:	jne    e8bff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x80f>
   e9cc7:	jmp    ea5f4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2204>
   e9ccc:	mov    0x10(%rsp),%rbp
   e9cd1:	mov    0x208(%rsp),%rax
   e9cd9:	mov    %rax,0x108(%rsp)
   e9ce1:	jmp    e9d09 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1919>
   e9ce3:	mov    0x110(%rsp),%rax
   e9ceb:	mov    %rax,(%rcx)
   e9cee:	mov    %r12,0x80(%rsp)
   e9cf6:	xor    %eax,%eax
   e9cf8:	mov    0xa8(%rsp),%r14
   e9d00:	test   %rax,%rax
   e9d03:	jne    ea8d4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24e4>
   e9d09:	cmpq   $0x0,0x8(%rsp)
   e9d0f:	je     ea4cf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20df>
   e9d15:	lea    0x8(%rbx),%rbp
   e9d19:	cmpq   $0x0,0x0(%rbp)
   e9d1e:	mov    0x8(%rsp),%r15
   e9d23:	je     e9d3f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x194f>
   e9d25:	mov    %rbp,%rdi
   e9d28:	call   e2460 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   e9d2d:	cmp    $0x1,%rax
   e9d31:	je     e9ff2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1c02>
   e9d37:	movq   $0x0,0x0(%rbp)
   e9d3f:	mov    0x48(%rbx),%r8
   e9d43:	test   %r8,%r8
   e9d46:	je     e9fd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1be0>
   e9d4c:	cmp    0x50(%rbx),%r8
   e9d50:	je     e9fd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1be0>
   e9d56:	lea    0x20(%r8),%rax
   e9d5a:	mov    %rax,0x48(%rbx)
   e9d5e:	mov    0x58(%rbx),%rax
   e9d62:	mov    (%r8),%r9
   e9d65:	dec    %r9
   e9d68:	cmp    %r9,%rax
   e9d6b:	jne    e9d7a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x198a>
   e9d6d:	mov    $0x1,%edi
   e9d72:	xor    %r9d,%r9d
   e9d75:	jmp    e9e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a87>
   e9d7a:	mov    %r9,%rcx
   e9d7d:	shr    $0x7,%rcx
   e9d81:	test   %rax,%rax
   e9d84:	je     e9ddf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19ef>
   e9d86:	mov    %r9d,%edx
   e9d89:	and    $0x7f,%edx
   e9d8c:	cmp    $0x1,%rdx
   e9d90:	mov    %rcx,%rsi
   e9d93:	sbb    $0xffffffffffffffff,%rsi
   e9d97:	cmp    $0x2,%rsi
   e9d9b:	jb     eb3aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fba>
   e9da1:	mov    0x18(%r8),%r10
   e9da5:	mov    %r10,%rdx
   e9da8:	cmp    $0x11,%r10
   e9dac:	jb     e9db2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19c2>
   e9dae:	mov    0x10(%r8),%rdx
   e9db2:	lea    0x8(,%rsi,8),%r14
   e9dba:	cmp    %rdx,%r14
   e9dbd:	ja     eb1ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e0f>
   e9dc3:	mov    %rax,%rdi
   e9dc6:	shr    $0x7,%rdi
   e9dca:	cmp    %rsi,%rdi
   e9dcd:	jae    eb3b3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fc3>
   e9dd3:	cmp    $0x11,%r10
   e9dd7:	jb     e9de3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19f3>
   e9dd9:	mov    0x8(%r8),%rdx
   e9ddd:	jmp    e9de7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19f7>
   e9ddf:	xor    %esi,%esi
   e9de1:	jmp    e9dec <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x19fc>
   e9de3:	lea    0x8(%r8),%rdx
   e9de7:	mov    0x8(%rdx,%rdi,8),%rsi
   e9dec:	and    $0x7f,%r9d
   e9df0:	cmp    $0x1,%r9
   e9df4:	sbb    $0xffffffffffffffff,%rcx
   e9df8:	mov    $0x8,%edi
   e9dfd:	cmp    $0x2,%rcx
   e9e01:	jae    e9e21 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a31>
   e9e03:	mov    0x18(%r8),%rcx
   e9e07:	mov    %rcx,%r14
   e9e0a:	cmp    $0x11,%rcx
   e9e0e:	jb     e9e49 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a59>
   e9e10:	mov    0x10(%r8),%r14
   e9e14:	mov    %r14,%rdx
   e9e17:	sub    %rdi,%rdx
   e9e1a:	jae    e9e55 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a65>
   e9e1c:	jmp    eb1c6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd6>
   e9e21:	dec    %rcx
   e9e24:	bsr    %rcx,%rcx
   e9e28:	not    %ecx
   e9e2a:	mov    $0xffffffffffffffff,%rdx
   e9e31:	shr    %cl,%rdx
   e9e34:	lea    0x10(,%rdx,8),%rdi
   e9e3c:	mov    0x18(%r8),%rcx
   e9e40:	mov    %rcx,%r14
   e9e43:	cmp    $0x11,%rcx
   e9e47:	jae    e9e10 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a20>
   e9e49:	mov    %r14,%rdx
   e9e4c:	sub    %rdi,%rdx
   e9e4f:	jb     eb1c6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd6>
   e9e55:	mov    %rdx,%r9
   e9e58:	sub    %rsi,%r9
   e9e5b:	jb     eb1d5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2de5>
   e9e61:	cmp    $0x11,%rcx
   e9e65:	jb     e9e6d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a7d>
   e9e67:	mov    0x8(%r8),%r8
   e9e6b:	jmp    e9e71 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a81>
   e9e6d:	add    $0x8,%r8
   e9e71:	add    %rsi,%rdi
   e9e74:	add    %r8,%rdi
   e9e77:	movq   $0x0,0x58(%rbx)
   e9e7f:	mov    %rdi,0x8(%rbx)
   e9e83:	mov    %r9,0x10(%rbx)
   e9e87:	movq   $0x0,0x18(%rbx)
   e9e8f:	mov    %rax,0x20(%rbx)
   e9e93:	mov    %rbp,%rdi
   e9e96:	call   e2460 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   e9e9b:	cmp    $0x1,%rax
   e9e9f:	je     e9ff2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1c02>
   e9ea5:	movq   $0x0,0x8(%rbx)
   e9ead:	mov    0x48(%rbx),%r8
   e9eb1:	test   %r8,%r8
   e9eb4:	je     e9fd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1be0>
   e9eba:	cmp    0x50(%rbx),%r8
   e9ebe:	je     e9fd0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1be0>
   e9ec4:	lea    0x20(%r8),%rax
   e9ec8:	mov    %rax,0x48(%rbx)
   e9ecc:	mov    0x58(%rbx),%rax
   e9ed0:	mov    (%r8),%rcx
   e9ed3:	dec    %rcx
   e9ed6:	mov    $0x1,%edi
   e9edb:	mov    $0x0,%r9d
   e9ee1:	cmp    %rcx,%rax
   e9ee4:	je     e9e77 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a87>
   e9ee6:	mov    %rcx,%r9
   e9ee9:	shr    $0x7,%r9
   e9eed:	test   %rax,%rax
   e9ef0:	je     e9f4a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b5a>
   e9ef2:	mov    %ecx,%edx
   e9ef4:	and    $0x7f,%edx
   e9ef7:	cmp    $0x1,%rdx
   e9efb:	mov    %r9,%rsi
   e9efe:	sbb    $0xffffffffffffffff,%rsi
   e9f02:	cmp    $0x1,%rsi
   e9f06:	jbe    eb3aa <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fba>
   e9f0c:	mov    0x18(%r8),%r10
   e9f10:	mov    %r10,%rdx
   e9f13:	cmp    $0x11,%r10
   e9f17:	jb     e9f1d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b2d>
   e9f19:	mov    0x10(%r8),%rdx
   e9f1d:	lea    0x8(,%rsi,8),%r14
   e9f25:	cmp    %rdx,%r14
   e9f28:	ja     eb1ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e0f>
   e9f2e:	mov    %rax,%rdi
   e9f31:	shr    $0x7,%rdi
   e9f35:	cmp    %rsi,%rdi
   e9f38:	jae    eb3b3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fc3>
   e9f3e:	cmp    $0x11,%r10
   e9f42:	jb     e9f4e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b5e>
   e9f44:	mov    0x8(%r8),%rdx
   e9f48:	jmp    e9f52 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b62>
   e9f4a:	xor    %esi,%esi
   e9f4c:	jmp    e9f57 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b67>
   e9f4e:	lea    0x8(%r8),%rdx
   e9f52:	mov    0x8(%rdx,%rdi,8),%rsi
   e9f57:	and    $0x7f,%ecx
   e9f5a:	cmp    $0x1,%rcx
   e9f5e:	sbb    $0xffffffffffffffff,%r9
   e9f62:	mov    $0x8,%edi
   e9f67:	cmp    $0x2,%r9
   e9f6b:	jae    e9f8b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b9b>
   e9f6d:	mov    0x18(%r8),%rcx
   e9f71:	mov    %rcx,%r14
   e9f74:	cmp    $0x11,%rcx
   e9f78:	jb     e9fb3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1bc3>
   e9f7a:	mov    0x10(%r8),%r14
   e9f7e:	mov    %r14,%rdx
   e9f81:	sub    %rdi,%rdx
   e9f84:	jae    e9fbf <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1bcf>
   e9f86:	jmp    eb1c6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd6>
   e9f8b:	dec    %r9
   e9f8e:	bsr    %r9,%rcx
   e9f92:	not    %ecx
   e9f94:	mov    $0xffffffffffffffff,%rdx
   e9f9b:	shr    %cl,%rdx
   e9f9e:	lea    0x10(,%rdx,8),%rdi
   e9fa6:	mov    0x18(%r8),%rcx
   e9faa:	mov    %rcx,%r14
   e9fad:	cmp    $0x11,%rcx
   e9fb1:	jae    e9f7a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1b8a>
   e9fb3:	mov    %r14,%rdx
   e9fb6:	sub    %rdi,%rdx
   e9fb9:	jb     eb1c6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dd6>
   e9fbf:	mov    %rdx,%r9
   e9fc2:	sub    %rsi,%r9
   e9fc5:	jae    e9e61 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1a71>
   e9fcb:	jmp    eb1d5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2de5>
   e9fd0:	lea    0x28(%rbx),%rdi
   e9fd4:	cmpq   $0x0,(%rdi)
   e9fd8:	je     ea130 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d40>
   e9fde:	call   e2460 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   e9fe3:	cmp    $0x1,%rax
   e9fe7:	jne    ea120 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d30>
   e9fed:	mov    0x8(%rsp),%r15
   e9ff2:	mov    (%rbx),%rsi
   e9ff5:	mov    %rdx,(%rbx)
   e9ff8:	cmp    $0x3,%r15
   e9ffc:	jb     ea182 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d92>
   ea002:	mov    $0x1,%eax
   ea007:	mov    $0x2,%edx
   ea00c:	xor    %ecx,%ecx
   ea00e:	xchg   %ax,%ax
   ea010:	lea    (%rdx,%rdx,2),%rdx
   ea014:	shl    $0x5,%rdx
   ea018:	lea    (%rax,%rax,2),%rdi
   ea01c:	shl    $0x5,%rdi
   ea020:	mov    (%rbx,%rdx,1),%rdx
   ea024:	cmp    (%rbx,%rdi,1),%rdx
   ea028:	mov    %rax,%rdi
   ea02b:	adc    $0x0,%rdi
   ea02f:	cmp    %r15,%rdi
   ea032:	jae    eb36a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f7a>
   ea038:	cmp    %r15,%rcx
   ea03b:	jae    eb36f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f7f>
   ea041:	lea    (%rdi,%rdi,2),%rax
   ea045:	shl    $0x5,%rax
   ea049:	add    %rbx,%rax
   ea04c:	lea    (%rcx,%rcx,2),%rcx
   ea050:	shl    $0x5,%rcx
   ea054:	mov    (%rax),%rdx
   ea057:	cmp    (%rbx,%rcx,1),%rdx
   ea05b:	jae    ea261 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e71>
   ea061:	add    %rbx,%rcx
   ea064:	movups 0x50(%rcx),%xmm0
   ea068:	movaps %xmm0,0x70(%rsp)
   ea06d:	movups 0x40(%rcx),%xmm0
   ea071:	movaps %xmm0,0x60(%rsp)
   ea076:	movups (%rcx),%xmm0
   ea079:	movups 0x10(%rcx),%xmm1
   ea07d:	movups 0x20(%rcx),%xmm2
   ea081:	movups 0x30(%rcx),%xmm3
   ea085:	movaps %xmm3,0x50(%rsp)
   ea08a:	movaps %xmm2,0x40(%rsp)
   ea08f:	movaps %xmm1,0x30(%rsp)
   ea094:	movaps %xmm0,0x20(%rsp)
   ea099:	movups 0x40(%rax),%xmm0
   ea09d:	movups 0x50(%rax),%xmm1
   ea0a1:	movups (%rax),%xmm2
   ea0a4:	movups 0x10(%rax),%xmm3
   ea0a8:	movups 0x20(%rax),%xmm4
   ea0ac:	movups 0x30(%rax),%xmm5
   ea0b0:	movups %xmm2,(%rcx)
   ea0b3:	movups %xmm1,0x50(%rcx)
   ea0b7:	movups %xmm0,0x40(%rcx)
   ea0bb:	movups %xmm5,0x30(%rcx)
   ea0bf:	movups %xmm4,0x20(%rcx)
   ea0c3:	movups %xmm3,0x10(%rcx)
   ea0c7:	movaps 0x70(%rsp),%xmm0
   ea0cc:	movups %xmm0,0x50(%rax)
   ea0d0:	movaps 0x60(%rsp),%xmm0
   ea0d5:	movups %xmm0,0x40(%rax)
   ea0d9:	movdqa 0x20(%rsp),%xmm0
   ea0df:	movdqa 0x30(%rsp),%xmm1
   ea0e5:	movdqa 0x40(%rsp),%xmm2
   ea0eb:	movaps 0x50(%rsp),%xmm3
   ea0f0:	movups %xmm3,0x30(%rax)
   ea0f4:	movdqu %xmm2,0x20(%rax)
   ea0f9:	movdqu %xmm1,0x10(%rax)
   ea0fe:	movdqu %xmm0,(%rax)
   ea102:	lea    0x1(,%rdi,2),%rax
   ea10a:	lea    0x2(,%rdi,2),%rdx
   ea112:	mov    %rdi,%rcx
   ea115:	cmp    %r15,%rdx
   ea118:	jb     ea010 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1c20>
   ea11e:	jmp    ea18e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1d9e>
   ea120:	lea    0x28(%rbx),%rax
   ea124:	movq   $0x0,(%rax)
   ea12b:	mov    0x8(%rsp),%r15
   ea130:	mov    (%rbx),%rsi
   ea133:	dec    %r15
   ea136:	lea    (%r15,%r15,2),%rax
   ea13a:	shl    $0x5,%rax
   ea13e:	movdqu (%rbx,%rax,1),%xmm0
   ea143:	movdqu 0x10(%rbx,%rax,1),%xmm1
   ea149:	movdqu 0x20(%rbx,%rax,1),%xmm2
   ea14f:	movups 0x30(%rbx,%rax,1),%xmm3
   ea154:	movups 0x40(%rbx,%rax,1),%xmm4
   ea159:	movups 0x50(%rbx,%rax,1),%xmm5
   ea15e:	movups %xmm5,0x50(%rbx)
   ea162:	movups %xmm4,0x40(%rbx)
   ea166:	movups %xmm3,0x30(%rbx)
   ea16a:	movdqu %xmm2,0x20(%rbx)
   ea16f:	movdqu %xmm1,0x10(%rbx)
   ea174:	movdqu %xmm0,(%rbx)
   ea178:	cmp    $0x3,%r15
   ea17c:	jae    ea002 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1c12>
   ea182:	mov    $0x2,%edx
   ea187:	mov    $0x1,%eax
   ea18c:	xor    %edi,%edi
   ea18e:	cmp    %r15,%rdx
   ea191:	jne    ea261 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e71>
   ea197:	cmp    %r15,%rdi
   ea19a:	jae    eb3d6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fe6>
   ea1a0:	lea    (%rax,%rax,2),%rax
   ea1a4:	shl    $0x5,%rax
   ea1a8:	add    %rbx,%rax
   ea1ab:	lea    (%rdi,%rdi,2),%rcx
   ea1af:	shl    $0x5,%rcx
   ea1b3:	mov    (%rax),%rdx
   ea1b6:	cmp    (%rbx,%rcx,1),%rdx
   ea1ba:	jae    ea261 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1e71>
   ea1c0:	add    %rbx,%rcx
   ea1c3:	movups 0x50(%rcx),%xmm0
   ea1c7:	movaps %xmm0,0x70(%rsp)
   ea1cc:	movups 0x40(%rcx),%xmm0
   ea1d0:	movaps %xmm0,0x60(%rsp)
   ea1d5:	movups (%rcx),%xmm0
   ea1d8:	movups 0x10(%rcx),%xmm1
   ea1dc:	movups 0x20(%rcx),%xmm2
   ea1e0:	movups 0x30(%rcx),%xmm3
   ea1e4:	movaps %xmm3,0x50(%rsp)
   ea1e9:	movaps %xmm2,0x40(%rsp)
   ea1ee:	movaps %xmm1,0x30(%rsp)
   ea1f3:	movaps %xmm0,0x20(%rsp)
   ea1f8:	movups 0x40(%rax),%xmm0
   ea1fc:	movups 0x50(%rax),%xmm1
   ea200:	movups (%rax),%xmm2
   ea203:	movups 0x10(%rax),%xmm3
   ea207:	movups 0x20(%rax),%xmm4
   ea20b:	movups 0x30(%rax),%xmm5
   ea20f:	movups %xmm2,(%rcx)
   ea212:	movups %xmm1,0x50(%rcx)
   ea216:	movups %xmm0,0x40(%rcx)
   ea21a:	movups %xmm5,0x30(%rcx)
   ea21e:	movups %xmm4,0x20(%rcx)
   ea222:	movups %xmm3,0x10(%rcx)
   ea226:	movaps 0x70(%rsp),%xmm0
   ea22b:	movups %xmm0,0x50(%rax)
   ea22f:	movaps 0x60(%rsp),%xmm0
   ea234:	movups %xmm0,0x40(%rax)
   ea238:	movdqa 0x20(%rsp),%xmm0
   ea23e:	movdqa 0x30(%rsp),%xmm1
   ea244:	movdqa 0x40(%rsp),%xmm2
   ea24a:	movaps 0x50(%rsp),%xmm3
   ea24f:	movups %xmm3,0x30(%rax)
   ea253:	movdqu %xmm2,0x20(%rax)
   ea258:	movdqu %xmm1,0x10(%rax)
   ea25d:	movdqu %xmm0,(%rax)
   ea261:	mov    0x80(%rsp),%rdx
   ea269:	test   %rdx,%rdx
   ea26c:	je     ea817 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2427>
   ea272:	mov    %rdx,%r12
   ea275:	inc    %r12
   ea278:	je     ead26 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2936>
   ea27e:	mov    %r15,0x8(%rsp)
   ea283:	mov    %edx,%eax
   ea285:	and    $0x7f,%eax
   ea288:	cmp    $0x1,%eax
   ea28b:	jne    ea2c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ed0>
   ea28d:	lea    0x80(%rsp),%rdi
   ea295:	mov    %r12,%rcx
   ea298:	call   ee1a0 <<tk_train::trainers::bpe::positions::Leaf>::push_restart>
   ea29d:	mov    0x10(%rsp),%rbp
   ea2a2:	mov    0xb8(%rsp),%r15
   ea2aa:	mov    0xa8(%rsp),%r14
   ea2b2:	test   %rax,%rax
   ea2b5:	je     e9d09 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1919>
   ea2bb:	jmp    ea8d4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24e4>
   ea2c0:	mov    0x98(%rsp),%rax
   ea2c8:	mov    %rax,%rdx
   ea2cb:	cmp    $0x11,%rax
   ea2cf:	jb     ea2d9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ee9>
   ea2d1:	mov    0x90(%rsp),%rdx
   ea2d9:	mov    $0x8,%r14d
   ea2df:	cmp    $0x7,%rdx
   ea2e3:	jbe    eb266 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e76>
   ea2e9:	lea    0x88(%rsp),%rcx
   ea2f1:	cmp    $0x11,%rax
   ea2f5:	jb     ea2ff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f0f>
   ea2f7:	mov    0x88(%rsp),%rcx
   ea2ff:	mov    %rsi,0x110(%rsp)
   ea307:	mov    %rsi,%rbp
   ea30a:	sub    (%rcx),%rbp
   ea30d:	jb     ea871 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2481>
   ea313:	mov    $0x7f,%eax
   ea318:	bsr    %rbp,%rax
   ea31c:	xor    $0x3f,%eax
   ea31f:	mov    $0x40,%cl
   ea321:	sub    %al,%cl
   ea323:	movzbl %cl,%eax
   ea326:	imul   $0x93,%eax,%r15d
   ea32d:	shr    $0xa,%r15d
   ea331:	imul   $0xffffffb7,%eax,%eax
   ea334:	xor    %ecx,%ecx
   ea336:	cmp    $0x25,%al
   ea338:	setae  %cl
   ea33b:	lea    (%rcx,%r15,1),%eax
   ea33f:	cmp    $0x1,%eax
   ea342:	adc    %ecx,%r15d
   ea345:	lea    0x80(%rsp),%rdi
   ea34d:	mov    %r15,%rsi
   ea350:	call   ee690 <<tk_train::trainers::bpe::positions::Leaf>::reserve>
   ea355:	test   %rax,%rax
   ea358:	jne    ea8d4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24e4>
   ea35e:	mov    0x98(%rsp),%rax
   ea366:	mov    %rax,%r13
   ea369:	cmp    $0x11,%rax
   ea36d:	jb     ea380 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f90>
   ea36f:	mov    0x90(%rsp),%r13
   ea377:	nopw   0x0(%rax,%rax,1)
   ea380:	cmp    $0x11,%rax
   ea384:	jb     ea3c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fd0>
   ea386:	mov    0x90(%rsp),%rcx
   ea38e:	cmp    %rax,%rcx
   ea391:	je     ea3ec <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1ffc>
   ea393:	mov    0x88(%rsp),%rax
   ea39b:	movb   $0x0,(%rax,%rcx,1)
   ea39f:	inc    %rcx
   ea3a2:	mov    %rcx,0x90(%rsp)
   ea3aa:	mov    0x98(%rsp),%rax
   ea3b2:	dec    %r15
   ea3b5:	jne    ea380 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f90>
   ea3b7:	jmp    ea42a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x203a>
   ea3b9:	nopl   0x0(%rax)
   ea3c0:	lea    0x88(%rsp),%rcx
   ea3c8:	lea    0x98(%rsp),%rdx
   ea3d0:	cmp    $0x10,%rax
   ea3d4:	je     ea403 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2013>
   ea3d6:	movb   $0x0,(%rcx,%rax,1)
   ea3da:	incq   (%rdx)
   ea3dd:	mov    0x98(%rsp),%rax
   ea3e5:	dec    %r15
   ea3e8:	jne    ea380 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1f90>
   ea3ea:	jmp    ea42a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x203a>
   ea3ec:	lea    0x88(%rsp),%rdi
   ea3f4:	call   ee7f0 <<smallvec::SmallVec<[u8; 16]>>::reserve_one_unchecked>
   ea3f9:	mov    0x90(%rsp),%rcx
   ea401:	jmp    ea393 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fa3>
   ea403:	lea    0x88(%rsp),%rdi
   ea40b:	call   ee7f0 <<smallvec::SmallVec<[u8; 16]>>::reserve_one_unchecked>
   ea410:	mov    0x88(%rsp),%rcx
   ea418:	mov    0x90(%rsp),%rax
   ea420:	lea    0x90(%rsp),%rdx
   ea428:	jmp    ea3d6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x1fe6>
   ea42a:	lea    0x88(%rsp),%rcx
   ea432:	cmp    $0x11,%rax
   ea436:	jb     ea448 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2058>
   ea438:	mov    0x88(%rsp),%rcx
   ea440:	mov    0x90(%rsp),%rax
   ea448:	cmp    %r13,%rax
   ea44b:	mov    0xb8(%rsp),%r15
   ea453:	jbe    eb323 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f33>
   ea459:	nopl   0x0(%rax)
   ea460:	mov    %ebp,%edx
   ea462:	shr    $0x7,%rbp
   ea466:	je     ea47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x208c>
   ea468:	or     $0x80,%dl
   ea46b:	mov    %dl,(%rcx,%r13,1)
   ea46f:	inc    %r13
   ea472:	cmp    %r13,%rax
   ea475:	jne    ea460 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2070>
   ea477:	jmp    eb323 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f33>
   ea47c:	mov    %dl,(%rcx,%r13,1)
   ea480:	mov    0x98(%rsp),%rax
   ea488:	mov    %rax,%rdx
   ea48b:	cmp    $0x11,%rax
   ea48f:	jb     ea499 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20a9>
   ea491:	mov    0x90(%rsp),%rdx
   ea499:	cmp    $0x8,%rdx
   ea49d:	mov    0x10(%rsp),%rbp
   ea4a2:	mov    0xf0(%rsp),%r13
   ea4aa:	jb     eb248 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e58>
   ea4b0:	lea    0x88(%rsp),%rcx
   ea4b8:	cmp    $0x11,%rax
   ea4bc:	jb     e9ce3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18f3>
   ea4c2:	mov    0x88(%rsp),%rcx
   ea4ca:	jmp    e9ce3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x18f3>
   ea4cf:	cmpq   $0x0,0x108(%rsp)
   ea4d8:	je     ea4e3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x20f3>
   ea4da:	mov    %rbx,%rdi
   ea4dd:	call   *0x19a0d(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea4e3:	movaps 0x80(%rsp),%xmm0
   ea4eb:	movaps %xmm0,0xf0(%rsp)
   ea4f3:	lea    0x90(%rsp),%rax
   ea4fb:	movups (%rax),%xmm0
   ea4fe:	movaps %xmm0,0x280(%rsp)
   ea506:	lea    0xc0(%rsp),%rdi
   ea50e:	call   c4010 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   ea513:	mov    0xb0(%rsp),%rbx
   ea51b:	jmp    e95a9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x11b9>
   ea520:	mov    %rax,(%rsp)
   ea524:	lea    0x1e0(%rsp),%rdi
   ea52c:	call   ed0f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ea531:	cmpq   $0x0,0x168(%rsp)
   ea53a:	je     ea547 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2157>
   ea53c:	mov    $0x8,%edi
   ea541:	call   *0x199a9(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea547:	test   %rbx,%rbx
   ea54a:	je     ea5dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21ed>
   ea550:	xor    %r15d,%r15d
   ea553:	mov    %r13,0xf0(%rsp)
   ea55b:	jmp    ea579 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2189>
   ea55d:	nopl   (%rax)
   ea560:	mov    0xf0(%rsp),%r13
   ea568:	mov    %r14,%rdi
   ea56b:	call   *0x1997f(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea571:	inc    %r15
   ea574:	cmp    %rbx,%r15
   ea577:	je     ea5dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21ed>
   ea579:	imul   $0x38,%r15,%rcx
   ea57d:	lea    (%rcx,%r13,1),%rax
   ea581:	cmpq   $0x0,0x18(%r13,%rcx,1)
   ea587:	je     ea5a0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21b0>
   ea589:	cmpq   $0x11,0x30(%rax)
   ea58e:	jb     ea571 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2181>
   ea590:	mov    0x20(%rax),%r14
   ea594:	jmp    ea568 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2178>
   ea596:	cs nopw 0x0(%rax,%rax,1)
   ea5a0:	mov    0x28(%rax),%rbp
   ea5a4:	test   %rbp,%rbp
   ea5a7:	je     ea571 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2181>
   ea5a9:	mov    0x20(%rax),%r14
   ea5ad:	lea    0x18(%r14),%r12
   ea5b1:	jmp    ea5c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21d9>
   ea5b3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ea5c0:	add    $0x20,%r12
   ea5c4:	dec    %rbp
   ea5c7:	je     ea560 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2170>
   ea5c9:	cmpq   $0x11,(%r12)
   ea5ce:	jb     ea5c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21d0>
   ea5d0:	mov    -0x10(%r12),%rdi
   ea5d5:	call   *0x19915(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea5db:	jmp    ea5c0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x21d0>
   ea5dd:	cmpq   $0x0,0x1c8(%rsp)
   ea5e6:	je     eaa11 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2621>
   ea5ec:	mov    %r13,%rdi
   ea5ef:	jmp    eaa0b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x261b>
   ea5f4:	mov    0x168(%rsp),%rbx
   ea5fc:	mov    0x170(%rsp),%r14
   ea604:	mov    0x178(%rsp),%r15
   ea60c:	lea    0x190(%rsp),%rdi
   ea614:	call   ed6f0 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)>>>
   ea619:	lea    0x1e0(%rsp),%rdi
   ea621:	call   ed0f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ea626:	cmp    $0xffffffffffffffff,%rbx
   ea62a:	je     ea671 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2281>
   ea62c:	mov    %rbx,0x1e0(%rsp)
   ea634:	mov    %r14,0x1e8(%rsp)
   ea63c:	mov    %r15,0x1f0(%rsp)
   ea644:	mov    0x130(%rsp),%rax
   ea64c:	mov    %rax,0xd0(%rsp)
   ea654:	movups 0x120(%rsp),%xmm0
   ea65c:	movaps %xmm0,0xc0(%rsp)
   ea664:	mov    0x158(%rsp),%r15
   ea66c:	jmp    eaae8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26f8>
   ea671:	mov    %r14,(%rsp)
   ea675:	mov    0x160(%rsp),%rax
   ea67d:	mov    (%rax),%rbx
   ea680:	jmp    eaa28 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2638>
   ea685:	mov    $0x2b,%edi
   ea68a:	mov    0x19877(%rip),%r14        # 103f08 <malloc@GLIBC_2.2.5>
   ea691:	call   *%r14
   ea694:	test   %rax,%rax
   ea697:	je     eb3e2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ff2>
   ea69d:	mov    %rax,%r15
   ea6a0:	movups -0xcc85b(%rip),%xmm0        # 1de4c <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x52f>
   ea6a7:	movups %xmm0,0x1b(%rax)
   ea6ab:	movups -0xcc871(%rip),%xmm0        # 1de41 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x524>
   ea6b2:	movups %xmm0,0x10(%rax)
   ea6b6:	movdqu -0xcc88d(%rip),%xmm0        # 1de31 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x514>
   ea6be:	movdqu %xmm0,(%rax)
   ea6c2:	mov    $0x18,%edi
   ea6c7:	call   *%r14
   ea6ca:	test   %rax,%rax
   ea6cd:	je     eb27c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e8c>
   ea6d3:	movq   $0x2b,(%rax)
   ea6da:	mov    %r15,0x8(%rax)
   ea6de:	mov    %rax,(%rsp)
   ea6e2:	movq   $0x2b,0x10(%rax)
   ea6ea:	lea    0x1e0(%rsp),%rdi
   ea6f2:	call   ed0f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ea6f7:	cmpq   $0x0,0x168(%rsp)
   ea700:	jne    ea53c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x214c>
   ea706:	jmp    ea547 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2157>
   ea70b:	mov    0x88(%rsp),%rbx
   ea713:	mov    0x90(%rsp),%r14
   ea71b:	mov    0xf0(%rsp),%rax
   ea723:	mov    %rax,0x198(%rsp)
   ea72b:	mov    $0x1a,%edi
   ea730:	mov    0x197d1(%rip),%r13        # 103f08 <malloc@GLIBC_2.2.5>
   ea737:	call   *%r13
   ea73a:	test   %rax,%rax
   ea73d:	je     eb3f6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3006>
   ea743:	mov    %rax,%r15
   ea746:	movups -0xccc4e(%rip),%xmm0        # 1daff <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1e2>
   ea74d:	movups %xmm0,0xa(%rax)
   ea751:	movdqu -0xccc64(%rip),%xmm0        # 1daf5 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1d8>
   ea759:	movdqu %xmm0,(%rax)
   ea75d:	mov    $0x18,%edi
   ea762:	call   *%r13
   ea765:	test   %rax,%rax
   ea768:	je     eb290 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ea0>
   ea76e:	movq   $0x1a,(%rax)
   ea775:	mov    %r15,0x8(%rax)
   ea779:	movq   $0x1a,0x10(%rax)
   ea781:	test   %r14,%r14
   ea784:	mov    %rax,(%rsp)
   ea788:	je     ea7b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23c4>
   ea78a:	lea    0x18(%rbx),%r15
   ea78e:	jmp    ea799 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23a9>
   ea790:	add    $0x20,%r15
   ea794:	dec    %r14
   ea797:	je     ea7ab <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23bb>
   ea799:	cmpq   $0x11,(%r15)
   ea79d:	jb     ea790 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23a0>
   ea79f:	mov    -0x10(%r15),%rdi
   ea7a3:	call   *0x19747(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea7a9:	jmp    ea790 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23a0>
   ea7ab:	mov    %rbx,%rdi
   ea7ae:	call   *0x1973c(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea7b4:	mov    0x40(%rsp),%rax
   ea7b9:	cmp    $0x1,%rax
   ea7bd:	jbe    ea7ef <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23ff>
   ea7bf:	mov    0x20(%rsp),%rbx
   ea7c4:	mov    0x28(%rsp),%r14
   ea7c9:	test   %r14,%r14
   ea7cc:	je     ea809 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2419>
   ea7ce:	lea    0x18(%rbx),%r15
   ea7d2:	jmp    ea7dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23ed>
   ea7d4:	add    $0x20,%r15
   ea7d8:	dec    %r14
   ea7db:	je     ea809 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2419>
   ea7dd:	cmpq   $0x11,(%r15)
   ea7e1:	jb     ea7d4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23e4>
   ea7e3:	mov    -0x10(%r15),%rdi
   ea7e7:	call   *0x19703(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea7ed:	jmp    ea7d4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23e4>
   ea7ef:	test   %rax,%rax
   ea7f2:	je     ea95e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x256e>
   ea7f8:	cmpq   $0x11,0x38(%rsp)
   ea7fe:	jb     ea95e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x256e>
   ea804:	mov    0x28(%rsp),%rbx
   ea809:	mov    %rbx,%rdi
   ea80c:	call   *0x196de(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea812:	jmp    ea95e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x256e>
   ea817:	mov    $0x1f,%r14d
   ea81d:	mov    $0x1f,%edi
   ea822:	mov    0x196df(%rip),%r13        # 103f08 <malloc@GLIBC_2.2.5>
   ea829:	call   *%r13
   ea82c:	test   %rax,%rax
   ea82f:	je     eb435 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3045>
   ea835:	mov    %rax,%r15
   ea838:	movups -0xcc90f(%rip),%xmm0        # 1df30 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x613>
   ea83f:	movups %xmm0,0xf(%rax)
   ea843:	movdqu -0xcc92a(%rip),%xmm0        # 1df21 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x604>
   ea84b:	movdqu %xmm0,(%rax)
   ea84f:	mov    $0x18,%edi
   ea854:	call   *%r13
   ea857:	test   %rax,%rax
   ea85a:	je     eb30f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f1f>
   ea860:	movq   $0x1f,(%rax)
   ea867:	mov    %rax,%rcx
   ea86a:	mov    $0x1f,%eax
   ea86f:	jmp    ea8c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24d9>
   ea871:	mov    $0x1c,%r14d
   ea877:	mov    $0x1c,%edi
   ea87c:	mov    0x19685(%rip),%r13        # 103f08 <malloc@GLIBC_2.2.5>
   ea883:	call   *%r13
   ea886:	test   %rax,%rax
   ea889:	je     eb435 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3045>
   ea88f:	mov    %rax,%r15
   ea892:	movups -0xccdb4(%rip),%xmm0        # 1dae5 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1c8>
   ea899:	movups %xmm0,0xc(%rax)
   ea89d:	movdqu -0xccdcc(%rip),%xmm0        # 1dad9 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1bc>
   ea8a5:	movdqu %xmm0,(%rax)
   ea8a9:	mov    $0x18,%edi
   ea8ae:	call   *%r13
   ea8b1:	test   %rax,%rax
   ea8b4:	je     eb340 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f50>
   ea8ba:	movq   $0x1c,(%rax)
   ea8c1:	mov    %rax,%rcx
   ea8c4:	mov    $0x1c,%eax
   ea8c9:	mov    %r15,0x8(%rcx)
   ea8cd:	mov    %rax,0x10(%rcx)
   ea8d1:	mov    %rcx,%rax
   ea8d4:	cmpq   $0x0,0x108(%rsp)
   ea8dd:	je     ea8ee <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24fe>
   ea8df:	mov    %rbx,%rdi
   ea8e2:	mov    %rax,%rbx
   ea8e5:	call   *0x19605(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea8eb:	mov    %rbx,%rax
   ea8ee:	cmpq   $0x0,0x80(%rsp)
   ea8f7:	mov    %rax,(%rsp)
   ea8fb:	je     ea91b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x252b>
   ea8fd:	cmpq   $0x11,0x98(%rsp)
   ea906:	jb     ea951 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2561>
   ea908:	mov    0x88(%rsp),%rbx
   ea910:	mov    %rbx,%rdi
   ea913:	call   *0x195d7(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea919:	jmp    ea951 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2561>
   ea91b:	mov    0x90(%rsp),%r14
   ea923:	test   %r14,%r14
   ea926:	je     ea951 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2561>
   ea928:	mov    0x88(%rsp),%rbx
   ea930:	lea    0x18(%rbx),%r15
   ea934:	jmp    ea93f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x254f>
   ea936:	add    $0x20,%r15
   ea93a:	dec    %r14
   ea93d:	je     ea910 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2520>
   ea93f:	cmpq   $0x11,(%r15)
   ea943:	jb     ea936 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2546>
   ea945:	mov    -0x10(%r15),%rdi
   ea949:	call   *0x195a1(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea94f:	jmp    ea936 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2546>
   ea951:	lea    0xc0(%rsp),%rdi
   ea959:	call   c4010 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   ea95e:	lea    0x190(%rsp),%rdi
   ea966:	call   ed6f0 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)>>>
   ea96b:	lea    0x1e0(%rsp),%rdi
   ea973:	call   ed0f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   ea978:	mov    0x170(%rsp),%rbx
   ea980:	mov    0x178(%rsp),%r13
   ea988:	test   %r13,%r13
   ea98b:	je     ea9fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x260d>
   ea98d:	xor    %ebp,%ebp
   ea98f:	jmp    ea9a2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25b2>
   ea991:	mov    %r14,%rdi
   ea994:	call   *0x19556(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea99a:	inc    %rbp
   ea99d:	cmp    %r13,%rbp
   ea9a0:	je     ea9fd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x260d>
   ea9a2:	lea    0x0(,%rbp,2),%rcx
   ea9aa:	add    %rbp,%rcx
   ea9ad:	shl    $0x4,%rcx
   ea9b1:	lea    (%rbx,%rcx,1),%rax
   ea9b5:	cmpq   $0x0,(%rbx,%rcx,1)
   ea9ba:	je     ea9c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25d9>
   ea9bc:	cmpq   $0x11,0x18(%rax)
   ea9c1:	jb     ea99a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25aa>
   ea9c3:	mov    0x8(%rax),%r14
   ea9c7:	jmp    ea991 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25a1>
   ea9c9:	mov    0x10(%rax),%r15
   ea9cd:	test   %r15,%r15
   ea9d0:	je     ea99a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25aa>
   ea9d2:	mov    0x8(%rax),%r14
   ea9d6:	lea    0x18(%r14),%r12
   ea9da:	jmp    ea9e9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25f9>
   ea9dc:	nopl   0x0(%rax)
   ea9e0:	add    $0x20,%r12
   ea9e4:	dec    %r15
   ea9e7:	je     ea991 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25a1>
   ea9e9:	cmpq   $0x11,(%r12)
   ea9ee:	jb     ea9e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25f0>
   ea9f0:	mov    -0x10(%r12),%rdi
   ea9f5:	call   *0x194f5(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ea9fb:	jmp    ea9e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x25f0>
   ea9fd:	cmpq   $0x0,0x168(%rsp)
   eaa06:	je     eaa11 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2621>
   eaa08:	mov    %rbx,%rdi
   eaa0b:	call   *0x194df(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eaa11:	mov    0x10(%rsp),%rbp
   eaa16:	mov    0x160(%rsp),%rax
   eaa1e:	mov    (%rax),%rbx
   eaa21:	lea    0x183d0(%rip),%r15        # 102df8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3300>
   eaa28:	xor    %eax,%eax
   eaa2a:	mov    $0x1,%ecx
   eaa2f:	lock cmpxchg %ecx,(%rbx)
   eaa33:	jne    eaa7b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x268b>
   eaa35:	mov    0x1b84c(%rip),%rax        # 106288 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   eaa3c:	shl    $1,%rax
   eaa3f:	test   %rax,%rax
   eaa42:	jne    ead73 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2983>
   eaa48:	movzbl 0x4(%rbx),%eax
   eaa4c:	test   %al,%al
   eaa4e:	je     eab5a <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x276a>
   eaa54:	mov    0x1b82d(%rip),%rax        # 106288 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   eaa5b:	movabs $0x7fffffffffffffff,%rcx
   eaa65:	test   %rcx,%rax
   eaa68:	jne    eadb1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29c1>
   eaa6e:	xor    %eax,%eax
   eaa70:	xchg   %eax,(%rbx)
   eaa72:	cmp    $0x2,%eax
   eaa75:	je     ead92 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29a2>
   eaa7b:	movq   $0xffffffffffffffff,0x1e0(%rsp)
   eaa87:	mov    (%r15),%rax
   eaa8a:	test   %rax,%rax
   eaa8d:	mov    0x118(%rsp),%rbx
   eaa95:	je     eaa9d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26ad>
   eaa97:	mov    (%rsp),%rdi
   eaa9b:	call   *%rax
   eaa9d:	cmpq   $0x0,0x8(%r15)
   eaaa2:	je     eaaae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x26be>
   eaaa4:	mov    (%rsp),%rdi
   eaaa8:	call   *0x19442(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eaaae:	mov    0x1e0(%rsp),%rax
   eaab6:	mov    0x130(%rsp),%rcx
   eaabe:	mov    %rcx,0xd0(%rsp)
   eaac6:	movups 0x120(%rsp),%xmm0
   eaace:	movaps %xmm0,0xc0(%rsp)
   eaad6:	cmp    $0xffffffffffffffff,%rax
   eaada:	mov    0x158(%rsp),%r15
   eaae2:	je     eabe1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x27f1>
   eaae8:	movups 0x120(%rsp),%xmm0
   eaaf0:	movaps %xmm0,0x20(%rsp)
   eaaf5:	mov    0x130(%rsp),%rbx
   eaafd:	mov    %rbx,0x30(%rsp)
   eab02:	cmp    0x20(%rsp),%rbx
   eab07:	jne    eab13 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2723>
   eab09:	lea    0x20(%rsp),%rdi
   eab0e:	call   ebc90 <<alloc::raw_vec::RawVec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>::grow_one>
   eab13:	mov    0x28(%rsp),%rax
   eab18:	lea    (%rbx,%rbx,2),%rcx
   eab1c:	mov    0x1f0(%rsp),%rdx
   eab24:	mov    %rdx,0x10(%rax,%rcx,8)
   eab29:	movups 0x1e0(%rsp),%xmm0
   eab31:	movups %xmm0,(%rax,%rcx,8)
   eab35:	inc    %rbx
   eab38:	mov    %rbx,0xd0(%rsp)
   eab40:	movaps 0x20(%rsp),%xmm0
   eab45:	movaps %xmm0,0xc0(%rsp)
   eab4d:	mov    0x118(%rsp),%rbx
   eab55:	jmp    eabe5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x27f5>
   eab5a:	xor    %eax,%eax
   eab5c:	mov    0x8(%rbx),%r14
   eab60:	test   %r14,%r14
   eab63:	jne    eab71 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2781>
   eab65:	mov    (%rsp),%rcx
   eab69:	mov    %rcx,0x8(%rbx)
   eab6d:	mov    %r15,0x10(%rbx)
   eab71:	test   %al,%al
   eab73:	jne    eab8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x279f>
   eab75:	mov    0x1b70c(%rip),%rax        # 106288 <std::panicking::panic_count::GLOBAL_PANIC_COUNT>
   eab7c:	movabs $0x7fffffffffffffff,%rcx
   eab86:	test   %rcx,%rax
   eab89:	jne    eae14 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a24>
   eab8f:	xor    %eax,%eax
   eab91:	xchg   %eax,(%rbx)
   eab93:	cmp    $0x2,%eax
   eab96:	je     eadc7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x29d7>
   eab9c:	movq   $0xffffffffffffffff,0x1e0(%rsp)
   eaba8:	test   %r14,%r14
   eabab:	jne    eaa87 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2697>
   eabb1:	mov    0x130(%rsp),%rax
   eabb9:	mov    %rax,0xd0(%rsp)
   eabc1:	movups 0x120(%rsp),%xmm0
   eabc9:	movaps %xmm0,0xc0(%rsp)
   eabd1:	mov    0x118(%rsp),%rbx
   eabd9:	mov    0x158(%rsp),%r15
   eabe1:	movb   $0x1,0x0(%rbp)
   eabe5:	mov    0xd0(%rsp),%rax
   eabed:	mov    %rax,0x30(%rsp)
   eabf2:	movaps 0xc0(%rsp),%xmm0
   eabfa:	movaps %xmm0,0x20(%rsp)
   eabff:	mov    0x260(%rsp),%rcx
   eac07:	add    $0x70,%rbx
   eac0b:	mov    0x30(%rsp),%rax
   eac10:	mov    %rax,0x130(%rsp)
   eac18:	movdqa 0x20(%rsp),%xmm0
   eac1e:	movdqa %xmm0,0x120(%rsp)
   eac27:	movzbl 0x0(%rbp),%eax
   eac2b:	test   %al,%al
   eac2d:	je     e8550 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x160>
   eac33:	jmp    eb117 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2d27>
   eac38:	mov    $0x22,%edi
   eac3d:	call   *0x192c5(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   eac43:	test   %rax,%rax
   eac46:	je     eb444 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3054>
   eac4c:	mov    %rax,%rbx
   eac4f:	movups -0xccce8(%rip),%xmm0        # 1df6e <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x651>
   eac56:	movups %xmm0,0x10(%rax)
   eac5a:	movdqu -0xccd04(%rip),%xmm0        # 1df5e <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x641>
   eac62:	movdqu %xmm0,(%rax)
   eac66:	movw   $0x3436,0x20(%rax)
   eac6c:	call   ed7e0 <alloc::boxed::box_new_uninit>
   eac71:	movq   $0x22,(%rax)
   eac78:	mov    %rbx,0x8(%rax)
   eac7c:	mov    %rax,(%rsp)
   eac80:	movq   $0x22,0x10(%rax)
   eac88:	jmp    ea95e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x256e>
   eac8d:	mov    %r8,0x198(%rsp)
   eac95:	mov    $0x1c,%r14d
   eac9b:	mov    $0x1c,%edi
   eaca0:	mov    0x19261(%rip),%r13        # 103f08 <malloc@GLIBC_2.2.5>
   eaca7:	call   *%r13
   eacaa:	test   %rax,%rax
   eacad:	je     eb455 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3065>
   eacb3:	mov    %rax,%rbx
   eacb6:	movups -0xcd1d8(%rip),%xmm0        # 1dae5 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1c8>
   eacbd:	movups %xmm0,0xc(%rax)
   eacc1:	movdqu -0xcd1f0(%rip),%xmm0        # 1dad9 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x1bc>
   eacc9:	movdqu %xmm0,(%rax)
   eaccd:	mov    $0x18,%edi
   eacd2:	call   *%r13
   eacd5:	test   %rax,%rax
   eacd8:	je     eb396 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2fa6>
   eacde:	movq   $0x1c,(%rax)
   eace5:	mov    %rbx,0x8(%rax)
   eace9:	mov    %rax,(%rsp)
   eaced:	movq   $0x1c,0x10(%rax)
   eacf5:	cmp    $0x11,%r15
   eacf9:	jb     ea7b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23c4>
   eacff:	mov    0xc8(%rsp),%rdi
   ead07:	jmp    ea7ae <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23be>
   ead0c:	cmp    $0x15,%rbx
   ead10:	jae    eaea5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ab5>
   ead16:	mov    %r13,%rdi
   ead19:	mov    %rbx,%rsi
   ead1c:	call   ed580 <core::slice::sort::shared::smallsort::insertion_sort_shift_left::<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions), <[((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)]>::sort_by_key<(usize, (u32, u32)), <tk_train::trainers::bpe::index::Shard>::commit::{closure#0}>::{closure#0}>>
   ead21:	jmp    e8bbe <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7ce>
   ead26:	mov    $0x1e,%r14d
   ead2c:	mov    $0x1e,%edi
   ead31:	call   *0x191d1(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   ead37:	test   %rax,%rax
   ead3a:	je     eb435 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3045>
   ead40:	mov    %rax,%r15
   ead43:	movups -0xcd000(%rip),%xmm0        # 1dd4a <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x42d>
   ead4a:	movups %xmm0,0xe(%rax)
   ead4e:	movdqu -0xcd01a(%rip),%xmm0        # 1dd3c <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x41f>
   ead56:	movdqu %xmm0,(%rax)
   ead5a:	call   ed7e0 <alloc::boxed::box_new_uninit>
   ead5f:	mov    %rax,%rcx
   ead62:	movq   $0x1e,(%rax)
   ead69:	mov    $0x1e,%eax
   ead6e:	jmp    ea8c9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x24d9>
   ead73:	call   8daf0 <std::panicking::panic_count::is_zero_slow_path>
   ead78:	movzbl 0x4(%rbx),%ecx
   ead7c:	test   %cl,%cl
   ead7e:	je     eadfb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2a0b>
   ead80:	test   %al,%al
   ead82:	mov    0x10(%rsp),%rbp
   ead87:	jne    eaa54 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2664>
   ead8d:	jmp    eaa6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x267e>
   ead92:	mov    $0xca,%edi
   ead97:	mov    %rbx,%rsi
   ead9a:	mov    $0x81,%edx
   ead9f:	mov    $0x1,%ecx
   eada4:	xor    %eax,%eax
   eada6:	call   *0x1918c(%rip)        # 103f38 <syscall@GLIBC_2.2.5>
   eadac:	jmp    eaa7b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x268b>
   eadb1:	call   8daf0 <std::panicking::panic_count::is_zero_slow_path>
   eadb6:	test   %al,%al
   eadb8:	jne    eaa6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x267e>
   eadbe:	movb   $0x1,0x4(%rbx)
   eadc2:	jmp    eaa6e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x267e>
   eadc7:	mov    $0xca,%edi
   eadcc:	mov    %rbx,%rsi
   eadcf:	mov    $0x81,%edx
   eadd4:	mov    $0x1,%ecx
   eadd9:	xor    %eax,%eax
   eaddb:	call   *0x19157(%rip)        # 103f38 <syscall@GLIBC_2.2.5>
   eade1:	movq   $0xffffffffffffffff,0x1e0(%rsp)
   eaded:	test   %r14,%r14
   eadf0:	jne    eaa87 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2697>
   eadf6:	jmp    eabb1 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x27c1>
   eadfb:	xor    $0x1,%al
   eadfd:	mov    0x10(%rsp),%rbp
   eae02:	mov    0x8(%rbx),%r14
   eae06:	test   %r14,%r14
   eae09:	jne    eab71 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2781>
   eae0f:	jmp    eab65 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2775>
   eae14:	call   8daf0 <std::panicking::panic_count::is_zero_slow_path>
   eae19:	test   %al,%al
   eae1b:	jne    eab8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x279f>
   eae21:	movb   $0x1,0x4(%rbx)
   eae25:	jmp    eab8f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x279f>
   eae2a:	mov    0xf0(%rsp),%rax
   eae32:	mov    %rax,0x198(%rsp)
   eae3a:	mov    $0x20,%r14d
   eae40:	mov    $0x20,%edi
   eae45:	call   *0x190bd(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   eae4b:	test   %rax,%rax
   eae4e:	je     eb455 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3065>
   eae54:	mov    %rax,%rbx
   eae57:	movups -0xd91ae(%rip),%xmm0        # 11cb0 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x5f30>
   eae5e:	movups %xmm0,0x10(%rax)
   eae62:	movdqu -0xd91ca(%rip),%xmm0        # 11ca0 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x5f20>
   eae6a:	movdqu %xmm0,(%rax)
   eae6e:	call   ed7e0 <alloc::boxed::box_new_uninit>
   eae73:	mov    %rax,(%rsp)
   eae77:	mov    (%rsp),%rax
   eae7b:	movq   $0x20,(%rax)
   eae82:	mov    %rbx,0x8(%rax)
   eae86:	movq   $0x20,0x10(%rax)
   eae8e:	mov    0xd8(%rsp),%r15
   eae96:	cmp    $0x11,%r15
   eae9a:	jb     ea7b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x23c4>
   eaea0:	jmp    eacff <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x290f>
   eaea5:	mov    %r13,%rdi
   eaea8:	mov    %rbx,%rsi
   eaeab:	call   c56c0 <core::slice::sort::stable::driftsort_main::<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions), <[((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)]>::sort_by_key<(usize, (u32, u32)), <tk_train::trainers::bpe::index::Shard>::commit::{closure#0}>::{closure#0}, alloc::vec::Vec<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)>>>
   eaeb0:	jmp    e8bbe <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7ce>
   eaeb5:	test   %dl,%dl
   eaeb7:	je     eaed9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ae9>
   eaeb9:	mov    %fs:0xffffffffffffffd0,%rax
   eaec2:	test   %rax,%rax
   eaec5:	je     eaee7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2af7>
   eaec7:	add    $0x110,%rax
   eaecd:	jmp    eaf0e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2b1e>
   eaecf:	call   693e0 <alloc::raw_vec::capacity_overflow>
   eaed4:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eaed9:	test   %rcx,%rcx
   eaedc:	je     e8497 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0xa7>
   eaee2:	shr    $1,%rcx
   eaee5:	jmp    eaf22 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2b32>
   eaee7:	mov    %r15,%rax
   eaeea:	mov    %rsi,%r15
   eaeed:	mov    %rax,0x158(%rsp)
   eaef5:	mov    %rcx,%r12
   eaef8:	call   7f700 <rayon_core::registry::global_registry>
   eaefd:	mov    %r12,%rcx
   eaf00:	mov    0x158(%rsp),%rdx
   eaf08:	mov    %r15,%rsi
   eaf0b:	mov    %rdx,%r15
   eaf0e:	mov    (%rax),%rax
   eaf11:	mov    0x208(%rax),%rax
   eaf18:	shr    $1,%rcx
   eaf1b:	cmp    %rax,%rcx
   eaf1e:	cmovbe %rax,%rcx
   eaf22:	mov    %rcx,0x270(%rsp)
   eaf2a:	mov    %rsi,0xc0(%rsp)
   eaf32:	sub    %rsi,%r14
   eaf35:	jb     eb1e7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2df7>
   eaf3b:	imul   $0x70,%rsi,%rax
   eaf3f:	add    %r13,%rax
   eaf42:	mov    0x8(%rbx),%rcx
   eaf46:	mov    0x10(%rbx),%rdx
   eaf4a:	mov    0x18(%rbx),%rdi
   eaf4e:	lea    0x268(%rsp),%r8
   eaf56:	mov    %r8,0x290(%rsp)
   eaf5e:	lea    0xc0(%rsp),%r8
   eaf66:	mov    %r8,0x298(%rsp)
   eaf6e:	lea    0x270(%rsp),%r9
   eaf76:	mov    %r9,0x2a0(%rsp)
   eaf7e:	mov    %rax,0x2a8(%rsp)
   eaf86:	mov    %r14,0x2b0(%rsp)
   eaf8e:	mov    %rbp,0x2b8(%rsp)
   eaf96:	mov    %rcx,0x2c0(%rsp)
   eaf9e:	mov    %rdx,0x2c8(%rsp)
   eafa6:	mov    %rdi,0x2d0(%rsp)
   eafae:	mov    %r8,0x2d8(%rsp)
   eafb6:	mov    %r9,0x2e0(%rsp)
   eafbe:	mov    %r13,0x2e8(%rsp)
   eafc6:	mov    %rsi,0x2f0(%rsp)
   eafce:	mov    %rbp,0x2f8(%rsp)
   eafd6:	mov    %rcx,0x300(%rsp)
   eafde:	mov    %rdx,0x308(%rsp)
   eafe6:	mov    %rdi,0x310(%rsp)
   eafee:	mov    %fs:0xffffffffffffffd0,%rdx
   eaff7:	test   %rdx,%rdx
   eaffa:	jne    eb023 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c33>
   eaffc:	call   7f700 <rayon_core::registry::global_registry>
   eb001:	mov    (%rax),%rsi
   eb004:	mov    %fs:0xffffffffffffffd0,%rdx
   eb00d:	test   %rdx,%rdx
   eb010:	je     eb212 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e22>
   eb016:	cmp    %rsi,0x110(%rdx)
   eb01d:	jne    eb22d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e3d>
   eb023:	lea    0x20(%rsp),%rdi
   eb028:	lea    0x290(%rsp),%rsi
   eb030:	xor    %ecx,%ecx
   eb032:	call   ebeb0 <rayon_core::join::join_context::<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}>
   eb037:	mov    0x20(%rsp),%rdx
   eb03c:	mov    0x28(%rsp),%rcx
   eb041:	mov    0x30(%rsp),%rax
   eb046:	mov    0x48(%rsp),%rsi
   eb04b:	mov    %rsi,0x130(%rsp)
   eb053:	movups 0x38(%rsp),%xmm0
   eb058:	movaps %xmm0,0x120(%rsp)
   eb060:	test   %rcx,%rcx
   eb063:	je     eb0a3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2cb3>
   eb065:	mov    0x120(%rsp),%rsi
   eb06d:	movq   $0x0,0x120(%rsp)
   eb079:	test   %rsi,%rsi
   eb07c:	je     eb0dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ced>
   eb07e:	mov    %rsi,0x18(%rcx)
   eb082:	mov    %rcx,0x20(%rsi)
   eb086:	mov    0x128(%rsp),%rcx
   eb08e:	xorps  %xmm0,%xmm0
   eb091:	add    0x130(%rsp),%rax
   eb099:	movups %xmm0,0x128(%rsp)
   eb0a1:	jmp    eb0dd <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2ced>
   eb0a3:	mov    0x120(%rsp),%rsi
   eb0ab:	mov    0x128(%rsp),%rcx
   eb0b3:	mov    %rdx,0x120(%rsp)
   eb0bb:	movq   $0x0,0x128(%rsp)
   eb0c7:	mov    0x130(%rsp),%rdi
   eb0cf:	mov    %rax,0x130(%rsp)
   eb0d7:	mov    %rsi,%rdx
   eb0da:	mov    %rdi,%rax
   eb0dd:	mov    %rdx,(%r15)
   eb0e0:	mov    %rcx,0x8(%r15)
   eb0e4:	mov    %rax,0x10(%r15)
   eb0e8:	lea    0x120(%rsp),%rdi
   eb0f0:	call   eb970 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   eb0f5:	jmp    e846e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x7e>
   eb0fa:	lea    -0xcd28b(%rip),%rdi        # 1de76 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x559>
   eb101:	lea    0x187d0(%rip),%rdx        # 1038d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   eb108:	mov    $0x16,%esi
   eb10d:	call   71e50 <core::option::expect_failed>
   eb112:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb117:	mov    %rbp,0x138(%rsp)
   eb11f:	mov    0x160(%rsp),%rax
   eb127:	mov    %rax,0x140(%rsp)
   eb12f:	mov    0x180(%rsp),%rax
   eb137:	mov    %rax,0x148(%rsp)
   eb13f:	movaps 0x120(%rsp),%xmm0
   eb147:	movaps %xmm0,0x20(%rsp)
   eb14c:	mov    0x1c0(%rsp),%rax
   eb154:	mov    %rax,0x50(%rsp)
   eb159:	mov    0x130(%rsp),%rax
   eb161:	mov    %rax,0x30(%rsp)
   eb166:	mov    0x138(%rsp),%rax
   eb16e:	mov    %rax,0x38(%rsp)
   eb173:	mov    0x140(%rsp),%rax
   eb17b:	mov    %rax,0x40(%rsp)
   eb180:	mov    0x148(%rsp),%rax
   eb188:	mov    %rax,0x48(%rsp)
   eb18d:	jmp    e8461 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x71>
   eb192:	lea    -0xcd33d(%rip),%rdi        # 1de5c <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x53f>
   eb199:	lea    0x18720(%rip),%rdx        # 1038c0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3dc8>
   eb1a0:	mov    $0x1a,%esi
   eb1a5:	call   71e50 <core::option::expect_failed>
   eb1aa:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb1af:	mov    %rsi,%r14
   eb1b2:	lea    0x183d7(%rip),%rcx        # 103590 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   eb1b9:	mov    %r14,%rdx
   eb1bc:	call   70c70 <core::slice::index::slice_index_fail>
   eb1c1:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb1c6:	mov    %r14,%rdx
   eb1c9:	lea    0x183c0(%rip),%rcx        # 103590 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   eb1d0:	jmp    eb26f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7f>
   eb1d5:	mov    %rsi,%rdi
   eb1d8:	mov    %rdx,%r14
   eb1db:	lea    0x183f6(%rip),%rcx        # 1035d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ae0>
   eb1e2:	jmp    eb26f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7f>
   eb1e7:	lea    -0xcd5e1(%rip),%rdi        # 1dc0d <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f0>
   eb1ee:	lea    0x189e3(%rip),%rdx        # 103bd8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x40e0>
   eb1f5:	mov    $0x13,%esi
   eb1fa:	call   6eef0 <core::panicking::panic_fmt>
   eb1ff:	mov    $0x8,%edi
   eb204:	lea    0x183e5(%rip),%rcx        # 1035f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3af8>
   eb20b:	jmp    eb26f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7f>
   eb20d:	mov    %r14,%rsi
   eb210:	jmp    eb1b2 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dc2>
   eb212:	sub    $0xffffffffffffff80,%rsi
   eb216:	lea    0x20(%rsp),%rdi
   eb21b:	lea    0x290(%rsp),%rdx
   eb223:	call   ebdc0 <<rayon_core::registry::Registry>::in_worker_cold::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>)>>
   eb228:	jmp    eb037 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c47>
   eb22d:	sub    $0xffffffffffffff80,%rsi
   eb231:	lea    0x20(%rsp),%rdi
   eb236:	lea    0x290(%rsp),%rcx
   eb23e:	call   ec370 <<rayon_core::registry::Registry>::in_worker_cross::<rayon_core::join::join_context<rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#0}, rayon::iter::plumbing::bridge_producer_consumer::helper<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>::{closure#1}, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>::{closure#0}, (alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>)>>
   eb243:	jmp    eb037 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2c47>
   eb248:	xor    %edi,%edi
   eb24a:	lea    0x18357(%rip),%rcx        # 1035a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ab0>
   eb251:	jmp    eb26f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2e7f>
   eb253:	mov    $0x8,%esi
   eb258:	xor    %edi,%edi
   eb25a:	lea    0x187c7(%rip),%rcx        # 103a28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f30>
   eb261:	jmp    eb1b9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2dc9>
   eb266:	xor    %edi,%edi
   eb268:	lea    0x187b9(%rip),%rcx        # 103a28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f30>
   eb26f:	mov    %r14,%rsi
   eb272:	call   70c70 <core::slice::index::slice_index_fail>
   eb277:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb27c:	mov    $0x8,%edi
   eb281:	mov    $0x18,%esi
   eb286:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb28b:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb290:	mov    $0x8,%edi
   eb295:	mov    $0x18,%esi
   eb29a:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb29f:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb2a4:	mov    $0x8,%esi
   eb2a9:	xor    %edi,%edi
   eb2ab:	jmp    eb2ed <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2efd>
   eb2ad:	test   %rax,%rax
   eb2b0:	je     eb464 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3074>
   eb2b6:	mov    %rax,%rdi
   eb2b9:	mov    %rdx,%rsi
   eb2bc:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb2c1:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb2c6:	lea    0x185db(%rip),%rdx        # 1038a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3db0>
   eb2cd:	mov    (%rsp),%rsi
   eb2d1:	call   6fd3b <core::panicking::panic_bounds_check>
   eb2d6:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb2db:	mov    %rsi,%rdx
   eb2de:	lea    0x182ab(%rip),%rax        # 103590 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   eb2e5:	mov    %rax,0x220(%rsp)
   eb2ed:	mov    0xf0(%rsp),%rax
   eb2f5:	mov    %rax,0x198(%rsp)
   eb2fd:	mov    0x220(%rsp),%rcx
   eb305:	call   70c70 <core::slice::index::slice_index_fail>
   eb30a:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb30f:	mov    $0x8,%edi
   eb314:	mov    $0x18,%esi
   eb319:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb31e:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb323:	lea    -0xcd352(%rip),%rdi        # 1dfd8 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x6bb>
   eb32a:	lea    0x186df(%rip),%rdx        # 103a10 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f18>
   eb331:	mov    $0x1c,%esi
   eb336:	call   71e50 <core::option::expect_failed>
   eb33b:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb340:	mov    $0x8,%edi
   eb345:	mov    $0x18,%esi
   eb34a:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb34f:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb354:	lea    0x17b5d(%rip),%rdx        # 102eb8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33c0>
   eb35b:	mov    0x8(%rsp),%rsi
   eb360:	call   6fd3b <core::panicking::panic_bounds_check>
   eb365:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb36a:	mov    %r15,%rsi
   eb36d:	jmp    eb384 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f94>
   eb36f:	mov    %r15,%rsi
   eb372:	mov    %rcx,%rdi
   eb375:	lea    0x17b54(%rip),%rax        # 102ed0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33d8>
   eb37c:	mov    %rax,0x1b0(%rsp)
   eb384:	mov    0x1b0(%rsp),%rdx
   eb38c:	call   6fd3b <core::panicking::panic_bounds_check>
   eb391:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb396:	mov    $0x8,%edi
   eb39b:	mov    $0x18,%esi
   eb3a0:	call   693c2 <alloc::alloc::handle_alloc_error>
   eb3a5:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb3aa:	shr    $0x7,%rax
   eb3ae:	xor    %esi,%esi
   eb3b0:	mov    %rax,%rdi
   eb3b3:	lea    0x18206(%rip),%rdx        # 1035c0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ac8>
   eb3ba:	call   6fd3b <core::panicking::panic_bounds_check>
   eb3bf:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb3c4:	mov    $0x8,%edi
   eb3c9:	mov    %rbx,%rsi
   eb3cc:	call   692b8 <alloc::raw_vec::handle_error>
   eb3d1:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb3d6:	mov    %r15,%rsi
   eb3d9:	lea    0x17ac0(%rip),%rax        # 102ea0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33a8>
   eb3e0:	jmp    eb37c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x2f8c>
   eb3e2:	mov    $0x1,%edi
   eb3e7:	mov    $0x2b,%esi
   eb3ec:	call   692b8 <alloc::raw_vec::handle_error>
   eb3f1:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb3f6:	mov    $0x1,%edi
   eb3fb:	mov    $0x1a,%esi
   eb400:	call   692b8 <alloc::raw_vec::handle_error>
   eb405:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb407:	mov    $0x8,%eax
   eb40c:	mov    %rax,0x1b8(%rsp)
   eb414:	mov    %r15,%rsi
   eb417:	mov    0x1b8(%rsp),%rdi
   eb41f:	call   692b8 <alloc::raw_vec::handle_error>
   eb424:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb426:	mov    $0x8,%edi
   eb42b:	mov    %r14,%rsi
   eb42e:	call   692b8 <alloc::raw_vec::handle_error>
   eb433:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb435:	mov    $0x1,%edi
   eb43a:	mov    %r14,%rsi
   eb43d:	call   692b8 <alloc::raw_vec::handle_error>
   eb442:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb444:	mov    $0x1,%edi
   eb449:	mov    $0x22,%esi
   eb44e:	call   692b8 <alloc::raw_vec::handle_error>
   eb453:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb455:	mov    $0x1,%edi
   eb45a:	mov    %r14,%rsi
   eb45d:	call   692b8 <alloc::raw_vec::handle_error>
   eb462:	jmp    eb47c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x308c>
   eb464:	lea    -0xcd9d3(%rip),%rdi        # 1da98 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x17b>
   eb46b:	lea    0x178ee(%rip),%rdx        # 102d60 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3268>
   eb472:	mov    $0x11,%esi
   eb477:	call   70d40 <core::panicking::panic>
   eb47c:	ud2
   eb47e:	jmp    eb609 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3219>
   eb483:	jmp    eb485 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3095>
   eb485:	mov    %rax,%r14
   eb488:	add    $0x38,%r13
   eb48c:	mov    %r13,0x198(%rsp)
   eb494:	lea    0x80(%rsp),%rdi
   eb49c:	call   e8260 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   eb4a1:	jmp    eb622 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3232>
   eb4a6:	jmp    eb5df <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x31ef>
   eb4ab:	mov    %rax,%r14
   eb4ae:	mov    0xf0(%rsp),%rax
   eb4b6:	mov    %rax,0x198(%rsp)
   eb4be:	cmp    $0x11,%r13
   eb4c2:	jb     eb674 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3284>
   eb4c8:	mov    0x8(%rsp),%rdi
   eb4cd:	jmp    eb66e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x327e>
   eb4d2:	mov    %rax,(%rsp)
   eb4d6:	jmp    eb6c5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32d5>
   eb4db:	jmp    eb683 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3293>
   eb4e0:	mov    %rax,%r14
   eb4e3:	mov    %rbx,%rdi
   eb4e6:	call   *0x18a04(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb4ec:	jmp    eb760 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3370>
   eb4f1:	mov    %rax,%rbx
   eb4f4:	lea    0xc0(%rsp),%rdi
   eb4fc:	call   edae0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::positions::Leaf>>>
   eb501:	mov    %rbx,%rax
   eb504:	jmp    eb765 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3375>
   eb509:	mov    %rax,%r14
   eb50c:	jmp    eb638 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3248>
   eb511:	jmp    eb61f <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x322f>
   eb516:	jmp    eb765 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3375>
   eb51b:	jmp    eb693 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32a3>
   eb520:	mov    %rax,%rbx
   eb523:	lea    0x20(%rsp),%rdi
   eb528:	call   db080 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   eb52d:	mov    %rbx,%rax
   eb530:	jmp    eb765 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3375>
   eb535:	mov    %rax,%rbx
   eb538:	lea    0x230(%rsp),%rdi
   eb540:	call   e8260 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   eb545:	mov    %rbx,%rax
   eb548:	jmp    eb765 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3375>
   eb54d:	mov    %rax,%r14
   eb550:	lea    0x1e0(%rsp),%rdi
   eb558:	call   dae80 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   eb55d:	lea    0x20(%rsp),%rdi
   eb562:	call   ebc30 <core::ptr::drop_glue::<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   eb567:	mov    %r14,%rdi
   eb56a:	call   fdf70 <_Unwind_Resume@plt>
   eb56f:	jmp    eb683 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3293>
   eb574:	mov    %rax,%r14
   eb577:	cmpq   $0x0,0x8(%r15)
   eb57c:	je     eb588 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3198>
   eb57e:	mov    (%rsp),%rdi
   eb582:	call   *0x18968(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb588:	mov    0x10(%rsp),%rax
   eb58d:	mov    %rax,0x138(%rsp)
   eb595:	mov    0x160(%rsp),%rax
   eb59d:	mov    %rax,0x140(%rsp)
   eb5a5:	mov    0x180(%rsp),%rax
   eb5ad:	mov    %rax,0x148(%rsp)
   eb5b5:	mov    0x1c0(%rsp),%rax
   eb5bd:	mov    %rax,0x150(%rsp)
   eb5c5:	lea    0x120(%rsp),%rdi
   eb5cd:	call   ebd00 <core::ptr::drop_glue::<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   eb5d2:	mov    %r14,%rdi
   eb5d5:	call   fdf70 <_Unwind_Resume@plt>
   eb5da:	jmp    eb683 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3293>
   eb5df:	mov    %rax,%r14
   eb5e2:	mov    %rbx,%rdi
   eb5e5:	call   *0x18905(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb5eb:	jmp    eb65b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x326b>
   eb5ed:	jmp    eb683 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3293>
   eb5f2:	mov    %rax,%rbx
   eb5f5:	lea    0x20(%rsp),%rdi
   eb5fa:	call   ed500 <core::ptr::drop_glue::<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)>>
   eb5ff:	mov    %rbx,%rax
   eb602:	jmp    eb6b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c4>
   eb607:	jmp    eb609 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3219>
   eb609:	mov    %rax,%r14
   eb60c:	mov    %r15,%rdi
   eb60f:	call   *0x188db(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb615:	mov    %r14,%rax
   eb618:	jmp    eb683 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3293>
   eb61a:	jmp    eb6b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c4>
   eb61f:	mov    %rax,%r14
   eb622:	lea    0x20(%rsp),%rbx
   eb627:	jmp    eb758 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3368>
   eb62c:	mov    %rax,%r14
   eb62f:	mov    %r15,%rdi
   eb632:	call   *0x188b8(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb638:	lea    0x80(%rsp),%rdi
   eb640:	call   edd80 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Repr>>
   eb645:	jmp    eb674 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3284>
   eb647:	mov    %rax,%rbx
   eb64a:	mov    %r15,%rdi
   eb64d:	call   *0x1889d(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb653:	mov    %rbx,%rax
   eb656:	jmp    eb6b4 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32c4>
   eb658:	mov    %rax,%r14
   eb65b:	cmpq   $0x11,0xd8(%rsp)
   eb664:	jb     eb674 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3284>
   eb666:	mov    0xc8(%rsp),%rdi
   eb66e:	call   *0x1887c(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb674:	lea    0x20(%rsp),%rdi
   eb679:	call   ede00 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Leaf; 1]>>>
   eb67e:	jmp    eb760 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3370>
   eb683:	cmpq   $0x0,0x108(%rsp)
   eb68c:	je     eb6d3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32e3>
   eb68e:	mov    %rbx,%rdi
   eb691:	jmp    eb6a6 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32b6>
   eb693:	cmpq   $0x0,0x208(%rsp)
   eb69c:	je     eb6d3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32e3>
   eb69e:	mov    0x210(%rsp),%rdi
   eb6a6:	mov    %rax,%rbx
   eb6a9:	call   *0x18841(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb6af:	mov    %rbx,%rax
   eb6b2:	jmp    eb6d3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x32e3>
   eb6b4:	mov    %rax,(%rsp)
   eb6b8:	lea    0xc0(%rsp),%rdi
   eb6c0:	call   ed230 <core::ptr::drop_glue::<alloc::vec::drain::Drain<(usize, bool, bool)>>>
   eb6c5:	mov    $0x1,%al
   eb6c7:	mov    %eax,0xb0(%rsp)
   eb6ce:	jmp    eb781 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3391>
   eb6d3:	cmpq   $0x0,0x80(%rsp)
   eb6dc:	jne    eb6f8 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3308>
   eb6de:	mov    0x90(%rsp),%r14
   eb6e6:	lea    0xc0(%rsp),%rbx
   eb6ee:	test   %r14,%r14
   eb6f1:	jne    eb71e <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x332e>
   eb6f3:	mov    %rax,%r14
   eb6f6:	jmp    eb758 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3368>
   eb6f8:	mov    %rax,%r14
   eb6fb:	lea    0xc0(%rsp),%rbx
   eb703:	cmpq   $0x11,0x98(%rsp)
   eb70c:	jb     eb758 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3368>
   eb70e:	mov    0x88(%rsp),%rdi
   eb716:	call   *0x187d4(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb71c:	jmp    eb758 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3368>
   eb71e:	mov    %rax,%r12
   eb721:	mov    0x88(%rsp),%r15
   eb729:	lea    0x18(%r15),%r13
   eb72d:	jmp    eb739 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3349>
   eb72f:	nop
   eb730:	add    $0x20,%r13
   eb734:	dec    %r14
   eb737:	je     eb74c <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x335c>
   eb739:	cmpq   $0x11,0x0(%r13)
   eb73e:	jb     eb730 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3340>
   eb740:	mov    -0x10(%r13),%rdi
   eb744:	call   *0x187a6(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb74a:	jmp    eb730 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3340>
   eb74c:	mov    %r15,%rdi
   eb74f:	call   *0x1879b(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb755:	mov    %r12,%r14
   eb758:	mov    %rbx,%rdi
   eb75b:	call   c4010 <core::ptr::drop_glue::<smallvec::SmallVec<[tk_train::trainers::bpe::positions::Positions; 1]>>>
   eb760:	mov    %r14,%rax
   eb763:	jmp    eb765 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3375>
   eb765:	mov    %rax,(%rsp)
   eb769:	lea    0x190(%rsp),%rdi
   eb771:	call   ed6f0 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<((usize, (u32, u32)), u64, tk_train::trainers::bpe::positions::Positions)>>>
   eb776:	movl   $0x0,0xb0(%rsp)
   eb781:	lea    0x1e0(%rsp),%rdi
   eb789:	call   ed0f0 <core::ptr::drop_glue::<alloc::vec::drain::Drain<tk_train::trainers::bpe::positions::Positions>>>
   eb78e:	mov    0x170(%rsp),%rbx
   eb796:	mov    0x178(%rsp),%r13
   eb79e:	test   %r13,%r13
   eb7a1:	jne    eb7fb <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x340b>
   eb7a3:	cmpq   $0x0,0x168(%rsp)
   eb7ac:	je     eb7b7 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33c7>
   eb7ae:	mov    %rbx,%rdi
   eb7b1:	call   *0x18739(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb7b7:	cmpb   $0x0,0xb0(%rsp)
   eb7bf:	je     eb87d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x348d>
   eb7c5:	mov    0x1d0(%rsp),%rbx
   eb7cd:	mov    0x1d8(%rsp),%r13
   eb7d5:	test   %r13,%r13
   eb7d8:	jne    eb88b <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x349b>
   eb7de:	cmpq   $0x0,0x1c8(%rsp)
   eb7e7:	mov    (%rsp),%r14
   eb7eb:	jne    eb90d <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x351d>
   eb7f1:	mov    0x10(%rsp),%rbp
   eb7f6:	jmp    eb920 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3530>
   eb7fb:	xor    %r14d,%r14d
   eb7fe:	jmp    eb811 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3421>
   eb800:	mov    %r15,%rdi
   eb803:	call   *0x186e7(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb809:	inc    %r14
   eb80c:	cmp    %r13,%r14
   eb80f:	je     eb7a3 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33b3>
   eb811:	lea    (%r14,%r14,2),%rcx
   eb815:	shl    $0x4,%rcx
   eb819:	lea    (%rbx,%rcx,1),%rax
   eb81d:	cmpq   $0x0,(%rbx,%rcx,1)
   eb822:	je     eb840 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3450>
   eb824:	cmpq   $0x11,0x18(%rax)
   eb829:	jb     eb809 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3419>
   eb82b:	mov    0x8(%rax),%r15
   eb82f:	jmp    eb800 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3410>
   eb831:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb840:	mov    0x10(%rax),%rbp
   eb844:	test   %rbp,%rbp
   eb847:	je     eb809 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3419>
   eb849:	mov    0x8(%rax),%r15
   eb84d:	lea    0x18(%r15),%r12
   eb851:	jmp    eb869 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3479>
   eb853:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb860:	add    $0x20,%r12
   eb864:	dec    %rbp
   eb867:	je     eb800 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3410>
   eb869:	cmpq   $0x11,(%r12)
   eb86e:	jb     eb860 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3470>
   eb870:	mov    -0x10(%r12),%rdi
   eb875:	call   *0x18675(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb87b:	jmp    eb860 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3470>
   eb87d:	mov    0x10(%rsp),%rbp
   eb882:	mov    (%rsp),%r14
   eb886:	jmp    eb920 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3530>
   eb88b:	xor    %r14d,%r14d
   eb88e:	jmp    eb8a5 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34b5>
   eb890:	mov    %r15,%rdi
   eb893:	call   *0x18657(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb899:	inc    %r14
   eb89c:	cmp    %r13,%r14
   eb89f:	je     eb7de <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x33ee>
   eb8a5:	imul   $0x38,%r14,%rcx
   eb8a9:	lea    (%rbx,%rcx,1),%rax
   eb8ad:	cmpq   $0x0,0x18(%rbx,%rcx,1)
   eb8b3:	je     eb8d0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34e0>
   eb8b5:	cmpq   $0x11,0x30(%rax)
   eb8ba:	jb     eb899 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34a9>
   eb8bc:	mov    0x20(%rax),%r15
   eb8c0:	jmp    eb890 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34a0>
   eb8c2:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb8d0:	mov    0x28(%rax),%rbp
   eb8d4:	test   %rbp,%rbp
   eb8d7:	je     eb899 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34a9>
   eb8d9:	mov    0x20(%rax),%r15
   eb8dd:	lea    0x18(%r15),%r12
   eb8e1:	jmp    eb8f9 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3509>
   eb8e3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   eb8f0:	add    $0x20,%r12
   eb8f4:	dec    %rbp
   eb8f7:	je     eb890 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x34a0>
   eb8f9:	cmpq   $0x11,(%r12)
   eb8fe:	jb     eb8f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3500>
   eb900:	mov    -0x10(%r12),%rdi
   eb905:	call   *0x185e5(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb90b:	jmp    eb8f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3500>
   eb90d:	mov    %rbx,%rdi
   eb910:	call   *0x185da(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   eb916:	mov    0x10(%rsp),%rbp
   eb91b:	jmp    eb920 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>+0x3530>
   eb91d:	mov    %rax,%r14
   eb920:	mov    %rbp,0x138(%rsp)
   eb928:	mov    0x160(%rsp),%rax
   eb930:	mov    %rax,0x140(%rsp)
   eb938:	mov    0x180(%rsp),%rax
   eb940:	mov    %rax,0x148(%rsp)
   eb948:	mov    0x1c0(%rsp),%rax
   eb950:	mov    %rax,0x150(%rsp)
   eb958:	lea    0x120(%rsp),%rdi
   eb960:	call   ebd60 <core::ptr::drop_glue::<rayon::iter::map::MapFolder<rayon::iter::while_some::WhileSomeFolder<rayon::iter::extend::ListVecFolder<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>>>
   eb965:	mov    %r14,%rdi
   eb968:	call   fdf70 <_Unwind_Resume@plt>
