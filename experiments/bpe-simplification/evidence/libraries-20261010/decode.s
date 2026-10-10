	.file	"decode.4afa0e5ba7445da2-cgu.0"
	.section	.text._RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64,"ax",@progbits
	.p2align	4
	.type	_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64,@function
_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64:
	.cfi_startproc
	testq	%rdx, %rdx
	je	.LBB0_3
	movzbl	(%rsi), %r9d
	movl	%r9d, %ecx
	andl	$127, %r9d
	movl	$1, %eax
	testb	%cl, %cl
	js	.LBB0_2
.LBB0_25:
	movq	%rdx, %rcx
	subq	%rax, %rcx
	jb	.LBB0_27
	addq	%rax, %rsi
	movq	%r9, (%rdi)
	movq	%rsi, 8(%rdi)
	movq	%rcx, 16(%rdi)
	retq
.LBB0_2:
	cmpq	$1, %rdx
	jne	.LBB0_6
.LBB0_3:
	movb	$0, (%rdi)
	movq	$0, 8(%rdi)
	retq
.LBB0_6:
	movzbl	1(%rsi), %r8d
	movl	%r8d, %ecx
	andl	$127, %ecx
	shll	$7, %ecx
	orq	%r9, %rcx
	movl	$2, %eax
	testb	%r8b, %r8b
	js	.LBB0_7
.LBB0_24:
	movq	%rcx, %r9
	testb	%r8b, %r8b
	jne	.LBB0_25
	movb	$2, (%rdi)
	movq	$0, 8(%rdi)
	retq
.LBB0_7:
	cmpq	$2, %rdx
	je	.LBB0_3
	movzbl	2(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shll	$14, %eax
	orq	%rax, %rcx
	movl	$3, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$3, %rdx
	je	.LBB0_3
	movzbl	3(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shll	$21, %eax
	orq	%rax, %rcx
	movl	$4, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$4, %rdx
	je	.LBB0_3
	movzbl	4(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shlq	$28, %rax
	orq	%rax, %rcx
	movl	$5, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$5, %rdx
	je	.LBB0_3
	movzbl	5(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shlq	$35, %rax
	orq	%rax, %rcx
	movl	$6, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$6, %rdx
	je	.LBB0_3
	movzbl	6(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shlq	$42, %rax
	orq	%rax, %rcx
	movl	$7, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$7, %rdx
	je	.LBB0_3
	movzbl	7(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shlq	$49, %rax
	orq	%rax, %rcx
	movl	$8, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$8, %rdx
	je	.LBB0_3
	movzbl	8(%rsi), %r8d
	movl	%r8d, %eax
	andl	$127, %eax
	shlq	$56, %rax
	orq	%rax, %rcx
	movl	$9, %eax
	testb	%r8b, %r8b
	jns	.LBB0_24
	cmpq	$9, %rdx
	je	.LBB0_3
	movzbl	9(%rsi), %r8d
	testb	%r8b, %r8b
	js	.LBB0_28
	movq	%r8, %rax
	shlq	$63, %rax
	orq	%rax, %rcx
	movl	$10, %eax
	jmp	.LBB0_24
.LBB0_27:
	pushq	%rax
	.cfi_def_cfa_offset 16
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.3(%rip), %rcx
	movq	%rax, %rdi
	movq	%rdx, %rsi
	callq	*_RNvNtNtCsgxBkk5gSRhY_4core5slice5index16slice_index_fail@GOTPCREL(%rip)
.LBB0_28:
	.cfi_def_cfa_offset 8
	movb	$1, (%rdi)
	movq	$0, 8(%rdi)
	retq
.Lfunc_end0:
	.size	_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64, .Lfunc_end0-_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64
	.cfi_endproc

	.section	.text._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,"ax",@progbits
	.p2align	4
	.type	_RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,@function
_RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt:
	.cfi_startproc
	movq	%rsi, %rax
	movzbl	(%rdi), %ecx
	leaq	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt(%rip), %rdx
	movq	(%rdx,%rcx,8), %rdx
	leaq	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel(%rip), %rdi
	movslq	(%rdi,%rcx,4), %rsi
	addq	%rdi, %rsi
	movq	%rax, %rdi
	jmpq	*_RNvMsa_NtCsgxBkk5gSRhY_4core3fmtNtB5_9Formatter9write_str@GOTPCREL(%rip)
.Lfunc_end1:
	.size	_RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt, .Lfunc_end1-_RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt
	.cfi_endproc

	.section	.text._RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,"ax",@progbits
	.p2align	4
	.type	_RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,@function
_RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt:
	.cfi_startproc
	movq	%rsi, %rax
	movzbl	(%rdi), %ecx
	movl	%ecx, %edx
	xorl	$1, %edx
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.8(%rip), %rdi
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.7(%rip), %rsi
	testl	%ecx, %ecx
	cmovneq	%rdi, %rsi
	leaq	9(,%rdx,4), %rdx
	movq	%rax, %rdi
	jmpq	*_RNvMsa_NtCsgxBkk5gSRhY_4core3fmtNtB5_9Formatter9write_str@GOTPCREL(%rip)
.Lfunc_end2:
	.size	_RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt, .Lfunc_end2-_RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt
	.cfi_endproc

	.section	.text.unsigned_checked,"ax",@progbits
	.globl	unsigned_checked
	.p2align	4
	.type	unsigned_checked,@function
unsigned_checked:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	leaq	8(%rsp), %rdi
	callq	_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64
	cmpq	$0, 16(%rsp)
	je	.LBB3_2
	movq	24(%rsp), %rax
	movq	%rax, 16(%rbx)
	movups	8(%rsp), %xmm0
	movups	%xmm0, (%rbx)
	movq	%rbx, %rax
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.LBB3_2:
	.cfi_def_cfa_offset 48
	movzbl	8(%rsp), %eax
	movb	%al, 7(%rsp)
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9(%rip), %rdi
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.0(%rip), %rcx
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.11(%rip), %r8
	leaq	7(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgxBkk5gSRhY_4core6result13unwrap_failed@GOTPCREL(%rip)
.Lfunc_end3:
	.size	unsigned_checked, .Lfunc_end3-unsigned_checked
	.cfi_endproc

	.section	.text.unsigned_trusted,"ax",@progbits
	.globl	unsigned_trusted
	.p2align	4
	.type	unsigned_trusted,@function
unsigned_trusted:
	.cfi_startproc
	pushq	%rbx
	.cfi_def_cfa_offset 16
	subq	$32, %rsp
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -16
	movq	%rdi, %rbx
	leaq	8(%rsp), %rdi
	callq	_RNvNtCsd7kW5qKcwAo_15unsigned_varint6decode3u64
	movq	24(%rsp), %rax
	movq	%rax, 16(%rbx)
	movups	8(%rsp), %xmm0
	movups	%xmm0, (%rbx)
	movq	%rbx, %rax
	addq	$32, %rsp
	.cfi_def_cfa_offset 16
	popq	%rbx
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end4:
	.size	unsigned_trusted, .Lfunc_end4-unsigned_trusted
	.cfi_endproc

	.section	.text.vint_checked,"ax",@progbits
	.globl	vint_checked
	.p2align	4
	.type	vint_checked,@function
vint_checked:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r13
	.cfi_def_cfa_offset 32
	pushq	%r12
	.cfi_def_cfa_offset 40
	pushq	%rbx
	.cfi_def_cfa_offset 48
	subq	$16, %rsp
	.cfi_def_cfa_offset 64
	.cfi_offset %rbx, -48
	.cfi_offset %r12, -40
	.cfi_offset %r13, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	8(%rdi), %r12
	movb	$1, %al
	testq	%r12, %r12
	je	.LBB5_7
	movq	%rdi, %rbx
	movq	(%rdi), %r14
	movzbl	(%r14), %ecx
	orl	$256, %ecx
	rep		bsfl	%ecx, %r13d
	cmpq	%r13, %r12
	jbe	.LBB5_7
	leaq	1(%r13), %r15
	cmpl	$9, %r15d
	jne	.LBB5_9
	movq	1(%r14), %rax
	testq	%r13, %r13
	jne	.LBB5_5
	jmp	.LBB5_8
.LBB5_9:
	movq	$0, 8(%rsp)
	leaq	8(%rsp), %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	*memcpy@GOTPCREL(%rip)
	movq	8(%rsp), %rax
	movl	%r15d, %ecx
	shrq	%cl, %rax
	testq	%r13, %r13
	je	.LBB5_8
.LBB5_5:
	leal	(,%r13,8), %ecx
	subl	%r13d, %ecx
	movq	%rax, %rdx
	shrq	%cl, %rdx
	testq	%rdx, %rdx
	je	.LBB5_6
.LBB5_8:
	subq	%r15, %r12
	addq	%r15, %r14
	movq	%r14, (%rbx)
	movq	%r12, 8(%rbx)
	addq	$16, %rsp
	.cfi_def_cfa_offset 48
	popq	%rbx
	.cfi_def_cfa_offset 40
	popq	%r12
	.cfi_def_cfa_offset 32
	popq	%r13
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.LBB5_6:
	.cfi_def_cfa_offset 64
	xorl	%eax, %eax
.LBB5_7:
	movb	%al, 7(%rsp)
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9(%rip), %rdi
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.1(%rip), %rcx
	leaq	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.12(%rip), %r8
	leaq	7(%rsp), %rdx
	movl	$22, %esi
	callq	*_RNvNtCsgxBkk5gSRhY_4core6result13unwrap_failed@GOTPCREL(%rip)
.Lfunc_end5:
	.size	vint_checked, .Lfunc_end5-vint_checked
	.cfi_endproc

	.section	.text.vint_trusted,"ax",@progbits
	.globl	vint_trusted
	.p2align	4
	.type	vint_trusted,@function
vint_trusted:
	.cfi_startproc
	pushq	%r15
	.cfi_def_cfa_offset 16
	pushq	%r14
	.cfi_def_cfa_offset 24
	pushq	%r12
	.cfi_def_cfa_offset 32
	pushq	%rbx
	.cfi_def_cfa_offset 40
	pushq	%rax
	.cfi_def_cfa_offset 48
	.cfi_offset %rbx, -40
	.cfi_offset %r12, -32
	.cfi_offset %r14, -24
	.cfi_offset %r15, -16
	movq	%rdi, %rbx
	movq	(%rdi), %r14
	movq	8(%rdi), %r12
	movzbl	(%r14), %eax
	orl	$256, %eax
	rep		bsfl	%eax, %r15d
	incq	%r15
	cmpq	$9, %r15
	jne	.LBB6_3
	movq	1(%r14), %rax
	jmp	.LBB6_2
.LBB6_3:
	movq	$0, (%rsp)
	movq	%rsp, %rdi
	movq	%r14, %rsi
	movq	%r15, %rdx
	callq	*memcpy@GOTPCREL(%rip)
	movq	(%rsp), %rax
	movl	%r15d, %ecx
	shrq	%cl, %rax
.LBB6_2:
	subq	%r15, %r12
	addq	%r15, %r14
	movq	%r14, (%rbx)
	movq	%r12, 8(%rbx)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%r12
	.cfi_def_cfa_offset 24
	popq	%r14
	.cfi_def_cfa_offset 16
	popq	%r15
	.cfi_def_cfa_offset 8
	retq
.Lfunc_end6:
	.size	vint_trusted, .Lfunc_end6-vint_trusted
	.cfi_endproc

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.0,@object
	.section	.data.rel.ro..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.0,"aw",@progbits
	.p2align	3, 0x0
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.0:
	.asciz	"\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000"
	.quad	_RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.0, 32

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.1,@object
	.section	.data.rel.ro..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.1,"aw",@progbits
	.p2align	3, 0x0
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.1:
	.asciz	"\000\000\000\000\000\000\000\000\001\000\000\000\000\000\000\000\001\000\000\000\000\000\000"
	.quad	_RNvXs2_NtCsdI0MOhZgPqQ_6vint645errorNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.1, 32

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.2,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.2:
	.asciz	"/root/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/unsigned-varint-0.8.0/src/decode.rs"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.2, 95

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.3,@object
	.section	.data.rel.ro..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.3,"aw",@progbits
	.p2align	3, 0x0
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.3:
	.quad	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.2
	.asciz	"^\000\000\000\000\000\000\000{\000\000\000\005\000\000"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.3, 24

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.4,@object
	.section	.rodata..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.4,"a",@progbits
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.4:
	.ascii	"Insufficient"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.4, 12

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.5,@object
	.section	.rodata.cst8,"aM",@progbits,8
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.5:
	.ascii	"Overflow"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.5, 8

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.6,@object
	.section	.rodata..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.6,"a",@progbits
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.6:
	.ascii	"NotMinimal"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.6, 10

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.7,@object
	.section	.rodata..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.7,"a",@progbits
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.7:
	.ascii	"LeadingZeroes"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.7, 13

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.8,@object
	.section	.rodata..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.8,"a",@progbits
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.8:
	.ascii	"Truncated"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.8, 9

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9,@object
	.section	.rodata..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9,"a",@progbits
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9:
	.ascii	"internal encoded bytes"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.9, 22

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.10,@object
	.section	.rodata.str1.1,"aMS",@progbits,1
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.10:
	.asciz	"/tmp/bpe-library-api-probe/decode.rs"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.10, 37

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.11,@object
	.section	.data.rel.ro..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.11,"aw",@progbits
	.p2align	3, 0x0
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.11:
	.quad	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.10
	.asciz	"$\000\000\000\000\000\000\000\002\000\000\000X\000\000"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.11, 24

	.type	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.12,@object
	.section	.data.rel.ro..Lanon.94bfee7f22fbc243a1b30153a24fe7c9.12,"aw",@progbits
	.p2align	3, 0x0
.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.12:
	.quad	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.10
	.asciz	"$\000\000\000\000\000\000\000\006\000\000\000C\000\000"
	.size	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.12, 24

	.type	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,@object
	.section	.rodata..Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt,"a",@progbits
	.p2align	3, 0x0
.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt:
	.quad	12
	.quad	8
	.quad	10
	.size	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt, 24

	.type	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel,@object
	.section	.rodata..Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel,"a",@progbits
	.p2align	2, 0x0
.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel:
	.long	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.4-.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel
	.long	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.5-.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel
	.long	.Lanon.94bfee7f22fbc243a1b30153a24fe7c9.6-.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel
	.size	.Lswitch.table._RNvXs0_NtCsd7kW5qKcwAo_15unsigned_varint6decodeNtB5_5ErrorNtNtCsgxBkk5gSRhY_4core3fmt5Debug3fmt.5.rel, 12

	.ident	"rustc version 1.98.1 (48a229cea 2026-09-01)"
	.section	".note.GNU-stack","",@progbits
