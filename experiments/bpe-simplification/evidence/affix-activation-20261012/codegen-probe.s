	.file	"bpe_affix_asm_probe.79718869180f1744-cgu.0"
	.section	.text.equal_choice,"ax",@progbits
	.globl	equal_choice
	.p2align	4
	.type	equal_choice,@function
equal_choice:
	.cfi_startproc
	andl	%ecx, %esi
	testl	%edi, %edi
	je	.LBB0_1
	cmpb	%sil, %dl
	je	.LBB0_4
	movl	$2, %eax
	testb	%dl, %dl
	jne	.LBB0_2
.LBB0_6:
	retq
.LBB0_1:
	movl	$1, %eax
	testb	%sil, %sil
	jne	.LBB0_6
.LBB0_2:
	movl	$3, %eax
	retq
.LBB0_4:
	movzbl	%dl, %eax
	retq
.Lfunc_end0:
	.size	equal_choice, .Lfunc_end0-equal_choice
	.cfi_endproc

	.section	.text.nested_choice,"ax",@progbits
	.globl	nested_choice
	.p2align	4
	.type	nested_choice,@function
nested_choice:
	.cfi_startproc
	andl	%ecx, %esi
	testl	%edi, %edi
	je	.LBB1_1
	testb	%sil, %sil
	je	.LBB1_4
	movl	$2, %eax
	testb	%dl, %dl
	jne	.LBB1_2
.LBB1_6:
	retq
.LBB1_1:
	movl	$3, %eax
	testb	%sil, %sil
	je	.LBB1_6
.LBB1_2:
	movl	$1, %eax
	retq
.LBB1_4:
	movzbl	%dl, %eax
	leaq	(%rax,%rax,2), %rax
	retq
.Lfunc_end1:
	.size	nested_choice, .Lfunc_end1-nested_choice
	.cfi_endproc

	.section	.text.paired_booland,"ax",@progbits
	.globl	paired_booland
	.p2align	4
	.type	paired_booland,@function
paired_booland:
	.cfi_startproc
	xorb	$1, %dl
	andb	%dil, %dl
	movzbl	%dl, %edx
	leaq	1(%rdx), %rax
	xorb	$1, %dl
	movzbl	%dl, %edx
	leaq	(%rdx,%rdx,2), %rdx
	testl	%ecx, %ecx
	cmoveq	%rdx, %rax
	testl	%esi, %esi
	cmoveq	%rdx, %rax
	retq
.Lfunc_end2:
	.size	paired_booland, .Lfunc_end2-paired_booland
	.cfi_endproc

	.section	.text.paired_choice,"ax",@progbits
	.globl	paired_choice
	.p2align	4
	.type	paired_choice,@function
paired_choice:
	.cfi_startproc
	xorb	$1, %dil
	orb	%dl, %dil
	movzbl	%dil, %edx
	movl	$2, %eax
	subq	%rdx, %rax
	leaq	(%rdx,%rdx,2), %rdx
	testl	%ecx, %ecx
	cmoveq	%rdx, %rax
	testl	%esi, %esi
	cmoveq	%rdx, %rax
	retq
.Lfunc_end3:
	.size	paired_choice, .Lfunc_end3-paired_choice
	.cfi_endproc

	.section	.text.staged_choice,"ax",@progbits
	.globl	staged_choice
	.p2align	4
	.type	staged_choice,@function
staged_choice:
	.cfi_startproc
	andl	%ecx, %esi
	leal	(%rsi,%rsi), %eax
	movl	%esi, %ecx
	xorb	$1, %cl
	movzbl	%cl, %ecx
	leaq	1(,%rcx,2), %rcx
	testl	%edi, %edi
	cmoveq	%rcx, %rax
	testl	%edx, %edx
	cmovneq	%rcx, %rax
	retq
.Lfunc_end4:
	.size	staged_choice, .Lfunc_end4-staged_choice
	.cfi_endproc

	.section	.text.tuple_match,"ax",@progbits
	.globl	tuple_match
	.p2align	4
	.type	tuple_match,@function
tuple_match:
	.cfi_startproc
	testl	%edi, %edi
	je	.LBB5_1
	testb	%sil, %sil
	je	.LBB5_3
	testb	%dl, %dl
	je	.LBB5_9
	movl	$3, %eax
	testb	%cl, %cl
	jne	.LBB5_7
.LBB5_8:
	retq
.LBB5_1:
	movl	$3, %eax
	testb	%cl, %sil
	je	.LBB5_8
.LBB5_7:
	movl	$1, %eax
	retq
.LBB5_3:
	movl	$3, %eax
	testb	%dl, %dl
	jne	.LBB5_8
	jmp	.LBB5_4
.LBB5_9:
	movl	$2, %eax
	testb	%cl, %cl
	jne	.LBB5_8
.LBB5_4:
	xorl	%eax, %eax
	retq
.Lfunc_end5:
	.size	tuple_match, .Lfunc_end5-tuple_match
	.cfi_endproc

	.section	.text.two_choices,"ax",@progbits
	.globl	two_choices
	.p2align	4
	.type	two_choices,@function
two_choices:
	.cfi_startproc
	movl	%edx, %edx
	movl	$2, %r8d
	subq	%rdx, %r8
	testl	%edi, %edi
	movl	$1, %eax
	cmovneq	%r8, %rax
	leaq	(%rdx,%rdx,2), %rdx
	movl	$3, %edi
	cmovneq	%rdx, %rdi
	testl	%ecx, %ecx
	cmoveq	%rdi, %rax
	testl	%esi, %esi
	cmoveq	%rdi, %rax
	retq
.Lfunc_end6:
	.size	two_choices, .Lfunc_end6-two_choices
	.cfi_endproc

	.ident	"rustc version 1.98.1 (48a229cea 2026-09-01)"
	.section	".note.GNU-stack","",@progbits
