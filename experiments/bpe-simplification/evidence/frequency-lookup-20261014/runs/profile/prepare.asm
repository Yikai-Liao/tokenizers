
/dev/shm/bpe-frequency-20261014/A-runner:     file format elf64-x86-64


Disassembly of section .text:

00000000000eba30 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_>:
   eba30:	55                   	push   %rbp
   eba31:	41 57                	push   %r15
   eba33:	41 56                	push   %r14
   eba35:	41 55                	push   %r13
   eba37:	41 54                	push   %r12
   eba39:	53                   	push   %rbx
   eba3a:	48 81 ec 88 02 00 00 	sub    $0x288,%rsp
   eba41:	49 89 d1             	mov    %rdx,%r9
   eba44:	48 89 fb             	mov    %rdi,%rbx
   eba47:	44 8b 39             	mov    (%rcx),%r15d
   eba4a:	44 8b 69 04          	mov    0x4(%rcx),%r13d
   eba4e:	8b 69 08             	mov    0x8(%rcx),%ebp
   eba51:	0f 10 41 60          	movups 0x60(%rcx),%xmm0
   eba55:	0f 29 84 24 70 02 00 	movaps %xmm0,0x270(%rsp)
   eba5c:	00 
   eba5d:	0f 10 41 50          	movups 0x50(%rcx),%xmm0
   eba61:	0f 29 84 24 60 02 00 	movaps %xmm0,0x260(%rsp)
   eba68:	00 
   eba69:	f3 0f 6f 41 10       	movdqu 0x10(%rcx),%xmm0
   eba6e:	f3 0f 6f 49 20       	movdqu 0x20(%rcx),%xmm1
   eba73:	f3 0f 6f 51 30       	movdqu 0x30(%rcx),%xmm2
   eba78:	0f 10 59 40          	movups 0x40(%rcx),%xmm3
   eba7c:	0f 29 9c 24 50 02 00 	movaps %xmm3,0x250(%rsp)
   eba83:	00 
   eba84:	66 0f 7f 94 24 40 02 	movdqa %xmm2,0x240(%rsp)
   eba8b:	00 00 
   eba8d:	66 0f 7f 8c 24 30 02 	movdqa %xmm1,0x230(%rsp)
   eba94:	00 00 
   eba96:	66 0f 7f 84 24 20 02 	movdqa %xmm0,0x220(%rsp)
   eba9d:	00 00 
   eba9f:	4c 8b 26             	mov    (%rsi),%r12
   ebaa2:	48 8b 46 08          	mov    0x8(%rsi),%rax
   ebaa6:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
   ebaad:	00 
   ebaae:	48 8b 46 10          	mov    0x10(%rsi),%rax
   ebab2:	48 89 84 24 b8 00 00 	mov    %rax,0xb8(%rsp)
   ebab9:	00 
   ebaba:	4d 8b 34 24          	mov    (%r12),%r14
   ebabe:	4d 8b 46 50          	mov    0x50(%r14),%r8
   ebac2:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
   ebac9:	00 
   ebaca:	44 89 fe             	mov    %r15d,%esi
   ebacd:	44 89 ea             	mov    %r13d,%edx
   ebad0:	89 e9                	mov    %ebp,%ecx
   ebad2:	e8 d9 81 00 00       	call   f3cb0 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges3new>
   ebad7:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
   ebade:	00 00 
   ebae0:	49 8b 76 50          	mov    0x50(%r14),%rsi
   ebae4:	4c 39 fe             	cmp    %r15,%rsi
   ebae7:	0f 86 30 13 00 00    	jbe    ece1d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13ed>
   ebaed:	4c 39 ee             	cmp    %r13,%rsi
   ebaf0:	0f 86 2c 13 00 00    	jbe    ece22 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f2>
   ebaf6:	4c 89 a4 24 80 00 00 	mov    %r12,0x80(%rsp)
   ebafd:	00 
   ebafe:	89 ac 24 d4 00 00 00 	mov    %ebp,0xd4(%rsp)
   ebb05:	48 89 9c 24 70 01 00 	mov    %rbx,0x170(%rsp)
   ebb0c:	00 
   ebb0d:	49 8b 46 48          	mov    0x48(%r14),%rax
   ebb11:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
   ebb16:	4a 8b 0c f8          	mov    (%rax,%r15,8),%rcx
   ebb1a:	48 89 8c 24 c8 00 00 	mov    %rcx,0xc8(%rsp)
   ebb21:	00 
   ebb22:	4c 89 ac 24 90 00 00 	mov    %r13,0x90(%rsp)
   ebb29:	00 
   ebb2a:	4a 8b 04 e8          	mov    (%rax,%r13,8),%rax
   ebb2e:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
   ebb35:	00 
   ebb36:	48 8d 84 24 20 02 00 	lea    0x220(%rsp),%rax
   ebb3d:	00 
   ebb3e:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
   ebb45:	00 
   ebb46:	4c 89 b4 24 88 01 00 	mov    %r14,0x188(%rsp)
   ebb4d:	00 
   ebb4e:	4c 89 74 24 28       	mov    %r14,0x28(%rsp)
   ebb53:	4c 89 b4 24 90 01 00 	mov    %r14,0x190(%rsp)
   ebb5a:	00 
   ebb5b:	bf 80 00 00 00       	mov    $0x80,%edi
   ebb60:	ff 15 9a d0 01 00    	call   *0x1d09a(%rip)        # 108c00 <malloc@GLIBC_2.2.5>
   ebb66:	48 85 c0             	test   %rax,%rax
   ebb69:	0f 84 c4 12 00 00    	je     ece33 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1403>
   ebb6f:	48 89 c5             	mov    %rax,%rbp
   ebb72:	66 0f ef c0          	pxor   %xmm0,%xmm0
   ebb76:	f3 0f 7f 84 24 a8 00 	movdqu %xmm0,0xa8(%rsp)
   ebb7d:	00 00 
   ebb7f:	48 c7 84 24 98 00 00 	movq   $0x10,0x98(%rsp)
   ebb86:	00 10 00 00 00 
   ebb8b:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
   ebb92:	00 
   ebb93:	48 8d 84 24 80 01 00 	lea    0x180(%rsp),%rax
   ebb9a:	00 
   ebb9b:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
   ebba2:	00 
   ebba3:	48 c7 84 24 e0 00 00 	movq   $0x10,0xe0(%rsp)
   ebbaa:	00 10 00 00 00 
   ebbaf:	41 bf 10 00 00 00    	mov    $0x10,%r15d
   ebbb5:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
   ebbbc:	00 00 
   ebbbe:	4c 8d a4 24 d8 00 00 	lea    0xd8(%rsp),%r12
   ebbc5:	00 
   ebbc6:	31 db                	xor    %ebx,%ebx
   ebbc8:	4c 89 e7             	mov    %r12,%rdi
   ebbcb:	e8 c0 a6 00 00       	call   f6290 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
   ebbd0:	48 83 f8 01          	cmp    $0x1,%rax
   ebbd4:	0f 85 b1 01 00 00    	jne    ebd8b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x35b>
   ebbda:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
   ebbde:	0f 84 74 11 00 00    	je     ecd58 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1328>
   ebbe4:	49 89 d5             	mov    %rdx,%r13
   ebbe7:	4c 89 f8             	mov    %r15,%rax
   ebbea:	48 29 d8             	sub    %rbx,%rax
   ebbed:	76 11                	jbe    ebc00 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1d0>
   ebbef:	49 89 de             	mov    %rbx,%r14
   ebbf2:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
   ebbf7:	e9 ff 00 00 00       	jmp    ebcfb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
   ebbfc:	0f 1f 40 00          	nopl   0x0(%rax)
   ebc00:	0f 84 99 00 00 00    	je     ebc9f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x26f>
   ebc06:	4c 89 fe             	mov    %r15,%rsi
   ebc09:	49 89 de             	mov    %rbx,%r14
   ebc0c:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
   ebc11:	48 39 c3             	cmp    %rax,%rbx
   ebc14:	0f 86 d6 00 00 00    	jbe    ebcf0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2c0>
   ebc1a:	4c 89 fa             	mov    %r15,%rdx
   ebc1d:	48 29 da             	sub    %rbx,%rdx
   ebc20:	4c 89 f0             	mov    %r14,%rax
   ebc23:	48 29 d0             	sub    %rdx,%rax
   ebc26:	48 39 c2             	cmp    %rax,%rdx
   ebc29:	76 36                	jbe    ebc61 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
   ebc2b:	48 89 f1             	mov    %rsi,%rcx
   ebc2e:	4c 29 f9             	sub    %r15,%rcx
   ebc31:	48 39 c1             	cmp    %rax,%rcx
   ebc34:	72 2b                	jb     ebc61 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
   ebc36:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
   ebc3d:	00 
   ebc3e:	4a 8d 3c fd 00 00 00 	lea    0x0(,%r15,8),%rdi
   ebc45:	00 
   ebc46:	48 01 ef             	add    %rbp,%rdi
   ebc49:	48 c1 e0 03          	shl    $0x3,%rax
   ebc4d:	49 89 f7             	mov    %rsi,%r15
   ebc50:	48 89 ee             	mov    %rbp,%rsi
   ebc53:	48 89 c2             	mov    %rax,%rdx
   ebc56:	ff 15 94 cf 01 00    	call   *0x1cf94(%rip)        # 108bf0 <memcpy@GLIBC_2.14>
   ebc5c:	e9 9a 00 00 00       	jmp    ebcfb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
   ebc61:	48 89 d8             	mov    %rbx,%rax
   ebc64:	48 89 f3             	mov    %rsi,%rbx
   ebc67:	48 29 d3             	sub    %rdx,%rbx
   ebc6a:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
   ebc71:	00 
   ebc72:	49 89 f7             	mov    %rsi,%r15
   ebc75:	48 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%rsi
   ebc7c:	00 
   ebc7d:	48 01 ee             	add    %rbp,%rsi
   ebc80:	48 8d 3c dd 00 00 00 	lea    0x0(,%rbx,8),%rdi
   ebc87:	00 
   ebc88:	48 01 ef             	add    %rbp,%rdi
   ebc8b:	48 c1 e2 03          	shl    $0x3,%rdx
   ebc8f:	ff 15 63 cf 01 00    	call   *0x1cf63(%rip)        # 108bf8 <memmove@GLIBC_2.2.5>
   ebc95:	48 89 9c 24 a8 00 00 	mov    %rbx,0xa8(%rsp)
   ebc9c:	00 
   ebc9d:	eb 5c                	jmp    ebcfb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
   ebc9f:	ba 01 00 00 00       	mov    $0x1,%edx
   ebca4:	b9 08 00 00 00       	mov    $0x8,%ecx
   ebca9:	41 b8 08 00 00 00    	mov    $0x8,%r8d
   ebcaf:	48 8d bc 24 98 00 00 	lea    0x98(%rsp),%rdi
   ebcb6:	00 
   ebcb7:	48 89 de             	mov    %rbx,%rsi
   ebcba:	e8 61 22 fe ff       	call   cdf20 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsgHLVOFInvdC_8tk_train>
   ebcbf:	48 8b b4 24 98 00 00 	mov    0x98(%rsp),%rsi
   ebcc6:	00 
   ebcc7:	48 8b 9c 24 a8 00 00 	mov    0xa8(%rsp),%rbx
   ebcce:	00 
   ebccf:	4c 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%r14
   ebcd6:	00 
   ebcd7:	4c 89 f8             	mov    %r15,%rax
   ebcda:	4c 29 f0             	sub    %r14,%rax
   ebcdd:	48 39 c3             	cmp    %rax,%rbx
   ebce0:	0f 87 34 ff ff ff    	ja     ebc1a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1ea>
   ebce6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
   ebced:	00 00 00 
   ebcf0:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
   ebcf7:	00 
   ebcf8:	49 89 f7             	mov    %rsi,%r15
   ebcfb:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
   ebd00:	49 8d 04 1e          	lea    (%r14,%rbx,1),%rax
   ebd04:	4c 39 f8             	cmp    %r15,%rax
   ebd07:	b9 00 00 00 00       	mov    $0x0,%ecx
   ebd0c:	49 0f 43 cf          	cmovae %r15,%rcx
   ebd10:	48 29 c8             	sub    %rcx,%rax
   ebd13:	4c 89 6c c5 00       	mov    %r13,0x0(%rbp,%rax,8)
   ebd18:	49 8d 5e 01          	lea    0x1(%r14),%rbx
   ebd1c:	48 89 9c 24 b0 00 00 	mov    %rbx,0xb0(%rsp)
   ebd23:	00 
   ebd24:	4c 39 fb             	cmp    %r15,%rbx
   ebd27:	0f 83 9b fe ff ff    	jae    ebbc8 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
   ebd2d:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
   ebd32:	49 01 c6             	add    %rax,%r14
   ebd35:	49 ff c6             	inc    %r14
   ebd38:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
   ebd3f:	00 
   ebd40:	4c 89 e7             	mov    %r12,%rdi
   ebd43:	e8 48 a5 00 00       	call   f6290 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
   ebd48:	48 83 f8 01          	cmp    $0x1,%rax
   ebd4c:	75 35                	jne    ebd83 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x353>
   ebd4e:	4d 39 fe             	cmp    %r15,%r14
   ebd51:	b8 00 00 00 00       	mov    $0x0,%eax
   ebd56:	49 0f 43 c7          	cmovae %r15,%rax
   ebd5a:	48 c1 e0 03          	shl    $0x3,%rax
   ebd5e:	48 89 e9             	mov    %rbp,%rcx
   ebd61:	48 29 c1             	sub    %rax,%rcx
   ebd64:	4a 89 14 f1          	mov    %rdx,(%rcx,%r14,8)
   ebd68:	48 ff c3             	inc    %rbx
   ebd6b:	49 ff c6             	inc    %r14
   ebd6e:	49 39 df             	cmp    %rbx,%r15
   ebd71:	75 cd                	jne    ebd40 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x310>
   ebd73:	4c 89 bc 24 b0 00 00 	mov    %r15,0xb0(%rsp)
   ebd7a:	00 
   ebd7b:	4c 89 fb             	mov    %r15,%rbx
   ebd7e:	e9 45 fe ff ff       	jmp    ebbc8 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
   ebd83:	48 89 9c 24 b0 00 00 	mov    %rbx,0xb0(%rsp)
   ebd8a:	00 
   ebd8b:	48 8b 8c 24 c8 00 00 	mov    0xc8(%rsp),%rcx
   ebd92:	00 
   ebd93:	48 8b 94 24 c0 00 00 	mov    0xc0(%rsp),%rdx
   ebd9a:	00 
   ebd9b:	48 01 ca             	add    %rcx,%rdx
   ebd9e:	f3 0f 6f 84 24 98 00 	movdqu 0x98(%rsp),%xmm0
   ebda5:	00 00 
   ebda7:	f3 0f 6f 8c 24 a8 00 	movdqu 0xa8(%rsp),%xmm1
   ebdae:	00 00 
   ebdb0:	66 0f 7f 84 24 e0 01 	movdqa %xmm0,0x1e0(%rsp)
   ebdb7:	00 00 
   ebdb9:	66 0f 7f 8c 24 f0 01 	movdqa %xmm1,0x1f0(%rsp)
   ebdc0:	00 00 
   ebdc2:	f3 0f 6f 94 24 80 01 	movdqu 0x180(%rsp),%xmm2
   ebdc9:	00 00 
   ebdcb:	66 0f 7f 94 24 00 02 	movdqa %xmm2,0x200(%rsp)
   ebdd2:	00 00 
   ebdd4:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
   ebddb:	00 
   ebddc:	48 89 84 24 10 02 00 	mov    %rax,0x210(%rsp)
   ebde3:	00 
   ebde4:	48 89 84 24 48 01 00 	mov    %rax,0x148(%rsp)
   ebdeb:	00 
   ebdec:	f3 0f 7f 94 24 38 01 	movdqu %xmm2,0x138(%rsp)
   ebdf3:	00 00 
   ebdf5:	f3 0f 7f 8c 24 28 01 	movdqu %xmm1,0x128(%rsp)
   ebdfc:	00 00 
   ebdfe:	f3 0f 7f 84 24 18 01 	movdqu %xmm0,0x118(%rsp)
   ebe05:	00 00 
   ebe07:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
   ebe0e:	00 00 00 00 00 
   ebe13:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ebe18:	48 89 84 24 f0 00 00 	mov    %rax,0xf0(%rsp)
   ebe1f:	00 
   ebe20:	48 89 8c 24 f8 00 00 	mov    %rcx,0xf8(%rsp)
   ebe27:	00 
   ebe28:	48 89 94 24 00 01 00 	mov    %rdx,0x100(%rsp)
   ebe2f:	00 
   ebe30:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
   ebe35:	89 8c 24 08 01 00 00 	mov    %ecx,0x108(%rsp)
   ebe3c:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
   ebe43:	00 
   ebe44:	89 8c 24 0c 01 00 00 	mov    %ecx,0x10c(%rsp)
   ebe4b:	48 89 84 24 10 01 00 	mov    %rax,0x110(%rsp)
   ebe52:	00 
   ebe53:	4c 8b b4 24 30 01 00 	mov    0x130(%rsp),%r14
   ebe5a:	00 
   ebe5b:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
   ebe62:	00 
   ebe63:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
   ebe68:	4d 85 f6             	test   %r14,%r14
   ebe6b:	0f 84 ac 0d 00 00    	je     ecc1d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
   ebe71:	4c 8b ac 24 28 01 00 	mov    0x128(%rsp),%r13
   ebe78:	00 
   ebe79:	48 8b 8c 24 b8 00 00 	mov    0xb8(%rsp),%rcx
   ebe80:	00 
   ebe81:	0f b6 01             	movzbl (%rcx),%eax
   ebe84:	88 44 24 17          	mov    %al,0x17(%rsp)
   ebe88:	48 8b 41 10          	mov    0x10(%rcx),%rax
   ebe8c:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
   ebe91:	48 8b 41 08          	mov    0x8(%rcx),%rax
   ebe95:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
   ebe9a:	48 8b 94 24 88 00 00 	mov    0x88(%rsp),%rdx
   ebea1:	00 
   ebea2:	48 8b 42 18          	mov    0x18(%rdx),%rax
   ebea6:	48 89 84 24 60 01 00 	mov    %rax,0x160(%rsp)
   ebead:	00 
   ebeae:	48 8b 42 28          	mov    0x28(%rdx),%rax
   ebeb2:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
   ebeb9:	00 
   ebeba:	48 8b 42 20          	mov    0x20(%rdx),%rax
   ebebe:	48 89 84 24 50 01 00 	mov    %rax,0x150(%rsp)
   ebec5:	00 
   ebec6:	48 8b 02             	mov    (%rdx),%rax
   ebec9:	48 89 84 24 68 01 00 	mov    %rax,0x168(%rsp)
   ebed0:	00 
   ebed1:	48 8b 42 08          	mov    0x8(%rdx),%rax
   ebed5:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
   ebeda:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
   ebee1:	00 
   ebee2:	48 8b 40 08          	mov    0x8(%rax),%rax
   ebee6:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
   ebeed:	00 
   ebeee:	48 8b 41 50          	mov    0x50(%rcx),%rax
   ebef2:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
   ebef9:	00 
   ebefa:	48 8b 41 48          	mov    0x48(%rcx),%rax
   ebefe:	48 89 84 24 78 01 00 	mov    %rax,0x178(%rsp)
   ebf05:	00 
   ebf06:	48 8d 05 23 c4 01 00 	lea    0x1c423(%rip),%rax        # 108330 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3a20>
   ebf0d:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
   ebf12:	4c 8b 8c 24 f0 00 00 	mov    0xf0(%rsp),%r9
   ebf19:	00 
   ebf1a:	48 8b 84 24 f8 00 00 	mov    0xf8(%rsp),%rax
   ebf21:	00 
   ebf22:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
   ebf29:	00 
   ebf2a:	8b 84 24 08 01 00 00 	mov    0x108(%rsp),%eax
   ebf31:	89 84 24 c0 00 00 00 	mov    %eax,0xc0(%rsp)
   ebf38:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
   ebf3f:	00 
   ebf40:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   ebf45:	8b 84 24 0c 01 00 00 	mov    0x10c(%rsp),%eax
   ebf4c:	89 84 24 b8 00 00 00 	mov    %eax,0xb8(%rsp)
   ebf53:	48 8b 5c 24 50       	mov    0x50(%rsp),%rbx
   ebf58:	4c 89 8c 24 c8 00 00 	mov    %r9,0xc8(%rsp)
   ebf5f:	00 
   ebf60:	eb 17                	jmp    ebf79 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x549>
   ebf62:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ebf69:	1f 84 00 00 00 00 00 
   ebf70:	4d 85 f6             	test   %r14,%r14
   ebf73:	0f 84 a4 0c 00 00    	je     ecc1d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
   ebf79:	4c 89 e8             	mov    %r13,%rax
   ebf7c:	49 8d 4d 01          	lea    0x1(%r13),%rcx
   ebf80:	48 39 d9             	cmp    %rbx,%rcx
   ebf83:	48 89 d9             	mov    %rbx,%rcx
   ebf86:	ba 00 00 00 00       	mov    $0x0,%edx
   ebf8b:	48 0f 42 ca          	cmovb  %rdx,%rcx
   ebf8f:	48 f7 d9             	neg    %rcx
   ebf92:	4c 01 e9             	add    %r13,%rcx
   ebf95:	48 ff c1             	inc    %rcx
   ebf98:	49 89 cd             	mov    %rcx,%r13
   ebf9b:	48 89 8c 24 28 01 00 	mov    %rcx,0x128(%rsp)
   ebfa2:	00 
   ebfa3:	4d 89 f7             	mov    %r14,%r15
   ebfa6:	49 ff cf             	dec    %r15
   ebfa9:	4c 89 bc 24 30 01 00 	mov    %r15,0x130(%rsp)
   ebfb0:	00 
   ebfb1:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
   ebfb8:	00 
   ebfb9:	4c 8b b4 24 38 01 00 	mov    0x138(%rsp),%r14
   ebfc0:	00 
   ebfc1:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
   ebfc5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
   ebfca:	49 8b 46 58          	mov    0x58(%r14),%rax
   ebfce:	bd 00 00 00 00       	mov    $0x0,%ebp
   ebfd3:	48 85 c0             	test   %rax,%rax
   ebfd6:	0f 84 c9 03 00 00    	je     ec3a5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x975>
   ebfdc:	48 ff c8             	dec    %rax
   ebfdf:	49 89 46 58          	mov    %rax,0x58(%r14)
   ebfe3:	49 8d 46 08          	lea    0x8(%r14),%rax
   ebfe7:	4d 8d 46 10          	lea    0x10(%r14),%r8
   ebfeb:	49 8b 36             	mov    (%r14),%rsi
   ebfee:	48 85 f6             	test   %rsi,%rsi
   ebff1:	74 3a                	je     ec02d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5fd>
   ebff3:	48 8b 10             	mov    (%rax),%rdx
   ebff6:	48 85 d2             	test   %rdx,%rdx
   ebff9:	74 2b                	je     ec026 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5f6>
   ebffb:	4d 8b 4e 18          	mov    0x18(%r14),%r9
   ebfff:	41 f6 c1 7f          	test   $0x7f,%r9b
   ec003:	0f 84 22 02 00 00    	je     ec22b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
   ec009:	4c 0f be 26          	movsbq (%rsi),%r12
   ec00d:	4d 85 e4             	test   %r12,%r12
   ec010:	0f 88 d0 00 00 00    	js     ec0e6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6b6>
   ec016:	4d 03 20             	add    (%r8),%r12
   ec019:	4d 89 20             	mov    %r12,(%r8)
   ec01c:	bf 01 00 00 00       	mov    $0x1,%edi
   ec021:	e9 3b 01 00 00       	jmp    ec161 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x731>
   ec026:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
   ec02d:	49 8b 6e 40          	mov    0x40(%r14),%rbp
   ec031:	48 85 ed             	test   %rbp,%rbp
   ec034:	0f 84 01 04 00 00    	je     ec43b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
   ec03a:	4d 8b 5e 48          	mov    0x48(%r14),%r11
   ec03e:	4c 39 dd             	cmp    %r11,%rbp
   ec041:	0f 84 f4 03 00 00    	je     ec43b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
   ec047:	4c 8d 65 20          	lea    0x20(%rbp),%r12
   ec04b:	4d 89 66 40          	mov    %r12,0x40(%r14)
   ec04f:	4d 8b 4e 50          	mov    0x50(%r14),%r9
   ec053:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
   ec057:	48 ff c9             	dec    %rcx
   ec05a:	49 39 c9             	cmp    %rcx,%r9
   ec05d:	75 18                	jne    ec077 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x647>
   ec05f:	49 c7 06 01 00 00 00 	movq   $0x1,(%r14)
   ec066:	66 0f ef c0          	pxor   %xmm0,%xmm0
   ec06a:	f3 0f 7f 00          	movdqu %xmm0,(%rax)
   ec06e:	4d 89 4e 18          	mov    %r9,0x18(%r14)
   ec072:	e9 4c 02 00 00       	jmp    ec2c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
   ec077:	49 89 ca             	mov    %rcx,%r10
   ec07a:	49 c1 ea 07          	shr    $0x7,%r10
   ec07e:	4d 85 c9             	test   %r9,%r9
   ec081:	0f 84 e2 00 00 00    	je     ec169 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x739>
   ec087:	89 c8                	mov    %ecx,%eax
   ec089:	83 e0 7f             	and    $0x7f,%eax
   ec08c:	48 83 f8 01          	cmp    $0x1,%rax
   ec090:	4c 89 d6             	mov    %r10,%rsi
   ec093:	48 83 de ff          	sbb    $0xffffffffffffffff,%rsi
   ec097:	48 83 fe 02          	cmp    $0x2,%rsi
   ec09b:	0f 82 6c 0d 00 00    	jb     ece0d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13dd>
   ec0a1:	48 8b 7d 18          	mov    0x18(%rbp),%rdi
   ec0a5:	48 89 fa             	mov    %rdi,%rdx
   ec0a8:	48 83 ff 11          	cmp    $0x11,%rdi
   ec0ac:	72 04                	jb     ec0b2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x682>
   ec0ae:	48 8b 55 10          	mov    0x10(%rbp),%rdx
   ec0b2:	48 8d 04 f5 08 00 00 	lea    0x8(,%rsi,8),%rax
   ec0b9:	00 
   ec0ba:	48 39 d0             	cmp    %rdx,%rax
   ec0bd:	0f 87 dc 0c 00 00    	ja     ecd9f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x136f>
   ec0c3:	4c 89 ca             	mov    %r9,%rdx
   ec0c6:	48 c1 ea 07          	shr    $0x7,%rdx
   ec0ca:	48 39 f2             	cmp    %rsi,%rdx
   ec0cd:	0f 83 45 0d 00 00    	jae    ece18 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13e8>
   ec0d3:	48 83 ff 11          	cmp    $0x11,%rdi
   ec0d7:	0f 82 90 00 00 00    	jb     ec16d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x73d>
   ec0dd:	48 8b 45 08          	mov    0x8(%rbp),%rax
   ec0e1:	e9 8b 00 00 00       	jmp    ec171 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x741>
   ec0e6:	48 83 fa 01          	cmp    $0x1,%rdx
   ec0ea:	0f 84 c0 0c 00 00    	je     ecdb0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1380>
   ec0f0:	41 83 e4 7f          	and    $0x7f,%r12d
   ec0f4:	4c 8d 52 ff          	lea    -0x1(%rdx),%r10
   ec0f8:	b9 07 00 00 00       	mov    $0x7,%ecx
   ec0fd:	31 ff                	xor    %edi,%edi
   ec0ff:	4c 89 e0             	mov    %r12,%rax
   ec102:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec109:	1f 84 00 00 00 00 00 
   ec110:	44 0f b6 5c 3e 01    	movzbl 0x1(%rsi,%rdi,1),%r11d
   ec116:	83 f9 3f             	cmp    $0x3f,%ecx
   ec119:	75 0a                	jne    ec125 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6f5>
   ec11b:	41 80 fb 01          	cmp    $0x1,%r11b
   ec11f:	0f 87 8f 0c 00 00    	ja     ecdb4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
   ec125:	45 0f b6 e3          	movzbl %r11b,%r12d
   ec129:	45 84 e4             	test   %r12b,%r12b
   ec12c:	79 1a                	jns    ec148 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x718>
   ec12e:	41 83 e4 7f          	and    $0x7f,%r12d
   ec132:	49 d3 e4             	shl    %cl,%r12
   ec135:	4c 09 e0             	or     %r12,%rax
   ec138:	83 c1 07             	add    $0x7,%ecx
   ec13b:	48 ff c7             	inc    %rdi
   ec13e:	49 39 fa             	cmp    %rdi,%r10
   ec141:	75 cd                	jne    ec110 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6e0>
   ec143:	e9 68 0c 00 00       	jmp    ecdb0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1380>
   ec148:	49 d3 e4             	shl    %cl,%r12
   ec14b:	49 09 c4             	or     %rax,%r12
   ec14e:	48 83 c7 02          	add    $0x2,%rdi
   ec152:	4d 03 20             	add    (%r8),%r12
   ec155:	4d 89 20             	mov    %r12,(%r8)
   ec158:	48 39 d7             	cmp    %rdx,%rdi
   ec15b:	0f 87 e8 0b 00 00    	ja     ecd49 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1319>
   ec161:	48 29 fa             	sub    %rdi,%rdx
   ec164:	e9 db 00 00 00       	jmp    ec244 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x814>
   ec169:	31 ff                	xor    %edi,%edi
   ec16b:	eb 09                	jmp    ec176 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x746>
   ec16d:	48 8d 45 08          	lea    0x8(%rbp),%rax
   ec171:	48 8b 7c d0 08       	mov    0x8(%rax,%rdx,8),%rdi
   ec176:	83 e1 7f             	and    $0x7f,%ecx
   ec179:	48 83 f9 01          	cmp    $0x1,%rcx
   ec17d:	49 83 da ff          	sbb    $0xffffffffffffffff,%r10
   ec181:	be 08 00 00 00       	mov    $0x8,%esi
   ec186:	49 83 fa 02          	cmp    $0x2,%r10
   ec18a:	73 1e                	jae    ec1aa <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x77a>
   ec18c:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
   ec190:	48 89 c8             	mov    %rcx,%rax
   ec193:	48 83 f9 11          	cmp    $0x11,%rcx
   ec197:	72 39                	jb     ec1d2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7a2>
   ec199:	48 8b 45 10          	mov    0x10(%rbp),%rax
   ec19d:	48 89 c2             	mov    %rax,%rdx
   ec1a0:	48 29 f2             	sub    %rsi,%rdx
   ec1a3:	73 39                	jae    ec1de <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7ae>
   ec1a5:	e9 de 0b 00 00       	jmp    ecd88 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1358>
   ec1aa:	49 ff ca             	dec    %r10
   ec1ad:	49 0f bd ca          	bsr    %r10,%rcx
   ec1b1:	f7 d1                	not    %ecx
   ec1b3:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
   ec1ba:	48 d3 e8             	shr    %cl,%rax
   ec1bd:	48 8d 34 c5 10 00 00 	lea    0x10(,%rax,8),%rsi
   ec1c4:	00 
   ec1c5:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
   ec1c9:	48 89 c8             	mov    %rcx,%rax
   ec1cc:	48 83 f9 11          	cmp    $0x11,%rcx
   ec1d0:	73 c7                	jae    ec199 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x769>
   ec1d2:	48 89 c2             	mov    %rax,%rdx
   ec1d5:	48 29 f2             	sub    %rsi,%rdx
   ec1d8:	0f 82 aa 0b 00 00    	jb     ecd88 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1358>
   ec1de:	48 39 fa             	cmp    %rdi,%rdx
   ec1e1:	0f 82 a9 0b 00 00    	jb     ecd90 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1360>
   ec1e7:	48 83 f9 11          	cmp    $0x11,%rcx
   ec1eb:	72 06                	jb     ec1f3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c3>
   ec1ed:	48 8b 6d 08          	mov    0x8(%rbp),%rbp
   ec1f1:	eb 04                	jmp    ec1f7 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c7>
   ec1f3:	48 83 c5 08          	add    $0x8,%rbp
   ec1f7:	48 01 fe             	add    %rdi,%rsi
   ec1fa:	48 01 ee             	add    %rbp,%rsi
   ec1fd:	48 29 fa             	sub    %rdi,%rdx
   ec200:	49 89 36             	mov    %rsi,(%r14)
   ec203:	49 89 56 08          	mov    %rdx,0x8(%r14)
   ec207:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
   ec20e:	00 
   ec20f:	4d 89 4e 18          	mov    %r9,0x18(%r14)
   ec213:	0f 84 aa 00 00 00    	je     ec2c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
   ec219:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
   ec220:	00 
   ec221:	41 f6 c1 7f          	test   $0x7f,%r9b
   ec225:	0f 85 de fd ff ff    	jne    ec009 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
   ec22b:	48 83 fa 07          	cmp    $0x7,%rdx
   ec22f:	0f 86 40 0b 00 00    	jbe    ecd75 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1345>
   ec235:	4c 8b 26             	mov    (%rsi),%r12
   ec238:	4d 89 20             	mov    %r12,(%r8)
   ec23b:	48 83 c2 f8          	add    $0xfffffffffffffff8,%rdx
   ec23f:	bf 08 00 00 00       	mov    $0x8,%edi
   ec244:	48 01 fe             	add    %rdi,%rsi
   ec247:	49 89 36             	mov    %rsi,(%r14)
   ec24a:	49 89 56 08          	mov    %rdx,0x8(%r14)
   ec24e:	49 ff c1             	inc    %r9
   ec251:	4d 89 4e 18          	mov    %r9,0x18(%r14)
   ec255:	48 8b 84 24 40 01 00 	mov    0x140(%rsp),%rax
   ec25c:	00 
   ec25d:	4c 3b 60 10          	cmp    0x10(%rax),%r12
   ec261:	0f 83 c5 0a 00 00    	jae    ecd2c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12fc>
   ec267:	48 8b 84 24 48 01 00 	mov    0x148(%rsp),%rax
   ec26e:	00 
   ec26f:	48 8b 48 10          	mov    0x10(%rax),%rcx
   ec273:	83 38 01             	cmpl   $0x1,(%rax)
   ec276:	0f 85 14 01 00 00    	jne    ec390 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x960>
   ec27c:	49 39 cc             	cmp    %rcx,%r12
   ec27f:	0f 83 1b 01 00 00    	jae    ec3a0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
   ec285:	4a 8d 0c a5 00 00 00 	lea    0x0(,%r12,4),%rcx
   ec28c:	00 
   ec28d:	e9 07 01 00 00       	jmp    ec399 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x969>
   ec292:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec299:	1f 84 00 00 00 00 00 
   ec2a0:	be 01 00 00 00       	mov    $0x1,%esi
   ec2a5:	31 d2                	xor    %edx,%edx
   ec2a7:	49 89 36             	mov    %rsi,(%r14)
   ec2aa:	49 89 56 08          	mov    %rdx,0x8(%r14)
   ec2ae:	66 0f ef c0          	pxor   %xmm0,%xmm0
   ec2b2:	f3 41 0f 7f 00       	movdqu %xmm0,(%r8)
   ec2b7:	4d 89 cc             	mov    %r9,%r12
   ec2ba:	48 85 d2             	test   %rdx,%rdx
   ec2bd:	0f 85 36 03 00 00    	jne    ec5f9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbc9>
   ec2c3:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
   ec2ca:	4d 39 dc             	cmp    %r11,%r12
   ec2cd:	0f 84 60 01 00 00    	je     ec433 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa03>
   ec2d3:	4d 8d 4c 24 20       	lea    0x20(%r12),%r9
   ec2d8:	4d 89 4e 40          	mov    %r9,0x40(%r14)
   ec2dc:	49 8b 04 24          	mov    (%r12),%rax
   ec2e0:	48 ff c8             	dec    %rax
   ec2e3:	74 bb                	je     ec2a0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x870>
   ec2e5:	89 c1                	mov    %eax,%ecx
   ec2e7:	83 e1 7f             	and    $0x7f,%ecx
   ec2ea:	48 c1 e8 07          	shr    $0x7,%rax
   ec2ee:	48 83 f9 01          	cmp    $0x1,%rcx
   ec2f2:	48 83 d8 ff          	sbb    $0xffffffffffffffff,%rax
   ec2f6:	bf 08 00 00 00       	mov    $0x8,%edi
   ec2fb:	48 83 f8 02          	cmp    $0x2,%rax
   ec2ff:	73 2f                	jae    ec330 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x900>
   ec301:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
   ec306:	48 89 c8             	mov    %rcx,%rax
   ec309:	48 83 f9 11          	cmp    $0x11,%rcx
   ec30d:	72 4a                	jb     ec359 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x929>
   ec30f:	49 8b 44 24 10       	mov    0x10(%r12),%rax
   ec314:	48 89 c2             	mov    %rax,%rdx
   ec317:	48 29 fa             	sub    %rdi,%rdx
   ec31a:	73 49                	jae    ec365 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x935>
   ec31c:	e9 80 09 00 00       	jmp    ecca1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1271>
   ec321:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec328:	0f 1f 84 00 00 00 00 
   ec32f:	00 
   ec330:	48 ff c8             	dec    %rax
   ec333:	48 0f bd c8          	bsr    %rax,%rcx
   ec337:	f7 d1                	not    %ecx
   ec339:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
   ec340:	48 d3 e8             	shr    %cl,%rax
   ec343:	48 8d 3c c5 10 00 00 	lea    0x10(,%rax,8),%rdi
   ec34a:	00 
   ec34b:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
   ec350:	48 89 c8             	mov    %rcx,%rax
   ec353:	48 83 f9 11          	cmp    $0x11,%rcx
   ec357:	73 b6                	jae    ec30f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x8df>
   ec359:	48 89 c2             	mov    %rax,%rdx
   ec35c:	48 29 fa             	sub    %rdi,%rdx
   ec35f:	0f 82 3c 09 00 00    	jb     ecca1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1271>
   ec365:	48 83 f9 11          	cmp    $0x11,%rcx
   ec369:	72 15                	jb     ec380 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x950>
   ec36b:	49 8b 74 24 08       	mov    0x8(%r12),%rsi
   ec370:	eb 15                	jmp    ec387 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x957>
   ec372:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec379:	1f 84 00 00 00 00 00 
   ec380:	49 83 c4 08          	add    $0x8,%r12
   ec384:	4c 89 e6             	mov    %r12,%rsi
   ec387:	48 01 fe             	add    %rdi,%rsi
   ec38a:	e9 18 ff ff ff       	jmp    ec2a7 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x877>
   ec38f:	90                   	nop
   ec390:	49 39 cc             	cmp    %rcx,%r12
   ec393:	73 0b                	jae    ec3a0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
   ec395:	4b 8d 0c 24          	lea    (%r12,%r12,1),%rcx
   ec399:	48 03 48 08          	add    0x8(%rax),%rcx
   ec39d:	0f 18 09             	prefetcht0 (%rcx)
   ec3a0:	bd 01 00 00 00       	mov    $0x1,%ebp
   ec3a5:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
   ec3a9:	48 39 d8             	cmp    %rbx,%rax
   ec3ac:	0f 86 ce 00 00 00    	jbe    ec480 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa50>
   ec3b2:	48 89 d8             	mov    %rbx,%rax
   ec3b5:	4c 29 f8             	sub    %r15,%rax
   ec3b8:	48 39 c5             	cmp    %rax,%rbp
   ec3bb:	0f 87 52 02 00 00    	ja     ec613 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbe3>
   ec3c1:	4d 89 ee             	mov    %r13,%r14
   ec3c4:	49 89 dd             	mov    %rbx,%r13
   ec3c7:	49 39 c6             	cmp    %rax,%r14
   ec3ca:	76 62                	jbe    ec42e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
   ec3cc:	48 89 da             	mov    %rbx,%rdx
   ec3cf:	4c 29 f2             	sub    %r14,%rdx
   ec3d2:	4c 89 f8             	mov    %r15,%rax
   ec3d5:	48 29 d0             	sub    %rdx,%rax
   ec3d8:	48 39 c2             	cmp    %rax,%rdx
   ec3db:	76 26                	jbe    ec403 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
   ec3dd:	4c 89 e9             	mov    %r13,%rcx
   ec3e0:	48 29 d9             	sub    %rbx,%rcx
   ec3e3:	48 39 c1             	cmp    %rax,%rcx
   ec3e6:	72 1b                	jb     ec403 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
   ec3e8:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
   ec3ef:	00 
   ec3f0:	48 8d 3c de          	lea    (%rsi,%rbx,8),%rdi
   ec3f4:	48 c1 e0 03          	shl    $0x3,%rax
   ec3f8:	48 89 c2             	mov    %rax,%rdx
   ec3fb:	ff 15 ef c7 01 00    	call   *0x1c7ef(%rip)        # 108bf0 <memcpy@GLIBC_2.14>
   ec401:	eb 2b                	jmp    ec42e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
   ec403:	4c 89 f1             	mov    %r14,%rcx
   ec406:	4d 89 ee             	mov    %r13,%r14
   ec409:	49 29 d6             	sub    %rdx,%r14
   ec40c:	48 8b 84 24 20 01 00 	mov    0x120(%rsp),%rax
   ec413:	00 
   ec414:	48 8d 34 c8          	lea    (%rax,%rcx,8),%rsi
   ec418:	4a 8d 3c f0          	lea    (%rax,%r14,8),%rdi
   ec41c:	48 c1 e2 03          	shl    $0x3,%rdx
   ec420:	ff 15 d2 c7 01 00    	call   *0x1c7d2(%rip)        # 108bf8 <memmove@GLIBC_2.2.5>
   ec426:	4c 89 b4 24 28 01 00 	mov    %r14,0x128(%rsp)
   ec42d:	00 
   ec42e:	4c 89 eb             	mov    %r13,%rbx
   ec431:	eb 50                	jmp    ec483 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa53>
   ec433:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
   ec43a:	00 
   ec43b:	49 83 7e 20 00       	cmpq   $0x0,0x20(%r14)
   ec440:	74 20                	je     ec462 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa32>
   ec442:	49 83 c6 20          	add    $0x20,%r14
   ec446:	4c 89 f7             	mov    %r14,%rdi
   ec449:	e8 72 d1 ff ff       	call   e95c0 <_RNvXs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions5codecNtB4_6CursorNtNtNtNtCsgxBkk5gSRhY_4core4iter6traits8iterator8Iterator4next>
   ec44e:	49 89 d4             	mov    %rdx,%r12
   ec451:	48 83 f8 01          	cmp    $0x1,%rax
   ec455:	0f 84 fa fd ff ff    	je     ec255 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x825>
   ec45b:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
   ec462:	31 ed                	xor    %ebp,%ebp
   ec464:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
   ec468:	48 39 d8             	cmp    %rbx,%rax
   ec46b:	0f 87 41 ff ff ff    	ja     ec3b2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x982>
   ec471:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec478:	0f 1f 84 00 00 00 00 
   ec47f:	00 
   ec480:	4d 89 ee             	mov    %r13,%r14
   ec483:	4b 8d 04 37          	lea    (%r15,%r14,1),%rax
   ec487:	48 39 d8             	cmp    %rbx,%rax
   ec48a:	b9 00 00 00 00       	mov    $0x0,%ecx
   ec48f:	48 0f 43 cb          	cmovae %rbx,%rcx
   ec493:	48 29 c8             	sub    %rcx,%rax
   ec496:	48 89 d9             	mov    %rbx,%rcx
   ec499:	48 29 c1             	sub    %rax,%rcx
   ec49c:	4d 89 f5             	mov    %r14,%r13
   ec49f:	48 39 e9             	cmp    %rbp,%rcx
   ec4a2:	73 2c                	jae    ec4d0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xaa0>
   ec4a4:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
   ec4ab:	00 
   ec4ac:	48 39 c3             	cmp    %rax,%rbx
   ec4af:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
   ec4b4:	75 3a                	jne    ec4f0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xac0>
   ec4b6:	4c 89 21             	mov    %r12,(%rcx)
   ec4b9:	41 be 01 00 00 00    	mov    $0x1,%r14d
   ec4bf:	eb 47                	jmp    ec508 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
   ec4c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ec4c8:	0f 1f 84 00 00 00 00 
   ec4cf:	00 
   ec4d0:	48 83 fd 01          	cmp    $0x1,%rbp
   ec4d4:	75 2a                	jne    ec500 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad0>
   ec4d6:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
   ec4dd:	00 
   ec4de:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
   ec4e2:	41 be 01 00 00 00    	mov    $0x1,%r14d
   ec4e8:	eb 19                	jmp    ec503 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad3>
   ec4ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   ec4f0:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
   ec4f4:	41 be 01 00 00 00    	mov    $0x1,%r14d
   ec4fa:	eb 0c                	jmp    ec508 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
   ec4fc:	0f 1f 40 00          	nopl   0x0(%rax)
   ec500:	45 31 f6             	xor    %r14d,%r14d
   ec503:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
   ec508:	4d 01 fe             	add    %r15,%r14
   ec50b:	4c 89 b4 24 30 01 00 	mov    %r14,0x130(%rsp)
   ec512:	00 
   ec513:	4c 8b 8c 24 c8 00 00 	mov    0xc8(%rsp),%r9
   ec51a:	00 
   ec51b:	49 8b 01             	mov    (%r9),%rax
   ec51e:	49 8b 71 10          	mov    0x10(%r9),%rsi
   ec522:	48 83 f8 01          	cmp    $0x1,%rax
   ec526:	75 18                	jne    ec540 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb10>
   ec528:	48 39 f5             	cmp    %rsi,%rbp
   ec52b:	0f 83 c1 08 00 00    	jae    ecdf2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c2>
   ec531:	49 8b 49 08          	mov    0x8(%r9),%rcx
   ec535:	8b 14 a9             	mov    (%rcx,%rbp,4),%edx
   ec538:	eb 26                	jmp    ec560 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb30>
   ec53a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   ec540:	48 39 f5             	cmp    %rsi,%rbp
   ec543:	0f 83 9d 08 00 00    	jae    ecde6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b6>
   ec549:	49 8b 49 08          	mov    0x8(%r9),%rcx
   ec54d:	0f b7 14 69          	movzwl (%rcx,%rbp,2),%edx
   ec551:	66 83 fa ff          	cmp    $0xffff,%dx
   ec555:	0f b7 d2             	movzwl %dx,%edx
   ec558:	bf ff ff ff ff       	mov    $0xffffffff,%edi
   ec55d:	0f 44 d7             	cmove  %edi,%edx
   ec560:	4c 8b 84 24 90 00 00 	mov    0x90(%rsp),%r8
   ec567:	00 
   ec568:	4c 39 44 24 20       	cmp    %r8,0x20(%rsp)
   ec56d:	40 0f 94 c7          	sete   %dil
   ec571:	4d 85 c0             	test   %r8,%r8
   ec574:	41 0f 94 c0          	sete   %r8b
   ec578:	3b 94 24 c0 00 00 00 	cmp    0xc0(%rsp),%edx
   ec57f:	0f 95 c2             	setne  %dl
   ec582:	41 08 f8             	or     %dil,%r8b
   ec585:	41 08 d0             	or     %dl,%r8b
   ec588:	0f 85 e2 f9 ff ff    	jne    ebf70 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
   ec58e:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
   ec593:	4c 8d 24 2a          	lea    (%rdx,%rbp,1),%r12
   ec597:	49 39 f4             	cmp    %rsi,%r12
   ec59a:	0f 83 d0 f9 ff ff    	jae    ebf70 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
   ec5a0:	48 8b 94 24 90 00 00 	mov    0x90(%rsp),%rdx
   ec5a7:	00 
   ec5a8:	48 8d 3c 2a          	lea    (%rdx,%rbp,1),%rdi
   ec5ac:	84 c0                	test   %al,%al
   ec5ae:	74 1e                	je     ec5ce <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb9e>
   ec5b0:	48 39 f7             	cmp    %rsi,%rdi
   ec5b3:	0f 83 3c 08 00 00    	jae    ecdf5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
   ec5b9:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
   ec5bc:	3b 84 24 b8 00 00 00 	cmp    0xb8(%rsp),%eax
   ec5c3:	0f 85 a7 f9 ff ff    	jne    ebf70 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
   ec5c9:	e9 96 00 00 00       	jmp    ec664 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
   ec5ce:	48 39 f7             	cmp    %rsi,%rdi
   ec5d1:	0f 83 12 08 00 00    	jae    ecde9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
   ec5d7:	0f b7 04 79          	movzwl (%rcx,%rdi,2),%eax
   ec5db:	66 83 f8 ff          	cmp    $0xffff,%ax
   ec5df:	0f b7 c0             	movzwl %ax,%eax
   ec5e2:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
   ec5e7:	0f 44 c1             	cmove  %ecx,%eax
   ec5ea:	3b 84 24 b8 00 00 00 	cmp    0xb8(%rsp),%eax
   ec5f1:	0f 85 79 f9 ff ff    	jne    ebf70 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
   ec5f7:	eb 6b                	jmp    ec664 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
   ec5f9:	45 31 c9             	xor    %r9d,%r9d
   ec5fc:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
   ec603:	00 
   ec604:	41 f6 c1 7f          	test   $0x7f,%r9b
   ec608:	0f 85 fb f9 ff ff    	jne    ec009 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
   ec60e:	e9 18 fc ff ff       	jmp    ec22b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
   ec613:	ba 01 00 00 00       	mov    $0x1,%edx
   ec618:	b9 08 00 00 00       	mov    $0x8,%ecx
   ec61d:	41 b8 08 00 00 00    	mov    $0x8,%r8d
   ec623:	48 8d bc 24 18 01 00 	lea    0x118(%rsp),%rdi
   ec62a:	00 
   ec62b:	4c 89 fe             	mov    %r15,%rsi
   ec62e:	e8 ed 18 fe ff       	call   cdf20 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsgHLVOFInvdC_8tk_train>
   ec633:	4c 8b ac 24 18 01 00 	mov    0x118(%rsp),%r13
   ec63a:	00 
   ec63b:	4c 8b b4 24 28 01 00 	mov    0x128(%rsp),%r14
   ec642:	00 
   ec643:	4c 8b bc 24 30 01 00 	mov    0x130(%rsp),%r15
   ec64a:	00 
   ec64b:	48 89 d8             	mov    %rbx,%rax
   ec64e:	4c 29 f8             	sub    %r15,%rax
   ec651:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
   ec656:	49 39 c6             	cmp    %rax,%r14
   ec659:	0f 87 6d fd ff ff    	ja     ec3cc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x99c>
   ec65f:	e9 ca fd ff ff       	jmp    ec42e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
   ec664:	83 bc 24 d8 00 00 00 	cmpl   $0x1,0xd8(%rsp)
   ec66b:	01 
   ec66c:	75 22                	jne    ec690 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc60>
   ec66e:	48 8b 94 24 e8 00 00 	mov    0xe8(%rsp),%rdx
   ec675:	00 
   ec676:	48 39 d5             	cmp    %rdx,%rbp
   ec679:	73 15                	jae    ec690 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc60>
   ec67b:	48 8b 9c 24 e0 00 00 	mov    0xe0(%rsp),%rbx
   ec682:	00 
   ec683:	4c 8b 94 24 68 01 00 	mov    0x168(%rsp),%r10
   ec68a:	00 
   ec68b:	e9 bd 00 00 00       	jmp    ec74d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
   ec690:	48 8b 84 24 10 01 00 	mov    0x110(%rsp),%rax
   ec697:	00 
   ec698:	48 8b 48 68          	mov    0x68(%rax),%rcx
   ec69c:	48 8b 70 70          	mov    0x70(%rax),%rsi
   ec6a0:	48 83 fe 01          	cmp    $0x1,%rsi
   ec6a4:	4c 8b 94 24 68 01 00 	mov    0x168(%rsp),%r10
   ec6ab:	00 
   ec6ac:	75 0d                	jne    ec6bb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc8b>
   ec6ae:	48 8b 59 08          	mov    0x8(%rcx),%rbx
   ec6b2:	48 8b 50 10          	mov    0x10(%rax),%rdx
   ec6b6:	e9 92 00 00 00       	jmp    ec74d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
   ec6bb:	83 78 18 01          	cmpl   $0x1,0x18(%rax)
   ec6bf:	75 14                	jne    ec6d5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xca5>
   ec6c1:	48 3b 68 20          	cmp    0x20(%rax),%rbp
   ec6c5:	72 0e                	jb     ec6d5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xca5>
   ec6c7:	48 8b 50 28          	mov    0x28(%rax),%rdx
   ec6cb:	bb 01 00 00 00       	mov    $0x1,%ebx
   ec6d0:	48 39 d5             	cmp    %rdx,%rbp
   ec6d3:	72 78                	jb     ec74d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
   ec6d5:	48 85 f6             	test   %rsi,%rsi
   ec6d8:	0f 84 b0 07 00 00    	je     ece8e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x145e>
   ec6de:	4d 89 d3             	mov    %r10,%r11
   ec6e1:	48 89 f7             	mov    %rsi,%rdi
   ec6e4:	31 d2                	xor    %edx,%edx
   ec6e6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
   ec6ed:	00 00 00 
   ec6f0:	49 89 d0             	mov    %rdx,%r8
   ec6f3:	49 89 f9             	mov    %rdi,%r9
   ec6f6:	49 d1 e9             	shr    $1,%r9
   ec6f9:	4c 01 ca             	add    %r9,%rdx
   ec6fc:	49 89 d2             	mov    %rdx,%r10
   ec6ff:	49 c1 e2 04          	shl    $0x4,%r10
   ec703:	4a 39 2c 11          	cmp    %rbp,(%rcx,%r10,1)
   ec707:	49 0f 47 d0          	cmova  %r8,%rdx
   ec70b:	4c 29 cf             	sub    %r9,%rdi
   ec70e:	48 83 ff 01          	cmp    $0x1,%rdi
   ec712:	77 dc                	ja     ec6f0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xcc0>
   ec714:	48 89 d7             	mov    %rdx,%rdi
   ec717:	48 c1 e7 04          	shl    $0x4,%rdi
   ec71b:	48 3b 2c 39          	cmp    (%rcx,%rdi,1),%rbp
   ec71f:	48 83 da ff          	sbb    $0xffffffffffffffff,%rdx
   ec723:	48 8d 7a ff          	lea    -0x1(%rdx),%rdi
   ec727:	48 39 f7             	cmp    %rsi,%rdi
   ec72a:	0f 83 67 07 00 00    	jae    ece97 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1467>
   ec730:	48 c1 e7 04          	shl    $0x4,%rdi
   ec734:	48 8b 5c 39 08       	mov    0x8(%rcx,%rdi,1),%rbx
   ec739:	48 39 f2             	cmp    %rsi,%rdx
   ec73c:	4d 89 da             	mov    %r11,%r10
   ec73f:	0f 83 6d ff ff ff    	jae    ec6b2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc82>
   ec745:	48 c1 e2 04          	shl    $0x4,%rdx
   ec749:	48 8b 14 11          	mov    (%rcx,%rdx,1),%rdx
   ec74d:	48 c7 84 24 d8 00 00 	movq   $0x1,0xd8(%rsp)
   ec754:	00 01 00 00 00 
   ec759:	48 89 9c 24 e0 00 00 	mov    %rbx,0xe0(%rsp)
   ec760:	00 
   ec761:	48 89 94 24 e8 00 00 	mov    %rdx,0xe8(%rsp)
   ec768:	00 
   ec769:	48 8d 4d ff          	lea    -0x1(%rbp),%rcx
   ec76d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ec772:	48 8b 38             	mov    (%rax),%rdi
   ec775:	48 8b 70 10          	mov    0x10(%rax),%rsi
   ec779:	48 83 ff 01          	cmp    $0x1,%rdi
   ec77d:	75 37                	jne    ec7b6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd86>
   ec77f:	48 39 f1             	cmp    %rsi,%rcx
   ec782:	0f 83 d5 06 00 00    	jae    ece5d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x142d>
   ec788:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
   ec78d:	48 8b 42 08          	mov    0x8(%rdx),%rax
   ec791:	8b 4c a8 fc          	mov    -0x4(%rax,%rbp,4),%ecx
   ec795:	83 f9 ff             	cmp    $0xffffffff,%ecx
   ec798:	75 41                	jne    ec7db <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdab>
   ec79a:	49 39 f4             	cmp    %rsi,%r12
   ec79d:	0f 83 c4 06 00 00    	jae    ece67 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1437>
   ec7a3:	46 8b 0c a0          	mov    (%rax,%r12,4),%r9d
   ec7a7:	41 83 f9 ff          	cmp    $0xffffffff,%r9d
   ec7ab:	0f 85 43 02 00 00    	jne    ec9f4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc4>
   ec7b1:	e9 e8 03 00 00       	jmp    ecb9e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
   ec7b6:	48 39 f1             	cmp    %rsi,%rcx
   ec7b9:	0f 83 a3 06 00 00    	jae    ece62 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1432>
   ec7bf:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ec7c4:	48 8b 40 08          	mov    0x8(%rax),%rax
   ec7c8:	0f b7 4c 68 fe       	movzwl -0x2(%rax,%rbp,2),%ecx
   ec7cd:	66 83 f9 ff          	cmp    $0xffff,%cx
   ec7d1:	74 50                	je     ec823 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdf3>
   ec7d3:	0f b7 c9             	movzwl %cx,%ecx
   ec7d6:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
   ec7db:	89 c9                	mov    %ecx,%ecx
   ec7dd:	48 8b 42 50          	mov    0x50(%rdx),%rax
   ec7e1:	48 39 c8             	cmp    %rcx,%rax
   ec7e4:	0f 86 5a 06 00 00    	jbe    ece44 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1414>
   ec7ea:	48 8b 42 48          	mov    0x48(%rdx),%rax
   ec7ee:	49 89 cb             	mov    %rcx,%r11
   ec7f1:	4c 8b 0c c8          	mov    (%rax,%rcx,8),%r9
   ec7f5:	49 89 e8             	mov    %rbp,%r8
   ec7f8:	4d 29 c8             	sub    %r9,%r8
   ec7fb:	0f 84 53 01 00 00    	je     ec954 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
   ec801:	49 8d 40 ff          	lea    -0x1(%r8),%rax
   ec805:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
   ec80a:	74 34                	je     ec840 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe10>
   ec80c:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
   ec811:	0f 83 5d 06 00 00    	jae    ece74 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1444>
   ec817:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
   ec81c:	46 8b 7c 80 fc       	mov    -0x4(%rax,%r8,4),%r15d
   ec821:	eb 44                	jmp    ec867 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe37>
   ec823:	49 39 f4             	cmp    %rsi,%r12
   ec826:	0f 83 40 06 00 00    	jae    ece6c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x143c>
   ec82c:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
   ec831:	66 83 f8 ff          	cmp    $0xffff,%ax
   ec835:	0f 85 b5 01 00 00    	jne    ec9f0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc0>
   ec83b:	e9 5e 03 00 00       	jmp    ecb9e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
   ec840:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
   ec845:	0f 83 36 06 00 00    	jae    ece81 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1451>
   ec84b:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
   ec850:	42 0f b7 44 40 fe    	movzwl -0x2(%rax,%r8,2),%eax
   ec856:	66 83 f8 ff          	cmp    $0xffff,%ax
   ec85a:	44 0f b7 f8          	movzwl %ax,%r15d
   ec85e:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
   ec863:	44 0f 44 f8          	cmove  %eax,%r15d
   ec867:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
   ec86e:	00 00 
   ec870:	0f 84 de 00 00 00    	je     ec954 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
   ec876:	44 89 f8             	mov    %r15d,%eax
   ec879:	48 33 84 24 58 01 00 	xor    0x158(%rsp),%rax
   ec880:	00 
   ec881:	48 b9 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rcx
   ec888:	f4 51 58 
   ec88b:	48 f7 e1             	mul    %rcx
   ec88e:	4c 31 da             	xor    %r11,%rdx
   ec891:	48 31 d0             	xor    %rdx,%rax
   ec894:	48 f7 e1             	mul    %rcx
   ec897:	48 89 c1             	mov    %rax,%rcx
   ec89a:	48 31 d1             	xor    %rdx,%rcx
   ec89d:	48 89 c8             	mov    %rcx,%rax
   ec8a0:	48 f7 a4 24 50 01 00 	mulq   0x150(%rsp)
   ec8a7:	00 
   ec8a8:	48 31 c2             	xor    %rax,%rdx
   ec8ab:	48 d3 c2             	rol    %cl,%rdx
   ec8ae:	48 89 d0             	mov    %rdx,%rax
   ec8b1:	48 c1 e8 39          	shr    $0x39,%rax
   ec8b5:	66 0f 6e c0          	movd   %eax,%xmm0
   ec8b9:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
   ec8bd:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
   ec8c2:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
   ec8c7:	31 c0                	xor    %eax,%eax
   ec8c9:	48 23 54 24 78       	and    0x78(%rsp),%rdx
   ec8ce:	f3 41 0f 6f 0c 12    	movdqu (%r10,%rdx,1),%xmm1
   ec8d4:	66 0f 6f d1          	movdqa %xmm1,%xmm2
   ec8d8:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
   ec8dc:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
   ec8e0:	85 c9                	test   %ecx,%ecx
   ec8e2:	74 50                	je     ec934 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf04>
   ec8e4:	44 89 7c 24 30       	mov    %r15d,0x30(%rsp)
   ec8e9:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   ec8ee:	44 89 f8             	mov    %r15d,%eax
   ec8f1:	4d 89 d7             	mov    %r10,%r15
   ec8f4:	f3 44 0f bc d1       	tzcnt  %ecx,%r10d
   ec8f9:	49 01 d2             	add    %rdx,%r10
   ec8fc:	4c 23 54 24 78       	and    0x78(%rsp),%r10
   ec901:	49 f7 da             	neg    %r10
   ec904:	4f 8d 14 52          	lea    (%r10,%r10,2),%r10
   ec908:	43 3b 44 97 f4       	cmp    -0xc(%r15,%r10,4),%eax
   ec90d:	75 0b                	jne    ec91a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xeea>
   ec90f:	47 3b 5c 97 f8       	cmp    -0x8(%r15,%r10,4),%r11d
   ec914:	0f 84 8c 00 00 00    	je     ec9a6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf76>
   ec91a:	44 8d 51 ff          	lea    -0x1(%rcx),%r10d
   ec91e:	66 41 21 ca          	and    %cx,%r10w
   ec922:	44 89 d1             	mov    %r10d,%ecx
   ec925:	4d 89 fa             	mov    %r15,%r10
   ec928:	44 8b 7c 24 30       	mov    0x30(%rsp),%r15d
   ec92d:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
   ec932:	75 ba                	jne    ec8ee <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xebe>
   ec934:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
   ec938:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
   ec93c:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
   ec940:	85 c9                	test   %ecx,%ecx
   ec942:	75 10                	jne    ec954 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
   ec944:	48 01 c2             	add    %rax,%rdx
   ec947:	48 83 c2 10          	add    $0x10,%rdx
   ec94b:	48 83 c0 10          	add    $0x10,%rax
   ec94f:	e9 75 ff ff ff       	jmp    ec8c9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe99>
   ec954:	4d 89 d7             	mov    %r10,%r15
   ec957:	4c 89 e0             	mov    %r12,%rax
   ec95a:	48 29 e8             	sub    %rbp,%rax
   ec95d:	b9 00 00 00 00       	mov    $0x0,%ecx
   ec962:	48 0f 43 c8          	cmovae %rax,%rcx
   ec966:	49 01 c9             	add    %rcx,%r9
   ec969:	31 c0                	xor    %eax,%eax
   ec96b:	4c 3b 8c 24 88 00 00 	cmp    0x88(%rsp),%r9
   ec972:	00 
   ec973:	0f 92 c0             	setb   %al
   ec976:	89 04 24             	mov    %eax,(%rsp)
   ec979:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
   ec980:	00 
   ec981:	31 f6                	xor    %esi,%esi
   ec983:	44 89 da             	mov    %r11d,%edx
   ec986:	44 89 d9             	mov    %r11d,%ecx
   ec989:	49 89 d9             	mov    %rbx,%r9
   ec98c:	e8 7f 76 00 00       	call   f4010 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
   ec991:	48 85 c0             	test   %rax,%rax
   ec994:	0f 85 26 03 00 00    	jne    eccc0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1290>
   ec99a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ec99f:	48 8b 38             	mov    (%rax),%rdi
   ec9a2:	48 8b 70 10          	mov    0x10(%rax),%rsi
   ec9a6:	48 85 ff             	test   %rdi,%rdi
   ec9a9:	4d 89 fa             	mov    %r15,%r10
   ec9ac:	74 21                	je     ec9cf <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf9f>
   ec9ae:	49 39 f4             	cmp    %rsi,%r12
   ec9b1:	0f 83 b0 04 00 00    	jae    ece67 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1437>
   ec9b7:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ec9bc:	48 8b 40 08          	mov    0x8(%rax),%rax
   ec9c0:	46 8b 0c a0          	mov    (%rax,%r12,4),%r9d
   ec9c4:	41 83 f9 ff          	cmp    $0xffffffff,%r9d
   ec9c8:	75 2a                	jne    ec9f4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc4>
   ec9ca:	e9 cf 01 00 00       	jmp    ecb9e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
   ec9cf:	49 39 f4             	cmp    %rsi,%r12
   ec9d2:	0f 83 94 04 00 00    	jae    ece6c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x143c>
   ec9d8:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
   ec9dd:	48 8b 40 08          	mov    0x8(%rax),%rax
   ec9e1:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
   ec9e6:	66 83 f8 ff          	cmp    $0xffff,%ax
   ec9ea:	0f 84 ae 01 00 00    	je     ecb9e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
   ec9f0:	44 0f b7 c8          	movzwl %ax,%r9d
   ec9f4:	44 89 cf             	mov    %r9d,%edi
   ec9f7:	48 39 bc 24 80 00 00 	cmp    %rdi,0x80(%rsp)
   ec9fe:	00 
   ec9ff:	0f 86 47 04 00 00    	jbe    ece4c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x141c>
   eca05:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
   eca0c:	00 
   eca0d:	48 8b 04 f8          	mov    (%rax,%rdi,8),%rax
   eca11:	4c 01 e0             	add    %r12,%rax
   eca14:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
   eca19:	0f 84 dd 00 00 00    	je     ecafc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10cc>
   eca1f:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
   eca24:	0f 83 4a 04 00 00    	jae    ece74 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1444>
   eca2a:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
   eca2f:	8b 34 81             	mov    (%rcx,%rax,4),%esi
   eca32:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
   eca39:	00 00 
   eca3b:	0f 84 ed 00 00 00    	je     ecb2e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10fe>
   eca41:	48 8b 84 24 58 01 00 	mov    0x158(%rsp),%rax
   eca48:	00 
   eca49:	49 89 f8             	mov    %rdi,%r8
   eca4c:	48 31 f8             	xor    %rdi,%rax
   eca4f:	48 bf 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rdi
   eca56:	f4 51 58 
   eca59:	48 f7 e7             	mul    %rdi
   eca5c:	48 31 d0             	xor    %rdx,%rax
   eca5f:	89 f1                	mov    %esi,%ecx
   eca61:	48 31 c8             	xor    %rcx,%rax
   eca64:	48 f7 e7             	mul    %rdi
   eca67:	48 89 c1             	mov    %rax,%rcx
   eca6a:	48 31 d1             	xor    %rdx,%rcx
   eca6d:	48 89 c8             	mov    %rcx,%rax
   eca70:	48 f7 a4 24 50 01 00 	mulq   0x150(%rsp)
   eca77:	00 
   eca78:	48 31 c2             	xor    %rax,%rdx
   eca7b:	48 d3 c2             	rol    %cl,%rdx
   eca7e:	48 89 d0             	mov    %rdx,%rax
   eca81:	48 c1 e8 39          	shr    $0x39,%rax
   eca85:	66 0f 6e c0          	movd   %eax,%xmm0
   eca89:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
   eca8d:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
   eca92:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
   eca97:	31 c0                	xor    %eax,%eax
   eca99:	48 23 54 24 78       	and    0x78(%rsp),%rdx
   eca9e:	f3 41 0f 6f 0c 12    	movdqu (%r10,%rdx,1),%xmm1
   ecaa4:	66 0f 6f d1          	movdqa %xmm1,%xmm2
   ecaa8:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
   ecaac:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
   ecab0:	85 c9                	test   %ecx,%ecx
   ecab2:	74 2b                	je     ecadf <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10af>
   ecab4:	f3 0f bc f9          	tzcnt  %ecx,%edi
   ecab8:	48 01 d7             	add    %rdx,%rdi
   ecabb:	48 23 7c 24 78       	and    0x78(%rsp),%rdi
   ecac0:	48 f7 df             	neg    %rdi
   ecac3:	48 8d 3c 7f          	lea    (%rdi,%rdi,2),%rdi
   ecac7:	45 3b 4c ba f4       	cmp    -0xc(%r10,%rdi,4),%r9d
   ecacc:	75 07                	jne    ecad5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10a5>
   ecace:	41 3b 74 ba f8       	cmp    -0x8(%r10,%rdi,4),%esi
   ecad3:	74 5e                	je     ecb33 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1103>
   ecad5:	8d 79 ff             	lea    -0x1(%rcx),%edi
   ecad8:	66 21 cf             	and    %cx,%di
   ecadb:	89 f9                	mov    %edi,%ecx
   ecadd:	75 d5                	jne    ecab4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1084>
   ecadf:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
   ecae3:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
   ecae7:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
   ecaeb:	85 c9                	test   %ecx,%ecx
   ecaed:	75 4f                	jne    ecb3e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x110e>
   ecaef:	48 01 c2             	add    %rax,%rdx
   ecaf2:	48 83 c2 10          	add    $0x10,%rdx
   ecaf6:	48 83 c0 10          	add    $0x10,%rax
   ecafa:	eb 9d                	jmp    eca99 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1069>
   ecafc:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
   ecb01:	0f 83 7a 03 00 00    	jae    ece81 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1451>
   ecb07:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
   ecb0c:	0f b7 04 41          	movzwl (%rcx,%rax,2),%eax
   ecb10:	66 83 f8 ff          	cmp    $0xffff,%ax
   ecb14:	0f b7 f0             	movzwl %ax,%esi
   ecb17:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
   ecb1c:	0f 44 f0             	cmove  %eax,%esi
   ecb1f:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
   ecb26:	00 00 
   ecb28:	0f 85 13 ff ff ff    	jne    eca41 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1011>
   ecb2e:	44 89 c9             	mov    %r9d,%ecx
   ecb31:	eb 11                	jmp    ecb44 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1114>
   ecb33:	49 8d 04 ba          	lea    (%r10,%rdi,4),%rax
   ecb37:	8b 78 fc             	mov    -0x4(%rax),%edi
   ecb3a:	89 f9                	mov    %edi,%ecx
   ecb3c:	eb 06                	jmp    ecb44 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1114>
   ecb3e:	44 89 c9             	mov    %r9d,%ecx
   ecb41:	4c 89 c7             	mov    %r8,%rdi
   ecb44:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
   ecb49:	48 8b 72 50          	mov    0x50(%rdx),%rsi
   ecb4d:	48 39 fe             	cmp    %rdi,%rsi
   ecb50:	0f 86 fe 02 00 00    	jbe    ece54 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1424>
   ecb56:	49 29 ec             	sub    %rbp,%r12
   ecb59:	b8 00 00 00 00       	mov    $0x0,%eax
   ecb5e:	49 0f 43 c4          	cmovae %r12,%rax
   ecb62:	48 8b 52 48          	mov    0x48(%rdx),%rdx
   ecb66:	48 03 04 fa          	add    (%rdx,%rdi,8),%rax
   ecb6a:	31 d2                	xor    %edx,%edx
   ecb6c:	48 3b 84 24 88 00 00 	cmp    0x88(%rsp),%rax
   ecb73:	00 
   ecb74:	0f 92 c2             	setb   %dl
   ecb77:	89 14 24             	mov    %edx,(%rsp)
   ecb7a:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
   ecb81:	00 
   ecb82:	be 01 00 00 00       	mov    $0x1,%esi
   ecb87:	44 89 ca             	mov    %r9d,%edx
   ecb8a:	49 89 e8             	mov    %rbp,%r8
   ecb8d:	49 89 d9             	mov    %rbx,%r9
   ecb90:	e8 7b 74 00 00       	call   f4010 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
   ecb95:	48 85 c0             	test   %rax,%rax
   ecb98:	0f 85 22 01 00 00    	jne    eccc0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1290>
   ecb9e:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
   ecba3:	48 83 f8 03          	cmp    $0x3,%rax
   ecba7:	72 26                	jb     ecbcf <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x119f>
   ecba9:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
   ecbae:	48 39 c1             	cmp    %rax,%rcx
   ecbb1:	74 3e                	je     ecbf1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11c1>
   ecbb3:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
   ecbb8:	48 89 2c c8          	mov    %rbp,(%rax,%rcx,8)
   ecbbc:	48 ff c1             	inc    %rcx
   ecbbf:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
   ecbc4:	4d 85 f6             	test   %r14,%r14
   ecbc7:	0f 85 45 f3 ff ff    	jne    ebf12 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
   ecbcd:	eb 4e                	jmp    ecc1d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
   ecbcf:	48 8d 4c 24 58       	lea    0x58(%rsp),%rcx
   ecbd4:	48 8d 54 24 68       	lea    0x68(%rsp),%rdx
   ecbd9:	48 83 f8 02          	cmp    $0x2,%rax
   ecbdd:	74 23                	je     ecc02 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11d2>
   ecbdf:	48 89 2c c1          	mov    %rbp,(%rcx,%rax,8)
   ecbe3:	48 ff 02             	incq   (%rdx)
   ecbe6:	4d 85 f6             	test   %r14,%r14
   ecbe9:	0f 85 23 f3 ff ff    	jne    ebf12 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
   ecbef:	eb 2c                	jmp    ecc1d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
   ecbf1:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
   ecbf6:	e8 c5 78 00 00       	call   f44c0 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsgHLVOFInvdC_8tk_train>
   ecbfb:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
   ecc00:	eb b1                	jmp    ecbb3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1183>
   ecc02:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
   ecc07:	e8 b4 78 00 00       	call   f44c0 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsgHLVOFInvdC_8tk_train>
   ecc0c:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
   ecc11:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
   ecc16:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
   ecc1b:	eb c2                	jmp    ecbdf <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11af>
   ecc1d:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
   ecc23:	74 0e                	je     ecc33 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1203>
   ecc25:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
   ecc2c:	00 
   ecc2d:	ff 15 b5 bf 01 00    	call   *0x1bfb5(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecc33:	4c 8b 74 24 58       	mov    0x58(%rsp),%r14
   ecc38:	0f 10 44 24 60       	movups 0x60(%rsp),%xmm0
   ecc3d:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
   ecc42:	48 8d bc 24 d8 00 00 	lea    0xd8(%rsp),%rdi
   ecc49:	00 
   ecc4a:	48 8d b4 24 98 01 00 	lea    0x198(%rsp),%rsi
   ecc51:	00 
   ecc52:	e8 89 7a 00 00       	call   f46e0 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6finish>
   ecc57:	48 8b 84 24 e8 00 00 	mov    0xe8(%rsp),%rax
   ecc5e:	00 
   ecc5f:	48 8b 8c 24 70 01 00 	mov    0x170(%rsp),%rcx
   ecc66:	00 
   ecc67:	48 89 41 10          	mov    %rax,0x10(%rcx)
   ecc6b:	0f 10 84 24 d8 00 00 	movups 0xd8(%rsp),%xmm0
   ecc72:	00 
   ecc73:	0f 11 01             	movups %xmm0,(%rcx)
   ecc76:	4c 89 71 18          	mov    %r14,0x18(%rcx)
   ecc7a:	66 0f 6f 44 24 30    	movdqa 0x30(%rsp),%xmm0
   ecc80:	f3 0f 7f 41 20       	movdqu %xmm0,0x20(%rcx)
   ecc85:	8b 84 24 d4 00 00 00 	mov    0xd4(%rsp),%eax
   ecc8c:	89 41 30             	mov    %eax,0x30(%rcx)
   ecc8f:	48 81 c4 88 02 00 00 	add    $0x288,%rsp
   ecc96:	5b                   	pop    %rbx
   ecc97:	41 5c                	pop    %r12
   ecc99:	41 5d                	pop    %r13
   ecc9b:	41 5e                	pop    %r14
   ecc9d:	41 5f                	pop    %r15
   ecc9f:	5d                   	pop    %rbp
   ecca0:	c3                   	ret
   ecca1:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
   ecca8:	00 
   ecca9:	48 89 c2             	mov    %rax,%rdx
   eccac:	48 8d 0d 4d b6 01 00 	lea    0x1b64d(%rip),%rcx        # 108300 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x39f0>
   eccb3:	48 89 c6             	mov    %rax,%rsi
   eccb6:	e8 a5 4d f8 ff       	call   71a60 <_RNvNtNtCsgxBkk5gSRhY_4core5slice5index16slice_index_fail>
   eccbb:	e9 82 01 00 00       	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   eccc0:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
   eccc6:	4c 8b b4 24 70 01 00 	mov    0x170(%rsp),%r14
   ecccd:	00 
   eccce:	74 14                	je     ecce4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12b4>
   eccd0:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
   eccd7:	00 
   eccd8:	48 89 c3             	mov    %rax,%rbx
   eccdb:	ff 15 07 bf 01 00    	call   *0x1bf07(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecce1:	48 89 d8             	mov    %rbx,%rax
   ecce4:	49 89 46 08          	mov    %rax,0x8(%r14)
   ecce8:	48 8d 05 09 af 01 00 	lea    0x1af09(%rip),%rax        # 107bf8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x32e8>
   eccef:	49 89 46 10          	mov    %rax,0x10(%r14)
   eccf3:	49 c7 06 ff ff ff ff 	movq   $0xffffffffffffffff,(%r14)
   eccfa:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
   ecd00:	72 0b                	jb     ecd0d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12dd>
   ecd02:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
   ecd07:	ff 15 db be 01 00    	call   *0x1bedb(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecd0d:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
   ecd14:	00 
   ecd15:	e8 d6 5b ff ff       	call   e28f0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
   ecd1a:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
   ecd21:	00 
   ecd22:	e8 c9 5b ff ff       	call   e28f0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
   ecd27:	e9 63 ff ff ff       	jmp    ecc8f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x125f>
   ecd2c:	48 8d 3d 77 12 f3 ff 	lea    -0xced89(%rip),%rdi        # 1dfaa <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x5bd>
   ecd33:	48 8d 15 3e b9 01 00 	lea    0x1b93e(%rip),%rdx        # 108678 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3d68>
   ecd3a:	be 53 00 00 00       	mov    $0x53,%esi
   ecd3f:	e8 4c 2e f8 ff       	call   6fb90 <_RNvNtCsgxBkk5gSRhY_4core9panicking9panic_fmt>
   ecd44:	e9 f9 00 00 00       	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   ecd49:	48 89 d0             	mov    %rdx,%rax
   ecd4c:	48 8d 0d 65 bb 01 00 	lea    0x1bb65(%rip),%rcx        # 1088b8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3fa8>
   ecd53:	e9 5b ff ff ff       	jmp    eccb3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
   ecd58:	48 8d 3d 09 0e f3 ff 	lea    -0xcf1f7(%rip),%rdi        # 1db68 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x17b>
   ecd5f:	48 8d 15 ea b6 01 00 	lea    0x1b6ea(%rip),%rdx        # 108450 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3b40>
   ecd66:	be 11 00 00 00       	mov    $0x11,%esi
   ecd6b:	e8 d0 5e f8 ff       	call   72c40 <_RNvNtCsgxBkk5gSRhY_4core6option13expect_failed>
   ecd70:	e9 cd 00 00 00       	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   ecd75:	b8 08 00 00 00       	mov    $0x8,%eax
   ecd7a:	31 ff                	xor    %edi,%edi
   ecd7c:	48 8d 0d fd b9 01 00 	lea    0x1b9fd(%rip),%rcx        # 108780 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3e70>
   ecd83:	e9 2b ff ff ff       	jmp    eccb3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
   ecd88:	48 89 f7             	mov    %rsi,%rdi
   ecd8b:	e9 19 ff ff ff       	jmp    ecca9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1279>
   ecd90:	48 89 d0             	mov    %rdx,%rax
   ecd93:	48 8d 0d ae b5 01 00 	lea    0x1b5ae(%rip),%rcx        # 108348 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3a38>
   ecd9a:	e9 14 ff ff ff       	jmp    eccb3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
   ecd9f:	bf 08 00 00 00       	mov    $0x8,%edi
   ecda4:	48 8d 0d b5 b5 01 00 	lea    0x1b5b5(%rip),%rcx        # 108360 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3a50>
   ecdab:	e9 03 ff ff ff       	jmp    eccb3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
   ecdb0:	31 c0                	xor    %eax,%eax
   ecdb2:	eb 02                	jmp    ecdb6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1386>
   ecdb4:	b0 01                	mov    $0x1,%al
   ecdb6:	88 84 24 e0 01 00 00 	mov    %al,0x1e0(%rsp)
   ecdbd:	48 8d 3d dc 13 f3 ff 	lea    -0xcec24(%rip),%rdi        # 1e1a0 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x7b3>
   ecdc4:	48 8d 0d a5 b2 01 00 	lea    0x1b2a5(%rip),%rcx        # 108070 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3760>
   ecdcb:	4c 8d 05 ce ba 01 00 	lea    0x1bace(%rip),%r8        # 1088a0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3f90>
   ecdd2:	48 8d 94 24 e0 01 00 	lea    0x1e0(%rsp),%rdx
   ecdd9:	00 
   ecdda:	be 13 00 00 00       	mov    $0x13,%esi
   ecddf:	e8 9c 5e f8 ff       	call   72c80 <_RNvNtCsgxBkk5gSRhY_4core6result13unwrap_failed>
   ecde4:	eb 5c                	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   ecde6:	48 89 ef             	mov    %rbp,%rdi
   ecde9:	48 8d 05 38 b4 01 00 	lea    0x1b438(%rip),%rax        # 108228 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3918>
   ecdf0:	eb 0a                	jmp    ecdfc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
   ecdf2:	48 89 ef             	mov    %rbp,%rdi
   ecdf5:	48 8d 05 44 b4 01 00 	lea    0x1b444(%rip),%rax        # 108240 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3930>
   ecdfc:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
   ece01:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
   ece06:	e8 20 3d f8 ff       	call   70b2b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
   ece0b:	eb 35                	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   ece0d:	49 c1 e9 07          	shr    $0x7,%r9
   ece11:	31 f6                	xor    %esi,%esi
   ece13:	4c 89 cf             	mov    %r9,%rdi
   ece16:	eb e9                	jmp    ece01 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
   ece18:	48 89 d7             	mov    %rdx,%rdi
   ece1b:	eb e4                	jmp    ece01 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
   ece1d:	4c 89 ff             	mov    %r15,%rdi
   ece20:	eb 03                	jmp    ece25 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f5>
   ece22:	4c 89 ef             	mov    %r13,%rdi
   ece25:	48 8d 15 7c ad 01 00 	lea    0x1ad7c(%rip),%rdx        # 107ba8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3298>
   ece2c:	e8 fa 3c f8 ff       	call   70b2b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
   ece31:	eb 0f                	jmp    ece42 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
   ece33:	bf 08 00 00 00       	mov    $0x8,%edi
   ece38:	be 80 00 00 00       	mov    $0x80,%esi
   ece3d:	e8 d0 d0 f7 ff       	call   69f12 <_RNvNtCs6i54tJFfzR_5alloc7raw_vec12handle_error>
   ece42:	0f 0b                	ud2
   ece44:	48 89 cf             	mov    %rcx,%rdi
   ece47:	48 89 c6             	mov    %rax,%rsi
   ece4a:	eb 08                	jmp    ece54 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1424>
   ece4c:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
   ece53:	00 
   ece54:	48 8d 05 4d ad 01 00 	lea    0x1ad4d(%rip),%rax        # 107ba8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3298>
   ece5b:	eb 9f                	jmp    ecdfc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
   ece5d:	48 89 cf             	mov    %rcx,%rdi
   ece60:	eb 93                	jmp    ecdf5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
   ece62:	48 89 cf             	mov    %rcx,%rdi
   ece65:	eb 82                	jmp    ecde9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
   ece67:	4c 89 e7             	mov    %r12,%rdi
   ece6a:	eb 89                	jmp    ecdf5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
   ece6c:	4c 89 e7             	mov    %r12,%rdi
   ece6f:	e9 75 ff ff ff       	jmp    ecde9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
   ece74:	48 89 c7             	mov    %rax,%rdi
   ece77:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
   ece7c:	e9 74 ff ff ff       	jmp    ecdf5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
   ece81:	48 89 c7             	mov    %rax,%rdi
   ece84:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
   ece89:	e9 5b ff ff ff       	jmp    ecde9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
   ece8e:	48 c7 c7 ff ff ff ff 	mov    $0xffffffffffffffff,%rdi
   ece95:	31 f6                	xor    %esi,%esi
   ece97:	48 8d 05 92 b7 01 00 	lea    0x1b792(%rip),%rax        # 108630 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x3d20>
   ece9e:	e9 59 ff ff ff       	jmp    ecdfc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
   ecea3:	eb 59                	jmp    ecefe <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14ce>
   ecea5:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   eceaa:	66 0f 70 44 24 30 ee 	pshufd $0xee,0x30(%rsp),%xmm0
   eceb1:	66 48 0f 7e c0       	movq   %xmm0,%rax
   eceb6:	48 83 f8 02          	cmp    $0x2,%rax
   eceba:	0f 87 b9 00 00 00    	ja     ecf79 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1549>
   ecec0:	e9 bd 00 00 00       	jmp    ecf82 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1552>
   ecec5:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   ececa:	eb 4d                	jmp    ecf19 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
   ececc:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   eced1:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
   eced8:	00 
   eced9:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
   ecede:	eb 23                	jmp    ecf03 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14d3>
   ecee0:	eb 02                	jmp    ecee4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14b4>
   ecee2:	eb 00                	jmp    ecee4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14b4>
   ecee4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   ecee9:	48 83 bc 24 98 00 00 	cmpq   $0x0,0x98(%rsp)
   ecef0:	00 00 
   ecef2:	74 25                	je     ecf19 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
   ecef4:	48 8b bc 24 a0 00 00 	mov    0xa0(%rsp),%rdi
   ecefb:	00 
   ecefc:	eb 15                	jmp    ecf13 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e3>
   ecefe:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
   ecf03:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
   ecf09:	74 0e                	je     ecf19 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
   ecf0b:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
   ecf12:	00 
   ecf13:	ff 15 cf bc 01 00    	call   *0x1bccf(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecf19:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
   ecf1f:	72 0b                	jb     ecf2c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14fc>
   ecf21:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
   ecf26:	ff 15 bc bc 01 00    	call   *0x1bcbc(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecf2c:	4c 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%r14
   ecf33:	00 
   ecf34:	4c 8b a4 24 a8 01 00 	mov    0x1a8(%rsp),%r12
   ecf3b:	00 
   ecf3c:	4d 85 e4             	test   %r12,%r12
   ecf3f:	75 4b                	jne    ecf8c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x155c>
   ecf41:	48 83 bc 24 98 01 00 	cmpq   $0x0,0x198(%rsp)
   ecf48:	00 00 
   ecf4a:	74 09                	je     ecf55 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1525>
   ecf4c:	4c 89 f7             	mov    %r14,%rdi
   ecf4f:	ff 15 93 bc 01 00    	call   *0x1bc93(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecf55:	4c 8b b4 24 b8 01 00 	mov    0x1b8(%rsp),%r14
   ecf5c:	00 
   ecf5d:	4c 8b a4 24 c0 01 00 	mov    0x1c0(%rsp),%r12
   ecf64:	00 
   ecf65:	4d 85 e4             	test   %r12,%r12
   ecf68:	0f 85 ab 00 00 00    	jne    ed019 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15e9>
   ecf6e:	48 83 bc 24 b0 01 00 	cmpq   $0x0,0x1b0(%rsp)
   ecf75:	00 00 
   ecf77:	74 09                	je     ecf82 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1552>
   ecf79:	4c 89 f7             	mov    %r14,%rdi
   ecf7c:	ff 15 66 bc 01 00    	call   *0x1bc66(%rip)        # 108be8 <free@GLIBC_2.2.5>
   ecf82:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
   ecf87:	e8 f4 5d 01 00       	call   102d80 <_Unwind_Resume@plt>
   ecf8c:	45 31 ed             	xor    %r13d,%r13d
   ecf8f:	48 8b 2d 52 bc 01 00 	mov    0x1bc52(%rip),%rbp        # 108be8 <free@GLIBC_2.2.5>
   ecf96:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
   ecf9b:	eb 15                	jmp    ecfb2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1582>
   ecf9d:	0f 1f 00             	nopl   (%rax)
   ecfa0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
   ecfa5:	4c 89 ff             	mov    %r15,%rdi
   ecfa8:	ff d5                	call   *%rbp
   ecfaa:	49 ff c5             	inc    %r13
   ecfad:	4d 39 e5             	cmp    %r12,%r13
   ecfb0:	74 8f                	je     ecf41 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1511>
   ecfb2:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
   ecfb9:	00 
   ecfba:	4c 01 e9             	add    %r13,%rcx
   ecfbd:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
   ecfc1:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
   ecfc6:	74 18                	je     ecfe0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15b0>
   ecfc8:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
   ecfcd:	72 db                	jb     ecfaa <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x157a>
   ecfcf:	4c 8b 78 08          	mov    0x8(%rax),%r15
   ecfd3:	eb d0                	jmp    ecfa5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1575>
   ecfd5:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
   ecfdc:	00 00 00 00 
   ecfe0:	48 8b 58 10          	mov    0x10(%rax),%rbx
   ecfe4:	48 85 db             	test   %rbx,%rbx
   ecfe7:	74 c1                	je     ecfaa <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x157a>
   ecfe9:	4c 8b 78 08          	mov    0x8(%rax),%r15
   ecfed:	4d 8d 67 18          	lea    0x18(%r15),%r12
   ecff1:	eb 16                	jmp    ed009 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d9>
   ecff3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ecffa:	84 00 00 00 00 00 
   ed000:	49 83 c4 20          	add    $0x20,%r12
   ed004:	48 ff cb             	dec    %rbx
   ed007:	74 97                	je     ecfa0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1570>
   ed009:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
   ed00e:	72 f0                	jb     ed000 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d0>
   ed010:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
   ed015:	ff d5                	call   *%rbp
   ed017:	eb e7                	jmp    ed000 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d0>
   ed019:	45 31 ed             	xor    %r13d,%r13d
   ed01c:	48 8b 2d c5 bb 01 00 	mov    0x1bbc5(%rip),%rbp        # 108be8 <free@GLIBC_2.2.5>
   ed023:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
   ed028:	eb 1c                	jmp    ed046 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1616>
   ed02a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
   ed030:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
   ed035:	4c 89 ff             	mov    %r15,%rdi
   ed038:	ff d5                	call   *%rbp
   ed03a:	49 ff c5             	inc    %r13
   ed03d:	4d 39 e5             	cmp    %r12,%r13
   ed040:	0f 84 28 ff ff ff    	je     ecf6e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x153e>
   ed046:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
   ed04d:	00 
   ed04e:	4c 01 e9             	add    %r13,%rcx
   ed051:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
   ed055:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
   ed05a:	74 14                	je     ed070 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1640>
   ed05c:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
   ed061:	72 d7                	jb     ed03a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x160a>
   ed063:	4c 8b 78 08          	mov    0x8(%rax),%r15
   ed067:	eb cc                	jmp    ed035 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1605>
   ed069:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
   ed070:	48 8b 58 10          	mov    0x10(%rax),%rbx
   ed074:	48 85 db             	test   %rbx,%rbx
   ed077:	74 c1                	je     ed03a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x160a>
   ed079:	4c 8b 78 08          	mov    0x8(%rax),%r15
   ed07d:	4d 8d 67 18          	lea    0x18(%r15),%r12
   ed081:	eb 16                	jmp    ed099 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1669>
   ed083:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ed08a:	84 00 00 00 00 00 
   ed090:	49 83 c4 20          	add    $0x20,%r12
   ed094:	48 ff cb             	dec    %rbx
   ed097:	74 97                	je     ed030 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1600>
   ed099:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
   ed09e:	72 f0                	jb     ed090 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1660>
   ed0a0:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
   ed0a5:	ff d5                	call   *%rbp
   ed0a7:	eb e7                	jmp    ed090 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1660>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
