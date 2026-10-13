
/dev/shm/bpe-frequency-20261014/A-full-runner:     file format elf64-x86-64


Disassembly of section .text:

0000000000144df0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_>:
  144df0:	55                   	push   %rbp
  144df1:	41 57                	push   %r15
  144df3:	41 56                	push   %r14
  144df5:	41 55                	push   %r13
  144df7:	41 54                	push   %r12
  144df9:	53                   	push   %rbx
  144dfa:	48 81 ec 88 02 00 00 	sub    $0x288,%rsp
  144e01:	49 89 d1             	mov    %rdx,%r9
  144e04:	48 89 fb             	mov    %rdi,%rbx
  144e07:	44 8b 39             	mov    (%rcx),%r15d
  144e0a:	44 8b 69 04          	mov    0x4(%rcx),%r13d
  144e0e:	8b 69 08             	mov    0x8(%rcx),%ebp
  144e11:	0f 10 41 60          	movups 0x60(%rcx),%xmm0
  144e15:	0f 29 84 24 70 02 00 	movaps %xmm0,0x270(%rsp)
  144e1c:	00 
  144e1d:	0f 10 41 50          	movups 0x50(%rcx),%xmm0
  144e21:	0f 29 84 24 60 02 00 	movaps %xmm0,0x260(%rsp)
  144e28:	00 
  144e29:	f3 0f 6f 41 10       	movdqu 0x10(%rcx),%xmm0
  144e2e:	f3 0f 6f 49 20       	movdqu 0x20(%rcx),%xmm1
  144e33:	f3 0f 6f 51 30       	movdqu 0x30(%rcx),%xmm2
  144e38:	0f 10 59 40          	movups 0x40(%rcx),%xmm3
  144e3c:	0f 29 9c 24 50 02 00 	movaps %xmm3,0x250(%rsp)
  144e43:	00 
  144e44:	66 0f 7f 94 24 40 02 	movdqa %xmm2,0x240(%rsp)
  144e4b:	00 00 
  144e4d:	66 0f 7f 8c 24 30 02 	movdqa %xmm1,0x230(%rsp)
  144e54:	00 00 
  144e56:	66 0f 7f 84 24 20 02 	movdqa %xmm0,0x220(%rsp)
  144e5d:	00 00 
  144e5f:	4c 8b 26             	mov    (%rsi),%r12
  144e62:	48 8b 46 08          	mov    0x8(%rsi),%rax
  144e66:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
  144e6d:	00 
  144e6e:	48 8b 46 10          	mov    0x10(%rsi),%rax
  144e72:	48 89 84 24 b8 00 00 	mov    %rax,0xb8(%rsp)
  144e79:	00 
  144e7a:	4d 8b 34 24          	mov    (%r12),%r14
  144e7e:	4d 8b 46 50          	mov    0x50(%r14),%r8
  144e82:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
  144e89:	00 
  144e8a:	44 89 fe             	mov    %r15d,%esi
  144e8d:	44 89 ea             	mov    %r13d,%edx
  144e90:	89 e9                	mov    %ebp,%ecx
  144e92:	e8 d9 81 00 00       	call   14d070 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges3new>
  144e97:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
  144e9e:	00 00 
  144ea0:	49 8b 76 50          	mov    0x50(%r14),%rsi
  144ea4:	4c 39 fe             	cmp    %r15,%rsi
  144ea7:	0f 86 30 13 00 00    	jbe    1461dd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13ed>
  144ead:	4c 39 ee             	cmp    %r13,%rsi
  144eb0:	0f 86 2c 13 00 00    	jbe    1461e2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f2>
  144eb6:	4c 89 a4 24 80 00 00 	mov    %r12,0x80(%rsp)
  144ebd:	00 
  144ebe:	89 ac 24 d4 00 00 00 	mov    %ebp,0xd4(%rsp)
  144ec5:	48 89 9c 24 70 01 00 	mov    %rbx,0x170(%rsp)
  144ecc:	00 
  144ecd:	49 8b 46 48          	mov    0x48(%r14),%rax
  144ed1:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
  144ed6:	4a 8b 0c f8          	mov    (%rax,%r15,8),%rcx
  144eda:	48 89 8c 24 c8 00 00 	mov    %rcx,0xc8(%rsp)
  144ee1:	00 
  144ee2:	4c 89 ac 24 90 00 00 	mov    %r13,0x90(%rsp)
  144ee9:	00 
  144eea:	4a 8b 04 e8          	mov    (%rax,%r13,8),%rax
  144eee:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
  144ef5:	00 
  144ef6:	48 8d 84 24 20 02 00 	lea    0x220(%rsp),%rax
  144efd:	00 
  144efe:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
  144f05:	00 
  144f06:	4c 89 b4 24 88 01 00 	mov    %r14,0x188(%rsp)
  144f0d:	00 
  144f0e:	4c 89 74 24 28       	mov    %r14,0x28(%rsp)
  144f13:	4c 89 b4 24 90 01 00 	mov    %r14,0x190(%rsp)
  144f1a:	00 
  144f1b:	bf 80 00 00 00       	mov    $0x80,%edi
  144f20:	ff 15 6a ec 01 00    	call   *0x1ec6a(%rip)        # 163b90 <malloc@GLIBC_2.2.5>
  144f26:	48 85 c0             	test   %rax,%rax
  144f29:	0f 84 c4 12 00 00    	je     1461f3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1403>
  144f2f:	48 89 c5             	mov    %rax,%rbp
  144f32:	66 0f ef c0          	pxor   %xmm0,%xmm0
  144f36:	f3 0f 7f 84 24 a8 00 	movdqu %xmm0,0xa8(%rsp)
  144f3d:	00 00 
  144f3f:	48 c7 84 24 98 00 00 	movq   $0x10,0x98(%rsp)
  144f46:	00 10 00 00 00 
  144f4b:	48 89 84 24 a0 00 00 	mov    %rax,0xa0(%rsp)
  144f52:	00 
  144f53:	48 8d 84 24 80 01 00 	lea    0x180(%rsp),%rax
  144f5a:	00 
  144f5b:	48 89 84 24 d8 00 00 	mov    %rax,0xd8(%rsp)
  144f62:	00 
  144f63:	48 c7 84 24 e0 00 00 	movq   $0x10,0xe0(%rsp)
  144f6a:	00 10 00 00 00 
  144f6f:	41 bf 10 00 00 00    	mov    $0x10,%r15d
  144f75:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  144f7c:	00 00 
  144f7e:	4c 8d a4 24 d8 00 00 	lea    0xd8(%rsp),%r12
  144f85:	00 
  144f86:	31 db                	xor    %ebx,%ebx
  144f88:	4c 89 e7             	mov    %r12,%rdi
  144f8b:	e8 c0 a6 00 00       	call   14f650 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
  144f90:	48 83 f8 01          	cmp    $0x1,%rax
  144f94:	0f 85 b1 01 00 00    	jne    14514b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x35b>
  144f9a:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  144f9e:	0f 84 74 11 00 00    	je     146118 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1328>
  144fa4:	49 89 d5             	mov    %rdx,%r13
  144fa7:	4c 89 f8             	mov    %r15,%rax
  144faa:	48 29 d8             	sub    %rbx,%rax
  144fad:	76 11                	jbe    144fc0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1d0>
  144faf:	49 89 de             	mov    %rbx,%r14
  144fb2:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  144fb7:	e9 ff 00 00 00       	jmp    1450bb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  144fbc:	0f 1f 40 00          	nopl   0x0(%rax)
  144fc0:	0f 84 99 00 00 00    	je     14505f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x26f>
  144fc6:	4c 89 fe             	mov    %r15,%rsi
  144fc9:	49 89 de             	mov    %rbx,%r14
  144fcc:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  144fd1:	48 39 c3             	cmp    %rax,%rbx
  144fd4:	0f 86 d6 00 00 00    	jbe    1450b0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2c0>
  144fda:	4c 89 fa             	mov    %r15,%rdx
  144fdd:	48 29 da             	sub    %rbx,%rdx
  144fe0:	4c 89 f0             	mov    %r14,%rax
  144fe3:	48 29 d0             	sub    %rdx,%rax
  144fe6:	48 39 c2             	cmp    %rax,%rdx
  144fe9:	76 36                	jbe    145021 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
  144feb:	48 89 f1             	mov    %rsi,%rcx
  144fee:	4c 29 f9             	sub    %r15,%rcx
  144ff1:	48 39 c1             	cmp    %rax,%rcx
  144ff4:	72 2b                	jb     145021 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
  144ff6:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
  144ffd:	00 
  144ffe:	4a 8d 3c fd 00 00 00 	lea    0x0(,%r15,8),%rdi
  145005:	00 
  145006:	48 01 ef             	add    %rbp,%rdi
  145009:	48 c1 e0 03          	shl    $0x3,%rax
  14500d:	49 89 f7             	mov    %rsi,%r15
  145010:	48 89 ee             	mov    %rbp,%rsi
  145013:	48 89 c2             	mov    %rax,%rdx
  145016:	ff 15 64 eb 01 00    	call   *0x1eb64(%rip)        # 163b80 <memcpy@GLIBC_2.14>
  14501c:	e9 9a 00 00 00       	jmp    1450bb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  145021:	48 89 d8             	mov    %rbx,%rax
  145024:	48 89 f3             	mov    %rsi,%rbx
  145027:	48 29 d3             	sub    %rdx,%rbx
  14502a:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
  145031:	00 
  145032:	49 89 f7             	mov    %rsi,%r15
  145035:	48 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%rsi
  14503c:	00 
  14503d:	48 01 ee             	add    %rbp,%rsi
  145040:	48 8d 3c dd 00 00 00 	lea    0x0(,%rbx,8),%rdi
  145047:	00 
  145048:	48 01 ef             	add    %rbp,%rdi
  14504b:	48 c1 e2 03          	shl    $0x3,%rdx
  14504f:	ff 15 33 eb 01 00    	call   *0x1eb33(%rip)        # 163b88 <memmove@GLIBC_2.2.5>
  145055:	48 89 9c 24 a8 00 00 	mov    %rbx,0xa8(%rsp)
  14505c:	00 
  14505d:	eb 5c                	jmp    1450bb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  14505f:	ba 01 00 00 00       	mov    $0x1,%edx
  145064:	b9 08 00 00 00       	mov    $0x8,%ecx
  145069:	41 b8 08 00 00 00    	mov    $0x8,%r8d
  14506f:	48 8d bc 24 98 00 00 	lea    0x98(%rsp),%rdi
  145076:	00 
  145077:	48 89 de             	mov    %rbx,%rsi
  14507a:	e8 51 22 fe ff       	call   1272d0 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsgHLVOFInvdC_8tk_train>
  14507f:	48 8b b4 24 98 00 00 	mov    0x98(%rsp),%rsi
  145086:	00 
  145087:	48 8b 9c 24 a8 00 00 	mov    0xa8(%rsp),%rbx
  14508e:	00 
  14508f:	4c 8b b4 24 b0 00 00 	mov    0xb0(%rsp),%r14
  145096:	00 
  145097:	4c 89 f8             	mov    %r15,%rax
  14509a:	4c 29 f0             	sub    %r14,%rax
  14509d:	48 39 c3             	cmp    %rax,%rbx
  1450a0:	0f 87 34 ff ff ff    	ja     144fda <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1ea>
  1450a6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  1450ad:	00 00 00 
  1450b0:	48 8b ac 24 a0 00 00 	mov    0xa0(%rsp),%rbp
  1450b7:	00 
  1450b8:	49 89 f7             	mov    %rsi,%r15
  1450bb:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  1450c0:	49 8d 04 1e          	lea    (%r14,%rbx,1),%rax
  1450c4:	4c 39 f8             	cmp    %r15,%rax
  1450c7:	b9 00 00 00 00       	mov    $0x0,%ecx
  1450cc:	49 0f 43 cf          	cmovae %r15,%rcx
  1450d0:	48 29 c8             	sub    %rcx,%rax
  1450d3:	4c 89 6c c5 00       	mov    %r13,0x0(%rbp,%rax,8)
  1450d8:	49 8d 5e 01          	lea    0x1(%r14),%rbx
  1450dc:	48 89 9c 24 b0 00 00 	mov    %rbx,0xb0(%rsp)
  1450e3:	00 
  1450e4:	4c 39 fb             	cmp    %r15,%rbx
  1450e7:	0f 83 9b fe ff ff    	jae    144f88 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
  1450ed:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  1450f2:	49 01 c6             	add    %rax,%r14
  1450f5:	49 ff c6             	inc    %r14
  1450f8:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  1450ff:	00 
  145100:	4c 89 e7             	mov    %r12,%rdi
  145103:	e8 48 a5 00 00       	call   14f650 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
  145108:	48 83 f8 01          	cmp    $0x1,%rax
  14510c:	75 35                	jne    145143 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x353>
  14510e:	4d 39 fe             	cmp    %r15,%r14
  145111:	b8 00 00 00 00       	mov    $0x0,%eax
  145116:	49 0f 43 c7          	cmovae %r15,%rax
  14511a:	48 c1 e0 03          	shl    $0x3,%rax
  14511e:	48 89 e9             	mov    %rbp,%rcx
  145121:	48 29 c1             	sub    %rax,%rcx
  145124:	4a 89 14 f1          	mov    %rdx,(%rcx,%r14,8)
  145128:	48 ff c3             	inc    %rbx
  14512b:	49 ff c6             	inc    %r14
  14512e:	49 39 df             	cmp    %rbx,%r15
  145131:	75 cd                	jne    145100 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x310>
  145133:	4c 89 bc 24 b0 00 00 	mov    %r15,0xb0(%rsp)
  14513a:	00 
  14513b:	4c 89 fb             	mov    %r15,%rbx
  14513e:	e9 45 fe ff ff       	jmp    144f88 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
  145143:	48 89 9c 24 b0 00 00 	mov    %rbx,0xb0(%rsp)
  14514a:	00 
  14514b:	48 8b 8c 24 c8 00 00 	mov    0xc8(%rsp),%rcx
  145152:	00 
  145153:	48 8b 94 24 c0 00 00 	mov    0xc0(%rsp),%rdx
  14515a:	00 
  14515b:	48 01 ca             	add    %rcx,%rdx
  14515e:	f3 0f 6f 84 24 98 00 	movdqu 0x98(%rsp),%xmm0
  145165:	00 00 
  145167:	f3 0f 6f 8c 24 a8 00 	movdqu 0xa8(%rsp),%xmm1
  14516e:	00 00 
  145170:	66 0f 7f 84 24 e0 01 	movdqa %xmm0,0x1e0(%rsp)
  145177:	00 00 
  145179:	66 0f 7f 8c 24 f0 01 	movdqa %xmm1,0x1f0(%rsp)
  145180:	00 00 
  145182:	f3 0f 6f 94 24 80 01 	movdqu 0x180(%rsp),%xmm2
  145189:	00 00 
  14518b:	66 0f 7f 94 24 00 02 	movdqa %xmm2,0x200(%rsp)
  145192:	00 00 
  145194:	48 8b 84 24 90 01 00 	mov    0x190(%rsp),%rax
  14519b:	00 
  14519c:	48 89 84 24 10 02 00 	mov    %rax,0x210(%rsp)
  1451a3:	00 
  1451a4:	48 89 84 24 48 01 00 	mov    %rax,0x148(%rsp)
  1451ab:	00 
  1451ac:	f3 0f 7f 94 24 38 01 	movdqu %xmm2,0x138(%rsp)
  1451b3:	00 00 
  1451b5:	f3 0f 7f 8c 24 28 01 	movdqu %xmm1,0x128(%rsp)
  1451bc:	00 00 
  1451be:	f3 0f 7f 84 24 18 01 	movdqu %xmm0,0x118(%rsp)
  1451c5:	00 00 
  1451c7:	48 c7 84 24 d8 00 00 	movq   $0x0,0xd8(%rsp)
  1451ce:	00 00 00 00 00 
  1451d3:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  1451d8:	48 89 84 24 f0 00 00 	mov    %rax,0xf0(%rsp)
  1451df:	00 
  1451e0:	48 89 8c 24 f8 00 00 	mov    %rcx,0xf8(%rsp)
  1451e7:	00 
  1451e8:	48 89 94 24 00 01 00 	mov    %rdx,0x100(%rsp)
  1451ef:	00 
  1451f0:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  1451f5:	89 8c 24 08 01 00 00 	mov    %ecx,0x108(%rsp)
  1451fc:	48 8b 8c 24 90 00 00 	mov    0x90(%rsp),%rcx
  145203:	00 
  145204:	89 8c 24 0c 01 00 00 	mov    %ecx,0x10c(%rsp)
  14520b:	48 89 84 24 10 01 00 	mov    %rax,0x110(%rsp)
  145212:	00 
  145213:	4c 8b b4 24 30 01 00 	mov    0x130(%rsp),%r14
  14521a:	00 
  14521b:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
  145222:	00 
  145223:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  145228:	4d 85 f6             	test   %r14,%r14
  14522b:	0f 84 ac 0d 00 00    	je     145fdd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
  145231:	4c 8b ac 24 28 01 00 	mov    0x128(%rsp),%r13
  145238:	00 
  145239:	48 8b 8c 24 b8 00 00 	mov    0xb8(%rsp),%rcx
  145240:	00 
  145241:	0f b6 01             	movzbl (%rcx),%eax
  145244:	88 44 24 17          	mov    %al,0x17(%rsp)
  145248:	48 8b 41 10          	mov    0x10(%rcx),%rax
  14524c:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  145251:	48 8b 41 08          	mov    0x8(%rcx),%rax
  145255:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  14525a:	48 8b 94 24 88 00 00 	mov    0x88(%rsp),%rdx
  145261:	00 
  145262:	48 8b 42 18          	mov    0x18(%rdx),%rax
  145266:	48 89 84 24 60 01 00 	mov    %rax,0x160(%rsp)
  14526d:	00 
  14526e:	48 8b 42 28          	mov    0x28(%rdx),%rax
  145272:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  145279:	00 
  14527a:	48 8b 42 20          	mov    0x20(%rdx),%rax
  14527e:	48 89 84 24 50 01 00 	mov    %rax,0x150(%rsp)
  145285:	00 
  145286:	48 8b 02             	mov    (%rdx),%rax
  145289:	48 89 84 24 68 01 00 	mov    %rax,0x168(%rsp)
  145290:	00 
  145291:	48 8b 42 08          	mov    0x8(%rdx),%rax
  145295:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  14529a:	48 8b 84 24 80 00 00 	mov    0x80(%rsp),%rax
  1452a1:	00 
  1452a2:	48 8b 40 08          	mov    0x8(%rax),%rax
  1452a6:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
  1452ad:	00 
  1452ae:	48 8b 41 50          	mov    0x50(%rcx),%rax
  1452b2:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  1452b9:	00 
  1452ba:	48 8b 41 48          	mov    0x48(%rcx),%rax
  1452be:	48 89 84 24 78 01 00 	mov    %rax,0x178(%rsp)
  1452c5:	00 
  1452c6:	48 8d 05 e3 df 01 00 	lea    0x1dfe3(%rip),%rax        # 1632b0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55d0>
  1452cd:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  1452d2:	4c 8b 8c 24 f0 00 00 	mov    0xf0(%rsp),%r9
  1452d9:	00 
  1452da:	48 8b 84 24 f8 00 00 	mov    0xf8(%rsp),%rax
  1452e1:	00 
  1452e2:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  1452e9:	00 
  1452ea:	8b 84 24 08 01 00 00 	mov    0x108(%rsp),%eax
  1452f1:	89 84 24 c0 00 00 00 	mov    %eax,0xc0(%rsp)
  1452f8:	48 8b 84 24 00 01 00 	mov    0x100(%rsp),%rax
  1452ff:	00 
  145300:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  145305:	8b 84 24 0c 01 00 00 	mov    0x10c(%rsp),%eax
  14530c:	89 84 24 b8 00 00 00 	mov    %eax,0xb8(%rsp)
  145313:	48 8b 5c 24 50       	mov    0x50(%rsp),%rbx
  145318:	4c 89 8c 24 c8 00 00 	mov    %r9,0xc8(%rsp)
  14531f:	00 
  145320:	eb 17                	jmp    145339 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x549>
  145322:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145329:	1f 84 00 00 00 00 00 
  145330:	4d 85 f6             	test   %r14,%r14
  145333:	0f 84 a4 0c 00 00    	je     145fdd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
  145339:	4c 89 e8             	mov    %r13,%rax
  14533c:	49 8d 4d 01          	lea    0x1(%r13),%rcx
  145340:	48 39 d9             	cmp    %rbx,%rcx
  145343:	48 89 d9             	mov    %rbx,%rcx
  145346:	ba 00 00 00 00       	mov    $0x0,%edx
  14534b:	48 0f 42 ca          	cmovb  %rdx,%rcx
  14534f:	48 f7 d9             	neg    %rcx
  145352:	4c 01 e9             	add    %r13,%rcx
  145355:	48 ff c1             	inc    %rcx
  145358:	49 89 cd             	mov    %rcx,%r13
  14535b:	48 89 8c 24 28 01 00 	mov    %rcx,0x128(%rsp)
  145362:	00 
  145363:	4d 89 f7             	mov    %r14,%r15
  145366:	49 ff cf             	dec    %r15
  145369:	4c 89 bc 24 30 01 00 	mov    %r15,0x130(%rsp)
  145370:	00 
  145371:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
  145378:	00 
  145379:	4c 8b b4 24 38 01 00 	mov    0x138(%rsp),%r14
  145380:	00 
  145381:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  145385:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  14538a:	49 8b 46 58          	mov    0x58(%r14),%rax
  14538e:	bd 00 00 00 00       	mov    $0x0,%ebp
  145393:	48 85 c0             	test   %rax,%rax
  145396:	0f 84 c9 03 00 00    	je     145765 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x975>
  14539c:	48 ff c8             	dec    %rax
  14539f:	49 89 46 58          	mov    %rax,0x58(%r14)
  1453a3:	49 8d 46 08          	lea    0x8(%r14),%rax
  1453a7:	4d 8d 46 10          	lea    0x10(%r14),%r8
  1453ab:	49 8b 36             	mov    (%r14),%rsi
  1453ae:	48 85 f6             	test   %rsi,%rsi
  1453b1:	74 3a                	je     1453ed <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5fd>
  1453b3:	48 8b 10             	mov    (%rax),%rdx
  1453b6:	48 85 d2             	test   %rdx,%rdx
  1453b9:	74 2b                	je     1453e6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5f6>
  1453bb:	4d 8b 4e 18          	mov    0x18(%r14),%r9
  1453bf:	41 f6 c1 7f          	test   $0x7f,%r9b
  1453c3:	0f 84 22 02 00 00    	je     1455eb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
  1453c9:	4c 0f be 26          	movsbq (%rsi),%r12
  1453cd:	4d 85 e4             	test   %r12,%r12
  1453d0:	0f 88 d0 00 00 00    	js     1454a6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6b6>
  1453d6:	4d 03 20             	add    (%r8),%r12
  1453d9:	4d 89 20             	mov    %r12,(%r8)
  1453dc:	bf 01 00 00 00       	mov    $0x1,%edi
  1453e1:	e9 3b 01 00 00       	jmp    145521 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x731>
  1453e6:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  1453ed:	49 8b 6e 40          	mov    0x40(%r14),%rbp
  1453f1:	48 85 ed             	test   %rbp,%rbp
  1453f4:	0f 84 01 04 00 00    	je     1457fb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
  1453fa:	4d 8b 5e 48          	mov    0x48(%r14),%r11
  1453fe:	4c 39 dd             	cmp    %r11,%rbp
  145401:	0f 84 f4 03 00 00    	je     1457fb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
  145407:	4c 8d 65 20          	lea    0x20(%rbp),%r12
  14540b:	4d 89 66 40          	mov    %r12,0x40(%r14)
  14540f:	4d 8b 4e 50          	mov    0x50(%r14),%r9
  145413:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
  145417:	48 ff c9             	dec    %rcx
  14541a:	49 39 c9             	cmp    %rcx,%r9
  14541d:	75 18                	jne    145437 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x647>
  14541f:	49 c7 06 01 00 00 00 	movq   $0x1,(%r14)
  145426:	66 0f ef c0          	pxor   %xmm0,%xmm0
  14542a:	f3 0f 7f 00          	movdqu %xmm0,(%rax)
  14542e:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  145432:	e9 4c 02 00 00       	jmp    145683 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
  145437:	49 89 ca             	mov    %rcx,%r10
  14543a:	49 c1 ea 07          	shr    $0x7,%r10
  14543e:	4d 85 c9             	test   %r9,%r9
  145441:	0f 84 e2 00 00 00    	je     145529 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x739>
  145447:	89 c8                	mov    %ecx,%eax
  145449:	83 e0 7f             	and    $0x7f,%eax
  14544c:	48 83 f8 01          	cmp    $0x1,%rax
  145450:	4c 89 d6             	mov    %r10,%rsi
  145453:	48 83 de ff          	sbb    $0xffffffffffffffff,%rsi
  145457:	48 83 fe 02          	cmp    $0x2,%rsi
  14545b:	0f 82 6c 0d 00 00    	jb     1461cd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13dd>
  145461:	48 8b 7d 18          	mov    0x18(%rbp),%rdi
  145465:	48 89 fa             	mov    %rdi,%rdx
  145468:	48 83 ff 11          	cmp    $0x11,%rdi
  14546c:	72 04                	jb     145472 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x682>
  14546e:	48 8b 55 10          	mov    0x10(%rbp),%rdx
  145472:	48 8d 04 f5 08 00 00 	lea    0x8(,%rsi,8),%rax
  145479:	00 
  14547a:	48 39 d0             	cmp    %rdx,%rax
  14547d:	0f 87 dc 0c 00 00    	ja     14615f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x136f>
  145483:	4c 89 ca             	mov    %r9,%rdx
  145486:	48 c1 ea 07          	shr    $0x7,%rdx
  14548a:	48 39 f2             	cmp    %rsi,%rdx
  14548d:	0f 83 45 0d 00 00    	jae    1461d8 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13e8>
  145493:	48 83 ff 11          	cmp    $0x11,%rdi
  145497:	0f 82 90 00 00 00    	jb     14552d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x73d>
  14549d:	48 8b 45 08          	mov    0x8(%rbp),%rax
  1454a1:	e9 8b 00 00 00       	jmp    145531 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x741>
  1454a6:	48 83 fa 01          	cmp    $0x1,%rdx
  1454aa:	0f 84 c0 0c 00 00    	je     146170 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1380>
  1454b0:	41 83 e4 7f          	and    $0x7f,%r12d
  1454b4:	4c 8d 52 ff          	lea    -0x1(%rdx),%r10
  1454b8:	b9 07 00 00 00       	mov    $0x7,%ecx
  1454bd:	31 ff                	xor    %edi,%edi
  1454bf:	4c 89 e0             	mov    %r12,%rax
  1454c2:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1454c9:	1f 84 00 00 00 00 00 
  1454d0:	44 0f b6 5c 3e 01    	movzbl 0x1(%rsi,%rdi,1),%r11d
  1454d6:	83 f9 3f             	cmp    $0x3f,%ecx
  1454d9:	75 0a                	jne    1454e5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6f5>
  1454db:	41 80 fb 01          	cmp    $0x1,%r11b
  1454df:	0f 87 8f 0c 00 00    	ja     146174 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
  1454e5:	45 0f b6 e3          	movzbl %r11b,%r12d
  1454e9:	45 84 e4             	test   %r12b,%r12b
  1454ec:	79 1a                	jns    145508 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x718>
  1454ee:	41 83 e4 7f          	and    $0x7f,%r12d
  1454f2:	49 d3 e4             	shl    %cl,%r12
  1454f5:	4c 09 e0             	or     %r12,%rax
  1454f8:	83 c1 07             	add    $0x7,%ecx
  1454fb:	48 ff c7             	inc    %rdi
  1454fe:	49 39 fa             	cmp    %rdi,%r10
  145501:	75 cd                	jne    1454d0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6e0>
  145503:	e9 68 0c 00 00       	jmp    146170 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1380>
  145508:	49 d3 e4             	shl    %cl,%r12
  14550b:	49 09 c4             	or     %rax,%r12
  14550e:	48 83 c7 02          	add    $0x2,%rdi
  145512:	4d 03 20             	add    (%r8),%r12
  145515:	4d 89 20             	mov    %r12,(%r8)
  145518:	48 39 d7             	cmp    %rdx,%rdi
  14551b:	0f 87 e8 0b 00 00    	ja     146109 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1319>
  145521:	48 29 fa             	sub    %rdi,%rdx
  145524:	e9 db 00 00 00       	jmp    145604 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x814>
  145529:	31 ff                	xor    %edi,%edi
  14552b:	eb 09                	jmp    145536 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x746>
  14552d:	48 8d 45 08          	lea    0x8(%rbp),%rax
  145531:	48 8b 7c d0 08       	mov    0x8(%rax,%rdx,8),%rdi
  145536:	83 e1 7f             	and    $0x7f,%ecx
  145539:	48 83 f9 01          	cmp    $0x1,%rcx
  14553d:	49 83 da ff          	sbb    $0xffffffffffffffff,%r10
  145541:	be 08 00 00 00       	mov    $0x8,%esi
  145546:	49 83 fa 02          	cmp    $0x2,%r10
  14554a:	73 1e                	jae    14556a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x77a>
  14554c:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
  145550:	48 89 c8             	mov    %rcx,%rax
  145553:	48 83 f9 11          	cmp    $0x11,%rcx
  145557:	72 39                	jb     145592 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7a2>
  145559:	48 8b 45 10          	mov    0x10(%rbp),%rax
  14555d:	48 89 c2             	mov    %rax,%rdx
  145560:	48 29 f2             	sub    %rsi,%rdx
  145563:	73 39                	jae    14559e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7ae>
  145565:	e9 de 0b 00 00       	jmp    146148 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1358>
  14556a:	49 ff ca             	dec    %r10
  14556d:	49 0f bd ca          	bsr    %r10,%rcx
  145571:	f7 d1                	not    %ecx
  145573:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  14557a:	48 d3 e8             	shr    %cl,%rax
  14557d:	48 8d 34 c5 10 00 00 	lea    0x10(,%rax,8),%rsi
  145584:	00 
  145585:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
  145589:	48 89 c8             	mov    %rcx,%rax
  14558c:	48 83 f9 11          	cmp    $0x11,%rcx
  145590:	73 c7                	jae    145559 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x769>
  145592:	48 89 c2             	mov    %rax,%rdx
  145595:	48 29 f2             	sub    %rsi,%rdx
  145598:	0f 82 aa 0b 00 00    	jb     146148 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1358>
  14559e:	48 39 fa             	cmp    %rdi,%rdx
  1455a1:	0f 82 a9 0b 00 00    	jb     146150 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1360>
  1455a7:	48 83 f9 11          	cmp    $0x11,%rcx
  1455ab:	72 06                	jb     1455b3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c3>
  1455ad:	48 8b 6d 08          	mov    0x8(%rbp),%rbp
  1455b1:	eb 04                	jmp    1455b7 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c7>
  1455b3:	48 83 c5 08          	add    $0x8,%rbp
  1455b7:	48 01 fe             	add    %rdi,%rsi
  1455ba:	48 01 ee             	add    %rbp,%rsi
  1455bd:	48 29 fa             	sub    %rdi,%rdx
  1455c0:	49 89 36             	mov    %rsi,(%r14)
  1455c3:	49 89 56 08          	mov    %rdx,0x8(%r14)
  1455c7:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  1455ce:	00 
  1455cf:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  1455d3:	0f 84 aa 00 00 00    	je     145683 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
  1455d9:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  1455e0:	00 
  1455e1:	41 f6 c1 7f          	test   $0x7f,%r9b
  1455e5:	0f 85 de fd ff ff    	jne    1453c9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
  1455eb:	48 83 fa 07          	cmp    $0x7,%rdx
  1455ef:	0f 86 40 0b 00 00    	jbe    146135 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1345>
  1455f5:	4c 8b 26             	mov    (%rsi),%r12
  1455f8:	4d 89 20             	mov    %r12,(%r8)
  1455fb:	48 83 c2 f8          	add    $0xfffffffffffffff8,%rdx
  1455ff:	bf 08 00 00 00       	mov    $0x8,%edi
  145604:	48 01 fe             	add    %rdi,%rsi
  145607:	49 89 36             	mov    %rsi,(%r14)
  14560a:	49 89 56 08          	mov    %rdx,0x8(%r14)
  14560e:	49 ff c1             	inc    %r9
  145611:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  145615:	48 8b 84 24 40 01 00 	mov    0x140(%rsp),%rax
  14561c:	00 
  14561d:	4c 3b 60 10          	cmp    0x10(%rax),%r12
  145621:	0f 83 c5 0a 00 00    	jae    1460ec <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12fc>
  145627:	48 8b 84 24 48 01 00 	mov    0x148(%rsp),%rax
  14562e:	00 
  14562f:	48 8b 48 10          	mov    0x10(%rax),%rcx
  145633:	83 38 01             	cmpl   $0x1,(%rax)
  145636:	0f 85 14 01 00 00    	jne    145750 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x960>
  14563c:	49 39 cc             	cmp    %rcx,%r12
  14563f:	0f 83 1b 01 00 00    	jae    145760 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
  145645:	4a 8d 0c a5 00 00 00 	lea    0x0(,%r12,4),%rcx
  14564c:	00 
  14564d:	e9 07 01 00 00       	jmp    145759 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x969>
  145652:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145659:	1f 84 00 00 00 00 00 
  145660:	be 01 00 00 00       	mov    $0x1,%esi
  145665:	31 d2                	xor    %edx,%edx
  145667:	49 89 36             	mov    %rsi,(%r14)
  14566a:	49 89 56 08          	mov    %rdx,0x8(%r14)
  14566e:	66 0f ef c0          	pxor   %xmm0,%xmm0
  145672:	f3 41 0f 7f 00       	movdqu %xmm0,(%r8)
  145677:	4d 89 cc             	mov    %r9,%r12
  14567a:	48 85 d2             	test   %rdx,%rdx
  14567d:	0f 85 36 03 00 00    	jne    1459b9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbc9>
  145683:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  14568a:	4d 39 dc             	cmp    %r11,%r12
  14568d:	0f 84 60 01 00 00    	je     1457f3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa03>
  145693:	4d 8d 4c 24 20       	lea    0x20(%r12),%r9
  145698:	4d 89 4e 40          	mov    %r9,0x40(%r14)
  14569c:	49 8b 04 24          	mov    (%r12),%rax
  1456a0:	48 ff c8             	dec    %rax
  1456a3:	74 bb                	je     145660 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x870>
  1456a5:	89 c1                	mov    %eax,%ecx
  1456a7:	83 e1 7f             	and    $0x7f,%ecx
  1456aa:	48 c1 e8 07          	shr    $0x7,%rax
  1456ae:	48 83 f9 01          	cmp    $0x1,%rcx
  1456b2:	48 83 d8 ff          	sbb    $0xffffffffffffffff,%rax
  1456b6:	bf 08 00 00 00       	mov    $0x8,%edi
  1456bb:	48 83 f8 02          	cmp    $0x2,%rax
  1456bf:	73 2f                	jae    1456f0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x900>
  1456c1:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
  1456c6:	48 89 c8             	mov    %rcx,%rax
  1456c9:	48 83 f9 11          	cmp    $0x11,%rcx
  1456cd:	72 4a                	jb     145719 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x929>
  1456cf:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  1456d4:	48 89 c2             	mov    %rax,%rdx
  1456d7:	48 29 fa             	sub    %rdi,%rdx
  1456da:	73 49                	jae    145725 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x935>
  1456dc:	e9 80 09 00 00       	jmp    146061 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1271>
  1456e1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1456e8:	0f 1f 84 00 00 00 00 
  1456ef:	00 
  1456f0:	48 ff c8             	dec    %rax
  1456f3:	48 0f bd c8          	bsr    %rax,%rcx
  1456f7:	f7 d1                	not    %ecx
  1456f9:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  145700:	48 d3 e8             	shr    %cl,%rax
  145703:	48 8d 3c c5 10 00 00 	lea    0x10(,%rax,8),%rdi
  14570a:	00 
  14570b:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
  145710:	48 89 c8             	mov    %rcx,%rax
  145713:	48 83 f9 11          	cmp    $0x11,%rcx
  145717:	73 b6                	jae    1456cf <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x8df>
  145719:	48 89 c2             	mov    %rax,%rdx
  14571c:	48 29 fa             	sub    %rdi,%rdx
  14571f:	0f 82 3c 09 00 00    	jb     146061 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1271>
  145725:	48 83 f9 11          	cmp    $0x11,%rcx
  145729:	72 15                	jb     145740 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x950>
  14572b:	49 8b 74 24 08       	mov    0x8(%r12),%rsi
  145730:	eb 15                	jmp    145747 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x957>
  145732:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145739:	1f 84 00 00 00 00 00 
  145740:	49 83 c4 08          	add    $0x8,%r12
  145744:	4c 89 e6             	mov    %r12,%rsi
  145747:	48 01 fe             	add    %rdi,%rsi
  14574a:	e9 18 ff ff ff       	jmp    145667 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x877>
  14574f:	90                   	nop
  145750:	49 39 cc             	cmp    %rcx,%r12
  145753:	73 0b                	jae    145760 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
  145755:	4b 8d 0c 24          	lea    (%r12,%r12,1),%rcx
  145759:	48 03 48 08          	add    0x8(%rax),%rcx
  14575d:	0f 18 09             	prefetcht0 (%rcx)
  145760:	bd 01 00 00 00       	mov    $0x1,%ebp
  145765:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
  145769:	48 39 d8             	cmp    %rbx,%rax
  14576c:	0f 86 ce 00 00 00    	jbe    145840 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa50>
  145772:	48 89 d8             	mov    %rbx,%rax
  145775:	4c 29 f8             	sub    %r15,%rax
  145778:	48 39 c5             	cmp    %rax,%rbp
  14577b:	0f 87 52 02 00 00    	ja     1459d3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbe3>
  145781:	4d 89 ee             	mov    %r13,%r14
  145784:	49 89 dd             	mov    %rbx,%r13
  145787:	49 39 c6             	cmp    %rax,%r14
  14578a:	76 62                	jbe    1457ee <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  14578c:	48 89 da             	mov    %rbx,%rdx
  14578f:	4c 29 f2             	sub    %r14,%rdx
  145792:	4c 89 f8             	mov    %r15,%rax
  145795:	48 29 d0             	sub    %rdx,%rax
  145798:	48 39 c2             	cmp    %rax,%rdx
  14579b:	76 26                	jbe    1457c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
  14579d:	4c 89 e9             	mov    %r13,%rcx
  1457a0:	48 29 d9             	sub    %rbx,%rcx
  1457a3:	48 39 c1             	cmp    %rax,%rcx
  1457a6:	72 1b                	jb     1457c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
  1457a8:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
  1457af:	00 
  1457b0:	48 8d 3c de          	lea    (%rsi,%rbx,8),%rdi
  1457b4:	48 c1 e0 03          	shl    $0x3,%rax
  1457b8:	48 89 c2             	mov    %rax,%rdx
  1457bb:	ff 15 bf e3 01 00    	call   *0x1e3bf(%rip)        # 163b80 <memcpy@GLIBC_2.14>
  1457c1:	eb 2b                	jmp    1457ee <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  1457c3:	4c 89 f1             	mov    %r14,%rcx
  1457c6:	4d 89 ee             	mov    %r13,%r14
  1457c9:	49 29 d6             	sub    %rdx,%r14
  1457cc:	48 8b 84 24 20 01 00 	mov    0x120(%rsp),%rax
  1457d3:	00 
  1457d4:	48 8d 34 c8          	lea    (%rax,%rcx,8),%rsi
  1457d8:	4a 8d 3c f0          	lea    (%rax,%r14,8),%rdi
  1457dc:	48 c1 e2 03          	shl    $0x3,%rdx
  1457e0:	ff 15 a2 e3 01 00    	call   *0x1e3a2(%rip)        # 163b88 <memmove@GLIBC_2.2.5>
  1457e6:	4c 89 b4 24 28 01 00 	mov    %r14,0x128(%rsp)
  1457ed:	00 
  1457ee:	4c 89 eb             	mov    %r13,%rbx
  1457f1:	eb 50                	jmp    145843 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa53>
  1457f3:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  1457fa:	00 
  1457fb:	49 83 7e 20 00       	cmpq   $0x0,0x20(%r14)
  145800:	74 20                	je     145822 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa32>
  145802:	49 83 c6 20          	add    $0x20,%r14
  145806:	4c 89 f7             	mov    %r14,%rdi
  145809:	e8 72 d1 ff ff       	call   142980 <_RNvXs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe9positions5codecNtB4_6CursorNtNtNtNtCsgxBkk5gSRhY_4core4iter6traits8iterator8Iterator4next>
  14580e:	49 89 d4             	mov    %rdx,%r12
  145811:	48 83 f8 01          	cmp    $0x1,%rax
  145815:	0f 84 fa fd ff ff    	je     145615 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x825>
  14581b:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  145822:	31 ed                	xor    %ebp,%ebp
  145824:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
  145828:	48 39 d8             	cmp    %rbx,%rax
  14582b:	0f 87 41 ff ff ff    	ja     145772 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x982>
  145831:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145838:	0f 1f 84 00 00 00 00 
  14583f:	00 
  145840:	4d 89 ee             	mov    %r13,%r14
  145843:	4b 8d 04 37          	lea    (%r15,%r14,1),%rax
  145847:	48 39 d8             	cmp    %rbx,%rax
  14584a:	b9 00 00 00 00       	mov    $0x0,%ecx
  14584f:	48 0f 43 cb          	cmovae %rbx,%rcx
  145853:	48 29 c8             	sub    %rcx,%rax
  145856:	48 89 d9             	mov    %rbx,%rcx
  145859:	48 29 c1             	sub    %rax,%rcx
  14585c:	4d 89 f5             	mov    %r14,%r13
  14585f:	48 39 e9             	cmp    %rbp,%rcx
  145862:	73 2c                	jae    145890 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xaa0>
  145864:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
  14586b:	00 
  14586c:	48 39 c3             	cmp    %rax,%rbx
  14586f:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
  145874:	75 3a                	jne    1458b0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xac0>
  145876:	4c 89 21             	mov    %r12,(%rcx)
  145879:	41 be 01 00 00 00    	mov    $0x1,%r14d
  14587f:	eb 47                	jmp    1458c8 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
  145881:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145888:	0f 1f 84 00 00 00 00 
  14588f:	00 
  145890:	48 83 fd 01          	cmp    $0x1,%rbp
  145894:	75 2a                	jne    1458c0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad0>
  145896:	48 8b 8c 24 20 01 00 	mov    0x120(%rsp),%rcx
  14589d:	00 
  14589e:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
  1458a2:	41 be 01 00 00 00    	mov    $0x1,%r14d
  1458a8:	eb 19                	jmp    1458c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad3>
  1458aa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  1458b0:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
  1458b4:	41 be 01 00 00 00    	mov    $0x1,%r14d
  1458ba:	eb 0c                	jmp    1458c8 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
  1458bc:	0f 1f 40 00          	nopl   0x0(%rax)
  1458c0:	45 31 f6             	xor    %r14d,%r14d
  1458c3:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
  1458c8:	4d 01 fe             	add    %r15,%r14
  1458cb:	4c 89 b4 24 30 01 00 	mov    %r14,0x130(%rsp)
  1458d2:	00 
  1458d3:	4c 8b 8c 24 c8 00 00 	mov    0xc8(%rsp),%r9
  1458da:	00 
  1458db:	49 8b 01             	mov    (%r9),%rax
  1458de:	49 8b 71 10          	mov    0x10(%r9),%rsi
  1458e2:	48 83 f8 01          	cmp    $0x1,%rax
  1458e6:	75 18                	jne    145900 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb10>
  1458e8:	48 39 f5             	cmp    %rsi,%rbp
  1458eb:	0f 83 c1 08 00 00    	jae    1461b2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c2>
  1458f1:	49 8b 49 08          	mov    0x8(%r9),%rcx
  1458f5:	8b 14 a9             	mov    (%rcx,%rbp,4),%edx
  1458f8:	eb 26                	jmp    145920 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb30>
  1458fa:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  145900:	48 39 f5             	cmp    %rsi,%rbp
  145903:	0f 83 9d 08 00 00    	jae    1461a6 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b6>
  145909:	49 8b 49 08          	mov    0x8(%r9),%rcx
  14590d:	0f b7 14 69          	movzwl (%rcx,%rbp,2),%edx
  145911:	66 83 fa ff          	cmp    $0xffff,%dx
  145915:	0f b7 d2             	movzwl %dx,%edx
  145918:	bf ff ff ff ff       	mov    $0xffffffff,%edi
  14591d:	0f 44 d7             	cmove  %edi,%edx
  145920:	4c 8b 84 24 90 00 00 	mov    0x90(%rsp),%r8
  145927:	00 
  145928:	4c 39 44 24 20       	cmp    %r8,0x20(%rsp)
  14592d:	40 0f 94 c7          	sete   %dil
  145931:	4d 85 c0             	test   %r8,%r8
  145934:	41 0f 94 c0          	sete   %r8b
  145938:	3b 94 24 c0 00 00 00 	cmp    0xc0(%rsp),%edx
  14593f:	0f 95 c2             	setne  %dl
  145942:	41 08 f8             	or     %dil,%r8b
  145945:	41 08 d0             	or     %dl,%r8b
  145948:	0f 85 e2 f9 ff ff    	jne    145330 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  14594e:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  145953:	4c 8d 24 2a          	lea    (%rdx,%rbp,1),%r12
  145957:	49 39 f4             	cmp    %rsi,%r12
  14595a:	0f 83 d0 f9 ff ff    	jae    145330 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  145960:	48 8b 94 24 90 00 00 	mov    0x90(%rsp),%rdx
  145967:	00 
  145968:	48 8d 3c 2a          	lea    (%rdx,%rbp,1),%rdi
  14596c:	84 c0                	test   %al,%al
  14596e:	74 1e                	je     14598e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb9e>
  145970:	48 39 f7             	cmp    %rsi,%rdi
  145973:	0f 83 3c 08 00 00    	jae    1461b5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
  145979:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
  14597c:	3b 84 24 b8 00 00 00 	cmp    0xb8(%rsp),%eax
  145983:	0f 85 a7 f9 ff ff    	jne    145330 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  145989:	e9 96 00 00 00       	jmp    145a24 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
  14598e:	48 39 f7             	cmp    %rsi,%rdi
  145991:	0f 83 12 08 00 00    	jae    1461a9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
  145997:	0f b7 04 79          	movzwl (%rcx,%rdi,2),%eax
  14599b:	66 83 f8 ff          	cmp    $0xffff,%ax
  14599f:	0f b7 c0             	movzwl %ax,%eax
  1459a2:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
  1459a7:	0f 44 c1             	cmove  %ecx,%eax
  1459aa:	3b 84 24 b8 00 00 00 	cmp    0xb8(%rsp),%eax
  1459b1:	0f 85 79 f9 ff ff    	jne    145330 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  1459b7:	eb 6b                	jmp    145a24 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
  1459b9:	45 31 c9             	xor    %r9d,%r9d
  1459bc:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  1459c3:	00 
  1459c4:	41 f6 c1 7f          	test   $0x7f,%r9b
  1459c8:	0f 85 fb f9 ff ff    	jne    1453c9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
  1459ce:	e9 18 fc ff ff       	jmp    1455eb <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
  1459d3:	ba 01 00 00 00       	mov    $0x1,%edx
  1459d8:	b9 08 00 00 00       	mov    $0x8,%ecx
  1459dd:	41 b8 08 00 00 00    	mov    $0x8,%r8d
  1459e3:	48 8d bc 24 18 01 00 	lea    0x118(%rsp),%rdi
  1459ea:	00 
  1459eb:	4c 89 fe             	mov    %r15,%rsi
  1459ee:	e8 dd 18 fe ff       	call   1272d0 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsgHLVOFInvdC_8tk_train>
  1459f3:	4c 8b ac 24 18 01 00 	mov    0x118(%rsp),%r13
  1459fa:	00 
  1459fb:	4c 8b b4 24 28 01 00 	mov    0x128(%rsp),%r14
  145a02:	00 
  145a03:	4c 8b bc 24 30 01 00 	mov    0x130(%rsp),%r15
  145a0a:	00 
  145a0b:	48 89 d8             	mov    %rbx,%rax
  145a0e:	4c 29 f8             	sub    %r15,%rax
  145a11:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
  145a16:	49 39 c6             	cmp    %rax,%r14
  145a19:	0f 87 6d fd ff ff    	ja     14578c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x99c>
  145a1f:	e9 ca fd ff ff       	jmp    1457ee <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  145a24:	83 bc 24 d8 00 00 00 	cmpl   $0x1,0xd8(%rsp)
  145a2b:	01 
  145a2c:	75 22                	jne    145a50 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc60>
  145a2e:	48 8b 94 24 e8 00 00 	mov    0xe8(%rsp),%rdx
  145a35:	00 
  145a36:	48 39 d5             	cmp    %rdx,%rbp
  145a39:	73 15                	jae    145a50 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc60>
  145a3b:	48 8b 9c 24 e0 00 00 	mov    0xe0(%rsp),%rbx
  145a42:	00 
  145a43:	4c 8b 94 24 68 01 00 	mov    0x168(%rsp),%r10
  145a4a:	00 
  145a4b:	e9 bd 00 00 00       	jmp    145b0d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
  145a50:	48 8b 84 24 10 01 00 	mov    0x110(%rsp),%rax
  145a57:	00 
  145a58:	48 8b 48 68          	mov    0x68(%rax),%rcx
  145a5c:	48 8b 70 70          	mov    0x70(%rax),%rsi
  145a60:	48 83 fe 01          	cmp    $0x1,%rsi
  145a64:	4c 8b 94 24 68 01 00 	mov    0x168(%rsp),%r10
  145a6b:	00 
  145a6c:	75 0d                	jne    145a7b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc8b>
  145a6e:	48 8b 59 08          	mov    0x8(%rcx),%rbx
  145a72:	48 8b 50 10          	mov    0x10(%rax),%rdx
  145a76:	e9 92 00 00 00       	jmp    145b0d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
  145a7b:	83 78 18 01          	cmpl   $0x1,0x18(%rax)
  145a7f:	75 14                	jne    145a95 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xca5>
  145a81:	48 3b 68 20          	cmp    0x20(%rax),%rbp
  145a85:	72 0e                	jb     145a95 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xca5>
  145a87:	48 8b 50 28          	mov    0x28(%rax),%rdx
  145a8b:	bb 01 00 00 00       	mov    $0x1,%ebx
  145a90:	48 39 d5             	cmp    %rdx,%rbp
  145a93:	72 78                	jb     145b0d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd1d>
  145a95:	48 85 f6             	test   %rsi,%rsi
  145a98:	0f 84 b0 07 00 00    	je     14624e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x145e>
  145a9e:	4d 89 d3             	mov    %r10,%r11
  145aa1:	48 89 f7             	mov    %rsi,%rdi
  145aa4:	31 d2                	xor    %edx,%edx
  145aa6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  145aad:	00 00 00 
  145ab0:	49 89 d0             	mov    %rdx,%r8
  145ab3:	49 89 f9             	mov    %rdi,%r9
  145ab6:	49 d1 e9             	shr    $1,%r9
  145ab9:	4c 01 ca             	add    %r9,%rdx
  145abc:	49 89 d2             	mov    %rdx,%r10
  145abf:	49 c1 e2 04          	shl    $0x4,%r10
  145ac3:	4a 39 2c 11          	cmp    %rbp,(%rcx,%r10,1)
  145ac7:	49 0f 47 d0          	cmova  %r8,%rdx
  145acb:	4c 29 cf             	sub    %r9,%rdi
  145ace:	48 83 ff 01          	cmp    $0x1,%rdi
  145ad2:	77 dc                	ja     145ab0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xcc0>
  145ad4:	48 89 d7             	mov    %rdx,%rdi
  145ad7:	48 c1 e7 04          	shl    $0x4,%rdi
  145adb:	48 3b 2c 39          	cmp    (%rcx,%rdi,1),%rbp
  145adf:	48 83 da ff          	sbb    $0xffffffffffffffff,%rdx
  145ae3:	48 8d 7a ff          	lea    -0x1(%rdx),%rdi
  145ae7:	48 39 f7             	cmp    %rsi,%rdi
  145aea:	0f 83 67 07 00 00    	jae    146257 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1467>
  145af0:	48 c1 e7 04          	shl    $0x4,%rdi
  145af4:	48 8b 5c 39 08       	mov    0x8(%rcx,%rdi,1),%rbx
  145af9:	48 39 f2             	cmp    %rsi,%rdx
  145afc:	4d 89 da             	mov    %r11,%r10
  145aff:	0f 83 6d ff ff ff    	jae    145a72 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc82>
  145b05:	48 c1 e2 04          	shl    $0x4,%rdx
  145b09:	48 8b 14 11          	mov    (%rcx,%rdx,1),%rdx
  145b0d:	48 c7 84 24 d8 00 00 	movq   $0x1,0xd8(%rsp)
  145b14:	00 01 00 00 00 
  145b19:	48 89 9c 24 e0 00 00 	mov    %rbx,0xe0(%rsp)
  145b20:	00 
  145b21:	48 89 94 24 e8 00 00 	mov    %rdx,0xe8(%rsp)
  145b28:	00 
  145b29:	48 8d 4d ff          	lea    -0x1(%rbp),%rcx
  145b2d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145b32:	48 8b 38             	mov    (%rax),%rdi
  145b35:	48 8b 70 10          	mov    0x10(%rax),%rsi
  145b39:	48 83 ff 01          	cmp    $0x1,%rdi
  145b3d:	75 37                	jne    145b76 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd86>
  145b3f:	48 39 f1             	cmp    %rsi,%rcx
  145b42:	0f 83 d5 06 00 00    	jae    14621d <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x142d>
  145b48:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145b4d:	48 8b 42 08          	mov    0x8(%rdx),%rax
  145b51:	8b 4c a8 fc          	mov    -0x4(%rax,%rbp,4),%ecx
  145b55:	83 f9 ff             	cmp    $0xffffffff,%ecx
  145b58:	75 41                	jne    145b9b <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdab>
  145b5a:	49 39 f4             	cmp    %rsi,%r12
  145b5d:	0f 83 c4 06 00 00    	jae    146227 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1437>
  145b63:	46 8b 0c a0          	mov    (%rax,%r12,4),%r9d
  145b67:	41 83 f9 ff          	cmp    $0xffffffff,%r9d
  145b6b:	0f 85 43 02 00 00    	jne    145db4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc4>
  145b71:	e9 e8 03 00 00       	jmp    145f5e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
  145b76:	48 39 f1             	cmp    %rsi,%rcx
  145b79:	0f 83 a3 06 00 00    	jae    146222 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1432>
  145b7f:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145b84:	48 8b 40 08          	mov    0x8(%rax),%rax
  145b88:	0f b7 4c 68 fe       	movzwl -0x2(%rax,%rbp,2),%ecx
  145b8d:	66 83 f9 ff          	cmp    $0xffff,%cx
  145b91:	74 50                	je     145be3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdf3>
  145b93:	0f b7 c9             	movzwl %cx,%ecx
  145b96:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145b9b:	89 c9                	mov    %ecx,%ecx
  145b9d:	48 8b 42 50          	mov    0x50(%rdx),%rax
  145ba1:	48 39 c8             	cmp    %rcx,%rax
  145ba4:	0f 86 5a 06 00 00    	jbe    146204 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1414>
  145baa:	48 8b 42 48          	mov    0x48(%rdx),%rax
  145bae:	49 89 cb             	mov    %rcx,%r11
  145bb1:	4c 8b 0c c8          	mov    (%rax,%rcx,8),%r9
  145bb5:	49 89 e8             	mov    %rbp,%r8
  145bb8:	4d 29 c8             	sub    %r9,%r8
  145bbb:	0f 84 53 01 00 00    	je     145d14 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
  145bc1:	49 8d 40 ff          	lea    -0x1(%r8),%rax
  145bc5:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
  145bca:	74 34                	je     145c00 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe10>
  145bcc:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145bd1:	0f 83 5d 06 00 00    	jae    146234 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1444>
  145bd7:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  145bdc:	46 8b 7c 80 fc       	mov    -0x4(%rax,%r8,4),%r15d
  145be1:	eb 44                	jmp    145c27 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe37>
  145be3:	49 39 f4             	cmp    %rsi,%r12
  145be6:	0f 83 40 06 00 00    	jae    14622c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x143c>
  145bec:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
  145bf1:	66 83 f8 ff          	cmp    $0xffff,%ax
  145bf5:	0f 85 b5 01 00 00    	jne    145db0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc0>
  145bfb:	e9 5e 03 00 00       	jmp    145f5e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
  145c00:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145c05:	0f 83 36 06 00 00    	jae    146241 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1451>
  145c0b:	48 8b 44 24 70       	mov    0x70(%rsp),%rax
  145c10:	42 0f b7 44 40 fe    	movzwl -0x2(%rax,%r8,2),%eax
  145c16:	66 83 f8 ff          	cmp    $0xffff,%ax
  145c1a:	44 0f b7 f8          	movzwl %ax,%r15d
  145c1e:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  145c23:	44 0f 44 f8          	cmove  %eax,%r15d
  145c27:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
  145c2e:	00 00 
  145c30:	0f 84 de 00 00 00    	je     145d14 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
  145c36:	44 89 f8             	mov    %r15d,%eax
  145c39:	48 33 84 24 58 01 00 	xor    0x158(%rsp),%rax
  145c40:	00 
  145c41:	48 b9 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rcx
  145c48:	f4 51 58 
  145c4b:	48 f7 e1             	mul    %rcx
  145c4e:	4c 31 da             	xor    %r11,%rdx
  145c51:	48 31 d0             	xor    %rdx,%rax
  145c54:	48 f7 e1             	mul    %rcx
  145c57:	48 89 c1             	mov    %rax,%rcx
  145c5a:	48 31 d1             	xor    %rdx,%rcx
  145c5d:	48 89 c8             	mov    %rcx,%rax
  145c60:	48 f7 a4 24 50 01 00 	mulq   0x150(%rsp)
  145c67:	00 
  145c68:	48 31 c2             	xor    %rax,%rdx
  145c6b:	48 d3 c2             	rol    %cl,%rdx
  145c6e:	48 89 d0             	mov    %rdx,%rax
  145c71:	48 c1 e8 39          	shr    $0x39,%rax
  145c75:	66 0f 6e c0          	movd   %eax,%xmm0
  145c79:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
  145c7d:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
  145c82:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
  145c87:	31 c0                	xor    %eax,%eax
  145c89:	48 23 54 24 78       	and    0x78(%rsp),%rdx
  145c8e:	f3 41 0f 6f 0c 12    	movdqu (%r10,%rdx,1),%xmm1
  145c94:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  145c98:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
  145c9c:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
  145ca0:	85 c9                	test   %ecx,%ecx
  145ca2:	74 50                	je     145cf4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf04>
  145ca4:	44 89 7c 24 30       	mov    %r15d,0x30(%rsp)
  145ca9:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  145cae:	44 89 f8             	mov    %r15d,%eax
  145cb1:	4d 89 d7             	mov    %r10,%r15
  145cb4:	f3 44 0f bc d1       	tzcnt  %ecx,%r10d
  145cb9:	49 01 d2             	add    %rdx,%r10
  145cbc:	4c 23 54 24 78       	and    0x78(%rsp),%r10
  145cc1:	49 f7 da             	neg    %r10
  145cc4:	4f 8d 14 52          	lea    (%r10,%r10,2),%r10
  145cc8:	43 3b 44 97 f4       	cmp    -0xc(%r15,%r10,4),%eax
  145ccd:	75 0b                	jne    145cda <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xeea>
  145ccf:	47 3b 5c 97 f8       	cmp    -0x8(%r15,%r10,4),%r11d
  145cd4:	0f 84 8c 00 00 00    	je     145d66 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf76>
  145cda:	44 8d 51 ff          	lea    -0x1(%rcx),%r10d
  145cde:	66 41 21 ca          	and    %cx,%r10w
  145ce2:	44 89 d1             	mov    %r10d,%ecx
  145ce5:	4d 89 fa             	mov    %r15,%r10
  145ce8:	44 8b 7c 24 30       	mov    0x30(%rsp),%r15d
  145ced:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  145cf2:	75 ba                	jne    145cae <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xebe>
  145cf4:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
  145cf8:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
  145cfc:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
  145d00:	85 c9                	test   %ecx,%ecx
  145d02:	75 10                	jne    145d14 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf24>
  145d04:	48 01 c2             	add    %rax,%rdx
  145d07:	48 83 c2 10          	add    $0x10,%rdx
  145d0b:	48 83 c0 10          	add    $0x10,%rax
  145d0f:	e9 75 ff ff ff       	jmp    145c89 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe99>
  145d14:	4d 89 d7             	mov    %r10,%r15
  145d17:	4c 89 e0             	mov    %r12,%rax
  145d1a:	48 29 e8             	sub    %rbp,%rax
  145d1d:	b9 00 00 00 00       	mov    $0x0,%ecx
  145d22:	48 0f 43 c8          	cmovae %rax,%rcx
  145d26:	49 01 c9             	add    %rcx,%r9
  145d29:	31 c0                	xor    %eax,%eax
  145d2b:	4c 3b 8c 24 88 00 00 	cmp    0x88(%rsp),%r9
  145d32:	00 
  145d33:	0f 92 c0             	setb   %al
  145d36:	89 04 24             	mov    %eax,(%rsp)
  145d39:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
  145d40:	00 
  145d41:	31 f6                	xor    %esi,%esi
  145d43:	44 89 da             	mov    %r11d,%edx
  145d46:	44 89 d9             	mov    %r11d,%ecx
  145d49:	49 89 d9             	mov    %rbx,%r9
  145d4c:	e8 7f 76 00 00       	call   14d3d0 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
  145d51:	48 85 c0             	test   %rax,%rax
  145d54:	0f 85 26 03 00 00    	jne    146080 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1290>
  145d5a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d5f:	48 8b 38             	mov    (%rax),%rdi
  145d62:	48 8b 70 10          	mov    0x10(%rax),%rsi
  145d66:	48 85 ff             	test   %rdi,%rdi
  145d69:	4d 89 fa             	mov    %r15,%r10
  145d6c:	74 21                	je     145d8f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf9f>
  145d6e:	49 39 f4             	cmp    %rsi,%r12
  145d71:	0f 83 b0 04 00 00    	jae    146227 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1437>
  145d77:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d7c:	48 8b 40 08          	mov    0x8(%rax),%rax
  145d80:	46 8b 0c a0          	mov    (%rax,%r12,4),%r9d
  145d84:	41 83 f9 ff          	cmp    $0xffffffff,%r9d
  145d88:	75 2a                	jne    145db4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfc4>
  145d8a:	e9 cf 01 00 00       	jmp    145f5e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
  145d8f:	49 39 f4             	cmp    %rsi,%r12
  145d92:	0f 83 94 04 00 00    	jae    14622c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x143c>
  145d98:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d9d:	48 8b 40 08          	mov    0x8(%rax),%rax
  145da1:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
  145da6:	66 83 f8 ff          	cmp    $0xffff,%ax
  145daa:	0f 84 ae 01 00 00    	je     145f5e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
  145db0:	44 0f b7 c8          	movzwl %ax,%r9d
  145db4:	44 89 cf             	mov    %r9d,%edi
  145db7:	48 39 bc 24 80 00 00 	cmp    %rdi,0x80(%rsp)
  145dbe:	00 
  145dbf:	0f 86 47 04 00 00    	jbe    14620c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x141c>
  145dc5:	48 8b 84 24 78 01 00 	mov    0x178(%rsp),%rax
  145dcc:	00 
  145dcd:	48 8b 04 f8          	mov    (%rax,%rdi,8),%rax
  145dd1:	4c 01 e0             	add    %r12,%rax
  145dd4:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
  145dd9:	0f 84 dd 00 00 00    	je     145ebc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10cc>
  145ddf:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145de4:	0f 83 4a 04 00 00    	jae    146234 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1444>
  145dea:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
  145def:	8b 34 81             	mov    (%rcx,%rax,4),%esi
  145df2:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
  145df9:	00 00 
  145dfb:	0f 84 ed 00 00 00    	je     145eee <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10fe>
  145e01:	48 8b 84 24 58 01 00 	mov    0x158(%rsp),%rax
  145e08:	00 
  145e09:	49 89 f8             	mov    %rdi,%r8
  145e0c:	48 31 f8             	xor    %rdi,%rax
  145e0f:	48 bf 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rdi
  145e16:	f4 51 58 
  145e19:	48 f7 e7             	mul    %rdi
  145e1c:	48 31 d0             	xor    %rdx,%rax
  145e1f:	89 f1                	mov    %esi,%ecx
  145e21:	48 31 c8             	xor    %rcx,%rax
  145e24:	48 f7 e7             	mul    %rdi
  145e27:	48 89 c1             	mov    %rax,%rcx
  145e2a:	48 31 d1             	xor    %rdx,%rcx
  145e2d:	48 89 c8             	mov    %rcx,%rax
  145e30:	48 f7 a4 24 50 01 00 	mulq   0x150(%rsp)
  145e37:	00 
  145e38:	48 31 c2             	xor    %rax,%rdx
  145e3b:	48 d3 c2             	rol    %cl,%rdx
  145e3e:	48 89 d0             	mov    %rdx,%rax
  145e41:	48 c1 e8 39          	shr    $0x39,%rax
  145e45:	66 0f 6e c0          	movd   %eax,%xmm0
  145e49:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
  145e4d:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
  145e52:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
  145e57:	31 c0                	xor    %eax,%eax
  145e59:	48 23 54 24 78       	and    0x78(%rsp),%rdx
  145e5e:	f3 41 0f 6f 0c 12    	movdqu (%r10,%rdx,1),%xmm1
  145e64:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  145e68:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
  145e6c:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
  145e70:	85 c9                	test   %ecx,%ecx
  145e72:	74 2b                	je     145e9f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10af>
  145e74:	f3 0f bc f9          	tzcnt  %ecx,%edi
  145e78:	48 01 d7             	add    %rdx,%rdi
  145e7b:	48 23 7c 24 78       	and    0x78(%rsp),%rdi
  145e80:	48 f7 df             	neg    %rdi
  145e83:	48 8d 3c 7f          	lea    (%rdi,%rdi,2),%rdi
  145e87:	45 3b 4c ba f4       	cmp    -0xc(%r10,%rdi,4),%r9d
  145e8c:	75 07                	jne    145e95 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10a5>
  145e8e:	41 3b 74 ba f8       	cmp    -0x8(%r10,%rdi,4),%esi
  145e93:	74 5e                	je     145ef3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1103>
  145e95:	8d 79 ff             	lea    -0x1(%rcx),%edi
  145e98:	66 21 cf             	and    %cx,%di
  145e9b:	89 f9                	mov    %edi,%ecx
  145e9d:	75 d5                	jne    145e74 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1084>
  145e9f:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
  145ea3:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
  145ea7:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
  145eab:	85 c9                	test   %ecx,%ecx
  145ead:	75 4f                	jne    145efe <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x110e>
  145eaf:	48 01 c2             	add    %rax,%rdx
  145eb2:	48 83 c2 10          	add    $0x10,%rdx
  145eb6:	48 83 c0 10          	add    $0x10,%rax
  145eba:	eb 9d                	jmp    145e59 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1069>
  145ebc:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145ec1:	0f 83 7a 03 00 00    	jae    146241 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1451>
  145ec7:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
  145ecc:	0f b7 04 41          	movzwl (%rcx,%rax,2),%eax
  145ed0:	66 83 f8 ff          	cmp    $0xffff,%ax
  145ed4:	0f b7 f0             	movzwl %ax,%esi
  145ed7:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  145edc:	0f 44 f0             	cmove  %eax,%esi
  145edf:	48 83 bc 24 60 01 00 	cmpq   $0x0,0x160(%rsp)
  145ee6:	00 00 
  145ee8:	0f 85 13 ff ff ff    	jne    145e01 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1011>
  145eee:	44 89 c9             	mov    %r9d,%ecx
  145ef1:	eb 11                	jmp    145f04 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1114>
  145ef3:	49 8d 04 ba          	lea    (%r10,%rdi,4),%rax
  145ef7:	8b 78 fc             	mov    -0x4(%rax),%edi
  145efa:	89 f9                	mov    %edi,%ecx
  145efc:	eb 06                	jmp    145f04 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1114>
  145efe:	44 89 c9             	mov    %r9d,%ecx
  145f01:	4c 89 c7             	mov    %r8,%rdi
  145f04:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145f09:	48 8b 72 50          	mov    0x50(%rdx),%rsi
  145f0d:	48 39 fe             	cmp    %rdi,%rsi
  145f10:	0f 86 fe 02 00 00    	jbe    146214 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1424>
  145f16:	49 29 ec             	sub    %rbp,%r12
  145f19:	b8 00 00 00 00       	mov    $0x0,%eax
  145f1e:	49 0f 43 c4          	cmovae %r12,%rax
  145f22:	48 8b 52 48          	mov    0x48(%rdx),%rdx
  145f26:	48 03 04 fa          	add    (%rdx,%rdi,8),%rax
  145f2a:	31 d2                	xor    %edx,%edx
  145f2c:	48 3b 84 24 88 00 00 	cmp    0x88(%rsp),%rax
  145f33:	00 
  145f34:	0f 92 c2             	setb   %dl
  145f37:	89 14 24             	mov    %edx,(%rsp)
  145f3a:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
  145f41:	00 
  145f42:	be 01 00 00 00       	mov    $0x1,%esi
  145f47:	44 89 ca             	mov    %r9d,%edx
  145f4a:	49 89 e8             	mov    %rbp,%r8
  145f4d:	49 89 d9             	mov    %rbx,%r9
  145f50:	e8 7b 74 00 00       	call   14d3d0 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
  145f55:	48 85 c0             	test   %rax,%rax
  145f58:	0f 85 22 01 00 00    	jne    146080 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1290>
  145f5e:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  145f63:	48 83 f8 03          	cmp    $0x3,%rax
  145f67:	72 26                	jb     145f8f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x119f>
  145f69:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  145f6e:	48 39 c1             	cmp    %rax,%rcx
  145f71:	74 3e                	je     145fb1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11c1>
  145f73:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  145f78:	48 89 2c c8          	mov    %rbp,(%rax,%rcx,8)
  145f7c:	48 ff c1             	inc    %rcx
  145f7f:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  145f84:	4d 85 f6             	test   %r14,%r14
  145f87:	0f 85 45 f3 ff ff    	jne    1452d2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
  145f8d:	eb 4e                	jmp    145fdd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
  145f8f:	48 8d 4c 24 58       	lea    0x58(%rsp),%rcx
  145f94:	48 8d 54 24 68       	lea    0x68(%rsp),%rdx
  145f99:	48 83 f8 02          	cmp    $0x2,%rax
  145f9d:	74 23                	je     145fc2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11d2>
  145f9f:	48 89 2c c1          	mov    %rbp,(%rcx,%rax,8)
  145fa3:	48 ff 02             	incq   (%rdx)
  145fa6:	4d 85 f6             	test   %r14,%r14
  145fa9:	0f 85 23 f3 ff ff    	jne    1452d2 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
  145faf:	eb 2c                	jmp    145fdd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ed>
  145fb1:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
  145fb6:	e8 c5 78 00 00       	call   14d880 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsgHLVOFInvdC_8tk_train>
  145fbb:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  145fc0:	eb b1                	jmp    145f73 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1183>
  145fc2:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
  145fc7:	e8 b4 78 00 00       	call   14d880 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsgHLVOFInvdC_8tk_train>
  145fcc:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  145fd1:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  145fd6:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
  145fdb:	eb c2                	jmp    145f9f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11af>
  145fdd:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  145fe3:	74 0e                	je     145ff3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1203>
  145fe5:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
  145fec:	00 
  145fed:	ff 15 85 db 01 00    	call   *0x1db85(%rip)        # 163b78 <free@GLIBC_2.2.5>
  145ff3:	4c 8b 74 24 58       	mov    0x58(%rsp),%r14
  145ff8:	0f 10 44 24 60       	movups 0x60(%rsp),%xmm0
  145ffd:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  146002:	48 8d bc 24 d8 00 00 	lea    0xd8(%rsp),%rdi
  146009:	00 
  14600a:	48 8d b4 24 98 01 00 	lea    0x198(%rsp),%rsi
  146011:	00 
  146012:	e8 89 7a 00 00       	call   14daa0 <_RNvMs_NtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6finish>
  146017:	48 8b 84 24 e8 00 00 	mov    0xe8(%rsp),%rax
  14601e:	00 
  14601f:	48 8b 8c 24 70 01 00 	mov    0x170(%rsp),%rcx
  146026:	00 
  146027:	48 89 41 10          	mov    %rax,0x10(%rcx)
  14602b:	0f 10 84 24 d8 00 00 	movups 0xd8(%rsp),%xmm0
  146032:	00 
  146033:	0f 11 01             	movups %xmm0,(%rcx)
  146036:	4c 89 71 18          	mov    %r14,0x18(%rcx)
  14603a:	66 0f 6f 44 24 30    	movdqa 0x30(%rsp),%xmm0
  146040:	f3 0f 7f 41 20       	movdqu %xmm0,0x20(%rcx)
  146045:	8b 84 24 d4 00 00 00 	mov    0xd4(%rsp),%eax
  14604c:	89 41 30             	mov    %eax,0x30(%rcx)
  14604f:	48 81 c4 88 02 00 00 	add    $0x288,%rsp
  146056:	5b                   	pop    %rbx
  146057:	41 5c                	pop    %r12
  146059:	41 5d                	pop    %r13
  14605b:	41 5e                	pop    %r14
  14605d:	41 5f                	pop    %r15
  14605f:	5d                   	pop    %rbp
  146060:	c3                   	ret
  146061:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  146068:	00 
  146069:	48 89 c2             	mov    %rax,%rdx
  14606c:	48 8d 0d 0d d2 01 00 	lea    0x1d20d(%rip),%rcx        # 163280 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55a0>
  146073:	48 89 c6             	mov    %rax,%rsi
  146076:	e8 75 e1 f3 ff       	call   841f0 <_RNvNtNtCsgxBkk5gSRhY_4core5slice5index16slice_index_fail>
  14607b:	e9 82 01 00 00       	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  146080:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  146086:	4c 8b b4 24 70 01 00 	mov    0x170(%rsp),%r14
  14608d:	00 
  14608e:	74 14                	je     1460a4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12b4>
  146090:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
  146097:	00 
  146098:	48 89 c3             	mov    %rax,%rbx
  14609b:	ff 15 d7 da 01 00    	call   *0x1dad7(%rip)        # 163b78 <free@GLIBC_2.2.5>
  1460a1:	48 89 d8             	mov    %rbx,%rax
  1460a4:	49 89 46 08          	mov    %rax,0x8(%r14)
  1460a8:	48 8d 05 81 ca 01 00 	lea    0x1ca81(%rip),%rax        # 162b30 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e50>
  1460af:	49 89 46 10          	mov    %rax,0x10(%r14)
  1460b3:	49 c7 06 ff ff ff ff 	movq   $0xffffffffffffffff,(%r14)
  1460ba:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
  1460c0:	72 0b                	jb     1460cd <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12dd>
  1460c2:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  1460c7:	ff 15 ab da 01 00    	call   *0x1daab(%rip)        # 163b78 <free@GLIBC_2.2.5>
  1460cd:	48 8d bc 24 98 01 00 	lea    0x198(%rsp),%rdi
  1460d4:	00 
  1460d5:	e8 d6 5b ff ff       	call   13bcb0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
  1460da:	48 8d bc 24 b0 01 00 	lea    0x1b0(%rsp),%rdi
  1460e1:	00 
  1460e2:	e8 c9 5b ff ff       	call   13bcb0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
  1460e7:	e9 63 ff ff ff       	jmp    14604f <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x125f>
  1460ec:	48 8d 3d dd e9 ed ff 	lea    -0x121623(%rip),%rdi        # 24ad0 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x7aa>
  1460f3:	48 8d 15 fe d4 01 00 	lea    0x1d4fe(%rip),%rdx        # 1635f8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5918>
  1460fa:	be 53 00 00 00       	mov    $0x53,%esi
  1460ff:	e8 ec c1 f3 ff       	call   822f0 <_RNvNtCsgxBkk5gSRhY_4core9panicking9panic_fmt>
  146104:	e9 f9 00 00 00       	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  146109:	48 89 d0             	mov    %rdx,%rax
  14610c:	48 8d 0d 25 d7 01 00 	lea    0x1d725(%rip),%rcx        # 163838 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5b58>
  146113:	e9 5b ff ff ff       	jmp    146073 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
  146118:	48 8d 3d 6f e5 ed ff 	lea    -0x121a91(%rip),%rdi        # 2468e <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x368>
  14611f:	48 8d 15 aa d2 01 00 	lea    0x1d2aa(%rip),%rdx        # 1633d0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x56f0>
  146126:	be 11 00 00 00       	mov    $0x11,%esi
  14612b:	e8 40 f4 f3 ff       	call   85570 <_RNvNtCsgxBkk5gSRhY_4core6option13expect_failed>
  146130:	e9 cd 00 00 00       	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  146135:	b8 08 00 00 00       	mov    $0x8,%eax
  14613a:	31 ff                	xor    %edi,%edi
  14613c:	48 8d 0d bd d5 01 00 	lea    0x1d5bd(%rip),%rcx        # 163700 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5a20>
  146143:	e9 2b ff ff ff       	jmp    146073 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
  146148:	48 89 f7             	mov    %rsi,%rdi
  14614b:	e9 19 ff ff ff       	jmp    146069 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1279>
  146150:	48 89 d0             	mov    %rdx,%rax
  146153:	48 8d 0d 6e d1 01 00 	lea    0x1d16e(%rip),%rcx        # 1632c8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55e8>
  14615a:	e9 14 ff ff ff       	jmp    146073 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
  14615f:	bf 08 00 00 00       	mov    $0x8,%edi
  146164:	48 8d 0d 75 d1 01 00 	lea    0x1d175(%rip),%rcx        # 1632e0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5600>
  14616b:	e9 03 ff ff ff       	jmp    146073 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1283>
  146170:	31 c0                	xor    %eax,%eax
  146172:	eb 02                	jmp    146176 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1386>
  146174:	b0 01                	mov    $0x1,%al
  146176:	88 84 24 e0 01 00 00 	mov    %al,0x1e0(%rsp)
  14617d:	48 8d 3d 42 eb ed ff 	lea    -0x1214be(%rip),%rdi        # 24cc6 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x9a0>
  146184:	48 8d 0d 65 ce 01 00 	lea    0x1ce65(%rip),%rcx        # 162ff0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5310>
  14618b:	4c 8d 05 8e d6 01 00 	lea    0x1d68e(%rip),%r8        # 163820 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5b40>
  146192:	48 8d 94 24 e0 01 00 	lea    0x1e0(%rsp),%rdx
  146199:	00 
  14619a:	be 13 00 00 00       	mov    $0x13,%esi
  14619f:	e8 0c f4 f3 ff       	call   855b0 <_RNvNtCsgxBkk5gSRhY_4core6result13unwrap_failed>
  1461a4:	eb 5c                	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  1461a6:	48 89 ef             	mov    %rbp,%rdi
  1461a9:	48 8d 05 f8 cf 01 00 	lea    0x1cff8(%rip),%rax        # 1631a8 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x54c8>
  1461b0:	eb 0a                	jmp    1461bc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
  1461b2:	48 89 ef             	mov    %rbp,%rdi
  1461b5:	48 8d 05 04 d0 01 00 	lea    0x1d004(%rip),%rax        # 1631c0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x54e0>
  1461bc:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  1461c1:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  1461c6:	e8 c0 d0 f3 ff       	call   8328b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
  1461cb:	eb 35                	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  1461cd:	49 c1 e9 07          	shr    $0x7,%r9
  1461d1:	31 f6                	xor    %esi,%esi
  1461d3:	4c 89 cf             	mov    %r9,%rdi
  1461d6:	eb e9                	jmp    1461c1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  1461d8:	48 89 d7             	mov    %rdx,%rdi
  1461db:	eb e4                	jmp    1461c1 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  1461dd:	4c 89 ff             	mov    %r15,%rdi
  1461e0:	eb 03                	jmp    1461e5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f5>
  1461e2:	4c 89 ef             	mov    %r13,%rdi
  1461e5:	48 8d 15 f4 c8 01 00 	lea    0x1c8f4(%rip),%rdx        # 162ae0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e00>
  1461ec:	e8 9a d0 f3 ff       	call   8328b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
  1461f1:	eb 0f                	jmp    146202 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1412>
  1461f3:	bf 08 00 00 00       	mov    $0x8,%edi
  1461f8:	be 80 00 00 00       	mov    $0x80,%esi
  1461fd:	e8 90 63 f3 ff       	call   7c592 <_RNvNtCs6i54tJFfzR_5alloc7raw_vec12handle_error>
  146202:	0f 0b                	ud2
  146204:	48 89 cf             	mov    %rcx,%rdi
  146207:	48 89 c6             	mov    %rax,%rsi
  14620a:	eb 08                	jmp    146214 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1424>
  14620c:	48 8b b4 24 80 00 00 	mov    0x80(%rsp),%rsi
  146213:	00 
  146214:	48 8d 05 c5 c8 01 00 	lea    0x1c8c5(%rip),%rax        # 162ae0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e00>
  14621b:	eb 9f                	jmp    1461bc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
  14621d:	48 89 cf             	mov    %rcx,%rdi
  146220:	eb 93                	jmp    1461b5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
  146222:	48 89 cf             	mov    %rcx,%rdi
  146225:	eb 82                	jmp    1461a9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
  146227:	4c 89 e7             	mov    %r12,%rdi
  14622a:	eb 89                	jmp    1461b5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
  14622c:	4c 89 e7             	mov    %r12,%rdi
  14622f:	e9 75 ff ff ff       	jmp    1461a9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
  146234:	48 89 c7             	mov    %rax,%rdi
  146237:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  14623c:	e9 74 ff ff ff       	jmp    1461b5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c5>
  146241:	48 89 c7             	mov    %rax,%rdi
  146244:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  146249:	e9 5b ff ff ff       	jmp    1461a9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b9>
  14624e:	48 c7 c7 ff ff ff ff 	mov    $0xffffffffffffffff,%rdi
  146255:	31 f6                	xor    %esi,%esi
  146257:	48 8d 05 52 d3 01 00 	lea    0x1d352(%rip),%rax        # 1635b0 <_RNvNtNtCshywbENnkSIr_9bitcannon8classify11atom_tables11ATOM_TABLES+0x58d0>
  14625e:	e9 59 ff ff ff       	jmp    1461bc <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13cc>
  146263:	eb 59                	jmp    1462be <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14ce>
  146265:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  14626a:	66 0f 70 44 24 30 ee 	pshufd $0xee,0x30(%rsp),%xmm0
  146271:	66 48 0f 7e c0       	movq   %xmm0,%rax
  146276:	48 83 f8 02          	cmp    $0x2,%rax
  14627a:	0f 87 b9 00 00 00    	ja     146339 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1549>
  146280:	e9 bd 00 00 00       	jmp    146342 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1552>
  146285:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  14628a:	eb 4d                	jmp    1462d9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
  14628c:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  146291:	48 8b 84 24 18 01 00 	mov    0x118(%rsp),%rax
  146298:	00 
  146299:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  14629e:	eb 23                	jmp    1462c3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14d3>
  1462a0:	eb 02                	jmp    1462a4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14b4>
  1462a2:	eb 00                	jmp    1462a4 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14b4>
  1462a4:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  1462a9:	48 83 bc 24 98 00 00 	cmpq   $0x0,0x98(%rsp)
  1462b0:	00 00 
  1462b2:	74 25                	je     1462d9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
  1462b4:	48 8b bc 24 a0 00 00 	mov    0xa0(%rsp),%rdi
  1462bb:	00 
  1462bc:	eb 15                	jmp    1462d3 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e3>
  1462be:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  1462c3:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  1462c9:	74 0e                	je     1462d9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e9>
  1462cb:	48 8b bc 24 20 01 00 	mov    0x120(%rsp),%rdi
  1462d2:	00 
  1462d3:	ff 15 9f d8 01 00    	call   *0x1d89f(%rip)        # 163b78 <free@GLIBC_2.2.5>
  1462d9:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
  1462df:	72 0b                	jb     1462ec <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14fc>
  1462e1:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  1462e6:	ff 15 8c d8 01 00    	call   *0x1d88c(%rip)        # 163b78 <free@GLIBC_2.2.5>
  1462ec:	4c 8b b4 24 a0 01 00 	mov    0x1a0(%rsp),%r14
  1462f3:	00 
  1462f4:	4c 8b a4 24 a8 01 00 	mov    0x1a8(%rsp),%r12
  1462fb:	00 
  1462fc:	4d 85 e4             	test   %r12,%r12
  1462ff:	75 4b                	jne    14634c <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x155c>
  146301:	48 83 bc 24 98 01 00 	cmpq   $0x0,0x198(%rsp)
  146308:	00 00 
  14630a:	74 09                	je     146315 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1525>
  14630c:	4c 89 f7             	mov    %r14,%rdi
  14630f:	ff 15 63 d8 01 00    	call   *0x1d863(%rip)        # 163b78 <free@GLIBC_2.2.5>
  146315:	4c 8b b4 24 b8 01 00 	mov    0x1b8(%rsp),%r14
  14631c:	00 
  14631d:	4c 8b a4 24 c0 01 00 	mov    0x1c0(%rsp),%r12
  146324:	00 
  146325:	4d 85 e4             	test   %r12,%r12
  146328:	0f 85 ab 00 00 00    	jne    1463d9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15e9>
  14632e:	48 83 bc 24 b0 01 00 	cmpq   $0x0,0x1b0(%rsp)
  146335:	00 00 
  146337:	74 09                	je     146342 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1552>
  146339:	4c 89 f7             	mov    %r14,%rdi
  14633c:	ff 15 36 d8 01 00    	call   *0x1d836(%rip)        # 163b78 <free@GLIBC_2.2.5>
  146342:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  146347:	e8 14 5e 01 00       	call   15c160 <_Unwind_Resume@plt>
  14634c:	45 31 ed             	xor    %r13d,%r13d
  14634f:	48 8b 2d 22 d8 01 00 	mov    0x1d822(%rip),%rbp        # 163b78 <free@GLIBC_2.2.5>
  146356:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
  14635b:	eb 15                	jmp    146372 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1582>
  14635d:	0f 1f 00             	nopl   (%rax)
  146360:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
  146365:	4c 89 ff             	mov    %r15,%rdi
  146368:	ff d5                	call   *%rbp
  14636a:	49 ff c5             	inc    %r13
  14636d:	4d 39 e5             	cmp    %r12,%r13
  146370:	74 8f                	je     146301 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1511>
  146372:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
  146379:	00 
  14637a:	4c 01 e9             	add    %r13,%rcx
  14637d:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
  146381:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
  146386:	74 18                	je     1463a0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15b0>
  146388:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
  14638d:	72 db                	jb     14636a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x157a>
  14638f:	4c 8b 78 08          	mov    0x8(%rax),%r15
  146393:	eb d0                	jmp    146365 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1575>
  146395:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  14639c:	00 00 00 00 
  1463a0:	48 8b 58 10          	mov    0x10(%rax),%rbx
  1463a4:	48 85 db             	test   %rbx,%rbx
  1463a7:	74 c1                	je     14636a <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x157a>
  1463a9:	4c 8b 78 08          	mov    0x8(%rax),%r15
  1463ad:	4d 8d 67 18          	lea    0x18(%r15),%r12
  1463b1:	eb 16                	jmp    1463c9 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d9>
  1463b3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1463ba:	84 00 00 00 00 00 
  1463c0:	49 83 c4 20          	add    $0x20,%r12
  1463c4:	48 ff cb             	dec    %rbx
  1463c7:	74 97                	je     146360 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1570>
  1463c9:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
  1463ce:	72 f0                	jb     1463c0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d0>
  1463d0:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
  1463d5:	ff d5                	call   *%rbp
  1463d7:	eb e7                	jmp    1463c0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15d0>
  1463d9:	45 31 ed             	xor    %r13d,%r13d
  1463dc:	48 8b 2d 95 d7 01 00 	mov    0x1d795(%rip),%rbp        # 163b78 <free@GLIBC_2.2.5>
  1463e3:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
  1463e8:	eb 1c                	jmp    146406 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1616>
  1463ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  1463f0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
  1463f5:	4c 89 ff             	mov    %r15,%rdi
  1463f8:	ff d5                	call   *%rbp
  1463fa:	49 ff c5             	inc    %r13
  1463fd:	4d 39 e5             	cmp    %r12,%r13
  146400:	0f 84 28 ff ff ff    	je     14632e <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x153e>
  146406:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
  14640d:	00 
  14640e:	4c 01 e9             	add    %r13,%rcx
  146411:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
  146415:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
  14641a:	74 14                	je     146430 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1640>
  14641c:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
  146421:	72 d7                	jb     1463fa <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x160a>
  146423:	4c 8b 78 08          	mov    0x8(%rax),%r15
  146427:	eb cc                	jmp    1463f5 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1605>
  146429:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  146430:	48 8b 58 10          	mov    0x10(%rax),%rbx
  146434:	48 85 db             	test   %rbx,%rbx
  146437:	74 c1                	je     1463fa <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x160a>
  146439:	4c 8b 78 08          	mov    0x8(%rax),%r15
  14643d:	4d 8d 67 18          	lea    0x18(%r15),%r12
  146441:	eb 16                	jmp    146459 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1669>
  146443:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  14644a:	84 00 00 00 00 00 
  146450:	49 83 c4 20          	add    $0x20,%r12
  146454:	48 ff cb             	dec    %rbx
  146457:	74 97                	je     1463f0 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1600>
  146459:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
  14645e:	72 f0                	jb     146450 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1660>
  146460:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
  146465:	ff d5                	call   *%rbp
  146467:	eb e7                	jmp    146450 <_RNCNvMNtNtNtNtCsgHLVOFInvdC_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1660>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
