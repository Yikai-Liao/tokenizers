
/dev/shm/bpe-frequency-20261014/C-full-runner:     file format elf64-x86-64


Disassembly of section .text:

0000000000144e30 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_>:
  144e30:	55                   	push   %rbp
  144e31:	41 57                	push   %r15
  144e33:	41 56                	push   %r14
  144e35:	41 55                	push   %r13
  144e37:	41 54                	push   %r12
  144e39:	53                   	push   %rbx
  144e3a:	48 81 ec 88 02 00 00 	sub    $0x288,%rsp
  144e41:	49 89 d1             	mov    %rdx,%r9
  144e44:	48 89 fb             	mov    %rdi,%rbx
  144e47:	44 8b 39             	mov    (%rcx),%r15d
  144e4a:	44 8b 69 04          	mov    0x4(%rcx),%r13d
  144e4e:	8b 69 08             	mov    0x8(%rcx),%ebp
  144e51:	0f 10 41 60          	movups 0x60(%rcx),%xmm0
  144e55:	0f 29 84 24 70 02 00 	movaps %xmm0,0x270(%rsp)
  144e5c:	00 
  144e5d:	0f 10 41 50          	movups 0x50(%rcx),%xmm0
  144e61:	0f 29 84 24 60 02 00 	movaps %xmm0,0x260(%rsp)
  144e68:	00 
  144e69:	f3 0f 6f 41 10       	movdqu 0x10(%rcx),%xmm0
  144e6e:	f3 0f 6f 49 20       	movdqu 0x20(%rcx),%xmm1
  144e73:	f3 0f 6f 51 30       	movdqu 0x30(%rcx),%xmm2
  144e78:	0f 10 59 40          	movups 0x40(%rcx),%xmm3
  144e7c:	0f 29 9c 24 50 02 00 	movaps %xmm3,0x250(%rsp)
  144e83:	00 
  144e84:	66 0f 7f 94 24 40 02 	movdqa %xmm2,0x240(%rsp)
  144e8b:	00 00 
  144e8d:	66 0f 7f 8c 24 30 02 	movdqa %xmm1,0x230(%rsp)
  144e94:	00 00 
  144e96:	66 0f 7f 84 24 20 02 	movdqa %xmm0,0x220(%rsp)
  144e9d:	00 00 
  144e9f:	4c 8b 26             	mov    (%rsi),%r12
  144ea2:	48 8b 46 08          	mov    0x8(%rsi),%rax
  144ea6:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  144ead:	00 
  144eae:	48 8b 46 10          	mov    0x10(%rsi),%rax
  144eb2:	48 89 84 24 c0 00 00 	mov    %rax,0xc0(%rsp)
  144eb9:	00 
  144eba:	4d 8b 34 24          	mov    (%r12),%r14
  144ebe:	4d 8b 46 58          	mov    0x58(%r14),%r8
  144ec2:	48 8d bc 24 d8 01 00 	lea    0x1d8(%rsp),%rdi
  144ec9:	00 
  144eca:	44 89 fe             	mov    %r15d,%esi
  144ecd:	44 89 ea             	mov    %r13d,%edx
  144ed0:	89 e9                	mov    %ebp,%ecx
  144ed2:	e8 f9 80 00 00       	call   14cfd0 <_RNvMs_NtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges3new>
  144ed7:	48 c7 44 24 68 00 00 	movq   $0x0,0x68(%rsp)
  144ede:	00 00 
  144ee0:	49 8b 76 58          	mov    0x58(%r14),%rsi
  144ee4:	4c 39 fe             	cmp    %r15,%rsi
  144ee7:	0f 86 ef 12 00 00    	jbe    1461dc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13ac>
  144eed:	4c 39 ee             	cmp    %r13,%rsi
  144ef0:	0f 86 eb 12 00 00    	jbe    1461e1 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b1>
  144ef6:	4c 89 a4 24 88 00 00 	mov    %r12,0x88(%rsp)
  144efd:	00 
  144efe:	89 ac 24 dc 00 00 00 	mov    %ebp,0xdc(%rsp)
  144f05:	48 89 9c 24 78 01 00 	mov    %rbx,0x178(%rsp)
  144f0c:	00 
  144f0d:	49 8b 46 50          	mov    0x50(%r14),%rax
  144f11:	4c 89 7c 24 20       	mov    %r15,0x20(%rsp)
  144f16:	4a 8b 0c f8          	mov    (%rax,%r15,8),%rcx
  144f1a:	48 89 8c 24 d0 00 00 	mov    %rcx,0xd0(%rsp)
  144f21:	00 
  144f22:	4c 89 ac 24 98 00 00 	mov    %r13,0x98(%rsp)
  144f29:	00 
  144f2a:	4a 8b 04 e8          	mov    (%rax,%r13,8),%rax
  144f2e:	48 89 84 24 c8 00 00 	mov    %rax,0xc8(%rsp)
  144f35:	00 
  144f36:	48 8d 84 24 20 02 00 	lea    0x220(%rsp),%rax
  144f3d:	00 
  144f3e:	48 89 84 24 88 01 00 	mov    %rax,0x188(%rsp)
  144f45:	00 
  144f46:	4c 89 b4 24 90 01 00 	mov    %r14,0x190(%rsp)
  144f4d:	00 
  144f4e:	4c 89 74 24 28       	mov    %r14,0x28(%rsp)
  144f53:	4c 89 b4 24 98 01 00 	mov    %r14,0x198(%rsp)
  144f5a:	00 
  144f5b:	bf 80 00 00 00       	mov    $0x80,%edi
  144f60:	ff 15 42 ec 01 00    	call   *0x1ec42(%rip)        # 163ba8 <malloc@GLIBC_2.2.5>
  144f66:	48 85 c0             	test   %rax,%rax
  144f69:	0f 84 83 12 00 00    	je     1461f2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13c2>
  144f6f:	48 89 c5             	mov    %rax,%rbp
  144f72:	66 0f ef c0          	pxor   %xmm0,%xmm0
  144f76:	f3 0f 7f 84 24 b0 00 	movdqu %xmm0,0xb0(%rsp)
  144f7d:	00 00 
  144f7f:	48 c7 84 24 a0 00 00 	movq   $0x10,0xa0(%rsp)
  144f86:	00 10 00 00 00 
  144f8b:	48 89 84 24 a8 00 00 	mov    %rax,0xa8(%rsp)
  144f92:	00 
  144f93:	48 8d 84 24 88 01 00 	lea    0x188(%rsp),%rax
  144f9a:	00 
  144f9b:	48 89 84 24 e0 00 00 	mov    %rax,0xe0(%rsp)
  144fa2:	00 
  144fa3:	48 c7 84 24 e8 00 00 	movq   $0x10,0xe8(%rsp)
  144faa:	00 10 00 00 00 
  144faf:	41 bf 10 00 00 00    	mov    $0x10,%r15d
  144fb5:	48 c7 44 24 30 00 00 	movq   $0x0,0x30(%rsp)
  144fbc:	00 00 
  144fbe:	4c 8d a4 24 e0 00 00 	lea    0xe0(%rsp),%r12
  144fc5:	00 
  144fc6:	31 db                	xor    %ebx,%ebx
  144fc8:	4c 89 e7             	mov    %r12,%rdi
  144fcb:	e8 f0 a6 00 00       	call   14f6c0 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
  144fd0:	48 83 f8 01          	cmp    $0x1,%rax
  144fd4:	0f 85 b1 01 00 00    	jne    14518b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x35b>
  144fda:	48 83 fb ff          	cmp    $0xffffffffffffffff,%rbx
  144fde:	0f 84 33 11 00 00    	je     146117 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12e7>
  144fe4:	49 89 d5             	mov    %rdx,%r13
  144fe7:	4c 89 f8             	mov    %r15,%rax
  144fea:	48 29 d8             	sub    %rbx,%rax
  144fed:	76 11                	jbe    145000 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1d0>
  144fef:	49 89 de             	mov    %rbx,%r14
  144ff2:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  144ff7:	e9 ff 00 00 00       	jmp    1450fb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  144ffc:	0f 1f 40 00          	nopl   0x0(%rax)
  145000:	0f 84 99 00 00 00    	je     14509f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x26f>
  145006:	4c 89 fe             	mov    %r15,%rsi
  145009:	49 89 de             	mov    %rbx,%r14
  14500c:	48 8b 5c 24 30       	mov    0x30(%rsp),%rbx
  145011:	48 39 c3             	cmp    %rax,%rbx
  145014:	0f 86 d6 00 00 00    	jbe    1450f0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2c0>
  14501a:	4c 89 fa             	mov    %r15,%rdx
  14501d:	48 29 da             	sub    %rbx,%rdx
  145020:	4c 89 f0             	mov    %r14,%rax
  145023:	48 29 d0             	sub    %rdx,%rax
  145026:	48 39 c2             	cmp    %rax,%rdx
  145029:	76 36                	jbe    145061 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
  14502b:	48 89 f1             	mov    %rsi,%rcx
  14502e:	4c 29 f9             	sub    %r15,%rcx
  145031:	48 39 c1             	cmp    %rax,%rcx
  145034:	72 2b                	jb     145061 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x231>
  145036:	48 8b ac 24 a8 00 00 	mov    0xa8(%rsp),%rbp
  14503d:	00 
  14503e:	4a 8d 3c fd 00 00 00 	lea    0x0(,%r15,8),%rdi
  145045:	00 
  145046:	48 01 ef             	add    %rbp,%rdi
  145049:	48 c1 e0 03          	shl    $0x3,%rax
  14504d:	49 89 f7             	mov    %rsi,%r15
  145050:	48 89 ee             	mov    %rbp,%rsi
  145053:	48 89 c2             	mov    %rax,%rdx
  145056:	ff 15 3c eb 01 00    	call   *0x1eb3c(%rip)        # 163b98 <memcpy@GLIBC_2.14>
  14505c:	e9 9a 00 00 00       	jmp    1450fb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  145061:	48 89 d8             	mov    %rbx,%rax
  145064:	48 89 f3             	mov    %rsi,%rbx
  145067:	48 29 d3             	sub    %rdx,%rbx
  14506a:	48 8b ac 24 a8 00 00 	mov    0xa8(%rsp),%rbp
  145071:	00 
  145072:	49 89 f7             	mov    %rsi,%r15
  145075:	48 8d 34 c5 00 00 00 	lea    0x0(,%rax,8),%rsi
  14507c:	00 
  14507d:	48 01 ee             	add    %rbp,%rsi
  145080:	48 8d 3c dd 00 00 00 	lea    0x0(,%rbx,8),%rdi
  145087:	00 
  145088:	48 01 ef             	add    %rbp,%rdi
  14508b:	48 c1 e2 03          	shl    $0x3,%rdx
  14508f:	ff 15 0b eb 01 00    	call   *0x1eb0b(%rip)        # 163ba0 <memmove@GLIBC_2.2.5>
  145095:	48 89 9c 24 b0 00 00 	mov    %rbx,0xb0(%rsp)
  14509c:	00 
  14509d:	eb 5c                	jmp    1450fb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x2cb>
  14509f:	ba 01 00 00 00       	mov    $0x1,%edx
  1450a4:	b9 08 00 00 00       	mov    $0x8,%ecx
  1450a9:	41 b8 08 00 00 00    	mov    $0x8,%r8d
  1450af:	48 8d bc 24 a0 00 00 	lea    0xa0(%rsp),%rdi
  1450b6:	00 
  1450b7:	48 89 de             	mov    %rbx,%rsi
  1450ba:	e8 81 22 fe ff       	call   127340 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsigKge7u8hP3_8tk_train>
  1450bf:	48 8b b4 24 a0 00 00 	mov    0xa0(%rsp),%rsi
  1450c6:	00 
  1450c7:	48 8b 9c 24 b0 00 00 	mov    0xb0(%rsp),%rbx
  1450ce:	00 
  1450cf:	4c 8b b4 24 b8 00 00 	mov    0xb8(%rsp),%r14
  1450d6:	00 
  1450d7:	4c 89 f8             	mov    %r15,%rax
  1450da:	4c 29 f0             	sub    %r14,%rax
  1450dd:	48 39 c3             	cmp    %rax,%rbx
  1450e0:	0f 87 34 ff ff ff    	ja     14501a <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1ea>
  1450e6:	66 2e 0f 1f 84 00 00 	cs nopw 0x0(%rax,%rax,1)
  1450ed:	00 00 00 
  1450f0:	48 8b ac 24 a8 00 00 	mov    0xa8(%rsp),%rbp
  1450f7:	00 
  1450f8:	49 89 f7             	mov    %rsi,%r15
  1450fb:	48 89 5c 24 30       	mov    %rbx,0x30(%rsp)
  145100:	49 8d 04 1e          	lea    (%r14,%rbx,1),%rax
  145104:	4c 39 f8             	cmp    %r15,%rax
  145107:	b9 00 00 00 00       	mov    $0x0,%ecx
  14510c:	49 0f 43 cf          	cmovae %r15,%rcx
  145110:	48 29 c8             	sub    %rcx,%rax
  145113:	4c 89 6c c5 00       	mov    %r13,0x0(%rbp,%rax,8)
  145118:	49 8d 5e 01          	lea    0x1(%r14),%rbx
  14511c:	48 89 9c 24 b8 00 00 	mov    %rbx,0xb8(%rsp)
  145123:	00 
  145124:	4c 39 fb             	cmp    %r15,%rbx
  145127:	0f 83 9b fe ff ff    	jae    144fc8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
  14512d:	48 8b 44 24 30       	mov    0x30(%rsp),%rax
  145132:	49 01 c6             	add    %rax,%r14
  145135:	49 ff c6             	inc    %r14
  145138:	0f 1f 84 00 00 00 00 	nopl   0x0(%rax,%rax,1)
  14513f:	00 
  145140:	4c 89 e7             	mov    %r12,%rdi
  145143:	e8 78 a5 00 00       	call   14f6c0 <_RNvXs_NtNtNtCsgxBkk5gSRhY_4core4iter8adapters4takeINtB4_4TakeQINtNtB6_7inspect7InspectINtNtB6_3map3MapQIBN_INtNtB6_7flatten7FlatMapINtNtNtBa_5slice4iter4IterNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe9positions4LeafENtNtB2x_5codec6CursorNCNvB2x_4read0EENCINvMs_NtB2z_6corpusNtB4c_6Corpus13fresh_matchesB1C_E0ENCB46_s_0EENtNtNtB8_6traits8iterator8Iterator4nextB2D_>
  145148:	48 83 f8 01          	cmp    $0x1,%rax
  14514c:	75 35                	jne    145183 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x353>
  14514e:	4d 39 fe             	cmp    %r15,%r14
  145151:	b8 00 00 00 00       	mov    $0x0,%eax
  145156:	49 0f 43 c7          	cmovae %r15,%rax
  14515a:	48 c1 e0 03          	shl    $0x3,%rax
  14515e:	48 89 e9             	mov    %rbp,%rcx
  145161:	48 29 c1             	sub    %rax,%rcx
  145164:	4a 89 14 f1          	mov    %rdx,(%rcx,%r14,8)
  145168:	48 ff c3             	inc    %rbx
  14516b:	49 ff c6             	inc    %r14
  14516e:	49 39 df             	cmp    %rbx,%r15
  145171:	75 cd                	jne    145140 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x310>
  145173:	4c 89 bc 24 b8 00 00 	mov    %r15,0xb8(%rsp)
  14517a:	00 
  14517b:	4c 89 fb             	mov    %r15,%rbx
  14517e:	e9 45 fe ff ff       	jmp    144fc8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x198>
  145183:	48 89 9c 24 b8 00 00 	mov    %rbx,0xb8(%rsp)
  14518a:	00 
  14518b:	48 8b 8c 24 d0 00 00 	mov    0xd0(%rsp),%rcx
  145192:	00 
  145193:	48 8b 94 24 c8 00 00 	mov    0xc8(%rsp),%rdx
  14519a:	00 
  14519b:	48 01 ca             	add    %rcx,%rdx
  14519e:	f3 0f 6f 84 24 a0 00 	movdqu 0xa0(%rsp),%xmm0
  1451a5:	00 00 
  1451a7:	f3 0f 6f 8c 24 b0 00 	movdqu 0xb0(%rsp),%xmm1
  1451ae:	00 00 
  1451b0:	66 0f 7f 84 24 a0 01 	movdqa %xmm0,0x1a0(%rsp)
  1451b7:	00 00 
  1451b9:	66 0f 7f 8c 24 b0 01 	movdqa %xmm1,0x1b0(%rsp)
  1451c0:	00 00 
  1451c2:	f3 0f 6f 94 24 88 01 	movdqu 0x188(%rsp),%xmm2
  1451c9:	00 00 
  1451cb:	66 0f 7f 94 24 c0 01 	movdqa %xmm2,0x1c0(%rsp)
  1451d2:	00 00 
  1451d4:	48 8b 84 24 98 01 00 	mov    0x198(%rsp),%rax
  1451db:	00 
  1451dc:	48 89 84 24 d0 01 00 	mov    %rax,0x1d0(%rsp)
  1451e3:	00 
  1451e4:	48 89 84 24 58 01 00 	mov    %rax,0x158(%rsp)
  1451eb:	00 
  1451ec:	f3 0f 7f 94 24 48 01 	movdqu %xmm2,0x148(%rsp)
  1451f3:	00 00 
  1451f5:	f3 0f 7f 8c 24 38 01 	movdqu %xmm1,0x138(%rsp)
  1451fc:	00 00 
  1451fe:	f3 0f 7f 84 24 28 01 	movdqu %xmm0,0x128(%rsp)
  145205:	00 00 
  145207:	48 c7 84 24 e0 00 00 	movq   $0x0,0xe0(%rsp)
  14520e:	00 00 00 00 00 
  145213:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145218:	48 89 84 24 00 01 00 	mov    %rax,0x100(%rsp)
  14521f:	00 
  145220:	48 89 8c 24 08 01 00 	mov    %rcx,0x108(%rsp)
  145227:	00 
  145228:	48 89 94 24 10 01 00 	mov    %rdx,0x110(%rsp)
  14522f:	00 
  145230:	48 8b 4c 24 20       	mov    0x20(%rsp),%rcx
  145235:	89 8c 24 18 01 00 00 	mov    %ecx,0x118(%rsp)
  14523c:	48 8b 8c 24 98 00 00 	mov    0x98(%rsp),%rcx
  145243:	00 
  145244:	89 8c 24 1c 01 00 00 	mov    %ecx,0x11c(%rsp)
  14524b:	48 89 84 24 20 01 00 	mov    %rax,0x120(%rsp)
  145252:	00 
  145253:	4c 8b b4 24 40 01 00 	mov    0x140(%rsp),%r14
  14525a:	00 
  14525b:	48 8b 84 24 28 01 00 	mov    0x128(%rsp),%rax
  145262:	00 
  145263:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  145268:	4d 85 f6             	test   %r14,%r14
  14526b:	0f 84 6b 0d 00 00    	je     145fdc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ac>
  145271:	4c 8b ac 24 38 01 00 	mov    0x138(%rsp),%r13
  145278:	00 
  145279:	48 8b 8c 24 c0 00 00 	mov    0xc0(%rsp),%rcx
  145280:	00 
  145281:	0f b6 01             	movzbl (%rcx),%eax
  145284:	88 44 24 17          	mov    %al,0x17(%rsp)
  145288:	48 8b 41 10          	mov    0x10(%rcx),%rax
  14528c:	48 89 44 24 48       	mov    %rax,0x48(%rsp)
  145291:	48 8b 41 08          	mov    0x8(%rcx),%rax
  145295:	48 89 44 24 78       	mov    %rax,0x78(%rsp)
  14529a:	48 8b 94 24 90 00 00 	mov    0x90(%rsp),%rdx
  1452a1:	00 
  1452a2:	48 8b 42 18          	mov    0x18(%rdx),%rax
  1452a6:	48 89 84 24 70 01 00 	mov    %rax,0x170(%rsp)
  1452ad:	00 
  1452ae:	48 8b 42 28          	mov    0x28(%rdx),%rax
  1452b2:	48 89 84 24 68 01 00 	mov    %rax,0x168(%rsp)
  1452b9:	00 
  1452ba:	48 8b 42 20          	mov    0x20(%rdx),%rax
  1452be:	48 89 84 24 60 01 00 	mov    %rax,0x160(%rsp)
  1452c5:	00 
  1452c6:	48 8b 02             	mov    (%rdx),%rax
  1452c9:	48 89 44 24 70       	mov    %rax,0x70(%rsp)
  1452ce:	48 8b 42 08          	mov    0x8(%rdx),%rax
  1452d2:	48 89 84 24 80 00 00 	mov    %rax,0x80(%rsp)
  1452d9:	00 
  1452da:	48 8b 84 24 88 00 00 	mov    0x88(%rsp),%rax
  1452e1:	00 
  1452e2:	48 8b 40 08          	mov    0x8(%rax),%rax
  1452e6:	48 89 84 24 90 00 00 	mov    %rax,0x90(%rsp)
  1452ed:	00 
  1452ee:	48 8b 41 58          	mov    0x58(%rcx),%rax
  1452f2:	48 89 84 24 88 00 00 	mov    %rax,0x88(%rsp)
  1452f9:	00 
  1452fa:	48 8b 41 50          	mov    0x50(%rcx),%rax
  1452fe:	48 89 84 24 80 01 00 	mov    %rax,0x180(%rsp)
  145305:	00 
  145306:	48 8d 05 a3 df 01 00 	lea    0x1dfa3(%rip),%rax        # 1632b0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55d0>
  14530d:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  145312:	4c 8b 8c 24 00 01 00 	mov    0x100(%rsp),%r9
  145319:	00 
  14531a:	48 8b 84 24 08 01 00 	mov    0x108(%rsp),%rax
  145321:	00 
  145322:	48 89 84 24 98 00 00 	mov    %rax,0x98(%rsp)
  145329:	00 
  14532a:	8b 84 24 18 01 00 00 	mov    0x118(%rsp),%eax
  145331:	89 84 24 c8 00 00 00 	mov    %eax,0xc8(%rsp)
  145338:	48 8b 84 24 10 01 00 	mov    0x110(%rsp),%rax
  14533f:	00 
  145340:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  145345:	8b 84 24 1c 01 00 00 	mov    0x11c(%rsp),%eax
  14534c:	89 84 24 c0 00 00 00 	mov    %eax,0xc0(%rsp)
  145353:	48 8b 5c 24 50       	mov    0x50(%rsp),%rbx
  145358:	4c 89 8c 24 d0 00 00 	mov    %r9,0xd0(%rsp)
  14535f:	00 
  145360:	eb 17                	jmp    145379 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x549>
  145362:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145369:	1f 84 00 00 00 00 00 
  145370:	4d 85 f6             	test   %r14,%r14
  145373:	0f 84 63 0c 00 00    	je     145fdc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ac>
  145379:	4c 89 e8             	mov    %r13,%rax
  14537c:	49 8d 4d 01          	lea    0x1(%r13),%rcx
  145380:	48 39 d9             	cmp    %rbx,%rcx
  145383:	48 89 d9             	mov    %rbx,%rcx
  145386:	ba 00 00 00 00       	mov    $0x0,%edx
  14538b:	48 0f 42 ca          	cmovb  %rdx,%rcx
  14538f:	48 f7 d9             	neg    %rcx
  145392:	4c 01 e9             	add    %r13,%rcx
  145395:	48 ff c1             	inc    %rcx
  145398:	49 89 cd             	mov    %rcx,%r13
  14539b:	48 89 8c 24 38 01 00 	mov    %rcx,0x138(%rsp)
  1453a2:	00 
  1453a3:	4d 89 f7             	mov    %r14,%r15
  1453a6:	49 ff cf             	dec    %r15
  1453a9:	4c 89 bc 24 40 01 00 	mov    %r15,0x140(%rsp)
  1453b0:	00 
  1453b1:	48 8b 8c 24 30 01 00 	mov    0x130(%rsp),%rcx
  1453b8:	00 
  1453b9:	4c 8b b4 24 48 01 00 	mov    0x148(%rsp),%r14
  1453c0:	00 
  1453c1:	48 8b 04 c1          	mov    (%rcx,%rax,8),%rax
  1453c5:	48 89 44 24 30       	mov    %rax,0x30(%rsp)
  1453ca:	49 8b 46 58          	mov    0x58(%r14),%rax
  1453ce:	bd 00 00 00 00       	mov    $0x0,%ebp
  1453d3:	48 85 c0             	test   %rax,%rax
  1453d6:	0f 84 c9 03 00 00    	je     1457a5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x975>
  1453dc:	48 ff c8             	dec    %rax
  1453df:	49 89 46 58          	mov    %rax,0x58(%r14)
  1453e3:	49 8d 46 08          	lea    0x8(%r14),%rax
  1453e7:	4d 8d 46 10          	lea    0x10(%r14),%r8
  1453eb:	49 8b 36             	mov    (%r14),%rsi
  1453ee:	48 85 f6             	test   %rsi,%rsi
  1453f1:	74 3a                	je     14542d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5fd>
  1453f3:	48 8b 10             	mov    (%rax),%rdx
  1453f6:	48 85 d2             	test   %rdx,%rdx
  1453f9:	74 2b                	je     145426 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5f6>
  1453fb:	4d 8b 4e 18          	mov    0x18(%r14),%r9
  1453ff:	41 f6 c1 7f          	test   $0x7f,%r9b
  145403:	0f 84 22 02 00 00    	je     14562b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
  145409:	4c 0f be 26          	movsbq (%rsi),%r12
  14540d:	4d 85 e4             	test   %r12,%r12
  145410:	0f 88 d0 00 00 00    	js     1454e6 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6b6>
  145416:	4d 03 20             	add    (%r8),%r12
  145419:	4d 89 20             	mov    %r12,(%r8)
  14541c:	bf 01 00 00 00       	mov    $0x1,%edi
  145421:	e9 3b 01 00 00       	jmp    145561 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x731>
  145426:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  14542d:	49 8b 6e 40          	mov    0x40(%r14),%rbp
  145431:	48 85 ed             	test   %rbp,%rbp
  145434:	0f 84 01 04 00 00    	je     14583b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
  14543a:	4d 8b 5e 48          	mov    0x48(%r14),%r11
  14543e:	4c 39 dd             	cmp    %r11,%rbp
  145441:	0f 84 f4 03 00 00    	je     14583b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa0b>
  145447:	4c 8d 65 20          	lea    0x20(%rbp),%r12
  14544b:	4d 89 66 40          	mov    %r12,0x40(%r14)
  14544f:	4d 8b 4e 50          	mov    0x50(%r14),%r9
  145453:	48 8b 4d 00          	mov    0x0(%rbp),%rcx
  145457:	48 ff c9             	dec    %rcx
  14545a:	49 39 c9             	cmp    %rcx,%r9
  14545d:	75 18                	jne    145477 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x647>
  14545f:	49 c7 06 01 00 00 00 	movq   $0x1,(%r14)
  145466:	66 0f ef c0          	pxor   %xmm0,%xmm0
  14546a:	f3 0f 7f 00          	movdqu %xmm0,(%rax)
  14546e:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  145472:	e9 4c 02 00 00       	jmp    1456c3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
  145477:	49 89 ca             	mov    %rcx,%r10
  14547a:	49 c1 ea 07          	shr    $0x7,%r10
  14547e:	4d 85 c9             	test   %r9,%r9
  145481:	0f 84 e2 00 00 00    	je     145569 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x739>
  145487:	89 c8                	mov    %ecx,%eax
  145489:	83 e0 7f             	and    $0x7f,%eax
  14548c:	48 83 f8 01          	cmp    $0x1,%rax
  145490:	4c 89 d6             	mov    %r10,%rsi
  145493:	48 83 de ff          	sbb    $0xffffffffffffffff,%rsi
  145497:	48 83 fe 02          	cmp    $0x2,%rsi
  14549b:	0f 82 2b 0d 00 00    	jb     1461cc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x139c>
  1454a1:	48 8b 7d 18          	mov    0x18(%rbp),%rdi
  1454a5:	48 89 fa             	mov    %rdi,%rdx
  1454a8:	48 83 ff 11          	cmp    $0x11,%rdi
  1454ac:	72 04                	jb     1454b2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x682>
  1454ae:	48 8b 55 10          	mov    0x10(%rbp),%rdx
  1454b2:	48 8d 04 f5 08 00 00 	lea    0x8(,%rsi,8),%rax
  1454b9:	00 
  1454ba:	48 39 d0             	cmp    %rdx,%rax
  1454bd:	0f 87 9b 0c 00 00    	ja     14615e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x132e>
  1454c3:	4c 89 ca             	mov    %r9,%rdx
  1454c6:	48 c1 ea 07          	shr    $0x7,%rdx
  1454ca:	48 39 f2             	cmp    %rsi,%rdx
  1454cd:	0f 83 04 0d 00 00    	jae    1461d7 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13a7>
  1454d3:	48 83 ff 11          	cmp    $0x11,%rdi
  1454d7:	0f 82 90 00 00 00    	jb     14556d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x73d>
  1454dd:	48 8b 45 08          	mov    0x8(%rbp),%rax
  1454e1:	e9 8b 00 00 00       	jmp    145571 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x741>
  1454e6:	48 83 fa 01          	cmp    $0x1,%rdx
  1454ea:	0f 84 7f 0c 00 00    	je     14616f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x133f>
  1454f0:	41 83 e4 7f          	and    $0x7f,%r12d
  1454f4:	4c 8d 52 ff          	lea    -0x1(%rdx),%r10
  1454f8:	b9 07 00 00 00       	mov    $0x7,%ecx
  1454fd:	31 ff                	xor    %edi,%edi
  1454ff:	4c 89 e0             	mov    %r12,%rax
  145502:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145509:	1f 84 00 00 00 00 00 
  145510:	44 0f b6 5c 3e 01    	movzbl 0x1(%rsi,%rdi,1),%r11d
  145516:	83 f9 3f             	cmp    $0x3f,%ecx
  145519:	75 0a                	jne    145525 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6f5>
  14551b:	41 80 fb 01          	cmp    $0x1,%r11b
  14551f:	0f 87 4e 0c 00 00    	ja     146173 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1343>
  145525:	45 0f b6 e3          	movzbl %r11b,%r12d
  145529:	45 84 e4             	test   %r12b,%r12b
  14552c:	79 1a                	jns    145548 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x718>
  14552e:	41 83 e4 7f          	and    $0x7f,%r12d
  145532:	49 d3 e4             	shl    %cl,%r12
  145535:	4c 09 e0             	or     %r12,%rax
  145538:	83 c1 07             	add    $0x7,%ecx
  14553b:	48 ff c7             	inc    %rdi
  14553e:	49 39 fa             	cmp    %rdi,%r10
  145541:	75 cd                	jne    145510 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x6e0>
  145543:	e9 27 0c 00 00       	jmp    14616f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x133f>
  145548:	49 d3 e4             	shl    %cl,%r12
  14554b:	49 09 c4             	or     %rax,%r12
  14554e:	48 83 c7 02          	add    $0x2,%rdi
  145552:	4d 03 20             	add    (%r8),%r12
  145555:	4d 89 20             	mov    %r12,(%r8)
  145558:	48 39 d7             	cmp    %rdx,%rdi
  14555b:	0f 87 a7 0b 00 00    	ja     146108 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12d8>
  145561:	48 29 fa             	sub    %rdi,%rdx
  145564:	e9 db 00 00 00       	jmp    145644 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x814>
  145569:	31 ff                	xor    %edi,%edi
  14556b:	eb 09                	jmp    145576 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x746>
  14556d:	48 8d 45 08          	lea    0x8(%rbp),%rax
  145571:	48 8b 7c d0 08       	mov    0x8(%rax,%rdx,8),%rdi
  145576:	83 e1 7f             	and    $0x7f,%ecx
  145579:	48 83 f9 01          	cmp    $0x1,%rcx
  14557d:	49 83 da ff          	sbb    $0xffffffffffffffff,%r10
  145581:	be 08 00 00 00       	mov    $0x8,%esi
  145586:	49 83 fa 02          	cmp    $0x2,%r10
  14558a:	73 1e                	jae    1455aa <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x77a>
  14558c:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
  145590:	48 89 c8             	mov    %rcx,%rax
  145593:	48 83 f9 11          	cmp    $0x11,%rcx
  145597:	72 39                	jb     1455d2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7a2>
  145599:	48 8b 45 10          	mov    0x10(%rbp),%rax
  14559d:	48 89 c2             	mov    %rax,%rdx
  1455a0:	48 29 f2             	sub    %rsi,%rdx
  1455a3:	73 39                	jae    1455de <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7ae>
  1455a5:	e9 9d 0b 00 00       	jmp    146147 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1317>
  1455aa:	49 ff ca             	dec    %r10
  1455ad:	49 0f bd ca          	bsr    %r10,%rcx
  1455b1:	f7 d1                	not    %ecx
  1455b3:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  1455ba:	48 d3 e8             	shr    %cl,%rax
  1455bd:	48 8d 34 c5 10 00 00 	lea    0x10(,%rax,8),%rsi
  1455c4:	00 
  1455c5:	48 8b 4d 18          	mov    0x18(%rbp),%rcx
  1455c9:	48 89 c8             	mov    %rcx,%rax
  1455cc:	48 83 f9 11          	cmp    $0x11,%rcx
  1455d0:	73 c7                	jae    145599 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x769>
  1455d2:	48 89 c2             	mov    %rax,%rdx
  1455d5:	48 29 f2             	sub    %rsi,%rdx
  1455d8:	0f 82 69 0b 00 00    	jb     146147 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1317>
  1455de:	48 39 fa             	cmp    %rdi,%rdx
  1455e1:	0f 82 68 0b 00 00    	jb     14614f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x131f>
  1455e7:	48 83 f9 11          	cmp    $0x11,%rcx
  1455eb:	72 06                	jb     1455f3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c3>
  1455ed:	48 8b 6d 08          	mov    0x8(%rbp),%rbp
  1455f1:	eb 04                	jmp    1455f7 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7c7>
  1455f3:	48 83 c5 08          	add    $0x8,%rbp
  1455f7:	48 01 fe             	add    %rdi,%rsi
  1455fa:	48 01 ee             	add    %rbp,%rsi
  1455fd:	48 29 fa             	sub    %rdi,%rdx
  145600:	49 89 36             	mov    %rsi,(%r14)
  145603:	49 89 56 08          	mov    %rdx,0x8(%r14)
  145607:	49 c7 46 10 00 00 00 	movq   $0x0,0x10(%r14)
  14560e:	00 
  14560f:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  145613:	0f 84 aa 00 00 00    	je     1456c3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x893>
  145619:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  145620:	00 
  145621:	41 f6 c1 7f          	test   $0x7f,%r9b
  145625:	0f 85 de fd ff ff    	jne    145409 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
  14562b:	48 83 fa 07          	cmp    $0x7,%rdx
  14562f:	0f 86 ff 0a 00 00    	jbe    146134 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1304>
  145635:	4c 8b 26             	mov    (%rsi),%r12
  145638:	4d 89 20             	mov    %r12,(%r8)
  14563b:	48 83 c2 f8          	add    $0xfffffffffffffff8,%rdx
  14563f:	bf 08 00 00 00       	mov    $0x8,%edi
  145644:	48 01 fe             	add    %rdi,%rsi
  145647:	49 89 36             	mov    %rsi,(%r14)
  14564a:	49 89 56 08          	mov    %rdx,0x8(%r14)
  14564e:	49 ff c1             	inc    %r9
  145651:	4d 89 4e 18          	mov    %r9,0x18(%r14)
  145655:	48 8b 84 24 50 01 00 	mov    0x150(%rsp),%rax
  14565c:	00 
  14565d:	4c 3b 60 10          	cmp    0x10(%rax),%r12
  145661:	0f 83 84 0a 00 00    	jae    1460eb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x12bb>
  145667:	48 8b 84 24 58 01 00 	mov    0x158(%rsp),%rax
  14566e:	00 
  14566f:	48 8b 48 10          	mov    0x10(%rax),%rcx
  145673:	83 38 01             	cmpl   $0x1,(%rax)
  145676:	0f 85 14 01 00 00    	jne    145790 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x960>
  14567c:	49 39 cc             	cmp    %rcx,%r12
  14567f:	0f 83 1b 01 00 00    	jae    1457a0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
  145685:	4a 8d 0c a5 00 00 00 	lea    0x0(,%r12,4),%rcx
  14568c:	00 
  14568d:	e9 07 01 00 00       	jmp    145799 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x969>
  145692:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145699:	1f 84 00 00 00 00 00 
  1456a0:	be 01 00 00 00       	mov    $0x1,%esi
  1456a5:	31 d2                	xor    %edx,%edx
  1456a7:	49 89 36             	mov    %rsi,(%r14)
  1456aa:	49 89 56 08          	mov    %rdx,0x8(%r14)
  1456ae:	66 0f ef c0          	pxor   %xmm0,%xmm0
  1456b2:	f3 41 0f 7f 00       	movdqu %xmm0,(%r8)
  1456b7:	4d 89 cc             	mov    %r9,%r12
  1456ba:	48 85 d2             	test   %rdx,%rdx
  1456bd:	0f 85 36 03 00 00    	jne    1459f9 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbc9>
  1456c3:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  1456ca:	4d 39 dc             	cmp    %r11,%r12
  1456cd:	0f 84 60 01 00 00    	je     145833 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa03>
  1456d3:	4d 8d 4c 24 20       	lea    0x20(%r12),%r9
  1456d8:	4d 89 4e 40          	mov    %r9,0x40(%r14)
  1456dc:	49 8b 04 24          	mov    (%r12),%rax
  1456e0:	48 ff c8             	dec    %rax
  1456e3:	74 bb                	je     1456a0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x870>
  1456e5:	89 c1                	mov    %eax,%ecx
  1456e7:	83 e1 7f             	and    $0x7f,%ecx
  1456ea:	48 c1 e8 07          	shr    $0x7,%rax
  1456ee:	48 83 f9 01          	cmp    $0x1,%rcx
  1456f2:	48 83 d8 ff          	sbb    $0xffffffffffffffff,%rax
  1456f6:	bf 08 00 00 00       	mov    $0x8,%edi
  1456fb:	48 83 f8 02          	cmp    $0x2,%rax
  1456ff:	73 2f                	jae    145730 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x900>
  145701:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
  145706:	48 89 c8             	mov    %rcx,%rax
  145709:	48 83 f9 11          	cmp    $0x11,%rcx
  14570d:	72 4a                	jb     145759 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x929>
  14570f:	49 8b 44 24 10       	mov    0x10(%r12),%rax
  145714:	48 89 c2             	mov    %rax,%rdx
  145717:	48 29 fa             	sub    %rdi,%rdx
  14571a:	73 49                	jae    145765 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x935>
  14571c:	e9 3f 09 00 00       	jmp    146060 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1230>
  145721:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145728:	0f 1f 84 00 00 00 00 
  14572f:	00 
  145730:	48 ff c8             	dec    %rax
  145733:	48 0f bd c8          	bsr    %rax,%rcx
  145737:	f7 d1                	not    %ecx
  145739:	48 c7 c0 ff ff ff ff 	mov    $0xffffffffffffffff,%rax
  145740:	48 d3 e8             	shr    %cl,%rax
  145743:	48 8d 3c c5 10 00 00 	lea    0x10(,%rax,8),%rdi
  14574a:	00 
  14574b:	49 8b 4c 24 18       	mov    0x18(%r12),%rcx
  145750:	48 89 c8             	mov    %rcx,%rax
  145753:	48 83 f9 11          	cmp    $0x11,%rcx
  145757:	73 b6                	jae    14570f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x8df>
  145759:	48 89 c2             	mov    %rax,%rdx
  14575c:	48 29 fa             	sub    %rdi,%rdx
  14575f:	0f 82 fb 08 00 00    	jb     146060 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1230>
  145765:	48 83 f9 11          	cmp    $0x11,%rcx
  145769:	72 15                	jb     145780 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x950>
  14576b:	49 8b 74 24 08       	mov    0x8(%r12),%rsi
  145770:	eb 15                	jmp    145787 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x957>
  145772:	66 66 66 66 66 2e 0f 	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145779:	1f 84 00 00 00 00 00 
  145780:	49 83 c4 08          	add    $0x8,%r12
  145784:	4c 89 e6             	mov    %r12,%rsi
  145787:	48 01 fe             	add    %rdi,%rsi
  14578a:	e9 18 ff ff ff       	jmp    1456a7 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x877>
  14578f:	90                   	nop
  145790:	49 39 cc             	cmp    %rcx,%r12
  145793:	73 0b                	jae    1457a0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x970>
  145795:	4b 8d 0c 24          	lea    (%r12,%r12,1),%rcx
  145799:	48 03 48 08          	add    0x8(%rax),%rcx
  14579d:	0f 18 09             	prefetcht0 (%rcx)
  1457a0:	bd 01 00 00 00       	mov    $0x1,%ebp
  1457a5:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
  1457a9:	48 39 d8             	cmp    %rbx,%rax
  1457ac:	0f 86 ce 00 00 00    	jbe    145880 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa50>
  1457b2:	48 89 d8             	mov    %rbx,%rax
  1457b5:	4c 29 f8             	sub    %r15,%rax
  1457b8:	48 39 c5             	cmp    %rax,%rbp
  1457bb:	0f 87 52 02 00 00    	ja     145a13 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xbe3>
  1457c1:	4d 89 ee             	mov    %r13,%r14
  1457c4:	49 89 dd             	mov    %rbx,%r13
  1457c7:	49 39 c6             	cmp    %rax,%r14
  1457ca:	76 62                	jbe    14582e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  1457cc:	48 89 da             	mov    %rbx,%rdx
  1457cf:	4c 29 f2             	sub    %r14,%rdx
  1457d2:	4c 89 f8             	mov    %r15,%rax
  1457d5:	48 29 d0             	sub    %rdx,%rax
  1457d8:	48 39 c2             	cmp    %rax,%rdx
  1457db:	76 26                	jbe    145803 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
  1457dd:	4c 89 e9             	mov    %r13,%rcx
  1457e0:	48 29 d9             	sub    %rbx,%rcx
  1457e3:	48 39 c1             	cmp    %rax,%rcx
  1457e6:	72 1b                	jb     145803 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9d3>
  1457e8:	48 8b b4 24 30 01 00 	mov    0x130(%rsp),%rsi
  1457ef:	00 
  1457f0:	48 8d 3c de          	lea    (%rsi,%rbx,8),%rdi
  1457f4:	48 c1 e0 03          	shl    $0x3,%rax
  1457f8:	48 89 c2             	mov    %rax,%rdx
  1457fb:	ff 15 97 e3 01 00    	call   *0x1e397(%rip)        # 163b98 <memcpy@GLIBC_2.14>
  145801:	eb 2b                	jmp    14582e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  145803:	4c 89 f1             	mov    %r14,%rcx
  145806:	4d 89 ee             	mov    %r13,%r14
  145809:	49 29 d6             	sub    %rdx,%r14
  14580c:	48 8b 84 24 30 01 00 	mov    0x130(%rsp),%rax
  145813:	00 
  145814:	48 8d 34 c8          	lea    (%rax,%rcx,8),%rsi
  145818:	4a 8d 3c f0          	lea    (%rax,%r14,8),%rdi
  14581c:	48 c1 e2 03          	shl    $0x3,%rdx
  145820:	ff 15 7a e3 01 00    	call   *0x1e37a(%rip)        # 163ba0 <memmove@GLIBC_2.2.5>
  145826:	4c 89 b4 24 38 01 00 	mov    %r14,0x138(%rsp)
  14582d:	00 
  14582e:	4c 89 eb             	mov    %r13,%rbx
  145831:	eb 50                	jmp    145883 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa53>
  145833:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  14583a:	00 
  14583b:	49 83 7e 20 00       	cmpq   $0x0,0x20(%r14)
  145840:	74 20                	je     145862 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xa32>
  145842:	49 83 c6 20          	add    $0x20,%r14
  145846:	4c 89 f7             	mov    %r14,%rdi
  145849:	e8 72 d1 ff ff       	call   1429c0 <_RNvXs_NtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe9positions5codecNtB4_6CursorNtNtNtNtCsgxBkk5gSRhY_4core4iter6traits8iterator8Iterator4next>
  14584e:	49 89 d4             	mov    %rdx,%r12
  145851:	48 83 f8 01          	cmp    $0x1,%rax
  145855:	0f 84 fa fd ff ff    	je     145655 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x825>
  14585b:	49 c7 06 00 00 00 00 	movq   $0x0,(%r14)
  145862:	31 ed                	xor    %ebp,%ebp
  145864:	49 8d 04 2f          	lea    (%r15,%rbp,1),%rax
  145868:	48 39 d8             	cmp    %rbx,%rax
  14586b:	0f 87 41 ff ff ff    	ja     1457b2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x982>
  145871:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  145878:	0f 1f 84 00 00 00 00 
  14587f:	00 
  145880:	4d 89 ee             	mov    %r13,%r14
  145883:	4b 8d 04 37          	lea    (%r15,%r14,1),%rax
  145887:	48 39 d8             	cmp    %rbx,%rax
  14588a:	b9 00 00 00 00       	mov    $0x0,%ecx
  14588f:	48 0f 43 cb          	cmovae %rbx,%rcx
  145893:	48 29 c8             	sub    %rcx,%rax
  145896:	48 89 d9             	mov    %rbx,%rcx
  145899:	48 29 c1             	sub    %rax,%rcx
  14589c:	4d 89 f5             	mov    %r14,%r13
  14589f:	48 39 e9             	cmp    %rbp,%rcx
  1458a2:	73 2c                	jae    1458d0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xaa0>
  1458a4:	48 8b 8c 24 30 01 00 	mov    0x130(%rsp),%rcx
  1458ab:	00 
  1458ac:	48 39 c3             	cmp    %rax,%rbx
  1458af:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
  1458b4:	75 3a                	jne    1458f0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xac0>
  1458b6:	4c 89 21             	mov    %r12,(%rcx)
  1458b9:	41 be 01 00 00 00    	mov    $0x1,%r14d
  1458bf:	eb 47                	jmp    145908 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
  1458c1:	66 66 66 66 66 66 2e 	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1458c8:	0f 1f 84 00 00 00 00 
  1458cf:	00 
  1458d0:	48 83 fd 01          	cmp    $0x1,%rbp
  1458d4:	75 2a                	jne    145900 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad0>
  1458d6:	48 8b 8c 24 30 01 00 	mov    0x130(%rsp),%rcx
  1458dd:	00 
  1458de:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
  1458e2:	41 be 01 00 00 00    	mov    $0x1,%r14d
  1458e8:	eb 19                	jmp    145903 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad3>
  1458ea:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  1458f0:	4c 89 24 c1          	mov    %r12,(%rcx,%rax,8)
  1458f4:	41 be 01 00 00 00    	mov    $0x1,%r14d
  1458fa:	eb 0c                	jmp    145908 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xad8>
  1458fc:	0f 1f 40 00          	nopl   0x0(%rax)
  145900:	45 31 f6             	xor    %r14d,%r14d
  145903:	48 8b 6c 24 30       	mov    0x30(%rsp),%rbp
  145908:	4d 01 fe             	add    %r15,%r14
  14590b:	4c 89 b4 24 40 01 00 	mov    %r14,0x140(%rsp)
  145912:	00 
  145913:	4c 8b 8c 24 d0 00 00 	mov    0xd0(%rsp),%r9
  14591a:	00 
  14591b:	49 8b 01             	mov    (%r9),%rax
  14591e:	49 8b 71 10          	mov    0x10(%r9),%rsi
  145922:	48 83 f8 01          	cmp    $0x1,%rax
  145926:	75 18                	jne    145940 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb10>
  145928:	48 39 f5             	cmp    %rsi,%rbp
  14592b:	0f 83 80 08 00 00    	jae    1461b1 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1381>
  145931:	49 8b 49 08          	mov    0x8(%r9),%rcx
  145935:	8b 14 a9             	mov    (%rcx,%rbp,4),%edx
  145938:	eb 26                	jmp    145960 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb30>
  14593a:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  145940:	48 39 f5             	cmp    %rsi,%rbp
  145943:	0f 83 5c 08 00 00    	jae    1461a5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1375>
  145949:	49 8b 49 08          	mov    0x8(%r9),%rcx
  14594d:	0f b7 14 69          	movzwl (%rcx,%rbp,2),%edx
  145951:	66 83 fa ff          	cmp    $0xffff,%dx
  145955:	0f b7 d2             	movzwl %dx,%edx
  145958:	bf ff ff ff ff       	mov    $0xffffffff,%edi
  14595d:	0f 44 d7             	cmove  %edi,%edx
  145960:	4c 8b 84 24 98 00 00 	mov    0x98(%rsp),%r8
  145967:	00 
  145968:	4c 39 44 24 20       	cmp    %r8,0x20(%rsp)
  14596d:	40 0f 94 c7          	sete   %dil
  145971:	4d 85 c0             	test   %r8,%r8
  145974:	41 0f 94 c0          	sete   %r8b
  145978:	3b 94 24 c8 00 00 00 	cmp    0xc8(%rsp),%edx
  14597f:	0f 95 c2             	setne  %dl
  145982:	41 08 f8             	or     %dil,%r8b
  145985:	41 08 d0             	or     %dl,%r8b
  145988:	0f 85 e2 f9 ff ff    	jne    145370 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  14598e:	48 8b 54 24 20       	mov    0x20(%rsp),%rdx
  145993:	4c 8d 24 2a          	lea    (%rdx,%rbp,1),%r12
  145997:	49 39 f4             	cmp    %rsi,%r12
  14599a:	0f 83 d0 f9 ff ff    	jae    145370 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  1459a0:	48 8b 94 24 98 00 00 	mov    0x98(%rsp),%rdx
  1459a7:	00 
  1459a8:	48 8d 3c 2a          	lea    (%rdx,%rbp,1),%rdi
  1459ac:	84 c0                	test   %al,%al
  1459ae:	74 1e                	je     1459ce <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xb9e>
  1459b0:	48 39 f7             	cmp    %rsi,%rdi
  1459b3:	0f 83 fb 07 00 00    	jae    1461b4 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
  1459b9:	8b 04 b9             	mov    (%rcx,%rdi,4),%eax
  1459bc:	3b 84 24 c0 00 00 00 	cmp    0xc0(%rsp),%eax
  1459c3:	0f 85 a7 f9 ff ff    	jne    145370 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  1459c9:	e9 96 00 00 00       	jmp    145a64 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
  1459ce:	48 39 f7             	cmp    %rsi,%rdi
  1459d1:	0f 83 d1 07 00 00    	jae    1461a8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1378>
  1459d7:	0f b7 04 79          	movzwl (%rcx,%rdi,2),%eax
  1459db:	66 83 f8 ff          	cmp    $0xffff,%ax
  1459df:	0f b7 c0             	movzwl %ax,%eax
  1459e2:	b9 ff ff ff ff       	mov    $0xffffffff,%ecx
  1459e7:	0f 44 c1             	cmove  %ecx,%eax
  1459ea:	3b 84 24 c0 00 00 00 	cmp    0xc0(%rsp),%eax
  1459f1:	0f 85 79 f9 ff ff    	jne    145370 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x540>
  1459f7:	eb 6b                	jmp    145a64 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc34>
  1459f9:	45 31 c9             	xor    %r9d,%r9d
  1459fc:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  145a03:	00 
  145a04:	41 f6 c1 7f          	test   $0x7f,%r9b
  145a08:	0f 85 fb f9 ff ff    	jne    145409 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x5d9>
  145a0e:	e9 18 fc ff ff       	jmp    14562b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x7fb>
  145a13:	ba 01 00 00 00       	mov    $0x1,%edx
  145a18:	b9 08 00 00 00       	mov    $0x8,%ecx
  145a1d:	41 b8 08 00 00 00    	mov    $0x8,%r8d
  145a23:	48 8d bc 24 28 01 00 	lea    0x128(%rsp),%rdi
  145a2a:	00 
  145a2b:	4c 89 fe             	mov    %r15,%rsi
  145a2e:	e8 0d 19 fe ff       	call   127340 <_RINvNvMs2_NtCs6i54tJFfzR_5alloc7raw_vecINtB8_11RawVecInnerpE7reserve21do_reserve_and_handleNtNtBa_5alloc6GlobalECsigKge7u8hP3_8tk_train>
  145a33:	4c 8b ac 24 28 01 00 	mov    0x128(%rsp),%r13
  145a3a:	00 
  145a3b:	4c 8b b4 24 38 01 00 	mov    0x138(%rsp),%r14
  145a42:	00 
  145a43:	4c 8b bc 24 40 01 00 	mov    0x140(%rsp),%r15
  145a4a:	00 
  145a4b:	48 89 d8             	mov    %rbx,%rax
  145a4e:	4c 29 f8             	sub    %r15,%rax
  145a51:	4c 89 6c 24 50       	mov    %r13,0x50(%rsp)
  145a56:	49 39 c6             	cmp    %rax,%r14
  145a59:	0f 87 6d fd ff ff    	ja     1457cc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x99c>
  145a5f:	e9 ca fd ff ff       	jmp    14582e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x9fe>
  145a64:	83 bc 24 e0 00 00 00 	cmpl   $0x1,0xe0(%rsp)
  145a6b:	01 
  145a6c:	75 33                	jne    145aa1 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc71>
  145a6e:	48 3b ac 24 f0 00 00 	cmp    0xf0(%rsp),%rbp
  145a75:	00 
  145a76:	48 8b 5c 24 70       	mov    0x70(%rsp),%rbx
  145a7b:	73 35                	jae    145ab2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc82>
  145a7d:	48 8d 8c 24 28 01 00 	lea    0x128(%rsp),%rcx
  145a84:	00 
  145a85:	48 8b 41 d0          	mov    -0x30(%rcx),%rax
  145a89:	48 89 84 24 b0 01 00 	mov    %rax,0x1b0(%rsp)
  145a90:	00 
  145a91:	f3 0f 6f 41 c0       	movdqu -0x40(%rcx),%xmm0
  145a96:	66 0f 7f 84 24 a0 01 	movdqa %xmm0,0x1a0(%rsp)
  145a9d:	00 00 
  145a9f:	eb 34                	jmp    145ad5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xca5>
  145aa1:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
  145aa8:	00 
  145aa9:	31 c9                	xor    %ecx,%ecx
  145aab:	48 8b 5c 24 70       	mov    0x70(%rsp),%rbx
  145ab0:	eb 13                	jmp    145ac5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xc95>
  145ab2:	48 8b 8c 24 f8 00 00 	mov    0xf8(%rsp),%rcx
  145ab9:	00 
  145aba:	48 8b b4 24 20 01 00 	mov    0x120(%rsp),%rsi
  145ac1:	00 
  145ac2:	48 ff c1             	inc    %rcx
  145ac5:	48 8d bc 24 a0 01 00 	lea    0x1a0(%rsp),%rdi
  145acc:	00 
  145acd:	48 89 ea             	mov    %rbp,%rdx
  145ad0:	e8 db 76 00 00       	call   14d1b0 <_RNvMs_NtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpusNtB4_6Corpus6weight>
  145ad5:	48 c7 84 24 e0 00 00 	movq   $0x1,0xe0(%rsp)
  145adc:	00 01 00 00 00 
  145ae1:	48 8b 84 24 b0 01 00 	mov    0x1b0(%rsp),%rax
  145ae8:	00 
  145ae9:	48 8d 8c 24 28 01 00 	lea    0x128(%rsp),%rcx
  145af0:	00 
  145af1:	48 89 41 d0          	mov    %rax,-0x30(%rcx)
  145af5:	66 0f 6f 84 24 a0 01 	movdqa 0x1a0(%rsp),%xmm0
  145afc:	00 00 
  145afe:	f3 0f 7f 41 c0       	movdqu %xmm0,-0x40(%rcx)
  145b03:	4c 8b 8c 24 a0 01 00 	mov    0x1a0(%rsp),%r9
  145b0a:	00 
  145b0b:	48 8d 4d ff          	lea    -0x1(%rbp),%rcx
  145b0f:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145b14:	48 8b 38             	mov    (%rax),%rdi
  145b17:	48 8b 70 10          	mov    0x10(%rax),%rsi
  145b1b:	48 83 ff 01          	cmp    $0x1,%rdi
  145b1f:	75 37                	jne    145b58 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd28>
  145b21:	48 39 f1             	cmp    %rsi,%rcx
  145b24:	0f 83 f2 06 00 00    	jae    14621c <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13ec>
  145b2a:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145b2f:	48 8b 42 08          	mov    0x8(%rdx),%rax
  145b33:	8b 4c a8 fc          	mov    -0x4(%rax,%rbp,4),%ecx
  145b37:	83 f9 ff             	cmp    $0xffffffff,%ecx
  145b3a:	75 41                	jne    145b7d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd4d>
  145b3c:	49 39 f4             	cmp    %rsi,%r12
  145b3f:	0f 83 e1 06 00 00    	jae    146226 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f6>
  145b45:	46 8b 14 a0          	mov    (%rax,%r12,4),%r10d
  145b49:	41 83 fa ff          	cmp    $0xffffffff,%r10d
  145b4d:	0f 85 5f 02 00 00    	jne    145db2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf82>
  145b53:	e9 05 04 00 00       	jmp    145f5d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x112d>
  145b58:	48 39 f1             	cmp    %rsi,%rcx
  145b5b:	0f 83 c0 06 00 00    	jae    146221 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f1>
  145b61:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145b66:	48 8b 40 08          	mov    0x8(%rax),%rax
  145b6a:	0f b7 4c 68 fe       	movzwl -0x2(%rax,%rbp,2),%ecx
  145b6f:	66 83 f9 ff          	cmp    $0xffff,%cx
  145b73:	74 50                	je     145bc5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xd95>
  145b75:	0f b7 c9             	movzwl %cx,%ecx
  145b78:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145b7d:	89 c9                	mov    %ecx,%ecx
  145b7f:	48 8b 42 58          	mov    0x58(%rdx),%rax
  145b83:	48 39 c8             	cmp    %rcx,%rax
  145b86:	0f 86 77 06 00 00    	jbe    146203 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d3>
  145b8c:	48 8b 42 50          	mov    0x50(%rdx),%rax
  145b90:	49 89 cb             	mov    %rcx,%r11
  145b93:	48 8b 1c c8          	mov    (%rax,%rcx,8),%rbx
  145b97:	49 89 e8             	mov    %rbp,%r8
  145b9a:	49 29 d8             	sub    %rbx,%r8
  145b9d:	0f 84 62 01 00 00    	je     145d05 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xed5>
  145ba3:	49 8d 40 ff          	lea    -0x1(%r8),%rax
  145ba7:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
  145bac:	74 34                	je     145be2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdb2>
  145bae:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145bb3:	0f 83 7a 06 00 00    	jae    146233 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1403>
  145bb9:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  145bbe:	46 8b 54 80 fc       	mov    -0x4(%rax,%r8,4),%r10d
  145bc3:	eb 44                	jmp    145c09 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xdd9>
  145bc5:	49 39 f4             	cmp    %rsi,%r12
  145bc8:	0f 83 5d 06 00 00    	jae    14622b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13fb>
  145bce:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
  145bd3:	66 83 f8 ff          	cmp    $0xffff,%ax
  145bd7:	0f 85 d1 01 00 00    	jne    145dae <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf7e>
  145bdd:	e9 7b 03 00 00       	jmp    145f5d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x112d>
  145be2:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145be7:	0f 83 53 06 00 00    	jae    146240 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1410>
  145bed:	48 8b 44 24 78       	mov    0x78(%rsp),%rax
  145bf2:	42 0f b7 44 40 fe    	movzwl -0x2(%rax,%r8,2),%eax
  145bf8:	66 83 f8 ff          	cmp    $0xffff,%ax
  145bfc:	44 0f b7 d0          	movzwl %ax,%r10d
  145c00:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  145c05:	44 0f 44 d0          	cmove  %eax,%r10d
  145c09:	48 83 bc 24 70 01 00 	cmpq   $0x0,0x170(%rsp)
  145c10:	00 00 
  145c12:	0f 84 ed 00 00 00    	je     145d05 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xed5>
  145c18:	44 89 d0             	mov    %r10d,%eax
  145c1b:	48 33 84 24 68 01 00 	xor    0x168(%rsp),%rax
  145c22:	00 
  145c23:	48 b9 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rcx
  145c2a:	f4 51 58 
  145c2d:	48 f7 e1             	mul    %rcx
  145c30:	4c 31 da             	xor    %r11,%rdx
  145c33:	48 31 d0             	xor    %rdx,%rax
  145c36:	48 f7 e1             	mul    %rcx
  145c39:	48 89 c1             	mov    %rax,%rcx
  145c3c:	48 31 d1             	xor    %rdx,%rcx
  145c3f:	48 89 c8             	mov    %rcx,%rax
  145c42:	48 f7 a4 24 60 01 00 	mulq   0x160(%rsp)
  145c49:	00 
  145c4a:	48 31 c2             	xor    %rax,%rdx
  145c4d:	48 d3 c2             	rol    %cl,%rdx
  145c50:	48 89 d0             	mov    %rdx,%rax
  145c53:	48 c1 e8 39          	shr    $0x39,%rax
  145c57:	66 0f 6e c0          	movd   %eax,%xmm0
  145c5b:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
  145c5f:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
  145c64:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
  145c69:	31 c0                	xor    %eax,%eax
  145c6b:	48 23 94 24 80 00 00 	and    0x80(%rsp),%rdx
  145c72:	00 
  145c73:	48 8b 4c 24 70       	mov    0x70(%rsp),%rcx
  145c78:	f3 0f 6f 0c 11       	movdqu (%rcx,%rdx,1),%xmm1
  145c7d:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  145c81:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
  145c85:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
  145c89:	85 c9                	test   %ecx,%ecx
  145c8b:	74 58                	je     145ce5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xeb5>
  145c8d:	44 89 54 24 30       	mov    %r10d,0x30(%rsp)
  145c92:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  145c97:	4d 89 cf             	mov    %r9,%r15
  145c9a:	44 89 d0             	mov    %r10d,%eax
  145c9d:	f3 44 0f bc d1       	tzcnt  %ecx,%r10d
  145ca2:	49 01 d2             	add    %rdx,%r10
  145ca5:	4c 23 94 24 80 00 00 	and    0x80(%rsp),%r10
  145cac:	00 
  145cad:	49 f7 da             	neg    %r10
  145cb0:	4f 8d 14 52          	lea    (%r10,%r10,2),%r10
  145cb4:	4c 8b 4c 24 70       	mov    0x70(%rsp),%r9
  145cb9:	43 3b 44 91 f4       	cmp    -0xc(%r9,%r10,4),%eax
  145cbe:	75 0b                	jne    145ccb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe9b>
  145cc0:	47 3b 5c 91 f8       	cmp    -0x8(%r9,%r10,4),%r11d
  145cc5:	0f 84 b7 00 00 00    	je     145d82 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf52>
  145ccb:	44 8d 51 ff          	lea    -0x1(%rcx),%r10d
  145ccf:	66 41 21 ca          	and    %cx,%r10w
  145cd3:	44 89 d1             	mov    %r10d,%ecx
  145cd6:	4d 89 f9             	mov    %r15,%r9
  145cd9:	44 8b 54 24 30       	mov    0x30(%rsp),%r10d
  145cde:	48 8b 44 24 20       	mov    0x20(%rsp),%rax
  145ce3:	75 b2                	jne    145c97 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe67>
  145ce5:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
  145ce9:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
  145ced:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
  145cf1:	85 c9                	test   %ecx,%ecx
  145cf3:	75 10                	jne    145d05 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xed5>
  145cf5:	48 01 c2             	add    %rax,%rdx
  145cf8:	48 83 c2 10          	add    $0x10,%rdx
  145cfc:	48 83 c0 10          	add    $0x10,%rax
  145d00:	e9 66 ff ff ff       	jmp    145c6b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xe3b>
  145d05:	4c 89 e0             	mov    %r12,%rax
  145d08:	48 29 e8             	sub    %rbp,%rax
  145d0b:	b9 00 00 00 00       	mov    $0x0,%ecx
  145d10:	48 0f 43 c8          	cmovae %rax,%rcx
  145d14:	48 01 cb             	add    %rcx,%rbx
  145d17:	31 c0                	xor    %eax,%eax
  145d19:	48 3b 9c 24 90 00 00 	cmp    0x90(%rsp),%rbx
  145d20:	00 
  145d21:	0f 92 c0             	setb   %al
  145d24:	89 04 24             	mov    %eax,(%rsp)
  145d27:	48 8d bc 24 d8 01 00 	lea    0x1d8(%rsp),%rdi
  145d2e:	00 
  145d2f:	31 f6                	xor    %esi,%esi
  145d31:	44 89 da             	mov    %r11d,%edx
  145d34:	44 89 d9             	mov    %r11d,%ecx
  145d37:	4d 89 cf             	mov    %r9,%r15
  145d3a:	e8 01 77 00 00       	call   14d440 <_RNvMs_NtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
  145d3f:	48 8b 5c 24 70       	mov    0x70(%rsp),%rbx
  145d44:	48 85 c0             	test   %rax,%rax
  145d47:	0f 85 32 03 00 00    	jne    14607f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x124f>
  145d4d:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d52:	48 8b 38             	mov    (%rax),%rdi
  145d55:	48 8b 70 10          	mov    0x10(%rax),%rsi
  145d59:	48 85 ff             	test   %rdi,%rdi
  145d5c:	4d 89 f9             	mov    %r15,%r9
  145d5f:	74 2c                	je     145d8d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf5d>
  145d61:	49 39 f4             	cmp    %rsi,%r12
  145d64:	0f 83 bc 04 00 00    	jae    146226 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13f6>
  145d6a:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d6f:	48 8b 40 08          	mov    0x8(%rax),%rax
  145d73:	46 8b 14 a0          	mov    (%rax,%r12,4),%r10d
  145d77:	41 83 fa ff          	cmp    $0xffffffff,%r10d
  145d7b:	75 35                	jne    145db2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf82>
  145d7d:	e9 db 01 00 00       	jmp    145f5d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x112d>
  145d82:	4c 89 cb             	mov    %r9,%rbx
  145d85:	48 85 ff             	test   %rdi,%rdi
  145d88:	4d 89 f9             	mov    %r15,%r9
  145d8b:	75 d4                	jne    145d61 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xf31>
  145d8d:	49 39 f4             	cmp    %rsi,%r12
  145d90:	0f 83 95 04 00 00    	jae    14622b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13fb>
  145d96:	48 8b 44 24 28       	mov    0x28(%rsp),%rax
  145d9b:	48 8b 40 08          	mov    0x8(%rax),%rax
  145d9f:	42 0f b7 04 60       	movzwl (%rax,%r12,2),%eax
  145da4:	66 83 f8 ff          	cmp    $0xffff,%ax
  145da8:	0f 84 af 01 00 00    	je     145f5d <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x112d>
  145dae:	44 0f b7 d0          	movzwl %ax,%r10d
  145db2:	44 89 d7             	mov    %r10d,%edi
  145db5:	48 39 bc 24 88 00 00 	cmp    %rdi,0x88(%rsp)
  145dbc:	00 
  145dbd:	0f 86 48 04 00 00    	jbe    14620b <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13db>
  145dc3:	48 8b 84 24 80 01 00 	mov    0x180(%rsp),%rax
  145dca:	00 
  145dcb:	48 8b 04 f8          	mov    (%rax,%rdi,8),%rax
  145dcf:	4c 01 e0             	add    %r12,%rax
  145dd2:	80 7c 24 17 00       	cmpb   $0x0,0x17(%rsp)
  145dd7:	0f 84 e1 00 00 00    	je     145ebe <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x108e>
  145ddd:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145de2:	0f 83 4b 04 00 00    	jae    146233 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1403>
  145de8:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
  145ded:	8b 34 81             	mov    (%rcx,%rax,4),%esi
  145df0:	48 83 bc 24 70 01 00 	cmpq   $0x0,0x170(%rsp)
  145df7:	00 00 
  145df9:	0f 84 f1 00 00 00    	je     145ef0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10c0>
  145dff:	48 8b 84 24 68 01 00 	mov    0x168(%rsp),%rax
  145e06:	00 
  145e07:	49 89 f8             	mov    %rdi,%r8
  145e0a:	48 31 f8             	xor    %rdi,%rax
  145e0d:	48 bf 2d 7f 95 4c 2d 	movabs $0x5851f42d4c957f2d,%rdi
  145e14:	f4 51 58 
  145e17:	48 f7 e7             	mul    %rdi
  145e1a:	48 31 d0             	xor    %rdx,%rax
  145e1d:	89 f1                	mov    %esi,%ecx
  145e1f:	48 31 c8             	xor    %rcx,%rax
  145e22:	48 f7 e7             	mul    %rdi
  145e25:	48 89 c1             	mov    %rax,%rcx
  145e28:	48 31 d1             	xor    %rdx,%rcx
  145e2b:	48 89 c8             	mov    %rcx,%rax
  145e2e:	48 f7 a4 24 60 01 00 	mulq   0x160(%rsp)
  145e35:	00 
  145e36:	48 31 c2             	xor    %rax,%rdx
  145e39:	48 d3 c2             	rol    %cl,%rdx
  145e3c:	48 89 d0             	mov    %rdx,%rax
  145e3f:	48 c1 e8 39          	shr    $0x39,%rax
  145e43:	66 0f 6e c0          	movd   %eax,%xmm0
  145e47:	66 0f 60 c0          	punpcklbw %xmm0,%xmm0
  145e4b:	f2 0f 70 c0 00       	pshuflw $0x0,%xmm0,%xmm0
  145e50:	66 0f 70 c0 44       	pshufd $0x44,%xmm0,%xmm0
  145e55:	31 c0                	xor    %eax,%eax
  145e57:	48 23 94 24 80 00 00 	and    0x80(%rsp),%rdx
  145e5e:	00 
  145e5f:	f3 0f 6f 0c 13       	movdqu (%rbx,%rdx,1),%xmm1
  145e64:	66 0f 6f d1          	movdqa %xmm1,%xmm2
  145e68:	66 0f 74 d0          	pcmpeqb %xmm0,%xmm2
  145e6c:	66 0f d7 ca          	pmovmskb %xmm2,%ecx
  145e70:	85 c9                	test   %ecx,%ecx
  145e72:	74 2d                	je     145ea1 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1071>
  145e74:	f3 0f bc f9          	tzcnt  %ecx,%edi
  145e78:	48 01 d7             	add    %rdx,%rdi
  145e7b:	48 23 bc 24 80 00 00 	and    0x80(%rsp),%rdi
  145e82:	00 
  145e83:	48 f7 df             	neg    %rdi
  145e86:	48 8d 3c 7f          	lea    (%rdi,%rdi,2),%rdi
  145e8a:	44 3b 54 bb f4       	cmp    -0xc(%rbx,%rdi,4),%r10d
  145e8f:	75 06                	jne    145e97 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1067>
  145e91:	3b 74 bb f8          	cmp    -0x8(%rbx,%rdi,4),%esi
  145e95:	74 5e                	je     145ef5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10c5>
  145e97:	8d 79 ff             	lea    -0x1(%rcx),%edi
  145e9a:	66 21 cf             	and    %cx,%di
  145e9d:	89 f9                	mov    %edi,%ecx
  145e9f:	75 d3                	jne    145e74 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1044>
  145ea1:	66 0f 76 d2          	pcmpeqd %xmm2,%xmm2
  145ea5:	66 0f 74 ca          	pcmpeqb %xmm2,%xmm1
  145ea9:	66 0f d7 c9          	pmovmskb %xmm1,%ecx
  145ead:	85 c9                	test   %ecx,%ecx
  145eaf:	75 4f                	jne    145f00 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10d0>
  145eb1:	48 01 c2             	add    %rax,%rdx
  145eb4:	48 83 c2 10          	add    $0x10,%rdx
  145eb8:	48 83 c0 10          	add    $0x10,%rax
  145ebc:	eb 99                	jmp    145e57 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1027>
  145ebe:	48 3b 44 24 48       	cmp    0x48(%rsp),%rax
  145ec3:	0f 83 77 03 00 00    	jae    146240 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1410>
  145ec9:	48 8b 4c 24 78       	mov    0x78(%rsp),%rcx
  145ece:	0f b7 04 41          	movzwl (%rcx,%rax,2),%eax
  145ed2:	66 83 f8 ff          	cmp    $0xffff,%ax
  145ed6:	0f b7 f0             	movzwl %ax,%esi
  145ed9:	b8 ff ff ff ff       	mov    $0xffffffff,%eax
  145ede:	0f 44 f0             	cmove  %eax,%esi
  145ee1:	48 83 bc 24 70 01 00 	cmpq   $0x0,0x170(%rsp)
  145ee8:	00 00 
  145eea:	0f 85 0f ff ff ff    	jne    145dff <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0xfcf>
  145ef0:	44 89 d1             	mov    %r10d,%ecx
  145ef3:	eb 11                	jmp    145f06 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10d6>
  145ef5:	48 8d 04 bb          	lea    (%rbx,%rdi,4),%rax
  145ef9:	8b 78 fc             	mov    -0x4(%rax),%edi
  145efc:	89 f9                	mov    %edi,%ecx
  145efe:	eb 06                	jmp    145f06 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x10d6>
  145f00:	44 89 d1             	mov    %r10d,%ecx
  145f03:	4c 89 c7             	mov    %r8,%rdi
  145f06:	48 8b 54 24 28       	mov    0x28(%rsp),%rdx
  145f0b:	48 8b 72 58          	mov    0x58(%rdx),%rsi
  145f0f:	48 39 fe             	cmp    %rdi,%rsi
  145f12:	0f 86 fb 02 00 00    	jbe    146213 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13e3>
  145f18:	49 29 ec             	sub    %rbp,%r12
  145f1b:	b8 00 00 00 00       	mov    $0x0,%eax
  145f20:	49 0f 43 c4          	cmovae %r12,%rax
  145f24:	48 8b 52 50          	mov    0x50(%rdx),%rdx
  145f28:	48 03 04 fa          	add    (%rdx,%rdi,8),%rax
  145f2c:	31 d2                	xor    %edx,%edx
  145f2e:	48 3b 84 24 90 00 00 	cmp    0x90(%rsp),%rax
  145f35:	00 
  145f36:	0f 92 c2             	setb   %dl
  145f39:	89 14 24             	mov    %edx,(%rsp)
  145f3c:	48 8d bc 24 d8 01 00 	lea    0x1d8(%rsp),%rdi
  145f43:	00 
  145f44:	be 01 00 00 00       	mov    $0x1,%esi
  145f49:	44 89 d2             	mov    %r10d,%edx
  145f4c:	49 89 e8             	mov    %rbp,%r8
  145f4f:	e8 ec 74 00 00       	call   14d440 <_RNvMs_NtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6record>
  145f54:	48 85 c0             	test   %rax,%rax
  145f57:	0f 85 22 01 00 00    	jne    14607f <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x124f>
  145f5d:	48 8b 44 24 68       	mov    0x68(%rsp),%rax
  145f62:	48 83 f8 03          	cmp    $0x3,%rax
  145f66:	72 26                	jb     145f8e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x115e>
  145f68:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  145f6d:	48 39 c1             	cmp    %rax,%rcx
  145f70:	74 3e                	je     145fb0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1180>
  145f72:	48 8b 44 24 58       	mov    0x58(%rsp),%rax
  145f77:	48 89 2c c8          	mov    %rbp,(%rax,%rcx,8)
  145f7b:	48 ff c1             	inc    %rcx
  145f7e:	48 89 4c 24 60       	mov    %rcx,0x60(%rsp)
  145f83:	4d 85 f6             	test   %r14,%r14
  145f86:	0f 85 86 f3 ff ff    	jne    145312 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
  145f8c:	eb 4e                	jmp    145fdc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ac>
  145f8e:	48 8d 4c 24 58       	lea    0x58(%rsp),%rcx
  145f93:	48 8d 54 24 68       	lea    0x68(%rsp),%rdx
  145f98:	48 83 f8 02          	cmp    $0x2,%rax
  145f9c:	74 23                	je     145fc1 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1191>
  145f9e:	48 89 2c c1          	mov    %rbp,(%rcx,%rax,8)
  145fa2:	48 ff 02             	incq   (%rdx)
  145fa5:	4d 85 f6             	test   %r14,%r14
  145fa8:	0f 85 64 f3 ff ff    	jne    145312 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x4e2>
  145fae:	eb 2c                	jmp    145fdc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11ac>
  145fb0:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
  145fb5:	e8 36 79 00 00       	call   14d8f0 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsigKge7u8hP3_8tk_train>
  145fba:	48 8b 4c 24 60       	mov    0x60(%rsp),%rcx
  145fbf:	eb b1                	jmp    145f72 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1142>
  145fc1:	48 8d 7c 24 58       	lea    0x58(%rsp),%rdi
  145fc6:	e8 25 79 00 00       	call   14d8f0 <_RNvMsc_Cs6eo6YbMmMB_8smallvecINtB5_8SmallVecAjj2_E21reserve_one_uncheckedCsigKge7u8hP3_8tk_train>
  145fcb:	48 8b 4c 24 58       	mov    0x58(%rsp),%rcx
  145fd0:	48 8b 44 24 60       	mov    0x60(%rsp),%rax
  145fd5:	48 8d 54 24 60       	lea    0x60(%rsp),%rdx
  145fda:	eb c2                	jmp    145f9e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x116e>
  145fdc:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  145fe2:	74 0e                	je     145ff2 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x11c2>
  145fe4:	48 8b bc 24 30 01 00 	mov    0x130(%rsp),%rdi
  145feb:	00 
  145fec:	ff 15 9e db 01 00    	call   *0x1db9e(%rip)        # 163b90 <free@GLIBC_2.2.5>
  145ff2:	4c 8b 74 24 58       	mov    0x58(%rsp),%r14
  145ff7:	0f 10 44 24 60       	movups 0x60(%rsp),%xmm0
  145ffc:	0f 29 44 24 30       	movaps %xmm0,0x30(%rsp)
  146001:	48 8d bc 24 e0 00 00 	lea    0xe0(%rsp),%rdi
  146008:	00 
  146009:	48 8d b4 24 d8 01 00 	lea    0x1d8(%rsp),%rsi
  146010:	00 
  146011:	e8 fa 7a 00 00       	call   14db10 <_RNvMs_NtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB4_15NeighborChanges6finish>
  146016:	48 8b 84 24 f0 00 00 	mov    0xf0(%rsp),%rax
  14601d:	00 
  14601e:	48 8b 8c 24 78 01 00 	mov    0x178(%rsp),%rcx
  146025:	00 
  146026:	48 89 41 10          	mov    %rax,0x10(%rcx)
  14602a:	0f 10 84 24 e0 00 00 	movups 0xe0(%rsp),%xmm0
  146031:	00 
  146032:	0f 11 01             	movups %xmm0,(%rcx)
  146035:	4c 89 71 18          	mov    %r14,0x18(%rcx)
  146039:	66 0f 6f 44 24 30    	movdqa 0x30(%rsp),%xmm0
  14603f:	f3 0f 7f 41 20       	movdqu %xmm0,0x20(%rcx)
  146044:	8b 84 24 dc 00 00 00 	mov    0xdc(%rsp),%eax
  14604b:	89 41 30             	mov    %eax,0x30(%rcx)
  14604e:	48 81 c4 88 02 00 00 	add    $0x288,%rsp
  146055:	5b                   	pop    %rbx
  146056:	41 5c                	pop    %r12
  146058:	41 5d                	pop    %r13
  14605a:	41 5e                	pop    %r14
  14605c:	41 5f                	pop    %r15
  14605e:	5d                   	pop    %rbp
  14605f:	c3                   	ret
  146060:	49 c7 46 50 00 00 00 	movq   $0x0,0x50(%r14)
  146067:	00 
  146068:	48 89 c2             	mov    %rax,%rdx
  14606b:	48 8d 0d 0e d2 01 00 	lea    0x1d20e(%rip),%rcx        # 163280 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55a0>
  146072:	48 89 c6             	mov    %rax,%rsi
  146075:	e8 16 ec f3 ff       	call   84c90 <_RNvNtNtCsgxBkk5gSRhY_4core5slice5index16slice_index_fail>
  14607a:	e9 82 01 00 00       	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  14607f:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  146085:	4c 8b b4 24 78 01 00 	mov    0x178(%rsp),%r14
  14608c:	00 
  14608d:	74 14                	je     1460a3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1273>
  14608f:	48 8b bc 24 30 01 00 	mov    0x130(%rsp),%rdi
  146096:	00 
  146097:	48 89 c3             	mov    %rax,%rbx
  14609a:	ff 15 f0 da 01 00    	call   *0x1daf0(%rip)        # 163b90 <free@GLIBC_2.2.5>
  1460a0:	48 89 d8             	mov    %rbx,%rax
  1460a3:	49 89 46 08          	mov    %rax,0x8(%r14)
  1460a7:	48 8d 05 82 ca 01 00 	lea    0x1ca82(%rip),%rax        # 162b30 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e50>
  1460ae:	49 89 46 10          	mov    %rax,0x10(%r14)
  1460b2:	49 c7 06 ff ff ff ff 	movq   $0xffffffffffffffff,(%r14)
  1460b9:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
  1460bf:	72 0b                	jb     1460cc <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x129c>
  1460c1:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  1460c6:	ff 15 c4 da 01 00    	call   *0x1dac4(%rip)        # 163b90 <free@GLIBC_2.2.5>
  1460cc:	48 8d bc 24 d8 01 00 	lea    0x1d8(%rsp),%rdi
  1460d3:	00 
  1460d4:	e8 17 5c ff ff       	call   13bcf0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
  1460d9:	48 8d bc 24 f0 01 00 	lea    0x1f0(%rsp),%rdi
  1460e0:	00 
  1460e1:	e8 0a 5c ff ff       	call   13bcf0 <_RINvNtCsgxBkk5gSRhY_4core3ptr9drop_glueINtNtCs6i54tJFfzR_5alloc3vec3VecINtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus10PairChangeNtNtB1c_9positions9PositionsEEEB1g_>
  1460e6:	e9 63 ff ff ff       	jmp    14604e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x121e>
  1460eb:	48 8d 3d 1e ea ed ff 	lea    -0x1215e2(%rip),%rdi        # 24b10 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x7aa>
  1460f2:	48 8d 15 17 d5 01 00 	lea    0x1d517(%rip),%rdx        # 163610 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5930>
  1460f9:	be 53 00 00 00       	mov    $0x53,%esi
  1460fe:	e8 8d cc f3 ff       	call   82d90 <_RNvNtCsgxBkk5gSRhY_4core9panicking9panic_fmt>
  146103:	e9 f9 00 00 00       	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  146108:	48 89 d0             	mov    %rdx,%rax
  14610b:	48 8d 0d 3e d7 01 00 	lea    0x1d73e(%rip),%rcx        # 163850 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5b70>
  146112:	e9 5b ff ff ff       	jmp    146072 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1242>
  146117:	48 8d 3d b0 e5 ed ff 	lea    -0x121a50(%rip),%rdi        # 246ce <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x368>
  14611e:	48 8d 15 ab d2 01 00 	lea    0x1d2ab(%rip),%rdx        # 1633d0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x56f0>
  146125:	be 11 00 00 00       	mov    $0x11,%esi
  14612a:	e8 e1 fe f3 ff       	call   86010 <_RNvNtCsgxBkk5gSRhY_4core6option13expect_failed>
  14612f:	e9 cd 00 00 00       	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  146134:	b8 08 00 00 00       	mov    $0x8,%eax
  146139:	31 ff                	xor    %edi,%edi
  14613b:	48 8d 0d d6 d5 01 00 	lea    0x1d5d6(%rip),%rcx        # 163718 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5a38>
  146142:	e9 2b ff ff ff       	jmp    146072 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1242>
  146147:	48 89 f7             	mov    %rsi,%rdi
  14614a:	e9 19 ff ff ff       	jmp    146068 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1238>
  14614f:	48 89 d0             	mov    %rdx,%rax
  146152:	48 8d 0d 6f d1 01 00 	lea    0x1d16f(%rip),%rcx        # 1632c8 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x55e8>
  146159:	e9 14 ff ff ff       	jmp    146072 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1242>
  14615e:	bf 08 00 00 00       	mov    $0x8,%edi
  146163:	48 8d 0d 76 d1 01 00 	lea    0x1d176(%rip),%rcx        # 1632e0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5600>
  14616a:	e9 03 ff ff ff       	jmp    146072 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1242>
  14616f:	31 c0                	xor    %eax,%eax
  146171:	eb 02                	jmp    146175 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1345>
  146173:	b0 01                	mov    $0x1,%al
  146175:	88 84 24 a0 01 00 00 	mov    %al,0x1a0(%rsp)
  14617c:	48 8d 3d 83 eb ed ff 	lea    -0x12147d(%rip),%rdi        # 24d06 <_RNvNtNtNtCsgxBkk5gSRhY_4core7unicode12unicode_data11white_space14WHITESPACE_MAP+0x9a0>
  146183:	48 8d 0d 66 ce 01 00 	lea    0x1ce66(%rip),%rcx        # 162ff0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5310>
  14618a:	4c 8d 05 a7 d6 01 00 	lea    0x1d6a7(%rip),%r8        # 163838 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x5b58>
  146191:	48 8d 94 24 a0 01 00 	lea    0x1a0(%rsp),%rdx
  146198:	00 
  146199:	be 13 00 00 00       	mov    $0x13,%esi
  14619e:	e8 ad fe f3 ff       	call   86050 <_RNvNtCsgxBkk5gSRhY_4core6result13unwrap_failed>
  1461a3:	eb 5c                	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  1461a5:	48 89 ef             	mov    %rbp,%rdi
  1461a8:	48 8d 05 f9 cf 01 00 	lea    0x1cff9(%rip),%rax        # 1631a8 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x54c8>
  1461af:	eb 0a                	jmp    1461bb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x138b>
  1461b1:	48 89 ef             	mov    %rbp,%rdi
  1461b4:	48 8d 05 05 d0 01 00 	lea    0x1d005(%rip),%rax        # 1631c0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x54e0>
  1461bb:	48 89 44 24 18       	mov    %rax,0x18(%rsp)
  1461c0:	48 8b 54 24 18       	mov    0x18(%rsp),%rdx
  1461c5:	e8 61 db f3 ff       	call   83d2b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
  1461ca:	eb 35                	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  1461cc:	49 c1 e9 07          	shr    $0x7,%r9
  1461d0:	31 f6                	xor    %esi,%esi
  1461d2:	4c 89 cf             	mov    %r9,%rdi
  1461d5:	eb e9                	jmp    1461c0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1390>
  1461d7:	48 89 d7             	mov    %rdx,%rdi
  1461da:	eb e4                	jmp    1461c0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1390>
  1461dc:	4c 89 ff             	mov    %r15,%rdi
  1461df:	eb 03                	jmp    1461e4 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13b4>
  1461e1:	4c 89 ef             	mov    %r13,%rdi
  1461e4:	48 8d 15 f5 c8 01 00 	lea    0x1c8f5(%rip),%rdx        # 162ae0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e00>
  1461eb:	e8 3b db f3 ff       	call   83d2b <_RNvNtCsgxBkk5gSRhY_4core9panicking18panic_bounds_check>
  1461f0:	eb 0f                	jmp    146201 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13d1>
  1461f2:	bf 08 00 00 00       	mov    $0x8,%edi
  1461f7:	be 80 00 00 00       	mov    $0x80,%esi
  1461fc:	e8 31 6e f3 ff       	call   7d032 <_RNvNtCs6i54tJFfzR_5alloc7raw_vec12handle_error>
  146201:	0f 0b                	ud2
  146203:	48 89 cf             	mov    %rcx,%rdi
  146206:	48 89 c6             	mov    %rax,%rsi
  146209:	eb 08                	jmp    146213 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x13e3>
  14620b:	48 8b b4 24 88 00 00 	mov    0x88(%rsp),%rsi
  146212:	00 
  146213:	48 8d 05 c6 c8 01 00 	lea    0x1c8c6(%rip),%rax        # 162ae0 <_RNvNtNtCsdbNqGqeMao8_9bitcannon8classify11atom_tables11ATOM_TABLES+0x4e00>
  14621a:	eb 9f                	jmp    1461bb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x138b>
  14621c:	48 89 cf             	mov    %rcx,%rdi
  14621f:	eb 93                	jmp    1461b4 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
  146221:	48 89 cf             	mov    %rcx,%rdi
  146224:	eb 82                	jmp    1461a8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1378>
  146226:	4c 89 e7             	mov    %r12,%rdi
  146229:	eb 89                	jmp    1461b4 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
  14622b:	4c 89 e7             	mov    %r12,%rdi
  14622e:	e9 75 ff ff ff       	jmp    1461a8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1378>
  146233:	48 89 c7             	mov    %rax,%rdi
  146236:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  14623b:	e9 74 ff ff ff       	jmp    1461b4 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1384>
  146240:	48 89 c7             	mov    %rax,%rdi
  146243:	48 8b 74 24 48       	mov    0x48(%rsp),%rsi
  146248:	e9 5b ff ff ff       	jmp    1461a8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1378>
  14624d:	eb 59                	jmp    1462a8 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1478>
  14624f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  146254:	66 0f 70 44 24 30 ee 	pshufd $0xee,0x30(%rsp),%xmm0
  14625b:	66 48 0f 7e c0       	movq   %xmm0,%rax
  146260:	48 83 f8 02          	cmp    $0x2,%rax
  146264:	0f 87 b9 00 00 00    	ja     146323 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14f3>
  14626a:	e9 bd 00 00 00       	jmp    14632c <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14fc>
  14626f:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  146274:	eb 4d                	jmp    1462c3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1493>
  146276:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  14627b:	48 8b 84 24 28 01 00 	mov    0x128(%rsp),%rax
  146282:	00 
  146283:	48 89 44 24 50       	mov    %rax,0x50(%rsp)
  146288:	eb 23                	jmp    1462ad <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x147d>
  14628a:	eb 02                	jmp    14628e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x145e>
  14628c:	eb 00                	jmp    14628e <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x145e>
  14628e:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  146293:	48 83 bc 24 a0 00 00 	cmpq   $0x0,0xa0(%rsp)
  14629a:	00 00 
  14629c:	74 25                	je     1462c3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1493>
  14629e:	48 8b bc 24 a8 00 00 	mov    0xa8(%rsp),%rdi
  1462a5:	00 
  1462a6:	eb 15                	jmp    1462bd <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x148d>
  1462a8:	48 89 44 24 20       	mov    %rax,0x20(%rsp)
  1462ad:	48 83 7c 24 50 00    	cmpq   $0x0,0x50(%rsp)
  1462b3:	74 0e                	je     1462c3 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1493>
  1462b5:	48 8b bc 24 30 01 00 	mov    0x130(%rsp),%rdi
  1462bc:	00 
  1462bd:	ff 15 cd d8 01 00    	call   *0x1d8cd(%rip)        # 163b90 <free@GLIBC_2.2.5>
  1462c3:	48 83 7c 24 68 03    	cmpq   $0x3,0x68(%rsp)
  1462c9:	72 0b                	jb     1462d6 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14a6>
  1462cb:	48 8b 7c 24 58       	mov    0x58(%rsp),%rdi
  1462d0:	ff 15 ba d8 01 00    	call   *0x1d8ba(%rip)        # 163b90 <free@GLIBC_2.2.5>
  1462d6:	4c 8b b4 24 e0 01 00 	mov    0x1e0(%rsp),%r14
  1462dd:	00 
  1462de:	4c 8b a4 24 e8 01 00 	mov    0x1e8(%rsp),%r12
  1462e5:	00 
  1462e6:	4d 85 e4             	test   %r12,%r12
  1462e9:	75 4b                	jne    146336 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1506>
  1462eb:	48 83 bc 24 d8 01 00 	cmpq   $0x0,0x1d8(%rsp)
  1462f2:	00 00 
  1462f4:	74 09                	je     1462ff <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14cf>
  1462f6:	4c 89 f7             	mov    %r14,%rdi
  1462f9:	ff 15 91 d8 01 00    	call   *0x1d891(%rip)        # 163b90 <free@GLIBC_2.2.5>
  1462ff:	4c 8b b4 24 f8 01 00 	mov    0x1f8(%rsp),%r14
  146306:	00 
  146307:	4c 8b a4 24 00 02 00 	mov    0x200(%rsp),%r12
  14630e:	00 
  14630f:	4d 85 e4             	test   %r12,%r12
  146312:	0f 85 b1 00 00 00    	jne    1463c9 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1599>
  146318:	48 83 bc 24 f0 01 00 	cmpq   $0x0,0x1f0(%rsp)
  14631f:	00 00 
  146321:	74 09                	je     14632c <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14fc>
  146323:	4c 89 f7             	mov    %r14,%rdi
  146326:	ff 15 64 d8 01 00    	call   *0x1d864(%rip)        # 163b90 <free@GLIBC_2.2.5>
  14632c:	48 8b 7c 24 20       	mov    0x20(%rsp),%rdi
  146331:	e8 fa 5d 01 00       	call   15c130 <_Unwind_Resume@plt>
  146336:	45 31 ed             	xor    %r13d,%r13d
  146339:	48 8b 2d 50 d8 01 00 	mov    0x1d850(%rip),%rbp        # 163b90 <free@GLIBC_2.2.5>
  146340:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
  146345:	eb 1b                	jmp    146362 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1532>
  146347:	66 0f 1f 84 00 00 00 	nopw   0x0(%rax,%rax,1)
  14634e:	00 00 
  146350:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
  146355:	4c 89 ff             	mov    %r15,%rdi
  146358:	ff d5                	call   *%rbp
  14635a:	49 ff c5             	inc    %r13
  14635d:	4d 39 e5             	cmp    %r12,%r13
  146360:	74 89                	je     1462eb <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14bb>
  146362:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
  146369:	00 
  14636a:	4c 01 e9             	add    %r13,%rcx
  14636d:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
  146371:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
  146376:	74 18                	je     146390 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1560>
  146378:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
  14637d:	72 db                	jb     14635a <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x152a>
  14637f:	4c 8b 78 08          	mov    0x8(%rax),%r15
  146383:	eb d0                	jmp    146355 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1525>
  146385:	66 66 2e 0f 1f 84 00 	data16 cs nopw 0x0(%rax,%rax,1)
  14638c:	00 00 00 00 
  146390:	48 8b 58 10          	mov    0x10(%rax),%rbx
  146394:	48 85 db             	test   %rbx,%rbx
  146397:	74 c1                	je     14635a <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x152a>
  146399:	4c 8b 78 08          	mov    0x8(%rax),%r15
  14639d:	4d 8d 67 18          	lea    0x18(%r15),%r12
  1463a1:	eb 16                	jmp    1463b9 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1589>
  1463a3:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  1463aa:	84 00 00 00 00 00 
  1463b0:	49 83 c4 20          	add    $0x20,%r12
  1463b4:	48 ff cb             	dec    %rbx
  1463b7:	74 97                	je     146350 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1520>
  1463b9:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
  1463be:	72 f0                	jb     1463b0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1580>
  1463c0:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
  1463c5:	ff d5                	call   *%rbp
  1463c7:	eb e7                	jmp    1463b0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1580>
  1463c9:	45 31 ed             	xor    %r13d,%r13d
  1463cc:	48 8b 2d bd d7 01 00 	mov    0x1d7bd(%rip),%rbp        # 163b90 <free@GLIBC_2.2.5>
  1463d3:	4c 89 64 24 30       	mov    %r12,0x30(%rsp)
  1463d8:	eb 1c                	jmp    1463f6 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15c6>
  1463da:	66 0f 1f 44 00 00    	nopw   0x0(%rax,%rax,1)
  1463e0:	4c 8b 64 24 30       	mov    0x30(%rsp),%r12
  1463e5:	4c 89 ff             	mov    %r15,%rdi
  1463e8:	ff d5                	call   *%rbp
  1463ea:	49 ff c5             	inc    %r13
  1463ed:	4d 39 e5             	cmp    %r12,%r13
  1463f0:	0f 84 22 ff ff ff    	je     146318 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x14e8>
  1463f6:	4a 8d 0c ed 00 00 00 	lea    0x0(,%r13,8),%rcx
  1463fd:	00 
  1463fe:	4c 01 e9             	add    %r13,%rcx
  146401:	49 8d 04 ce          	lea    (%r14,%rcx,8),%rax
  146405:	49 83 3c ce 00       	cmpq   $0x0,(%r14,%rcx,8)
  14640a:	74 14                	je     146420 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15f0>
  14640c:	48 83 78 18 11       	cmpq   $0x11,0x18(%rax)
  146411:	72 d7                	jb     1463ea <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15ba>
  146413:	4c 8b 78 08          	mov    0x8(%rax),%r15
  146417:	eb cc                	jmp    1463e5 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15b5>
  146419:	0f 1f 80 00 00 00 00 	nopl   0x0(%rax)
  146420:	48 8b 58 10          	mov    0x10(%rax),%rbx
  146424:	48 85 db             	test   %rbx,%rbx
  146427:	74 c1                	je     1463ea <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15ba>
  146429:	4c 8b 78 08          	mov    0x8(%rax),%r15
  14642d:	4d 8d 67 18          	lea    0x18(%r15),%r12
  146431:	eb 16                	jmp    146449 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1619>
  146433:	66 66 66 66 2e 0f 1f 	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
  14643a:	84 00 00 00 00 00 
  146440:	49 83 c4 20          	add    $0x20,%r12
  146444:	48 ff cb             	dec    %rbx
  146447:	74 97                	je     1463e0 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x15b0>
  146449:	49 83 3c 24 11       	cmpq   $0x11,(%r12)
  14644e:	72 f0                	jb     146440 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1610>
  146450:	49 8b 7c 24 f0       	mov    -0x10(%r12),%rdi
  146455:	ff d5                	call   *%rbp
  146457:	eb e7                	jmp    146440 <_RNCNvMNtNtNtNtCsigKge7u8hP3_8tk_train8trainers3bpe6corpus5mergeNtB6_6Corpus13prepare_freshs2_0Bc_+0x1610>

Disassembly of section .init:

Disassembly of section .fini:

Disassembly of section .plt:
