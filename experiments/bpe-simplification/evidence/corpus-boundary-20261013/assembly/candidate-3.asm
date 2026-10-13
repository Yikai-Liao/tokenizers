
/tmp/bpe-corpus-binaries-20261013/final-candidate-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000dbce0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next>:
   dbce0:	push   %rbp
   dbce1:	push   %r15
   dbce3:	push   %r14
   dbce5:	push   %r13
   dbce7:	push   %r12
   dbce9:	push   %rbx
   dbcea:	sub    $0x3e8,%rsp
   dbcf1:	mov    %rcx,0x20(%rsp)
   dbcf6:	mov    %rdx,%r12
   dbcf9:	mov    %rdi,0x2a0(%rsp)
   dbd01:	mov    %r8,0xa0(%rsp)
   dbd09:	movzbl 0x30(%r8),%ebp
   dbd0e:	movq   $0x0,0x2a8(%rsp)
   dbd1a:	movq   $0x8,0x2b0(%rsp)
   dbd26:	movq   $0x0,0x2b8(%rsp)
   dbd32:	mov    $0x1,%r13d
   dbd38:	test   %bpl,%bpl
   dbd3b:	jne    dbd53 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x73>
   dbd3d:	sub    0x30(%r12),%rsi
   dbd42:	cmp    $0x100,%rsi
   dbd49:	mov    $0x100,%r13d
   dbd4f:	cmovb  %rsi,%r13
   dbd53:	mov    0x2deee(%rip),%rax        # 109c48 <ahash::random_state::RAND_SOURCE>
   dbd5a:	test   %rax,%rax
   dbd5d:	je     e0597 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48b7>
   dbd63:	mov    (%rax),%r14
   dbd66:	mov    0x8(%rax),%r15
   dbd6a:	mov    0x2dedf(%rip),%rbx        # 109c50 <ahash::random_state::get_fixed_seeds::SEEDS>
   dbd71:	test   %rbx,%rbx
   dbd74:	je     e05a1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48c1>
   dbd7a:	mov    %r14,%rdi
   dbd7d:	call   *0x18(%r15)
   dbd81:	xor    0x8(%rbx),%rax
   dbd85:	movabs $0x5851f42d4c957f2d,%r15
   dbd8f:	mul    %r15
   dbd92:	mov    %rax,%rdi
   dbd95:	xor    %rdx,%rdi
   dbd98:	mov    (%rbx),%r10
   dbd9b:	mov    0x20(%rbx),%r11
   dbd9f:	mov    0x30(%rbx),%r9
   dbda3:	mov    %r11,%rax
   dbda6:	xor    %rdi,%rax
   dbda9:	mul    %r15
   dbdac:	xor    %r9,%rdx
   dbdaf:	xor    %rdx,%rax
   dbdb2:	mul    %r15
   dbdb5:	mov    %rax,%rcx
   dbdb8:	xor    %rdx,%rcx
   dbdbb:	mov    %rcx,%rax
   dbdbe:	mul    %r10
   dbdc1:	mov    %rdx,%rsi
   dbdc4:	xor    %rax,%rsi
   dbdc7:	rol    %cl,%rsi
   dbdca:	mov    0x28(%rbx),%r14
   dbdce:	mov    0x38(%rbx),%rbx
   dbdd2:	mov    %r14,%rax
   dbdd5:	xor    %rdi,%rax
   dbdd8:	mul    %r15
   dbddb:	xor    %rbx,%rdx
   dbdde:	xor    %rdx,%rax
   dbde1:	mul    %r15
   dbde4:	mov    %rax,%rcx
   dbde7:	xor    %rdx,%rcx
   dbdea:	mov    %rcx,%rax
   dbded:	mul    %r10
   dbdf0:	mov    %rdx,%r8
   dbdf3:	xor    %rax,%r8
   dbdf6:	rol    %cl,%r8
   dbdf9:	xor    %rdi,%r9
   dbdfc:	mov    %r9,%rax
   dbdff:	mul    %r15
   dbe02:	xor    %rdx,%r14
   dbe05:	xor    %r14,%rax
   dbe08:	mul    %r15
   dbe0b:	mov    %rax,%rcx
   dbe0e:	xor    %rdx,%rcx
   dbe11:	mov    %rcx,%rax
   dbe14:	mul    %r10
   dbe17:	mov    %rdx,%r9
   dbe1a:	xor    %rax,%r9
   dbe1d:	rol    %cl,%r9
   dbe20:	xor    %rbx,%rdi
   dbe23:	mov    %rdi,%rax
   dbe26:	mul    %r15
   dbe29:	xor    %rdx,%r11
   dbe2c:	xor    %r11,%rax
   dbe2f:	mul    %r15
   dbe32:	mov    %rax,%rcx
   dbe35:	xor    %rdx,%rcx
   dbe38:	mov    %rcx,%rax
   dbe3b:	mul    %r10
   dbe3e:	xor    %rax,%rdx
   dbe41:	rol    %cl,%rdx
   dbe44:	movaps 0x2aa45(%rip),%xmm0        # 106890 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   dbe4b:	movaps %xmm0,0x370(%rsp)
   dbe53:	movdqa 0x2aa25(%rip),%xmm0        # 106880 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   dbe5b:	movdqa %xmm0,0x360(%rsp)
   dbe64:	mov    %rsi,0x380(%rsp)
   dbe6c:	mov    %r8,0x388(%rsp)
   dbe74:	mov    %r9,0x390(%rsp)
   dbe7c:	mov    %rdx,0x398(%rsp)
   dbe84:	mov    0x2ddbd(%rip),%rax        # 109c48 <ahash::random_state::RAND_SOURCE>
   dbe8b:	test   %rax,%rax
   dbe8e:	je     e05ae <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48ce>
   dbe94:	mov    (%rax),%r14
   dbe97:	mov    0x8(%rax),%r15
   dbe9b:	mov    0x2ddae(%rip),%rbx        # 109c50 <ahash::random_state::get_fixed_seeds::SEEDS>
   dbea2:	test   %rbx,%rbx
   dbea5:	je     e05b8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48d8>
   dbeab:	mov    %r14,%rdi
   dbeae:	call   *0x18(%r15)
   dbeb2:	mov    %r12,0x300(%rsp)
   dbeba:	xor    0x8(%rbx),%rax
   dbebe:	movabs $0x5851f42d4c957f2d,%r15
   dbec8:	mul    %r15
   dbecb:	mov    %rax,%rdi
   dbece:	xor    %rdx,%rdi
   dbed1:	mov    (%rbx),%r10
   dbed4:	mov    0x20(%rbx),%r11
   dbed8:	mov    0x30(%rbx),%r9
   dbedc:	mov    %r11,%rax
   dbedf:	xor    %rdi,%rax
   dbee2:	mul    %r15
   dbee5:	xor    %r9,%rdx
   dbee8:	xor    %rdx,%rax
   dbeeb:	mul    %r15
   dbeee:	mov    %rax,%rcx
   dbef1:	xor    %rdx,%rcx
   dbef4:	mov    %rcx,%rax
   dbef7:	mul    %r10
   dbefa:	mov    %rdx,%rsi
   dbefd:	xor    %rax,%rsi
   dbf00:	rol    %cl,%rsi
   dbf03:	mov    0x28(%rbx),%r14
   dbf07:	mov    0x38(%rbx),%rbx
   dbf0b:	mov    %r14,%rax
   dbf0e:	xor    %rdi,%rax
   dbf11:	mul    %r15
   dbf14:	xor    %rbx,%rdx
   dbf17:	xor    %rdx,%rax
   dbf1a:	mul    %r15
   dbf1d:	mov    %rax,%rcx
   dbf20:	xor    %rdx,%rcx
   dbf23:	mov    %rcx,%rax
   dbf26:	mul    %r10
   dbf29:	mov    %rdx,%r8
   dbf2c:	xor    %rax,%r8
   dbf2f:	rol    %cl,%r8
   dbf32:	xor    %rdi,%r9
   dbf35:	mov    %r9,%rax
   dbf38:	mul    %r15
   dbf3b:	xor    %rdx,%r14
   dbf3e:	xor    %r14,%rax
   dbf41:	mul    %r15
   dbf44:	mov    %rax,%rcx
   dbf47:	xor    %rdx,%rcx
   dbf4a:	mov    %rcx,%rax
   dbf4d:	mul    %r10
   dbf50:	mov    %rdx,%r9
   dbf53:	xor    %rax,%r9
   dbf56:	rol    %cl,%r9
   dbf59:	xor    %rbx,%rdi
   dbf5c:	mov    %rdi,%rax
   dbf5f:	mul    %r15
   dbf62:	xor    %rdx,%r11
   dbf65:	xor    %r11,%rax
   dbf68:	mul    %r15
   dbf6b:	mov    %rax,%rcx
   dbf6e:	xor    %rdx,%rcx
   dbf71:	mov    %rcx,%rax
   dbf74:	mul    %r10
   dbf77:	xor    %rax,%rdx
   dbf7a:	rol    %cl,%rdx
   dbf7d:	movaps 0x2a90c(%rip),%xmm0        # 106890 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   dbf84:	movaps %xmm0,0x3b0(%rsp)
   dbf8c:	movdqa 0x2a8ec(%rip),%xmm0        # 106880 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   dbf94:	movdqa %xmm0,0x3a0(%rsp)
   dbf9d:	mov    %rsi,0x3c0(%rsp)
   dbfa5:	mov    %r8,0x3c8(%rsp)
   dbfad:	mov    %r9,0x3d0(%rsp)
   dbfb5:	mov    %rdx,0x3d8(%rsp)
   dbfbd:	mov    0x2b8(%rsp),%r15
   dbfc5:	cmp    %r13,%r15
   dbfc8:	jae    dcfa1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12c1>
   dbfce:	movabs $0x7ffffffffffffff1,%rcx
   dbfd8:	movabs $0xbe5466cf34e90c6c,%r12
   dbfe2:	mov    0x20(%rsp),%rax
   dbfe7:	add    $0x40,%rax
   dbfeb:	mov    %rax,0x1a0(%rsp)
   dbff3:	lea    0x2ae2e(%rip),%rax        # 106e28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3930>
   dbffa:	mov    %rax,0x2f8(%rsp)
   dc002:	lea    0x30(%rsp),%rax
   dc007:	mov    %rax,0x290(%rsp)
   dc00f:	add    $0x7,%rcx
   dc013:	mov    %rcx,0x238(%rsp)
   dc01b:	lea    0x2a76e(%rip),%rax        # 106790 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   dc022:	mov    %rax,0x110(%rsp)
   dc02a:	lea    0x2ae3f(%rip),%rax        # 106e70 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3978>
   dc031:	mov    %rax,0x288(%rsp)
   dc039:	lea    0x2b088(%rip),%rax        # 1070c8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3bd0>
   dc040:	mov    %rax,0x138(%rsp)
   dc048:	lea    -0xbddfe(%rip),%rax        # 1e251 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x504>
   dc04f:	mov    %rax,0x198(%rsp)
   dc057:	mov    %bpl,0x118(%rsp)
   dc05f:	mov    %r13,0x250(%rsp)
   dc067:	mov    %r15,0xa8(%rsp)
   dc06f:	mov    0xa0(%rsp),%rax
   dc077:	mov    0x10(%rax),%r15
   dc07b:	test   %r15,%r15
   dc07e:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc084:	mov    0x20(%rax),%rdi
   dc088:	test   %rdi,%rdi
   dc08b:	je     e096f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c8f>
   dc091:	mov    0xa0(%rsp),%rax
   dc099:	mov    0x8(%rax),%rbp
   dc09d:	jmp    dc0c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e8>
   dc09f:	nop
   dc0a0:	mov    %r14,%rdi
   dc0a3:	call   *%r15
   dc0a6:	mov    0xa0(%rsp),%rax
   dc0ae:	mov    0x10(%rax),%r15
   dc0b2:	test   %r15,%r15
   dc0b5:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc0bb:	mov    0x20(%rax),%rdi
   dc0bf:	test   %rdi,%rdi
   dc0c2:	je     e096f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c8f>
   dc0c8:	mov    0x28(%rbp),%r14d
   dc0cc:	mov    0x2c(%rbp),%ebx
   dc0cf:	mov    %r14,%rax
   dc0d2:	xor    %r12,%rax
   dc0d5:	movabs $0x5851f42d4c957f2d,%rcx
   dc0df:	mul    %rcx
   dc0e2:	xor    %rbx,%rdx
   dc0e5:	xor    %rdx,%rax
   dc0e8:	mul    %rcx
   dc0eb:	mov    %rax,%rcx
   dc0ee:	xor    %rdx,%rcx
   dc0f1:	mov    %rcx,%rax
   dc0f4:	movabs $0x452821e638d01377,%rdx
   dc0fe:	mul    %rdx
   dc101:	mov    %rdx,%r13
   dc104:	xor    %rax,%r13
   dc107:	rol    %cl,%r13
   dc10a:	mov    0x20(%rbp),%rsi
   dc10e:	mov    %r13,%rax
   dc111:	or     %rdi,%rax
   dc114:	shr    $0x20,%rax
   dc118:	je     dc130 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x450>
   dc11a:	mov    %r13,%rax
   dc11d:	xor    %edx,%edx
   dc11f:	div    %rdi
   dc122:	jmp    dc137 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x457>
   dc124:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   dc130:	mov    %r13d,%eax
   dc133:	xor    %edx,%edx
   dc135:	div    %edi
   dc137:	mov    0xa0(%rsp),%rax
   dc13f:	mov    0x18(%rax),%rdi
   dc143:	imul   $0x70,%rdx,%rax
   dc147:	cmpq   $0x0,0x48(%rdi,%rax,1)
   dc14d:	je     dc220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x540>
   dc153:	add    %rax,%rdi
   dc156:	mov    0x58(%rdi),%rax
   dc15a:	xor    %r14,%rax
   dc15d:	movabs $0x5851f42d4c957f2d,%rcx
   dc167:	mul    %rcx
   dc16a:	xor    %rbx,%rdx
   dc16d:	xor    %rdx,%rax
   dc170:	mul    %rcx
   dc173:	mov    %rax,%rcx
   dc176:	xor    %rdx,%rcx
   dc179:	mov    %rcx,%rax
   dc17c:	mulq   0x50(%rdi)
   dc180:	xor    %rax,%rdx
   dc183:	rol    %cl,%rdx
   dc186:	mov    %rdx,%r8
   dc189:	shr    $0x39,%r8
   dc18d:	mov    0x30(%rdi),%rax
   dc191:	mov    0x38(%rdi),%rcx
   dc195:	movd   %r8d,%xmm0
   dc19a:	punpcklbw %xmm0,%xmm0
   dc19e:	pshuflw $0x0,%xmm0,%xmm0
   dc1a3:	pshufd $0x44,%xmm0,%xmm0
   dc1a8:	xor    %edi,%edi
   dc1aa:	and    %rcx,%rdx
   dc1ad:	movdqu (%rax,%rdx,1),%xmm1
   dc1b2:	movdqa %xmm1,%xmm2
   dc1b6:	pcmpeqb %xmm0,%xmm2
   dc1ba:	pmovmskb %xmm2,%r8d
   dc1bf:	test   %r8d,%r8d
   dc1c2:	je     dc200 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x520>
   dc1c4:	tzcnt  %r8d,%r10d
   dc1c9:	add    %rdx,%r10
   dc1cc:	and    %rcx,%r10
   dc1cf:	shl    $0x4,%r10
   dc1d3:	mov    %rax,%r9
   dc1d6:	sub    %r10,%r9
   dc1d9:	cmp    -0x10(%r9),%r14d
   dc1dd:	jne    dc1e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x509>
   dc1df:	cmp    -0xc(%r9),%ebx
   dc1e3:	je     dc3d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6f0>
   dc1e9:	lea    -0x1(%r8),%r9d
   dc1ed:	and    %r8w,%r9w
   dc1f1:	mov    %r9d,%r8d
   dc1f4:	jne    dc1c4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e4>
   dc1f6:	cs nopw 0x0(%rax,%rax,1)
   dc200:	pcmpeqd %xmm2,%xmm2
   dc204:	pcmpeqb %xmm2,%xmm1
   dc208:	pmovmskb %xmm1,%r8d
   dc20d:	test   %r8d,%r8d
   dc210:	jne    dc220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x540>
   dc212:	add    %rdi,%rdx
   dc215:	add    $0x10,%rdx
   dc219:	add    $0x10,%rdi
   dc21d:	jmp    dc1aa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ca>
   dc21f:	nop
   dc220:	lea    -0x1(%r15),%r14
   dc224:	mov    0xa0(%rsp),%rax
   dc22c:	mov    %r14,0x10(%rax)
   dc230:	lea    (%r14,%r14,2),%rsi
   dc234:	shl    $0x4,%rsi
   dc238:	mov    0x0(%rbp,%rsi,1),%rdi
   dc23d:	mov    0x8(%rbp,%rsi,1),%rdx
   dc242:	mov    0x10(%rbp,%rsi,1),%rcx
   dc247:	mov    0x18(%rbp,%rsi,1),%rax
   dc24c:	test   %r14,%r14
   dc24f:	je     dc560 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x880>
   dc255:	add    %rbp,%rsi
   dc258:	mov    0x0(%rbp),%r8
   dc25c:	mov    %r8,0x10(%rsp)
   dc261:	mov    0x8(%rbp),%r8
   dc265:	mov    %r8,0x8(%rsp)
   dc26a:	mov    0x10(%rbp),%r13
   dc26e:	mov    0x18(%rbp),%r8
   dc272:	mov    %r8,0x240(%rsp)
   dc27a:	movups 0x20(%rsi),%xmm0
   dc27e:	mov    %rdi,0x0(%rbp)
   dc282:	mov    %rdx,0x8(%rbp)
   dc286:	mov    %rcx,0x10(%rbp)
   dc28a:	mov    %rax,0x18(%rbp)
   dc28e:	movups %xmm0,0x20(%rbp)
   dc292:	movdqu 0x0(%rbp),%xmm0
   dc297:	movdqu 0x10(%rbp),%xmm1
   dc29c:	movdqu 0x20(%rbp),%xmm2
   dc2a1:	movdqa %xmm2,0x50(%rsp)
   dc2a7:	movdqa %xmm1,0x40(%rsp)
   dc2ad:	mov    %r14,%r12
   dc2b0:	sub    $0x8,%r12
   dc2b4:	mov    $0x0,%eax
   dc2b9:	cmovb  %rax,%r12
   dc2bd:	movdqa %xmm0,0x30(%rsp)
   dc2c3:	mov    $0x1,%edx
   dc2c8:	mov    $0x0,%eax
   dc2cd:	cmp    $0xa,%r15
   dc2d1:	jb     dc336 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x656>
   dc2d3:	xor    %eax,%eax
   dc2d5:	data16 cs nopw 0x0(%rax,%rax,1)
   dc2e0:	mov    %rax,%rbx
   dc2e3:	mov    %rbp,%rdi
   dc2e6:	mov    %r14,%rsi
   dc2e9:	call   db130 <<dary_heap::Hole<tk_train::trainers::bpe::index::Candidate>>::max_sibling::<8>>
   dc2ee:	lea    (%rax,%rax,2),%rcx
   dc2f2:	shl    $0x4,%rcx
   dc2f6:	lea    (%rbx,%rbx,2),%rdx
   dc2fa:	shl    $0x4,%rdx
   dc2fe:	movdqu 0x0(%rbp,%rcx,1),%xmm0
   dc304:	movdqu 0x10(%rbp,%rcx,1),%xmm1
   dc30a:	movdqu 0x20(%rbp,%rcx,1),%xmm2
   dc310:	movdqu %xmm0,0x0(%rbp,%rdx,1)
   dc316:	movdqu %xmm1,0x10(%rbp,%rdx,1)
   dc31c:	movdqu %xmm2,0x20(%rbp,%rdx,1)
   dc322:	lea    0x1(,%rax,8),%rdx
   dc32a:	mov    %rax,%rcx
   dc32d:	shl    $0x3,%rcx
   dc331:	cmp    %r12,%rcx
   dc334:	jb     dc2e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x600>
   dc336:	lea    0x1(%rdx),%rsi
   dc33a:	cmp    %r14,%rsi
   dc33d:	jae    dc5ac <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8cc>
   dc343:	lea    (%rdx,%rdx,2),%rcx
   dc347:	shl    $0x4,%rcx
   dc34b:	lea    (%rcx,%rbp,1),%rdi
   dc34f:	add    $0x5c,%rdi
   dc353:	mov    $0x2,%r8d
   dc359:	sub    %r15,%r8
   dc35c:	movabs $0xbe5466cf34e90c6c,%r12
   dc366:	mov    0x2b53b(%rip),%r15        # 1078a8 <free@GLIBC_2.2.5>
   dc36d:	jmp    dc38b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6ab>
   dc36f:	nop
   dc370:	add    $0x30,%rdi
   dc374:	lea    (%r8,%rsi,1),%r9
   dc378:	inc    %r9
   dc37b:	inc    %rsi
   dc37e:	mov    %rcx,%rdx
   dc381:	cmp    $0x1,%r9
   dc385:	je     dc5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8e0>
   dc38b:	lea    (%rdx,%rdx,2),%rcx
   dc38f:	shl    $0x4,%rcx
   dc393:	mov    0x20(%rbp,%rcx,1),%r9
   dc398:	cmp    -0xc(%rdi),%r9
   dc39c:	setbe  %r9b
   dc3a0:	jne    dc3be <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6de>
   dc3a2:	add    %rbp,%rcx
   dc3a5:	mov    -0x4(%rdi),%r9d
   dc3a9:	cmp    0x28(%rcx),%r9d
   dc3ad:	setbe  %r9b
   dc3b1:	jne    dc3be <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6de>
   dc3b3:	mov    (%rdi),%r9d
   dc3b6:	cmp    0x2c(%rcx),%r9d
   dc3ba:	setbe  %r9b
   dc3be:	mov    %rsi,%rcx
   dc3c1:	test   %r9b,%r9b
   dc3c4:	jne    dc370 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x690>
   dc3c6:	mov    %rdx,%rcx
   dc3c9:	jmp    dc370 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x690>
   dc3cb:	nopl   0x0(%rax,%rax,1)
   dc3d0:	mov    -0x8(%r9),%r15
   dc3d4:	cmp    %r15,%rsi
   dc3d7:	je     dc710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xa30>
   dc3dd:	lea    0x30(%rsp),%rdi
   dc3e2:	mov    0xa0(%rsp),%r14
   dc3ea:	mov    %r14,%rsi
   dc3ed:	call   e1380 <<dary_heap::DaryHeap<tk_train::trainers::bpe::index::Candidate, 8>>::pop>
   dc3f2:	cmpb   $0x0,0x30(%rsp)
   dc3f7:	je     e0518 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4838>
   dc3fd:	lea    0x38(%rsp),%rax
   dc402:	movups (%rax),%xmm0
   dc405:	movups 0x10(%rax),%xmm1
   dc409:	movups 0x20(%rax),%xmm2
   dc40d:	movaps %xmm2,0x1e0(%rsp)
   dc415:	movaps %xmm1,0x1d0(%rsp)
   dc41d:	movaps %xmm0,0x1c0(%rsp)
   dc425:	mov    %r15,0x1e0(%rsp)
   dc42d:	mov    0x10(%r14),%rbx
   dc431:	cmp    (%r14),%rbx
   dc434:	jne    dc43e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x75e>
   dc436:	mov    %r14,%rdi
   dc439:	call   e1700 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   dc43e:	mov    0x8(%r14),%rbp
   dc442:	lea    (%rbx,%rbx,2),%rdx
   dc446:	shl    $0x4,%rdx
   dc44a:	movdqa 0x1c0(%rsp),%xmm0
   dc453:	movdqa 0x1d0(%rsp),%xmm1
   dc45c:	movdqa 0x1e0(%rsp),%xmm2
   dc465:	movdqu %xmm2,0x20(%rbp,%rdx,1)
   dc46b:	movdqu %xmm1,0x10(%rbp,%rdx,1)
   dc471:	movdqu %xmm0,0x0(%rbp,%rdx,1)
   dc477:	lea    0x1(%rbx),%r15
   dc47b:	mov    %r15,0x10(%r14)
   dc47f:	mov    0x20(%rbp,%rdx,1),%rax
   dc484:	mov    0x28(%rbp,%rdx,1),%ecx
   dc488:	mov    0x2c(%rbp,%rdx,1),%edx
   dc48c:	test   %rbx,%rbx
   dc48f:	jne    dc52c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x84c>
   dc495:	xor    %ebx,%ebx
   dc497:	lea    (%rbx,%rbx,2),%rsi
   dc49b:	shl    $0x4,%rsi
   dc49f:	movdqa 0x1c0(%rsp),%xmm0
   dc4a8:	movdqa 0x1d0(%rsp),%xmm1
   dc4b1:	movdqu %xmm1,0x10(%rbp,%rsi,1)
   dc4b7:	movdqu %xmm0,0x0(%rbp,%rsi,1)
   dc4bd:	mov    %rax,0x20(%rbp,%rsi,1)
   dc4c2:	mov    %ecx,0x28(%rbp,%rsi,1)
   dc4c6:	mov    %edx,0x2c(%rbp,%rsi,1)
   dc4ca:	mov    0xa0(%rsp),%rax
   dc4d2:	mov    0x20(%rax),%rdi
   dc4d6:	test   %rdi,%rdi
   dc4d9:	jne    dc0c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e8>
   dc4df:	jmp    e096f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c8f>
   dc4e4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   dc4f0:	test   %r8b,%r8b
   dc4f3:	jne    dc497 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7b7>
   dc4f5:	lea    (%rbx,%rbx,2),%r8
   dc4f9:	shl    $0x4,%r8
   dc4fd:	movdqu (%rdi),%xmm0
   dc501:	movdqu 0x10(%rdi),%xmm1
   dc506:	movdqu 0x20(%rdi),%xmm2
   dc50b:	movdqu %xmm2,0x20(%rbp,%r8,1)
   dc512:	movdqu %xmm1,0x10(%rbp,%r8,1)
   dc519:	movdqu %xmm0,0x0(%rbp,%r8,1)
   dc520:	mov    %rsi,%rbx
   dc523:	test   %rsi,%rsi
   dc526:	je     dc495 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7b5>
   dc52c:	lea    -0x1(%rbx),%rsi
   dc530:	shr    $0x3,%rsi
   dc534:	lea    (%rsi,%rsi,2),%r8
   dc538:	shl    $0x4,%r8
   dc53c:	lea    (%r8,%rbp,1),%rdi
   dc540:	cmp    0x20(%rbp,%r8,1),%rax
   dc545:	setbe  %r8b
   dc549:	jne    dc4f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x810>
   dc54b:	cmp    %ecx,0x28(%rdi)
   dc54e:	setbe  %r8b
   dc552:	jne    dc4f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x810>
   dc554:	cmp    %edx,0x2c(%rdi)
   dc557:	ja     dc4f5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x815>
   dc559:	jmp    dc497 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7b7>
   dc55e:	xchg   %ax,%ax
   dc560:	mov    %rdx,%r14
   dc563:	mov    %rcx,%r13
   dc566:	mov    0x2b33b(%rip),%r15        # 1078a8 <free@GLIBC_2.2.5>
   dc56d:	test   %rdi,%rdi
   dc570:	jne    dc68a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9aa>
   dc576:	test   %r13,%r13
   dc579:	je     dc0a6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c6>
   dc57f:	lea    0x18(%r14),%rbx
   dc583:	jmp    dc59d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8bd>
   dc585:	data16 cs nopw 0x0(%rax,%rax,1)
   dc590:	add    $0x20,%rbx
   dc594:	dec    %r13
   dc597:	je     dc0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c0>
   dc59d:	cmpq   $0x11,(%rbx)
   dc5a1:	jb     dc590 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8b0>
   dc5a3:	mov    -0x10(%rbx),%rdi
   dc5a7:	call   *%r15
   dc5aa:	jmp    dc590 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8b0>
   dc5ac:	mov    %rdx,%rcx
   dc5af:	movabs $0xbe5466cf34e90c6c,%r12
   dc5b9:	mov    0x2b2e8(%rip),%r15        # 1078a8 <free@GLIBC_2.2.5>
   dc5c0:	cmp    %r14,%rcx
   dc5c3:	jae    dc5fb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x91b>
   dc5c5:	lea    (%rcx,%rcx,2),%rdx
   dc5c9:	shl    $0x4,%rdx
   dc5cd:	lea    (%rax,%rax,2),%rax
   dc5d1:	shl    $0x4,%rax
   dc5d5:	movdqu 0x0(%rbp,%rdx,1),%xmm0
   dc5db:	movdqu 0x10(%rbp,%rdx,1),%xmm1
   dc5e1:	movdqu 0x20(%rbp,%rdx,1),%xmm2
   dc5e7:	movdqu %xmm2,0x20(%rbp,%rax,1)
   dc5ed:	movdqu %xmm1,0x10(%rbp,%rax,1)
   dc5f3:	movdqu %xmm0,0x0(%rbp,%rax,1)
   dc5f9:	jmp    dc5fe <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x91e>
   dc5fb:	mov    %rax,%rcx
   dc5fe:	mov    0x8(%rsp),%r14
   dc603:	lea    (%rcx,%rcx,2),%rsi
   dc607:	shl    $0x4,%rsi
   dc60b:	movdqa 0x30(%rsp),%xmm0
   dc611:	movdqa 0x40(%rsp),%xmm1
   dc617:	movdqa 0x50(%rsp),%xmm2
   dc61d:	movdqu %xmm2,0x20(%rbp,%rsi,1)
   dc623:	movdqu %xmm1,0x10(%rbp,%rsi,1)
   dc629:	movdqu %xmm0,0x0(%rbp,%rsi,1)
   dc62f:	mov    0x20(%rbp,%rsi,1),%rax
   dc634:	mov    0x28(%rbp,%rsi,1),%edx
   dc638:	mov    0x2c(%rbp,%rsi,1),%esi
   dc63c:	test   %rcx,%rcx
   dc63f:	jne    dc6dc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9fc>
   dc645:	xor    %ecx,%ecx
   dc647:	lea    (%rcx,%rcx,2),%rcx
   dc64b:	shl    $0x4,%rcx
   dc64f:	movdqa 0x30(%rsp),%xmm0
   dc655:	movdqa 0x40(%rsp),%xmm1
   dc65b:	movdqu %xmm1,0x10(%rbp,%rcx,1)
   dc661:	movdqu %xmm0,0x0(%rbp,%rcx,1)
   dc667:	mov    %rax,0x20(%rbp,%rcx,1)
   dc66c:	mov    %edx,0x28(%rbp,%rcx,1)
   dc670:	mov    %esi,0x2c(%rbp,%rcx,1)
   dc674:	mov    0x10(%rsp),%rdi
   dc679:	mov    0x240(%rsp),%rax
   dc681:	test   %rdi,%rdi
   dc684:	je     dc576 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x896>
   dc68a:	cmp    $0x11,%rax
   dc68e:	jae    dc0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c0>
   dc694:	jmp    dc0a6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c6>
   dc699:	nopl   0x0(%rax)
   dc6a0:	test   %r9b,%r9b
   dc6a3:	jne    dc647 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x967>
   dc6a5:	lea    (%rcx,%rcx,2),%rcx
   dc6a9:	shl    $0x4,%rcx
   dc6ad:	movdqu (%r8),%xmm0
   dc6b2:	movdqu 0x10(%r8),%xmm1
   dc6b8:	movdqu 0x20(%r8),%xmm2
   dc6be:	movdqu %xmm2,0x20(%rbp,%rcx,1)
   dc6c4:	movdqu %xmm1,0x10(%rbp,%rcx,1)
   dc6ca:	movdqu %xmm0,0x0(%rbp,%rcx,1)
   dc6d0:	mov    %rdi,%rcx
   dc6d3:	test   %rdi,%rdi
   dc6d6:	je     dc645 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x965>
   dc6dc:	lea    -0x1(%rcx),%rdi
   dc6e0:	shr    $0x3,%rdi
   dc6e4:	lea    (%rdi,%rdi,2),%r9
   dc6e8:	shl    $0x4,%r9
   dc6ec:	lea    (%r9,%rbp,1),%r8
   dc6f0:	cmp    0x20(%rbp,%r9,1),%rax
   dc6f5:	setbe  %r9b
   dc6f9:	jne    dc6a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9c0>
   dc6fb:	cmp    %edx,0x28(%r8)
   dc6ff:	setbe  %r9b
   dc703:	jne    dc6a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9c0>
   dc705:	cmp    %esi,0x2c(%r8)
   dc709:	ja     dc6a5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9c5>
   dc70b:	jmp    dc647 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x967>
   dc710:	mov    0xa0(%rsp),%rax
   dc718:	cmp    0x28(%rax),%rsi
   dc71c:	jb     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc722:	cmpq   $0x0,0xa8(%rsp)
   dc72b:	je     dc8ce <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbee>
   dc731:	cmp    %ebx,%r14d
   dc734:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc73a:	cmpq   $0x0,0x3b8(%rsp)
   dc743:	je     dc804 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb24>
   dc749:	mov    0x3c8(%rsp),%rax
   dc751:	xor    %r14,%rax
   dc754:	movabs $0x5851f42d4c957f2d,%rcx
   dc75e:	mul    %rcx
   dc761:	mov    %rax,%rcx
   dc764:	xor    %rdx,%rcx
   dc767:	mov    %rcx,%rax
   dc76a:	mulq   0x3c0(%rsp)
   dc772:	xor    %rax,%rdx
   dc775:	rol    %cl,%rdx
   dc778:	mov    %rdx,%rsi
   dc77b:	shr    $0x39,%rsi
   dc77f:	mov    0x3a0(%rsp),%rax
   dc787:	mov    0x3a8(%rsp),%rcx
   dc78f:	movd   %esi,%xmm0
   dc793:	punpcklbw %xmm0,%xmm0
   dc797:	pshuflw $0x0,%xmm0,%xmm0
   dc79c:	pshufd $0x44,%xmm0,%xmm0
   dc7a1:	xor    %esi,%esi
   dc7a3:	and    %rcx,%rdx
   dc7a6:	movdqu (%rax,%rdx,1),%xmm1
   dc7ab:	movdqa %xmm1,%xmm2
   dc7af:	pcmpeqb %xmm0,%xmm2
   dc7b3:	pmovmskb %xmm2,%edi
   dc7b7:	test   %edi,%edi
   dc7b9:	je     dc7e7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb07>
   dc7bb:	tzcnt  %edi,%r8d
   dc7c0:	add    %rdx,%r8
   dc7c3:	and    %rcx,%r8
   dc7c6:	shl    $0x2,%r8
   dc7ca:	mov    %rax,%r9
   dc7cd:	sub    %r8,%r9
   dc7d0:	cmp    -0x4(%r9),%r14d
   dc7d4:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc7da:	lea    -0x1(%rdi),%r8d
   dc7de:	and    %di,%r8w
   dc7e2:	mov    %r8d,%edi
   dc7e5:	jne    dc7bb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xadb>
   dc7e7:	pcmpeqd %xmm2,%xmm2
   dc7eb:	pcmpeqb %xmm2,%xmm1
   dc7ef:	pmovmskb %xmm1,%edi
   dc7f3:	test   %edi,%edi
   dc7f5:	jne    dc804 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb24>
   dc7f7:	add    %rsi,%rdx
   dc7fa:	add    $0x10,%rdx
   dc7fe:	add    $0x10,%rsi
   dc802:	jmp    dc7a3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xac3>
   dc804:	cmpq   $0x0,0x378(%rsp)
   dc80d:	je     dc8ce <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbee>
   dc813:	mov    0x388(%rsp),%rax
   dc81b:	xor    %rbx,%rax
   dc81e:	movabs $0x5851f42d4c957f2d,%rcx
   dc828:	mul    %rcx
   dc82b:	mov    %rax,%rcx
   dc82e:	xor    %rdx,%rcx
   dc831:	mov    %rcx,%rax
   dc834:	mulq   0x380(%rsp)
   dc83c:	xor    %rax,%rdx
   dc83f:	rol    %cl,%rdx
   dc842:	mov    %rdx,%rsi
   dc845:	shr    $0x39,%rsi
   dc849:	mov    0x360(%rsp),%rax
   dc851:	mov    0x368(%rsp),%rcx
   dc859:	movd   %esi,%xmm0
   dc85d:	punpcklbw %xmm0,%xmm0
   dc861:	pshuflw $0x0,%xmm0,%xmm0
   dc866:	pshufd $0x44,%xmm0,%xmm0
   dc86b:	xor    %esi,%esi
   dc86d:	and    %rcx,%rdx
   dc870:	movdqu (%rax,%rdx,1),%xmm1
   dc875:	movdqa %xmm1,%xmm2
   dc879:	pcmpeqb %xmm0,%xmm2
   dc87d:	pmovmskb %xmm2,%edi
   dc881:	test   %edi,%edi
   dc883:	je     dc8b1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbd1>
   dc885:	tzcnt  %edi,%r8d
   dc88a:	add    %rdx,%r8
   dc88d:	and    %rcx,%r8
   dc890:	shl    $0x2,%r8
   dc894:	mov    %rax,%r9
   dc897:	sub    %r8,%r9
   dc89a:	cmp    -0x4(%r9),%ebx
   dc89e:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dc8a4:	lea    -0x1(%rdi),%r8d
   dc8a8:	and    %di,%r8w
   dc8ac:	mov    %r8d,%edi
   dc8af:	jne    dc885 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xba5>
   dc8b1:	pcmpeqd %xmm2,%xmm2
   dc8b5:	pcmpeqb %xmm2,%xmm1
   dc8b9:	pmovmskb %xmm1,%edi
   dc8bd:	test   %edi,%edi
   dc8bf:	jne    dc8ce <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbee>
   dc8c1:	add    %rsi,%rdx
   dc8c4:	add    $0x10,%rdx
   dc8c8:	add    $0x10,%rsi
   dc8cc:	jmp    dc86d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb8d>
   dc8ce:	mov    %r14,0xe0(%rsp)
   dc8d6:	mov    0x300(%rsp),%r9
   dc8de:	mov    0x30(%r9),%rcx
   dc8e2:	mov    %rcx,0x1c0(%rsp)
   dc8ea:	cmp    %r14,%rcx
   dc8ed:	jbe    e07f3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b13>
   dc8f3:	mov    0x10(%r9),%rsi
   dc8f7:	cmp    %r14,%rsi
   dc8fa:	jbe    e0ac7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4de7>
   dc900:	mov    0x8(%r9),%rax
   dc904:	mov    %r14,%r11
   dc907:	shl    $0x5,%r11
   dc90b:	lea    (%rax,%r11,1),%r12
   dc90f:	add    $0x8,%r12
   dc913:	movzbl 0x1f(%rax,%r11,1),%edi
   dc919:	lea    0x40(%rdi),%r8d
   dc91d:	movzbl %r8b,%r8d
   dc921:	cmp    $0x18,%r8b
   dc925:	mov    $0x18,%r10d
   dc92b:	cmovae %r10d,%r8d
   dc92f:	cmp    $0xd8,%dil
   dc933:	cmovae 0x8(%rax,%r11,1),%r12
   dc939:	movzbl %r8b,%edx
   dc93d:	cmovae 0x10(%rax,%r11,1),%rdx
   dc943:	mov    %rbx,0xe0(%rsp)
   dc94b:	mov    %rcx,0x1c0(%rsp)
   dc953:	cmp    %rbx,%rcx
   dc956:	jbe    e0809 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b29>
   dc95c:	cmp    %rbx,%rsi
   dc95f:	jbe    e0ab5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4dd5>
   dc965:	mov    %rbx,%rcx
   dc968:	shl    $0x5,%rcx
   dc96c:	lea    (%rax,%rcx,1),%rsi
   dc970:	add    $0x8,%rsi
   dc974:	movzbl 0x1f(%rax,%rcx,1),%r8d
   dc97a:	lea    0x40(%r8),%edi
   dc97e:	movzbl %dil,%r10d
   dc982:	cmp    $0x18,%r10b
   dc986:	mov    $0x18,%edi
   dc98b:	cmovae %edi,%r10d
   dc98f:	cmp    $0xd8,%r8b
   dc993:	cmovae 0x8(%rax,%rcx,1),%rsi
   dc999:	movzbl %r10b,%r15d
   dc99d:	cmovae 0x10(%rax,%rcx,1),%r15
   dc9a3:	cmpq   $0xffffffffffffffff,0x70(%r9)
   dc9a8:	je     dc9b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xcd9>
   dc9aa:	mov    0x80(%r9),%rbp
   dc9b1:	mov    %r15,%rax
   dc9b4:	sub    %rbp,%rax
   dc9b7:	jae    dc9be <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xcde>
   dc9b9:	mov    %r15,%rax
   dc9bc:	jmp    dca08 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd28>
   dc9be:	mov    %rax,0x10(%rsp)
   dc9c3:	mov    %rdx,0x240(%rsp)
   dc9cb:	mov    0x78(%r9),%rdi
   dc9cf:	mov    %rbp,%rdx
   dc9d2:	mov    %rsi,0x8(%rsp)
   dc9d7:	call   *0x2af23(%rip)        # 107900 <bcmp@GLIBC_2.2.5>
   dc9dd:	test   %eax,%eax
   dc9df:	je     dc9f3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd13>
   dc9e1:	mov    %r15,%rax
   dc9e4:	mov    0x240(%rsp),%rdx
   dc9ec:	mov    0x8(%rsp),%rsi
   dc9f1:	jmp    dca08 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd28>
   dc9f3:	mov    0x8(%rsp),%rsi
   dc9f8:	add    %rbp,%rsi
   dc9fb:	mov    0x240(%rsp),%rdx
   dca03:	mov    0x10(%rsp),%rax
   dca08:	lea    (%rax,%rdx,1),%rbp
   dca0c:	cmp    $0x19,%rbp
   dca10:	lea    0x348(%rsp),%r15
   dca18:	mov    %rsi,0x8(%rsp)
   dca1d:	mov    %rax,0x10(%rsp)
   dca22:	jae    dca32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd52>
   dca24:	xor    %eax,%eax
   dca26:	movabs $0xc000000000000000,%rbp
   dca30:	jmp    dca75 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd95>
   dca32:	cmp    $0x21,%rbp
   dca36:	mov    $0x20,%eax
   dca3b:	cmovb  %rax,%rbp
   dca3f:	test   %rbp,%rbp
   dca42:	js     e0adc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4dfc>
   dca48:	mov    %rdx,%r15
   dca4b:	mov    %rbp,%rdi
   dca4e:	call   *0x2ae6c(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   dca54:	test   %rax,%rax
   dca57:	je     e0905 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c25>
   dca5d:	movabs $0xd800000000000000,%rcx
   dca67:	or     %rcx,%rbp
   dca6a:	mov    %r15,%rdx
   dca6d:	lea    0x348(%rsp),%r15
   dca75:	mov    %rax,0x348(%rsp)
   dca7d:	movq   $0x0,0x350(%rsp)
   dca89:	mov    %rbp,0x358(%rsp)
   dca91:	mov    %r15,%rdi
   dca94:	mov    %r12,%rsi
   dca97:	call   e1770 <<compact_str::repr::Repr>::push_str>
   dca9c:	mov    %r15,%rdi
   dca9f:	mov    0x8(%rsp),%rsi
   dcaa4:	mov    0x10(%rsp),%rdx
   dcaa9:	call   e1770 <<compact_str::repr::Repr>::push_str>
   dcaae:	lea    0x30(%rsp),%rdi
   dcab3:	mov    0x300(%rsp),%rsi
   dcabb:	mov    %r15,%rdx
   dcabe:	call   e1c30 <<tk_train::trainers::bpe::vocabulary::Vocabulary>::insert>
   dcac3:	mov    0x30(%rsp),%rax
   dcac8:	test   %rax,%rax
   dcacb:	jne    e04e5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4805>
   dcad1:	mov    0x38(%rsp),%r15d
   dcad6:	movzbl 0x3c(%rsp),%ebp
   dcadb:	mov    %ebp,%eax
   dcadd:	not    %al
   dcadf:	or     0x118(%rsp),%al
   dcae6:	test   $0x1,%al
   dcae8:	je     dcb00 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe20>
   dcaea:	cmpq   $0x0,0xa8(%rsp)
   dcaf3:	setne  %al
   dcaf6:	test   %bpl,%al
   dcaf9:	je     dcb29 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe49>
   dcafb:	jmp    dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dcb00:	mov    0x20(%rsp),%rax
   dcb05:	cmp    %r15,0x50(%rax)
   dcb09:	jbe    dcb1a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe3a>
   dcb0b:	mov    0x48(%rax),%rax
   dcb0f:	cmpq   $0x0,(%rax,%r15,8)
   dcb14:	jne    e0572 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4892>
   dcb1a:	cmpq   $0x0,0xa8(%rsp)
   dcb23:	jne    dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dcb29:	lea    0x30(%rsp),%rdi
   dcb2e:	mov    0xa0(%rsp),%r12
   dcb36:	mov    %r12,%rsi
   dcb39:	call   e1380 <<dary_heap::DaryHeap<tk_train::trainers::bpe::index::Candidate, 8>>::pop>
   dcb3e:	cmpl   $0x1,0x30(%rsp)
   dcb43:	jne    e0851 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b71>
   dcb49:	lea    0x38(%rsp),%rax
   dcb4e:	movups (%rax),%xmm0
   dcb51:	movups 0x10(%rax),%xmm1
   dcb55:	movups 0x20(%rax),%xmm2
   dcb59:	movaps %xmm2,0x1e0(%rsp)
   dcb61:	movaps %xmm1,0x1d0(%rsp)
   dcb69:	movaps %xmm0,0x1c0(%rsp)
   dcb71:	cmpb   $0x0,0x30(%r12)
   dcb77:	jne    dcbd6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xef6>
   dcb79:	mov    0xa0(%rsp),%rax
   dcb81:	mov    0x20(%rax),%rcx
   dcb85:	test   %rcx,%rcx
   dcb88:	je     e0b40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e60>
   dcb8e:	mov    %r13,%rax
   dcb91:	or     %rcx,%rax
   dcb94:	shr    $0x20,%rax
   dcb98:	je     dcba4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xec4>
   dcb9a:	mov    %r13,%rax
   dcb9d:	xor    %edx,%edx
   dcb9f:	div    %rcx
   dcba2:	jmp    dcbab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xecb>
   dcba4:	mov    %r13d,%eax
   dcba7:	xor    %edx,%edx
   dcba9:	div    %ecx
   dcbab:	mov    0xa0(%rsp),%rax
   dcbb3:	mov    0x18(%rax),%rax
   dcbb7:	imul   $0x70,%rdx,%rcx
   dcbbb:	lea    (%rax,%rcx,1),%rdi
   dcbbf:	add    $0x30,%rdi
   dcbc3:	mov    %r14d,%esi
   dcbc6:	mov    %ebx,%edx
   dcbc8:	call   e2300 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::remove::<(u32, u32)>>
   dcbcd:	test   %rax,%rax
   dcbd0:	je     e0952 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c72>
   dcbd6:	mov    %bpl,0xa8(%rsp)
   dcbde:	movdqa 0x1c0(%rsp),%xmm0
   dcbe7:	movdqa 0x1d0(%rsp),%xmm1
   dcbf0:	movdqa 0x1e0(%rsp),%xmm2
   dcbf9:	movdqa %xmm2,0x100(%rsp)
   dcc02:	movdqa %xmm1,0xf0(%rsp)
   dcc0b:	movdqa %xmm0,0xe0(%rsp)
   dcc14:	mov    0x20(%rsp),%rcx
   dcc19:	mov    0x50(%rcx),%r12
   dcc1d:	cmp    %r15,%r12
   dcc20:	jbe    dcc30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf50>
   dcc22:	mov    0x48(%rcx),%rax
   dcc26:	cmpq   $0x0,(%rax,%r15,8)
   dcc2b:	setne  %al
   dcc2e:	jmp    dcc32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf52>
   dcc30:	xor    %eax,%eax
   dcc32:	mov    0x250(%rsp),%r13
   dcc3a:	or     %al,0x88(%rcx)
   dcc40:	mov    0x78(%rcx),%rax
   dcc44:	mov    $0x1,%ecx
   dcc49:	test   %rax,%rax
   dcc4c:	je     dcc8a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xfaa>
   dcc4e:	cmp    %r15,%r12
   dcc51:	jne    dccb5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xfd5>
   dcc53:	mov    %rcx,%r12
   dcc56:	mov    0x1a0(%rsp),%rax
   dcc5e:	cmp    %r15,(%rax)
   dcc61:	jne    dcc70 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf90>
   dcc63:	mov    0x1a0(%rsp),%rdi
   dcc6b:	call   75220 <<alloc::raw_vec::RawVec<usize>>::grow_one>
   dcc70:	mov    0x20(%rsp),%rcx
   dcc75:	mov    0x48(%rcx),%rax
   dcc79:	mov    %r12,(%rax,%r15,8)
   dcc7d:	lea    0x1(%r15),%rax
   dcc81:	mov    %rax,0x50(%rcx)
   dcc85:	jmp    dced1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11f1>
   dcc8a:	cmp    %r14,%r12
   dcc8d:	jbe    e0b2b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e4b>
   dcc93:	cmp    %rbx,%r12
   dcc96:	jbe    e0b14 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e34>
   dcc9c:	mov    0x20(%rsp),%rcx
   dcca1:	mov    0x48(%rcx),%rcx
   dcca5:	mov    (%rcx,%rbx,8),%rdx
   dcca9:	add    (%rcx,%r14,8),%rdx
   dccad:	mov    %rdx,%rcx
   dccb0:	cmp    %r15,%r12
   dccb3:	je     dcc53 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf73>
   dccb5:	jbe    e0b08 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e28>
   dccbb:	test   %rax,%rax
   dccbe:	jne    dcebb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11db>
   dccc4:	mov    0x20(%rsp),%rax
   dccc9:	mov    0x48(%rax),%rax
   dcccd:	mov    (%rax,%r15,8),%rax
   dccd1:	test   %rax,%rax
   dccd4:	je     dcebb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11db>
   dccda:	cmp    %rcx,%rax
   dccdd:	je     dcebb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11db>
   dcce3:	mov    %rcx,0x8(%rsp)
   dcce8:	mov    0x20(%rsp),%rax
   dcced:	mov    0x10(%rax),%rbp
   dccf1:	lea    0x0(,%rbp,8),%rdx
   dccf9:	mov    %rbp,%rax
   dccfc:	shr    $0x3d,%rax
   dcd00:	setne  %al
   dcd03:	cmp    0x238(%rsp),%rdx
   dcd0b:	seta   %cl
   dcd0e:	or     %al,%cl
   dcd10:	jne    e04d6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47f6>
   dcd16:	test   %rdx,%rdx
   dcd19:	je     dcd45 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1065>
   dcd1b:	mov    $0x8,%esi
   dcd20:	mov    %rdx,%rdi
   dcd23:	mov    %rdx,%r13
   dcd26:	call   68f90 <__rustc::__rust_alloc>
   dcd2b:	mov    $0x8,%edi
   dcd30:	mov    %r13,%rdx
   dcd33:	mov    %rax,%r13
   dcd36:	mov    %rbp,0x10(%rsp)
   dcd3b:	test   %rax,%rax
   dcd3e:	jne    dcd54 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1074>
   dcd40:	jmp    e04d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47f8>
   dcd45:	mov    $0x8,%r13d
   dcd4b:	movq   $0x0,0x10(%rsp)
   dcd54:	test   %rbp,%rbp
   dcd57:	je     dcd64 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1084>
   dcd59:	mov    %r13,%rdi
   dcd5c:	xor    %esi,%esi
   dcd5e:	call   *0x2ab6c(%rip)        # 1078d0 <memset@GLIBC_2.2.5>
   dcd64:	mov    0x20(%rsp),%rax
   dcd69:	mov    0x60(%rax),%rax
   dcd6d:	test   %rax,%rax
   dcd70:	je     dce7c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x119c>
   dcd76:	mov    0x20(%rsp),%rsi
   dcd7b:	mov    0x58(%rsi),%rcx
   dcd7f:	lea    (%rcx,%rax,8),%rdx
   dcd83:	cmpb   $0x0,(%rsi)
   dcd86:	je     dcdfd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x111d>
   dcd88:	jmp    dcd93 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10b3>
   dcd8a:	cmp    %rdx,%rcx
   dcd8d:	je     dce7c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x119c>
   dcd93:	mov    (%rcx),%rdi
   dcd96:	cmp    %rbp,%rdi
   dcd99:	jae    e09dd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cfd>
   dcd9f:	add    $0x8,%rcx
   dcda3:	mov    0x20(%rsp),%rax
   dcda8:	mov    0x8(%rax),%rsi
   dcdac:	nopl   0x0(%rax)
   dcdb0:	mov    (%rsi,%rdi,4),%eax
   dcdb3:	cmp    $0xffffffff,%eax
   dcdb6:	je     dcd8a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10aa>
   dcdb8:	mov    (%rsi,%rdi,4),%eax
   dcdbb:	cmp    %rax,%r12
   dcdbe:	jbe    e09c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ce5>
   dcdc4:	mov    0x20(%rsp),%r8
   dcdc9:	mov    0x48(%r8),%r8
   dcdcd:	mov    (%r8,%rax,8),%r8
   dcdd1:	mov    %r8,0x0(%r13,%rdi,8)
   dcdd6:	lea    (%r8,%rdi,1),%rax
   dcdda:	dec    %rax
   dcddd:	cmp    %rbp,%rax
   dcde0:	jae    e09b3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cd3>
   dcde6:	add    %r8,%rdi
   dcde9:	mov    %r8,-0x8(%r13,%rdi,8)
   dcdee:	cmp    %rbp,%rdi
   dcdf1:	jb     dcdb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10d0>
   dcdf3:	jmp    e09dd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cfd>
   dcdf8:	cmp    %rdx,%rcx
   dcdfb:	je     dce7c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x119c>
   dcdfd:	mov    (%rcx),%rdi
   dce00:	cmp    %rbp,%rdi
   dce03:	jae    e09ec <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d0c>
   dce09:	add    $0x8,%rcx
   dce0d:	mov    0x20(%rsp),%rax
   dce12:	mov    0x8(%rax),%rsi
   dce16:	cs nopw 0x0(%rax,%rax,1)
   dce20:	movzwl (%rsi,%rdi,2),%eax
   dce24:	cmp    $0xffff,%ax
   dce28:	je     dcdf8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1118>
   dce2a:	movzwl (%rsi,%rdi,2),%eax
   dce2e:	cmp    $0xffff,%ax
   dce32:	movzwl %ax,%eax
   dce35:	mov    $0xffffffff,%r8d
   dce3b:	cmove  %r8,%rax
   dce3f:	cmp    %rax,%r12
   dce42:	jbe    e09c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ce5>
   dce48:	mov    0x20(%rsp),%r8
   dce4d:	mov    0x48(%r8),%r8
   dce51:	mov    (%r8,%rax,8),%r8
   dce55:	mov    %r8,0x0(%r13,%rdi,8)
   dce5a:	lea    (%r8,%rdi,1),%rax
   dce5e:	dec    %rax
   dce61:	cmp    %rbp,%rax
   dce64:	jae    e09b3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cd3>
   dce6a:	add    %r8,%rdi
   dce6d:	mov    %r8,-0x8(%r13,%rdi,8)
   dce72:	cmp    %rbp,%rdi
   dce75:	jb     dce20 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1140>
   dce77:	jmp    e09ec <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d0c>
   dce7c:	mov    0x20(%rsp),%r12
   dce81:	mov    0x78(%r12),%rdi
   dce86:	test   %rdi,%rdi
   dce89:	je     dce9c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11bc>
   dce8b:	cmpq   $0x0,0x80(%r12)
   dce94:	je     dce9c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11bc>
   dce96:	call   *0x2aa0c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dce9c:	mov    %r13,0x78(%r12)
   dcea1:	mov    %rbp,0x80(%r12)
   dcea9:	mov    0x50(%r12),%r12
   dceae:	mov    0x250(%rsp),%r13
   dceb6:	mov    0x8(%rsp),%rcx
   dcebb:	cmp    %r15,%r12
   dcebe:	jbe    e0b19 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e39>
   dcec4:	mov    0x20(%rsp),%rax
   dcec9:	mov    0x48(%rax),%rax
   dcecd:	mov    %rcx,(%rax,%r15,8)
   dced1:	movabs $0xbe5466cf34e90c6c,%r12
   dcedb:	movaps 0xe0(%rsp),%xmm0
   dcee3:	movaps 0xf0(%rsp),%xmm1
   dceeb:	movaps %xmm1,0x40(%rsp)
   dcef0:	movaps %xmm0,0x30(%rsp)
   dcef5:	mov    %r14d,0x50(%rsp)
   dcefa:	mov    %ebx,0x54(%rsp)
   dcefe:	mov    %r15d,0x58(%rsp)
   dcf03:	mov    0x2b8(%rsp),%r15
   dcf0b:	cmp    0x2a8(%rsp),%r15
   dcf13:	jne    dcf22 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1242>
   dcf15:	lea    0x2a8(%rsp),%rdi
   dcf1d:	call   e1700 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   dcf22:	mov    0x2b0(%rsp),%rax
   dcf2a:	lea    (%r15,%r15,2),%rcx
   dcf2e:	shl    $0x4,%rcx
   dcf32:	movdqa 0x30(%rsp),%xmm0
   dcf38:	movdqa 0x40(%rsp),%xmm1
   dcf3e:	movdqa 0x50(%rsp),%xmm2
   dcf44:	movdqu %xmm2,0x20(%rax,%rcx,1)
   dcf4a:	movdqu %xmm1,0x10(%rax,%rcx,1)
   dcf50:	movdqu %xmm0,(%rax,%rcx,1)
   dcf55:	inc    %r15
   dcf58:	mov    %r15,0x2b8(%rsp)
   dcf60:	cmp    %ebx,%r14d
   dcf63:	je     dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dcf65:	testb  $0x1,0xa8(%rsp)
   dcf6d:	jne    dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12b9>
   dcf6f:	lea    0x360(%rsp),%rdi
   dcf77:	mov    %r14d,%esi
   dcf7a:	call   e2440 <<hashbrown::map::HashMap<u32, (), ahash::random_state::RandomState>>::insert>
   dcf7f:	lea    0x3a0(%rsp),%rdi
   dcf87:	mov    %ebx,%esi
   dcf89:	call   e2440 <<hashbrown::map::HashMap<u32, (), ahash::random_state::RandomState>>::insert>
   dcf8e:	cmp    %r13,%r15
   dcf91:	jb     dc067 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x387>
   dcf97:	jmp    dcfa1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12c1>
   dcf99:	mov    0x2b8(%rsp),%r15
   dcfa1:	test   %r15,%r15
   dcfa4:	je     dd09e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13be>
   dcfaa:	mov    0x368(%rsp),%rcx
   dcfb2:	test   %rcx,%rcx
   dcfb5:	je     dcfdd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12fd>
   dcfb7:	lea    0x13(,%rcx,4),%rax
   dcfbf:	and    $0xfffffffffffffff0,%rax
   dcfc3:	add    %rax,%rcx
   dcfc6:	cmp    $0xffffffffffffffef,%rcx
   dcfca:	je     dcfdd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12fd>
   dcfcc:	mov    0x360(%rsp),%rdi
   dcfd4:	sub    %rax,%rdi
   dcfd7:	call   *0x2a8cb(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dcfdd:	mov    0x3a8(%rsp),%rcx
   dcfe5:	test   %rcx,%rcx
   dcfe8:	je     dd010 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1330>
   dcfea:	lea    0x13(,%rcx,4),%rax
   dcff2:	and    $0xfffffffffffffff0,%rax
   dcff6:	add    %rax,%rcx
   dcff9:	cmp    $0xffffffffffffffef,%rcx
   dcffd:	je     dd010 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1330>
   dcfff:	mov    0x3a0(%rsp),%rdi
   dd007:	sub    %rax,%rdi
   dd00a:	call   *0x2a898(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dd010:	mov    0x2b0(%rsp),%r14
   dd018:	lea    0x0(,%r15,8),%rbx
   dd020:	mov    %rbx,%rdi
   dd023:	call   *0x2a897(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   dd029:	mov    %rax,0x198(%rsp)
   dd031:	test   %rax,%rax
   dd034:	je     e0a01 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d21>
   dd03a:	movabs $0x3fffffffffffffe,%rax
   dd044:	mov    %r15,%r13
   dd047:	dec    %r13
   dd04a:	je     dd1df <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14ff>
   dd050:	mov    %r15,%rcx
   dd053:	and    %rax,%rcx
   dd056:	lea    0x50(%r14),%rdx
   dd05a:	xor    %eax,%eax
   dd05c:	mov    0x198(%rsp),%rsi
   dd064:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   dd070:	movq   (%rdx),%xmm0
   dd074:	movq   -0x30(%rdx),%xmm1
   dd079:	punpcklqdq %xmm0,%xmm1
   dd07d:	movdqu %xmm1,(%rsi,%rax,8)
   dd082:	add    $0x2,%rax
   dd086:	add    $0x60,%rdx
   dd08a:	cmp    %rax,%rcx
   dd08d:	jne    dd070 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1390>
   dd08f:	test   $0x1,%r15b
   dd093:	jne    dd1e1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1501>
   dd099:	jmp    dd1fa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x151a>
   dd09e:	mov    0x2a0(%rsp),%rdx
   dd0a6:	movq   $0x0,0x8(%rdx)
   dd0ae:	xor    %eax,%eax
   dd0b0:	mov    %rax,(%rdx)
   dd0b3:	mov    0x3a8(%rsp),%rcx
   dd0bb:	test   %rcx,%rcx
   dd0be:	je     dd0e6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1406>
   dd0c0:	lea    0x13(,%rcx,4),%rax
   dd0c8:	and    $0xfffffffffffffff0,%rax
   dd0cc:	add    %rax,%rcx
   dd0cf:	cmp    $0xffffffffffffffef,%rcx
   dd0d3:	je     dd0e6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1406>
   dd0d5:	mov    0x3a0(%rsp),%rdi
   dd0dd:	sub    %rax,%rdi
   dd0e0:	call   *0x2a7c2(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dd0e6:	mov    0x368(%rsp),%rcx
   dd0ee:	test   %rcx,%rcx
   dd0f1:	je     dd119 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1439>
   dd0f3:	lea    0x13(,%rcx,4),%rax
   dd0fb:	and    $0xfffffffffffffff0,%rax
   dd0ff:	add    %rax,%rcx
   dd102:	cmp    $0xffffffffffffffef,%rcx
   dd106:	je     dd119 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1439>
   dd108:	mov    0x360(%rsp),%rdi
   dd110:	sub    %rax,%rdi
   dd113:	call   *0x2a78f(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dd119:	mov    0x2b0(%rsp),%rbx
   dd121:	mov    0x2b8(%rsp),%r15
   dd129:	test   %r15,%r15
   dd12c:	je     dd1c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14e8>
   dd132:	xor    %r12d,%r12d
   dd135:	mov    0x2a76c(%rip),%r13        # 1078a8 <free@GLIBC_2.2.5>
   dd13c:	mov    %r15,0xa8(%rsp)
   dd144:	jmp    dd166 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1486>
   dd146:	cs nopw 0x0(%rax,%rax,1)
   dd150:	mov    0xa8(%rsp),%r15
   dd158:	mov    %r14,%rdi
   dd15b:	call   *%r13
   dd15e:	inc    %r12
   dd161:	cmp    %r15,%r12
   dd164:	je     dd1c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14e8>
   dd166:	lea    (%r12,%r12,2),%rcx
   dd16a:	shl    $0x4,%rcx
   dd16e:	lea    (%rbx,%rcx,1),%rax
   dd172:	cmpq   $0x0,(%rbx,%rcx,1)
   dd177:	je     dd190 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14b0>
   dd179:	cmpq   $0x11,0x18(%rax)
   dd17e:	jb     dd15e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x147e>
   dd180:	mov    0x8(%rax),%r14
   dd184:	jmp    dd158 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1478>
   dd186:	cs nopw 0x0(%rax,%rax,1)
   dd190:	mov    0x10(%rax),%rbp
   dd194:	test   %rbp,%rbp
   dd197:	je     dd15e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x147e>
   dd199:	mov    0x8(%rax),%r14
   dd19d:	lea    0x18(%r14),%r15
   dd1a1:	jmp    dd1b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14d9>
   dd1a3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dd1b0:	add    $0x20,%r15
   dd1b4:	dec    %rbp
   dd1b7:	je     dd150 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1470>
   dd1b9:	cmpq   $0x11,(%r15)
   dd1bd:	jb     dd1b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14d0>
   dd1bf:	mov    -0x10(%r15),%rdi
   dd1c3:	call   *%r13
   dd1c6:	jmp    dd1b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14d0>
   dd1c8:	cmpq   $0x0,0x2a8(%rsp)
   dd1d1:	je     e03c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e5>
   dd1d7:	mov    %rbx,%rdi
   dd1da:	jmp    e03bf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46df>
   dd1df:	xor    %eax,%eax
   dd1e1:	lea    (%rax,%rax,2),%rcx
   dd1e5:	shl    $0x4,%rcx
   dd1e9:	mov    0x20(%r14,%rcx,1),%rcx
   dd1ee:	mov    0x198(%rsp),%rdx
   dd1f6:	mov    %rcx,(%rdx,%rax,8)
   dd1fa:	mov    %rbx,0x2f8(%rsp)
   dd202:	mov    0x2a8(%rsp),%rcx
   dd20a:	mov    0x20(%rsp),%rdx
   dd20f:	cmpb   $0x1,0x89(%rdx)
   dd216:	mov    %r15,0xa8(%rsp)
   dd21e:	jne    dd43c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x175c>
   dd224:	lea    (%r15,%r15,2),%rax
   dd228:	shl    $0x4,%rax
   dd22c:	add    %r14,%rax
   dd22f:	mov    %r14,0x140(%rsp)
   dd237:	mov    %rcx,0x150(%rsp)
   dd23f:	mov    %rax,0x158(%rsp)
   dd247:	lea    0x30(%r14),%rax
   dd24b:	mov    %rax,0x148(%rsp)
   dd253:	movdqu (%r14),%xmm0
   dd258:	movdqu 0x10(%r14),%xmm1
   dd25e:	movdqu 0x20(%r14),%xmm2
   dd264:	movdqa %xmm2,0x100(%rsp)
   dd26d:	movdqa %xmm1,0xf0(%rsp)
   dd276:	movdqa %xmm0,0xe0(%rsp)
   dd27f:	cmpq   $0x0,0xe0(%rsp)
   dd288:	mov    $0x1,%eax
   dd28d:	cmove  0xf0(%rsp),%rax
   dd296:	lea    0xe0(%rsp),%r15
   dd29e:	mov    0xe8(%rsp),%rcx
   dd2a6:	cmovne %r15,%rcx
   dd2aa:	shl    $0x5,%rax
   dd2ae:	add    %rcx,%rax
   dd2b1:	mov    %rdx,0x1d0(%rsp)
   dd2b9:	movq   $0x0,0x1d8(%rsp)
   dd2c5:	movq   $0x0,0x1f8(%rsp)
   dd2d1:	mov    %rcx,0x218(%rsp)
   dd2d9:	mov    %rax,0x220(%rsp)
   dd2e1:	movq   $0x0,0x228(%rsp)
   dd2ed:	movq   $0x2,0x1c0(%rsp)
   dd2f9:	lea    0x1c0(%rsp),%r14
   dd301:	mov    %r14,%rdi
   dd304:	call   e6bd0 <<itertools::adaptors::coalesce::CoalesceBy<core::iter::adapters::map::Map<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::prepare_reuse::{closure#0}>, itertools::adaptors::coalesce::DedupPred2CoalescePred<itertools::adaptors::coalesce::DedupEq>, itertools::adaptors::coalesce::NoCount> as core::iter::traits::iterator::Iterator>::next>
   dd309:	cmp    $0x1,%rax
   dd30d:	mov    %r14,0x250(%rsp)
   dd315:	jne    ddea3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21c3>
   dd31b:	mov    %rdx,%rbx
   dd31e:	mov    $0x20,%edi
   dd323:	call   *0x2a597(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   dd329:	test   %rax,%rax
   dd32c:	je     e0a52 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d72>
   dd332:	mov    %rax,%r14
   dd335:	mov    %rbx,(%rax)
   dd338:	movq   $0x4,0x160(%rsp)
   dd344:	mov    %rax,0x168(%rsp)
   dd34c:	movq   $0x1,0x170(%rsp)
   dd358:	movups 0x220(%rsp),%xmm0
   dd360:	movaps %xmm0,0x90(%rsp)
   dd368:	movups 0x210(%rsp),%xmm0
   dd370:	movaps %xmm0,0x80(%rsp)
   dd378:	movups 0x200(%rsp),%xmm0
   dd380:	movaps %xmm0,0x70(%rsp)
   dd385:	movdqu 0x1c0(%rsp),%xmm0
   dd38e:	movdqu 0x1d0(%rsp),%xmm1
   dd397:	movdqu 0x1e0(%rsp),%xmm2
   dd3a0:	movups 0x1f0(%rsp),%xmm3
   dd3a8:	movaps %xmm3,0x60(%rsp)
   dd3ad:	movdqa %xmm2,0x50(%rsp)
   dd3b3:	movdqa %xmm1,0x40(%rsp)
   dd3b9:	movdqa %xmm0,0x30(%rsp)
   dd3bf:	mov    $0x1,%r13d
   dd3c5:	lea    0x30(%rsp),%rbp
   dd3ca:	lea    0x160(%rsp),%r12
   dd3d2:	jmp    dd3f7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1717>
   dd3d4:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   dd3e0:	mov    0x168(%rsp),%r14
   dd3e8:	mov    %rbx,(%r14,%r13,8)
   dd3ec:	inc    %r13
   dd3ef:	mov    %r13,0x170(%rsp)
   dd3f7:	mov    %rbp,%rdi
   dd3fa:	call   e6bd0 <<itertools::adaptors::coalesce::CoalesceBy<core::iter::adapters::map::Map<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>, <tk_train::trainers::bpe::corpus::Corpus>::prepare_reuse::{closure#0}>, itertools::adaptors::coalesce::DedupPred2CoalescePred<itertools::adaptors::coalesce::DedupEq>, itertools::adaptors::coalesce::NoCount> as core::iter::traits::iterator::Iterator>::next>
   dd3ff:	cmp    $0x1,%rax
   dd403:	jne    ddf64 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2284>
   dd409:	mov    %rdx,%rbx
   dd40c:	cmp    0x160(%rsp),%r13
   dd414:	jne    dd3e8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1708>
   dd416:	xor    %edx,%edx
   dd418:	cmpq   $0x1,0x30(%rsp)
   dd41e:	sete   %dl
   dd421:	inc    %rdx
   dd424:	mov    $0x8,%ecx
   dd429:	mov    $0x8,%r8d
   dd42f:	mov    %r12,%rdi
   dd432:	mov    %r13,%rsi
   dd435:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   dd43a:	jmp    dd3e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1700>
   dd43c:	mov    0x20(%r14),%ebx
   dd440:	cmp    0x24(%r14),%ebx
   dd444:	jne    ddb31 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e51>
   dd44a:	lea    (%r15,%r15,2),%rax
   dd44e:	shl    $0x4,%rax
   dd452:	add    %r14,%rax
   dd455:	mov    %r14,0x30(%rsp)
   dd45a:	mov    %rcx,0x40(%rsp)
   dd45f:	mov    %rax,0x48(%rsp)
   dd464:	lea    0x30(%r14),%rax
   dd468:	mov    %rax,0x38(%rsp)
   dd46d:	movups (%r14),%xmm0
   dd471:	movups 0x10(%r14),%xmm1
   dd476:	movaps %xmm0,0x2c0(%rsp)
   dd47e:	movaps %xmm1,0x2d0(%rsp)
   dd486:	mov    0x28(%r14),%eax
   dd48a:	mov    %eax,0x250(%rsp)
   dd491:	lea    0x30(%rsp),%rdi
   dd496:	call   e2640 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::corpus::SelectedMerge>>>
   dd49b:	mov    0x20(%rsp),%rcx
   dd4a0:	cmpb   $0x0,0x30(%rcx)
   dd4a4:	mov    $0xffffffffffffffff,%rax
   dd4ab:	je     dd4b1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x17d1>
   dd4ad:	mov    0x38(%rcx),%rax
   dd4b1:	mov    %rcx,0x328(%rsp)
   dd4b9:	mov    %rax,0x330(%rsp)
   dd4c1:	movdqa 0x2c0(%rsp),%xmm0
   dd4ca:	movdqa 0x2d0(%rsp),%xmm1
   dd4d3:	movdqa %xmm0,0x140(%rsp)
   dd4dc:	movdqa %xmm1,0x150(%rsp)
   dd4e5:	mov    0x140(%rsp),%rcx
   dd4ed:	mov    %rcx,%rax
   dd4f0:	sub    $0x1,%rax
   dd4f4:	jae    dd4fe <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x181e>
   dd4f6:	mov    0x158(%rsp),%rax
   dd4fe:	mov    %rax,0x240(%rsp)
   dd506:	movq   $0x0,0x1a8(%rsp)
   dd512:	movq   $0x8,0x1b0(%rsp)
   dd51e:	movq   $0x0,0x1b8(%rsp)
   dd52a:	mov    0x20(%rsp),%rdx
   dd52f:	mov    0x50(%rdx),%rax
   dd533:	cmp    %rbx,%rax
   dd536:	jbe    e0a66 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d86>
   dd53c:	mov    0x48(%rdx),%rax
   dd540:	mov    (%rax,%rbx,8),%r15
   dd544:	test   %rcx,%rcx
   dd547:	mov    $0x1,%r9d
   dd54d:	cmove  0x150(%rsp),%r9
   dd556:	lea    0x140(%rsp),%r8
   dd55e:	cmove  0x148(%rsp),%r8
   dd567:	shl    $0x5,%r9
   dd56b:	add    %r8,%r9
   dd56e:	movq   $0x0,0x30(%rsp)
   dd577:	movq   $0x0,0x50(%rsp)
   dd580:	mov    %r8,0x70(%rsp)
   dd585:	mov    %r9,0x78(%rsp)
   dd58a:	movq   $0x0,0x80(%rsp)
   dd596:	mov    0x10(%rdx),%r14
   dd59a:	movzbl (%rdx),%eax
   dd59d:	mov    %al,0x8(%rsp)
   dd5a1:	mov    0x8(%rdx),%rax
   dd5a5:	mov    %rax,0x10(%rsp)
   dd5aa:	lea    0x299c7(%rip),%rax        # 106f78 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a80>
   dd5b1:	mov    %rax,0x118(%rsp)
   dd5b9:	xor    %r10d,%r10d
   dd5bc:	xor    %eax,%eax
   dd5be:	xor    %r13d,%r13d
   dd5c1:	jmp    dd5fc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x191c>
   dd5c3:	mov    0x1b0(%rsp),%rax
   dd5cb:	mov    %rbp,(%rax,%r13,8)
   dd5cf:	inc    %r13
   dd5d2:	mov    %r13,0x1b8(%rsp)
   dd5da:	mov    %r12,%r13
   dd5dd:	nopl   (%rax)
   dd5e0:	mov    0x30(%rsp),%rax
   dd5e5:	mov    0x38(%rsp),%rdx
   dd5ea:	mov    0x70(%rsp),%r8
   dd5ef:	mov    0x80(%rsp),%r10
   dd5f7:	mov    0x78(%rsp),%r9
   dd5fc:	test   %rax,%rax
   dd5ff:	je     dd610 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1930>
   dd601:	test   %rdx,%rdx
   dd604:	je     dd610 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1930>
   dd606:	mov    0x48(%rsp),%rsi
   dd60b:	jmp    dd91b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c3b>
   dd610:	test   %r8,%r8
   dd613:	je     dda15 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d35>
   dd619:	cmp    %r9,%r8
   dd61c:	je     dda15 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d35>
   dd622:	mov    (%r8),%rdi
   dd625:	dec    %rdi
   dd628:	cmp    %rdi,%r10
   dd62b:	jne    dd692 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19b2>
   dd62d:	movq   $0x0,0x40(%rsp)
   dd636:	lea    0x20(%r8),%r11
   dd63a:	cmp    %r9,%r11
   dd63d:	je     dd7c1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ae1>
   dd643:	mov    (%r11),%rax
   dd646:	dec    %rax
   dd649:	je     dd7c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ae9>
   dd64f:	mov    %eax,%ecx
   dd651:	and    $0x7f,%ecx
   dd654:	shr    $0x7,%rax
   dd658:	cmp    $0x1,%rcx
   dd65c:	sbb    $0xffffffffffffffff,%rax
   dd660:	mov    $0x8,%edi
   dd665:	cmp    $0x2,%rax
   dd669:	jae    dd7cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1aef>
   dd66f:	mov    0x38(%r8),%rax
   dd673:	mov    %rax,%rdx
   dd676:	cmp    $0x11,%rax
   dd67a:	jb     dd7fb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b1b>
   dd680:	mov    0x30(%r8),%rdx
   dd684:	cmp    %rdi,%rdx
   dd687:	jae    dd804 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b24>
   dd68d:	jmp    e05ed <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x490d>
   dd692:	mov    %rdi,%rcx
   dd695:	shr    $0x7,%rcx
   dd699:	test   %r10,%r10
   dd69c:	je     dd704 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a24>
   dd69e:	mov    %edi,%eax
   dd6a0:	and    $0x7f,%eax
   dd6a3:	cmp    $0x1,%rax
   dd6a7:	mov    %rcx,%rax
   dd6aa:	sbb    $0xffffffffffffffff,%rax
   dd6ae:	cmp    $0x2,%rax
   dd6b2:	jb     e0a13 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d33>
   dd6b8:	mov    0x18(%r8),%r11
   dd6bc:	mov    %r11,%rdx
   dd6bf:	cmp    $0x11,%r11
   dd6c3:	jb     dd6c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19e9>
   dd6c5:	mov    0x10(%r8),%rdx
   dd6c9:	lea    0x8(,%rax,8),%rsi
   dd6d1:	cmp    %rdx,%rsi
   dd6d4:	ja     e05d4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48f4>
   dd6da:	mov    %r10,%rdx
   dd6dd:	shr    $0x7,%rdx
   dd6e1:	cmp    %rax,%rdx
   dd6e4:	jae    e0a1e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d3e>
   dd6ea:	lea    0x10(%r8),%rax
   dd6ee:	mov    0x8(%r8),%rsi
   dd6f2:	add    $0x8,%rsi
   dd6f6:	cmp    $0x11,%r11
   dd6fa:	cmovb  %rax,%rsi
   dd6fe:	mov    (%rsi,%rdx,8),%r11
   dd702:	jmp    dd707 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a27>
   dd704:	xor    %r11d,%r11d
   dd707:	and    $0x7f,%edi
   dd70a:	cmp    $0x1,%rdi
   dd70e:	sbb    $0xffffffffffffffff,%rcx
   dd712:	mov    $0x8,%edi
   dd717:	cmp    $0x2,%rcx
   dd71b:	jae    dd73b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a5b>
   dd71d:	mov    0x18(%r8),%rax
   dd721:	mov    %rax,%rsi
   dd724:	cmp    $0x11,%rax
   dd728:	jb     dd763 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a83>
   dd72a:	mov    0x10(%r8),%rsi
   dd72e:	mov    %rsi,%rdx
   dd731:	sub    %rdi,%rdx
   dd734:	jae    dd76f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a8f>
   dd736:	jmp    e0537 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4857>
   dd73b:	dec    %rcx
   dd73e:	bsr    %rcx,%rcx
   dd742:	not    %ecx
   dd744:	mov    $0xffffffffffffffff,%rax
   dd74b:	shr    %cl,%rax
   dd74e:	lea    0x10(,%rax,8),%rdi
   dd756:	mov    0x18(%r8),%rax
   dd75a:	mov    %rax,%rsi
   dd75d:	cmp    $0x11,%rax
   dd761:	jae    dd72a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a4a>
   dd763:	mov    %rsi,%rdx
   dd766:	sub    %rdi,%rdx
   dd769:	jb     e0537 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4857>
   dd76f:	cmp    %r11,%rdx
   dd772:	jb     e05c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48e5>
   dd778:	cmp    $0x11,%rax
   dd77c:	jb     dd79f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1abf>
   dd77e:	mov    0x8(%r8),%rax
   dd782:	lea    0x20(%r8),%rcx
   dd786:	movq   $0x0,0x40(%rsp)
   dd78f:	sub    %r11,%rdx
   dd792:	je     dd7b5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ad5>
   dd794:	add    %r11,%rdi
   dd797:	add    %rdi,%rax
   dd79a:	jmp    dd90d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c2d>
   dd79f:	lea    0x8(%r8),%rax
   dd7a3:	lea    0x20(%r8),%rcx
   dd7a7:	movq   $0x0,0x40(%rsp)
   dd7b0:	sub    %r11,%rdx
   dd7b3:	jne    dd794 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ab4>
   dd7b5:	mov    %rcx,%r11
   dd7b8:	cmp    %r9,%r11
   dd7bb:	jne    dd643 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1963>
   dd7c1:	mov    %r11,%r8
   dd7c4:	jmp    dda0b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d2b>
   dd7c9:	add    $0x40,%r8
   dd7cd:	jmp    dd832 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b52>
   dd7cf:	dec    %rax
   dd7d2:	bsr    %rax,%rcx
   dd7d6:	not    %ecx
   dd7d8:	mov    $0xffffffffffffffff,%rax
   dd7df:	shr    %cl,%rax
   dd7e2:	lea    0x10(,%rax,8),%rdi
   dd7ea:	mov    0x38(%r8),%rax
   dd7ee:	mov    %rax,%rdx
   dd7f1:	cmp    $0x11,%rax
   dd7f5:	jae    dd680 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19a0>
   dd7fb:	cmp    %rdi,%rdx
   dd7fe:	jb     e05ed <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x490d>
   dd804:	cmp    $0x11,%rax
   dd808:	jb     dd825 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b45>
   dd80a:	mov    0x28(%r8),%rax
   dd80e:	add    $0x40,%r8
   dd812:	sub    %rdi,%rdx
   dd815:	je     dd832 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b52>
   dd817:	add    %rdi,%rax
   dd81a:	xor    %r10d,%r10d
   dd81d:	mov    %r8,%rcx
   dd820:	jmp    dd90d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c2d>
   dd825:	lea    0x28(%r8),%rax
   dd829:	add    $0x40,%r8
   dd82d:	sub    %rdi,%rdx
   dd830:	jne    dd817 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b37>
   dd832:	mov    %r8,%rcx
   dd835:	jmp    dd857 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b77>
   dd837:	nopw   0x0(%rax,%rax,1)
   dd840:	mov    $0x1,%eax
   dd845:	xor    %edx,%edx
   dd847:	lea    0x20(%r8),%rcx
   dd84b:	mov    %r8,%r11
   dd84e:	test   %rdx,%rdx
   dd851:	jne    dd90a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c2a>
   dd857:	mov    %rcx,%r8
   dd85a:	cmp    %r9,%rcx
   dd85d:	je     dda08 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d28>
   dd863:	mov    (%r8),%rax
   dd866:	dec    %rax
   dd869:	je     dd840 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b60>
   dd86b:	mov    %eax,%ecx
   dd86d:	and    $0x7f,%ecx
   dd870:	shr    $0x7,%rax
   dd874:	cmp    $0x1,%rcx
   dd878:	sbb    $0xffffffffffffffff,%rax
   dd87c:	mov    $0x8,%edi
   dd881:	cmp    $0x2,%rax
   dd885:	jae    dd8b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bd0>
   dd887:	mov    0x38(%r11),%rax
   dd88b:	mov    %rax,%rsi
   dd88e:	cmp    $0x11,%rax
   dd892:	jb     dd8d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bf8>
   dd894:	mov    0x30(%r11),%rsi
   dd898:	mov    %rsi,%rdx
   dd89b:	sub    %rdi,%rdx
   dd89e:	jae    dd8e4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c04>
   dd8a0:	jmp    e0537 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4857>
   dd8a5:	data16 cs nopw 0x0(%rax,%rax,1)
   dd8b0:	dec    %rax
   dd8b3:	bsr    %rax,%rcx
   dd8b7:	not    %ecx
   dd8b9:	mov    $0xffffffffffffffff,%rax
   dd8c0:	shr    %cl,%rax
   dd8c3:	lea    0x10(,%rax,8),%rdi
   dd8cb:	mov    0x38(%r11),%rax
   dd8cf:	mov    %rax,%rsi
   dd8d2:	cmp    $0x11,%rax
   dd8d6:	jae    dd894 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bb4>
   dd8d8:	mov    %rsi,%rdx
   dd8db:	sub    %rdi,%rdx
   dd8de:	jb     e0537 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4857>
   dd8e4:	cmp    $0x11,%rax
   dd8e8:	jb     dd8f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c10>
   dd8ea:	mov    0x28(%r11),%rax
   dd8ee:	jmp    dd8f7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c17>
   dd8f0:	add    $0x28,%r11
   dd8f4:	mov    %r11,%rax
   dd8f7:	add    %rdi,%rax
   dd8fa:	lea    0x20(%r8),%rcx
   dd8fe:	mov    %r8,%r11
   dd901:	test   %rdx,%rdx
   dd904:	je     dd857 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b77>
   dd90a:	xor    %r10d,%r10d
   dd90d:	mov    %r10,0x48(%rsp)
   dd912:	mov    %r10,%rsi
   dd915:	xor    %r10d,%r10d
   dd918:	mov    %rcx,%r8
   dd91b:	mov    %rax,0x30(%rsp)
   dd920:	mov    %rdx,0x38(%rsp)
   dd925:	mov    %r8,0x70(%rsp)
   dd92a:	mov    %r10,0x80(%rsp)
   dd932:	test   $0x7f,%sil
   dd936:	je     dd950 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c70>
   dd938:	movsbq (%rax),%rbp
   dd93c:	test   %rbp,%rbp
   dd93f:	js     dd970 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c90>
   dd941:	mov    $0x1,%edi
   dd946:	jmp    dd9d2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cf2>
   dd94b:	nopl   0x0(%rax,%rax,1)
   dd950:	cmp    $0x7,%rdx
   dd954:	jbe    e0587 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48a7>
   dd95a:	mov    (%rax),%rbp
   dd95d:	mov    %rbp,0x40(%rsp)
   dd962:	add    $0xfffffffffffffff8,%rdx
   dd966:	mov    $0x8,%edi
   dd96b:	mov    %rdx,%rcx
   dd96e:	jmp    dd9e8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d08>
   dd970:	cmp    $0x1,%rdx
   dd974:	je     e0916 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c36>
   dd97a:	and    $0x7f,%ebp
   dd97d:	lea    -0x1(%rdx),%r8
   dd981:	mov    $0x7,%ecx
   dd986:	xor    %edi,%edi
   dd988:	nopl   0x0(%rax,%rax,1)
   dd990:	movzbl 0x1(%rax,%rdi,1),%r9d
   dd996:	cmp    $0x3f,%ecx
   dd999:	jne    dd9a5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cc5>
   dd99b:	cmp    $0x1,%r9b
   dd99f:	ja     e091a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c3a>
   dd9a5:	movzbl %r9b,%r9d
   dd9a9:	test   %r9b,%r9b
   dd9ac:	jns    dd9c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ce8>
   dd9ae:	and    $0x7f,%r9d
   dd9b2:	shl    %cl,%r9
   dd9b5:	or     %r9,%rbp
   dd9b8:	add    $0x7,%ecx
   dd9bb:	inc    %rdi
   dd9be:	cmp    %rdi,%r8
   dd9c1:	jne    dd990 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cb0>
   dd9c3:	jmp    e0916 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c36>
   dd9c8:	shl    %cl,%r9
   dd9cb:	or     %r9,%rbp
   dd9ce:	add    $0x2,%rdi
   dd9d2:	add    0x40(%rsp),%rbp
   dd9d7:	mov    %rbp,0x40(%rsp)
   dd9dc:	mov    %rdx,%rcx
   dd9df:	sub    %rdi,%rcx
   dd9e2:	jb     e0566 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4886>
   dd9e8:	add    %rdi,%rax
   dd9eb:	mov    %rax,0x30(%rsp)
   dd9f0:	mov    %rcx,0x38(%rsp)
   dd9f5:	inc    %rsi
   dd9f8:	mov    %rsi,0x48(%rsp)
   dd9fd:	cmp    %r13,%rbp
   dda00:	jb     dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   dda06:	jmp    dda5c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d7c>
   dda08:	xor    %r10d,%r10d
   dda0b:	mov    %r10,0x48(%rsp)
   dda10:	xor    %r10d,%r10d
   dda13:	xor    %edx,%edx
   dda15:	movq   $0x0,0x30(%rsp)
   dda1e:	mov    %rdx,0x38(%rsp)
   dda23:	mov    %r8,0x70(%rsp)
   dda28:	mov    %r10,0x80(%rsp)
   dda30:	cmpq   $0x0,0x50(%rsp)
   dda36:	je     ddfc4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x22e4>
   dda3c:	lea    0x50(%rsp),%rdi
   dda41:	call   e2770 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   dda46:	cmp    $0x1,%rax
   dda4a:	jne    ddfc4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x22e4>
   dda50:	mov    %rdx,%rbp
   dda53:	cmp    %r13,%rbp
   dda56:	jb     dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   dda5c:	cmp    %r14,%rbp
   dda5f:	jae    e0546 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4866>
   dda65:	cmpb   $0x0,0x8(%rsp)
   dda6a:	je     dda80 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1da0>
   dda6c:	mov    0x10(%rsp),%rax
   dda71:	mov    (%rax,%rbp,4),%eax
   dda74:	test   %r15,%r15
   dda77:	jne    ddaa1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1dc1>
   dda79:	jmp    dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   dda7e:	xchg   %ax,%ax
   dda80:	mov    0x10(%rsp),%rax
   dda85:	movzwl (%rax,%rbp,2),%eax
   dda89:	cmp    $0xffff,%ax
   dda8d:	movzwl %ax,%eax
   dda90:	mov    $0xffffffff,%ecx
   dda95:	cmove  %ecx,%eax
   dda98:	test   %r15,%r15
   dda9b:	je     dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   ddaa1:	cmp    %ebx,%eax
   ddaa3:	jne    dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   ddaa9:	lea    (%r15,%r15,1),%rax
   ddaad:	lea    (%rax,%rbp,1),%r12
   ddab1:	cmp    %r14,%r12
   ddab4:	jae    dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   ddaba:	lea    (%r15,%rbp,1),%rax
   ddabe:	cmpb   $0x0,0x8(%rsp)
   ddac3:	je     ddae0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e00>
   ddac5:	cmp    %r14,%rax
   ddac8:	jae    e0a23 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d43>
   ddace:	mov    0x10(%rsp),%rcx
   ddad3:	mov    (%rcx,%rax,4),%eax
   ddad6:	cmp    %ebx,%eax
   ddad8:	jne    dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   ddade:	jmp    ddb09 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e29>
   ddae0:	cmp    %r14,%rax
   ddae3:	jae    e0a32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d52>
   ddae9:	mov    0x10(%rsp),%rcx
   ddaee:	movzwl (%rcx,%rax,2),%eax
   ddaf2:	cmp    $0xffff,%ax
   ddaf6:	movzwl %ax,%eax
   ddaf9:	mov    $0xffffffff,%ecx
   ddafe:	cmove  %ecx,%eax
   ddb01:	cmp    %ebx,%eax
   ddb03:	jne    dd5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1900>
   ddb09:	mov    0x1b8(%rsp),%r13
   ddb11:	cmp    0x1a8(%rsp),%r13
   ddb19:	jne    dd5c3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x18e3>
   ddb1f:	lea    0x1a8(%rsp),%rdi
   ddb27:	call   79bd0 <<alloc::raw_vec::RawVec<u64>>::grow_one>
   ddb2c:	jmp    dd5c3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x18e3>
   ddb31:	mov    %rcx,0x290(%rsp)
   ddb39:	mov    0x2b8(%rsp),%rax
   ddb41:	mov    %rax,0x2d0(%rsp)
   ddb49:	movdqu 0x2a8(%rsp),%xmm0
   ddb52:	movdqa %xmm0,0x2c0(%rsp)
   ddb5b:	mov    0x20(%rsp),%rcx
   ddb60:	cmpb   $0x0,0x30(%rcx)
   ddb64:	mov    $0xffffffffffffffff,%rax
   ddb6b:	je     ddb71 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e91>
   ddb6d:	mov    0x38(%rcx),%rax
   ddb71:	mov    %rcx,0x328(%rsp)
   ddb79:	mov    %rax,0x330(%rsp)
   ddb81:	mov    0x2c0c0(%rip),%rax        # 109c48 <ahash::random_state::RAND_SOURCE>
   ddb88:	test   %rax,%rax
   ddb8b:	je     e0733 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a53>
   ddb91:	mov    (%rax),%r15
   ddb94:	mov    0x8(%rax),%r12
   ddb98:	mov    0x2c0b1(%rip),%rbx        # 109c50 <ahash::random_state::get_fixed_seeds::SEEDS>
   ddb9f:	test   %rbx,%rbx
   ddba2:	je     e073d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a5d>
   ddba8:	mov    %r15,%rdi
   ddbab:	call   *0x18(%r12)
   ddbb0:	mov    %r13,0x10(%rsp)
   ddbb5:	xor    0x8(%rbx),%rax
   ddbb9:	movabs $0x5851f42d4c957f2d,%r12
   ddbc3:	mul    %r12
   ddbc6:	mov    %rax,%rdi
   ddbc9:	xor    %rdx,%rdi
   ddbcc:	mov    (%rbx),%r10
   ddbcf:	mov    0x20(%rbx),%r11
   ddbd3:	mov    0x30(%rbx),%r9
   ddbd7:	mov    %r11,%rax
   ddbda:	xor    %rdi,%rax
   ddbdd:	mul    %r12
   ddbe0:	xor    %r9,%rdx
   ddbe3:	xor    %rdx,%rax
   ddbe6:	mul    %r12
   ddbe9:	mov    %rax,%rcx
   ddbec:	xor    %rdx,%rcx
   ddbef:	mov    %rcx,%rax
   ddbf2:	mul    %r10
   ddbf5:	mov    %rdx,%rsi
   ddbf8:	xor    %rax,%rsi
   ddbfb:	rol    %cl,%rsi
   ddbfe:	mov    0x28(%rbx),%r15
   ddc02:	mov    0x38(%rbx),%rbx
   ddc06:	mov    %r15,%rax
   ddc09:	xor    %rdi,%rax
   ddc0c:	mul    %r12
   ddc0f:	xor    %rbx,%rdx
   ddc12:	xor    %rdx,%rax
   ddc15:	mul    %r12
   ddc18:	mov    %rax,%rcx
   ddc1b:	xor    %rdx,%rcx
   ddc1e:	mov    %rcx,%rax
   ddc21:	mul    %r10
   ddc24:	mov    %rdx,%r8
   ddc27:	xor    %rax,%r8
   ddc2a:	rol    %cl,%r8
   ddc2d:	xor    %rdi,%r9
   ddc30:	mov    %r9,%rax
   ddc33:	mul    %r12
   ddc36:	xor    %rdx,%r15
   ddc39:	xor    %r15,%rax
   ddc3c:	mul    %r12
   ddc3f:	mov    %rax,%rcx
   ddc42:	xor    %rdx,%rcx
   ddc45:	mov    %rcx,%rax
   ddc48:	mul    %r10
   ddc4b:	mov    %rdx,%r9
   ddc4e:	xor    %rax,%r9
   ddc51:	rol    %cl,%r9
   ddc54:	xor    %rbx,%rdi
   ddc57:	mov    %rdi,%rax
   ddc5a:	mul    %r12
   ddc5d:	xor    %rdx,%r11
   ddc60:	xor    %r11,%rax
   ddc63:	mul    %r12
   ddc66:	mov    %rax,%rcx
   ddc69:	xor    %rdx,%rcx
   ddc6c:	mov    %rcx,%rax
   ddc6f:	mul    %r10
   ddc72:	xor    %rax,%rdx
   ddc75:	rol    %cl,%rdx
   ddc78:	movaps 0x28c11(%rip),%xmm0        # 106890 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   ddc7f:	movaps %xmm0,0x40(%rsp)
   ddc84:	movdqa 0x28bf4(%rip),%xmm0        # 106880 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   ddc8c:	movdqa %xmm0,0x30(%rsp)
   ddc92:	lea    0x50(%rsp),%r12
   ddc97:	mov    %rsi,0x50(%rsp)
   ddc9c:	mov    %r8,0x58(%rsp)
   ddca1:	mov    %r9,0x60(%rsp)
   ddca6:	mov    %rdx,0x68(%rsp)
   ddcab:	mov    0x48(%rsp),%rax
   ddcb0:	mov    0xa8(%rsp),%rsi
   ddcb8:	mov    %rsi,%rcx
   ddcbb:	shr    $1,%rcx
   ddcbe:	test   %rax,%rax
   ddcc1:	cmove  %rax,%rcx
   ddcc5:	sub    %rcx,%rsi
   ddcc8:	cmp    0x40(%rsp),%rsi
   ddccd:	ja     e074a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a6a>
   ddcd3:	xor    %ebx,%ebx
   ddcd5:	jmp    dddf5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2115>
   ddcda:	nopw   0x0(%rax,%rax,1)
   ddce0:	and    %rdx,%rbp
   ddce3:	movdqu (%rax,%rbp,1),%xmm1
   ddce8:	movdqa %xmm1,%xmm2
   ddcec:	pcmpeqb %xmm0,%xmm2
   ddcf0:	pmovmskb %xmm2,%r9d
   ddcf5:	test   %r9d,%r9d
   ddcf8:	je     ddd30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2050>
   ddcfa:	tzcnt  %r9d,%r10d
   ddcff:	add    %rbp,%r10
   ddd02:	and    %rdx,%r10
   ddd05:	neg    %r10
   ddd08:	lea    (%r10,%r10,2),%r11
   ddd0c:	cmp    -0xc(%rax,%r11,4),%r15d
   ddd11:	jne    ddd1e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x203e>
   ddd13:	cmp    -0x8(%rax,%r11,4),%r13d
   ddd18:	je     dddd5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x20f5>
   ddd1e:	lea    -0x1(%r9),%r10d
   ddd22:	and    %r9w,%r10w
   ddd26:	mov    %r10d,%r9d
   ddd29:	jne    ddcfa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x201a>
   ddd2b:	nopl   0x0(%rax,%rax,1)
   ddd30:	cmp    $0x1,%r8
   ddd34:	je     ddd48 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2068>
   ddd36:	pmovmskb %xmm1,%esi
   ddd3a:	test   %esi,%esi
   ddd3c:	je     ddd62 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2082>
   ddd3e:	tzcnt  %esi,%esi
   ddd42:	add    %rbp,%rsi
   ddd45:	and    %rdx,%rsi
   ddd48:	pcmpeqd %xmm2,%xmm2
   ddd4c:	pcmpeqb %xmm2,%xmm1
   ddd50:	pmovmskb %xmm1,%r8d
   ddd55:	test   %r8d,%r8d
   ddd58:	jne    ddd80 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x20a0>
   ddd5a:	mov    $0x1,%r8d
   ddd60:	jmp    ddd65 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2085>
   ddd62:	xor    %r8d,%r8d
   ddd65:	add    %rdi,%rbp
   ddd68:	add    $0x10,%rbp
   ddd6c:	add    $0x10,%rdi
   ddd70:	jmp    ddce0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2000>
   ddd75:	data16 cs nopw 0x0(%rax,%rax,1)
   ddd80:	movzbl (%rax,%rsi,1),%edi
   ddd84:	test   %dil,%dil
   ddd87:	jns    dde8e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21ae>
   ddd8d:	and    $0x1,%dil
   ddd91:	lea    -0x10(%rsi),%r8
   ddd95:	and    %rdx,%r8
   ddd98:	mov    %cl,(%rax,%rsi,1)
   ddd9b:	mov    %cl,0x10(%rax,%r8,1)
   ddda0:	movdqa 0x40(%rsp),%xmm0
   ddda6:	movzbl %dil,%ecx
   dddaa:	movd   %ecx,%xmm1
   dddae:	shufpd $0x2,-0xd2447(%rip),%xmm1        # b970 <GCC_except_table1289+0xe8>
   dddb7:	psubq  %xmm1,%xmm0
   dddbb:	movdqa %xmm0,0x40(%rsp)
   dddc1:	neg    %rsi
   dddc4:	lea    (%rsi,%rsi,2),%rcx
   dddc8:	mov    %r15d,-0xc(%rax,%rcx,4)
   dddcd:	mov    %r13d,-0x8(%rax,%rcx,4)
   dddd2:	mov    %rsi,%r10
   dddd5:	lea    (%r10,%r10,2),%rcx
   dddd9:	mov    0x8(%rsp),%edx
   ddddd:	mov    %edx,-0x4(%rax,%rcx,4)
   ddde1:	inc    %rbx
   ddde4:	mov    0xa8(%rsp),%r15
   dddec:	cmp    %r15,%rbx
   dddef:	je     dded3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21f3>
   dddf5:	lea    (%rbx,%rbx,2),%rsi
   dddf9:	shl    $0x4,%rsi
   dddfd:	mov    0x20(%r14,%rsi,1),%r15d
   dde02:	mov    0x58(%rsp),%rax
   dde07:	xor    %r15,%rax
   dde0a:	movabs $0x5851f42d4c957f2d,%rcx
   dde14:	mul    %rcx
   dde17:	mov    0x24(%r14,%rsi,1),%r13d
   dde1c:	xor    %r13,%rdx
   dde1f:	xor    %rdx,%rax
   dde22:	mul    %rcx
   dde25:	mov    %rax,%rcx
   dde28:	xor    %rdx,%rcx
   dde2b:	mov    %rcx,%rax
   dde2e:	mulq   0x50(%rsp)
   dde33:	mov    %rdx,%rbp
   dde36:	xor    %rax,%rbp
   dde39:	rol    %cl,%rbp
   dde3c:	mov    0x28(%r14,%rsi,1),%eax
   dde41:	mov    %eax,0x8(%rsp)
   dde45:	cmpq   $0x0,0x40(%rsp)
   dde4b:	je     dde7a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x219a>
   dde4d:	mov    0x30(%rsp),%rax
   dde52:	mov    0x38(%rsp),%rdx
   dde57:	mov    %rbp,%rcx
   dde5a:	shr    $0x39,%rcx
   dde5e:	movd   %ecx,%xmm0
   dde62:	punpcklbw %xmm0,%xmm0
   dde66:	pshuflw $0x0,%xmm0,%xmm0
   dde6b:	pshufd $0x44,%xmm0,%xmm0
   dde70:	xor    %r8d,%r8d
   dde73:	xor    %edi,%edi
   dde75:	jmp    ddce0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2000>
   dde7a:	mov    $0x1,%esi
   dde7f:	lea    0x30(%rsp),%rdi
   dde84:	mov    %r12,%rdx
   dde87:	call   c4660 <<hashbrown::raw::RawTable<((u32, u32), u32)>>::reserve_rehash::<hashbrown::map::make_hasher<(u32, u32), u32, ahash::random_state::RandomState>::{closure#0}>>
   dde8c:	jmp    dde4d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x216d>
   dde8e:	movdqa (%rax),%xmm0
   dde92:	pmovmskb %xmm0,%esi
   dde96:	tzcnt  %esi,%esi
   dde9a:	movzbl (%rax,%rsi,1),%edi
   dde9e:	jmp    ddd8d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x20ad>
   ddea3:	mov    $0x1,%r14b
   ddea6:	mov    $0x8,%ebx
   ddeab:	xor    %r13d,%r13d
   ddeae:	mov    %fs:0xffffffffffffffd0,%rax
   ddeb7:	test   %rax,%rax
   ddeba:	mov    %rbx,0x240(%rsp)
   ddec2:	je     ddf93 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x22b3>
   ddec8:	add    $0x110,%rax
   ddece:	jmp    ddf98 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x22b8>
   dded3:	movdqa 0x30(%rsp),%xmm0
   dded9:	movdqa 0x40(%rsp),%xmm1
   ddedf:	movdqa 0x50(%rsp),%xmm2
   ddee5:	movaps 0x60(%rsp),%xmm3
   ddeea:	movaps %xmm3,0x1f0(%rsp)
   ddef2:	movdqa %xmm2,0x1e0(%rsp)
   ddefb:	movdqa %xmm1,0x1d0(%rsp)
   ddf04:	movdqa %xmm0,0x1c0(%rsp)
   ddf0d:	cmpq   $0x0,0x10(%rsp)
   ddf13:	je     de518 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2838>
   ddf19:	movabs $0x3fffffffffffffe,%rdi
   ddf23:	and    %r15,%rdi
   ddf26:	xor    %eax,%eax
   ddf28:	mov    %r14,%rcx
   ddf2b:	xor    %ebx,%ebx
   ddf2d:	jmp    ddf47 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2267>
   ddf2f:	nop
   ddf30:	add    %rdx,%rbx
   ddf33:	add    %rsi,%rbx
   ddf36:	add    $0x2,%rax
   ddf3a:	add    $0x60,%rcx
   ddf3e:	cmp    %rax,%rdi
   ddf41:	je     de360 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2680>
   ddf47:	mov    (%rcx),%rdx
   ddf4a:	sub    $0x1,%rdx
   ddf4e:	jae    ddf54 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2274>
   ddf50:	mov    0x18(%rcx),%rdx
   ddf54:	mov    0x30(%rcx),%rsi
   ddf58:	sub    $0x1,%rsi
   ddf5c:	jae    ddf30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2250>
   ddf5e:	mov    0x48(%rcx),%rsi
   ddf62:	jmp    ddf30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2250>
   ddf64:	mov    0x168(%rsp),%rbx
   ddf6c:	cmpq   $0x0,0x160(%rsp)
   ddf75:	sete   %r14b
   ddf79:	mov    %fs:0xffffffffffffffd0,%rax
   ddf82:	test   %rax,%rax
   ddf85:	mov    %rbx,0x240(%rsp)
   ddf8d:	jne    ddec8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21e8>
   ddf93:	call   7f660 <rayon_core::registry::global_registry>
   ddf98:	mov    (%rax),%rax
   ddf9b:	mov    0x208(%rax),%rcx
   ddfa2:	test   %rcx,%rcx
   ddfa5:	je     e0a41 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d61>
   ddfab:	mov    %r13,%rax
   ddfae:	or     %rcx,%rax
   ddfb1:	shr    $0x20,%rax
   ddfb5:	je     ddff8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2318>
   ddfb7:	mov    %r13,%rax
   ddfba:	xor    %edx,%edx
   ddfbc:	div    %rcx
   ddfbf:	mov    %rax,%rcx
   ddfc2:	jmp    de001 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2321>
   ddfc4:	mov    0x148(%rsp),%r13
   ddfcc:	cmpq   $0x0,0x140(%rsp)
   ddfd5:	je     de937 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c57>
   ddfdb:	cmpq   $0x10,0x158(%rsp)
   ddfe4:	jbe    de97c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c9c>
   ddfea:	mov    %r13,%rdi
   ddfed:	call   *0x298b5(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ddff3:	jmp    de97c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c9c>
   ddff8:	mov    %r13d,%eax
   ddffb:	xor    %edx,%edx
   ddffd:	div    %ecx
   ddfff:	mov    %eax,%ecx
   de001:	xor    %eax,%eax
   de003:	test   %rdx,%rdx
   de006:	setne  %al
   de009:	lea    (%rcx,%rax,1),%rdx
   de00d:	cmp    $0x1,%rdx
   de011:	adc    %rax,%rcx
   de014:	mov    0x20(%rsp),%r8
   de019:	cmpb   $0x0,0x30(%r8)
   de01e:	mov    $0xffffffffffffffff,%rax
   de025:	je     de02b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x234b>
   de027:	mov    0x38(%r8),%rax
   de02b:	mov    %rax,0x270(%rsp)
   de033:	movl   $0x0,0xc8(%rsp)
   de03e:	movb   $0x0,0xcc(%rsp)
   de046:	movq   $0x0,0xd0(%rsp)
   de052:	movq   $0x0,0x120(%rsp)
   de05e:	movq   $0x8,0x128(%rsp)
   de06a:	movq   $0x0,0x130(%rsp)
   de076:	movb   $0x0,0x1a8(%rsp)
   de07e:	mov    %rbx,0x30(%rsp)
   de083:	mov    %r13,0x38(%rsp)
   de088:	mov    %rcx,0x40(%rsp)
   de08d:	mov    %r15,0x48(%rsp)
   de092:	mov    %r8,0x50(%rsp)
   de097:	lea    0x270(%rsp),%rax
   de09f:	mov    %rax,0x58(%rsp)
   de0a4:	lea    0x60(%rsp),%rsi
   de0a9:	lea    0xc8(%rsp),%rdx
   de0b1:	mov    %rdx,0x60(%rsp)
   de0b6:	mov    %rbx,0x1c0(%rsp)
   de0be:	mov    %r13,0x1c8(%rsp)
   de0c6:	mov    %rcx,0x1d0(%rsp)
   de0ce:	lea    0x1d8(%rsp),%rdi
   de0d6:	mov    %r15,0x1d8(%rsp)
   de0de:	mov    %r8,0x1e0(%rsp)
   de0e6:	mov    %rax,0x1e8(%rsp)
   de0ee:	lea    0x1f0(%rsp),%r8
   de0f6:	mov    %r13,%rax
   de0f9:	or     %rcx,%rax
   de0fc:	shr    $0x20,%rax
   de100:	je     de10f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x242f>
   de102:	mov    %r13,%rax
   de105:	xor    %edx,%edx
   de107:	div    %rcx
   de10a:	mov    %rax,%rbp
   de10d:	jmp    de118 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2438>
   de10f:	mov    %r13d,%eax
   de112:	xor    %edx,%edx
   de114:	div    %ecx
   de116:	mov    %eax,%ebp
   de118:	cmp    $0x1,%rdx
   de11c:	sbb    $0xffffffffffffffff,%rbp
   de120:	mov    %rbp,0x180(%rsp)
   de128:	lea    0x1a8(%rsp),%rax
   de130:	mov    %rax,0x160(%rsp)
   de138:	mov    %rsi,0x168(%rsp)
   de140:	mov    %r8,0x170(%rsp)
   de148:	mov    %rdi,0x178(%rsp)
   de150:	mov    %rcx,0x2d0(%rsp)
   de158:	mov    %rbx,0x2c0(%rsp)
   de160:	mov    %r13,0x2c8(%rsp)
   de168:	mov    %fs:0xffffffffffffffd0,%rax
   de171:	test   %rax,%rax
   de174:	mov    %r14b,0x10(%rsp)
   de179:	je     de183 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x24a3>
   de17b:	add    $0x110,%rax
   de181:	jmp    de188 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x24a8>
   de183:	call   7f660 <rayon_core::registry::global_registry>
   de188:	mov    (%rax),%rax
   de18b:	mov    0x208(%rax),%rcx
   de192:	sub    $0x8,%rsp
   de196:	lea    0x168(%rsp),%rax
   de19e:	lea    0xb8(%rsp),%r13
   de1a6:	lea    0x2c8(%rsp),%r9
   de1ae:	mov    $0x1,%r8d
   de1b4:	mov    %r13,%rdi
   de1b7:	mov    %rbp,%rsi
   de1ba:	xor    %edx,%edx
   de1bc:	push   %rax
   de1bd:	call   e70f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::chunks::ChunksProducer<usize>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_reuse::{closure#1}>>>
   de1c2:	add    $0x10,%rsp
   de1c6:	mov    0xc0(%rsp),%rax
   de1ce:	test   %rax,%rax
   de1d1:	je     de20b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x252b>
   de1d3:	mov    0xb0(%rsp),%rcx
   de1db:	test   %rcx,%rcx
   de1de:	je     de20b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x252b>
   de1e0:	xor    %edx,%edx
   de1e2:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de1f0:	add    0x10(%rcx),%rdx
   de1f4:	dec    %rax
   de1f7:	je     de202 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2522>
   de1f9:	mov    0x18(%rcx),%rcx
   de1fd:	test   %rcx,%rcx
   de200:	jne    de1f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2510>
   de202:	test   %rdx,%rdx
   de205:	jne    e05f5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4915>
   de20b:	movdqu 0xb0(%rsp),%xmm0
   de214:	movdqa %xmm0,0x1c0(%rsp)
   de21d:	mov    0xc0(%rsp),%r15
   de225:	mov    %r15,0x1d0(%rsp)
   de22d:	mov    0x1c0(%rsp),%rdi
   de235:	mov    0x1c8(%rsp),%rax
   de23d:	test   %rdi,%rdi
   de240:	je     de35b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x267b>
   de246:	mov    %rax,0x8(%rsp)
   de24b:	dec    %r15
   de24e:	mov    %r15,%rax
   de251:	jmp    de270 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2590>
   de253:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de260:	lea    -0x1(%r15),%rax
   de264:	mov    %r14,%rdi
   de267:	test   %r14,%r14
   de26a:	je     de387 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x26a7>
   de270:	mov    %rax,%r15
   de273:	mov    0x18(%rdi),%r14
   de277:	test   %r14,%r14
   de27a:	je     de290 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x25b0>
   de27c:	movq   $0x0,0x20(%r14)
   de284:	jmp    de299 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x25b9>
   de286:	cs nopw 0x0(%rax,%rax,1)
   de290:	movq   $0x0,0x8(%rsp)
   de299:	mov    (%rdi),%r12
   de29c:	movdqu 0x8(%rdi),%xmm0
   de2a1:	movdqa %xmm0,0x160(%rsp)
   de2aa:	call   *0x295f8(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   de2b0:	cmp    $0xffffffffffffffff,%r12
   de2b4:	je     de38a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x26aa>
   de2ba:	mov    %r12,0x30(%rsp)
   de2bf:	movdqa 0x160(%rsp),%xmm0
   de2c8:	lea    0x38(%rsp),%rax
   de2cd:	movdqu %xmm0,(%rax)
   de2d1:	mov    0x38(%rsp),%rbx
   de2d6:	mov    0x40(%rsp),%rbp
   de2db:	mov    0x120(%rsp),%rax
   de2e3:	mov    0x130(%rsp),%r13
   de2eb:	sub    %r13,%rax
   de2ee:	cmp    %rax,%rbp
   de2f1:	ja     de333 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2653>
   de2f3:	test   %rbp,%rbp
   de2f6:	je     de311 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2631>
   de2f8:	imul   $0x38,%r13,%rdi
   de2fc:	add    0x128(%rsp),%rdi
   de304:	imul   $0x38,%rbp,%rdx
   de308:	mov    %rbx,%rsi
   de30b:	call   *0x2959f(%rip)        # 1078b0 <memcpy@GLIBC_2.14>
   de311:	add    %rbp,%r13
   de314:	mov    %r13,0x130(%rsp)
   de31c:	test   %r12,%r12
   de31f:	je     de260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2580>
   de325:	mov    %rbx,%rdi
   de328:	call   *0x2957a(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   de32e:	jmp    de260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2580>
   de333:	mov    $0x8,%ecx
   de338:	mov    $0x38,%r8d
   de33e:	lea    0x120(%rsp),%rdi
   de346:	mov    %r13,%rsi
   de349:	mov    %rbp,%rdx
   de34c:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   de351:	mov    0x130(%rsp),%r13
   de359:	jmp    de2f8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2618>
   de35b:	xor    %r14d,%r14d
   de35e:	jmp    de38f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x26af>
   de360:	test   $0x1,%r15b
   de364:	jne    de51c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x283c>
   de36a:	mov    %fs:0xffffffffffffffd0,%rax
   de373:	test   %rax,%rax
   de376:	je     de548 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2868>
   de37c:	add    $0x110,%rax
   de382:	jmp    de54d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x286d>
   de387:	xor    %r14d,%r14d
   de38a:	mov    0x8(%rsp),%rax
   de38f:	mov    %r14,0x1c0(%rsp)
   de397:	mov    %rax,0x1c8(%rsp)
   de39f:	mov    %r15,0x1d0(%rsp)
   de3a7:	lea    0x1c0(%rsp),%rdi
   de3af:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   de3b4:	movdqu 0x120(%rsp),%xmm0
   de3bd:	movdqa %xmm0,0x30(%rsp)
   de3c3:	mov    0x130(%rsp),%rax
   de3cb:	mov    %rax,0x40(%rsp)
   de3d0:	mov    0xd0(%rsp),%r13
   de3d8:	mov    0xd8(%rsp),%rbp
   de3e0:	cmpb   $0x0,0xcc(%rsp)
   de3e8:	jne    e06d2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x49f2>
   de3ee:	test   %r13,%r13
   de3f1:	je     de491 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27b1>
   de3f7:	mov    0x38(%rsp),%rbx
   de3fc:	mov    0x40(%rsp),%r14
   de401:	test   %r14,%r14
   de404:	je     de441 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2761>
   de406:	mov    0x2949b(%rip),%r12        # 1078a8 <free@GLIBC_2.2.5>
   de40d:	mov    %rbx,%r15
   de410:	jmp    de431 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2751>
   de412:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de420:	mov    %r15,%rdi
   de423:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   de428:	add    $0x38,%r15
   de42c:	dec    %r14
   de42f:	je     de441 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2761>
   de431:	cmpq   $0x3,0x28(%r15)
   de436:	jb     de420 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2740>
   de438:	mov    0x18(%r15),%rdi
   de43c:	call   *%r12
   de43f:	jmp    de420 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2740>
   de441:	mov    $0xffffffffffffffff,%r15
   de448:	cmpq   $0x0,0x30(%rsp)
   de44e:	mov    0x240(%rsp),%r14
   de456:	je     de461 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2781>
   de458:	mov    %rbx,%rdi
   de45b:	call   *0x29447(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   de461:	cmpb   $0x0,0x10(%rsp)
   de466:	jne    de471 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2791>
   de468:	mov    %r14,%rdi
   de46b:	call   *0x29437(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   de471:	cmpq   $0x0,0xe0(%rsp)
   de47a:	je     de4b1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27d1>
   de47c:	cmpq   $0x11,0xf8(%rsp)
   de485:	jb     de506 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2826>
   de487:	mov    0xe8(%rsp),%rbx
   de48f:	jmp    de4fd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x281d>
   de491:	mov    0x30(%rsp),%r15
   de496:	mov    0x38(%rsp),%r13
   de49b:	mov    0x40(%rsp),%rbp
   de4a0:	mov    0x240(%rsp),%r14
   de4a8:	cmpb   $0x0,0x10(%rsp)
   de4ad:	je     de468 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2788>
   de4af:	jmp    de471 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2791>
   de4b1:	mov    0xf0(%rsp),%r14
   de4b9:	test   %r14,%r14
   de4bc:	je     de506 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2826>
   de4be:	mov    %r15,0x10(%rsp)
   de4c3:	mov    0xe8(%rsp),%rbx
   de4cb:	lea    0x18(%rbx),%r15
   de4cf:	mov    0x293d2(%rip),%r12        # 1078a8 <free@GLIBC_2.2.5>
   de4d6:	jmp    de4e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2809>
   de4d8:	nopl   0x0(%rax,%rax,1)
   de4e0:	add    $0x20,%r15
   de4e4:	dec    %r14
   de4e7:	je     de4f8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2818>
   de4e9:	cmpq   $0x11,(%r15)
   de4ed:	jb     de4e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2800>
   de4ef:	mov    -0x10(%r15),%rdi
   de4f3:	call   *%r12
   de4f6:	jmp    de4e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2800>
   de4f8:	mov    0x10(%rsp),%r15
   de4fd:	mov    %rbx,%rdi
   de500:	call   *0x293a2(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   de506:	lea    0x140(%rsp),%rdi
   de50e:	call   e2640 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::corpus::SelectedMerge>>>
   de513:	jmp    df302 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3622>
   de518:	xor    %eax,%eax
   de51a:	xor    %ebx,%ebx
   de51c:	lea    (%rax,%rax,2),%rcx
   de520:	shl    $0x4,%rcx
   de524:	mov    (%r14,%rcx,1),%rax
   de528:	sub    $0x1,%rax
   de52c:	jae    de533 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2853>
   de52e:	mov    0x18(%r14,%rcx,1),%rax
   de533:	add    %rax,%rbx
   de536:	mov    %fs:0xffffffffffffffd0,%rax
   de53f:	test   %rax,%rax
   de542:	jne    de37c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x269c>
   de548:	call   7f660 <rayon_core::registry::global_registry>
   de54d:	mov    (%rax),%rax
   de550:	mov    0x208(%rax),%rcx
   de557:	test   %rcx,%rcx
   de55a:	je     e0a90 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4db0>
   de560:	lea    (%r15,%r15,2),%rax
   de564:	shl    $0x4,%rax
   de568:	mov    %rax,0x138(%rsp)
   de570:	mov    %rbx,%rax
   de573:	or     %rcx,%rax
   de576:	shr    $0x20,%rax
   de57a:	je     de586 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28a6>
   de57c:	mov    %rbx,%rax
   de57f:	xor    %edx,%edx
   de581:	div    %rcx
   de584:	jmp    de58c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28ac>
   de586:	mov    %ebx,%eax
   de588:	xor    %edx,%edx
   de58a:	div    %ecx
   de58c:	add    %r14,0x138(%rsp)
   de594:	cmp    $0x1,%rdx
   de598:	sbb    $0xffffffffffffffff,%rax
   de59c:	cmp    $0x4000000,%rax
   de5a2:	mov    $0x4000000,%r8d
   de5a8:	cmovb  %rax,%r8
   de5ac:	test   %rax,%rax
   de5af:	mov    $0x1,%eax
   de5b4:	cmove  %rax,%r8
   de5b8:	movq   $0x0,0x1a8(%rsp)
   de5c4:	movq   $0x8,0x1b0(%rsp)
   de5d0:	movq   $0x0,0x1b8(%rsp)
   de5dc:	mov    $0x8,%eax
   de5e1:	movq   $0x0,0x8(%rsp)
   de5ea:	mov    %r14,%rdx
   de5ed:	mov    %r8,0x238(%rsp)
   de5f5:	jmp    de622 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2942>
   de5f7:	nopw   0x0(%rax,%rax,1)
   de600:	mov    0x238(%rsp),%r8
   de608:	mov    0x1a0(%rsp),%rdx
   de610:	add    $0x30,%rdx
   de614:	cmp    0x138(%rsp),%rdx
   de61c:	je     de7ba <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ada>
   de622:	mov    (%rdx),%rcx
   de625:	mov    $0x1,%r13d
   de62b:	test   %rcx,%rcx
   de62e:	jne    de634 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2954>
   de630:	mov    0x10(%rdx),%r13
   de634:	test   %r13,%r13
   de637:	mov    %rdx,0x1a0(%rsp)
   de63f:	je     de608 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2928>
   de641:	mov    %rdx,%rbx
   de644:	test   %rcx,%rcx
   de647:	jne    de64d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x296d>
   de649:	mov    0x8(%rdx),%rbx
   de64d:	xor    %r12d,%r12d
   de650:	jmp    de6e5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a05>
   de655:	data16 cs nopw 0x0(%rax,%rax,1)
   de660:	mov    0x1b0(%rsp),%rax
   de668:	shl    $0x5,%rbp
   de66c:	mov    0x240(%rsp),%rsi
   de674:	add    %rsi,%rbp
   de677:	mov    0x8(%rsp),%rdx
   de67c:	imul   $0x70,%rdx,%rcx
   de680:	movdqa 0x250(%rsp),%xmm0
   de689:	movq   %xmm0,(%rax,%rcx,1)
   de68e:	mov    0x118(%rsp),%edi
   de695:	mov    %edi,0x8(%rax,%rcx,1)
   de699:	movq   $0x0,0x10(%rax,%rcx,1)
   de6a2:	movq   $0x0,0x30(%rax,%rcx,1)
   de6ab:	mov    %rsi,0x50(%rax,%rcx,1)
   de6b0:	mov    %rbp,0x58(%rax,%rcx,1)
   de6b5:	mov    0x10(%rsp),%rsi
   de6ba:	mov    %rsi,0x60(%rax,%rcx,1)
   de6bf:	mov    %r15,0x68(%rax,%rcx,1)
   de6c4:	inc    %rdx
   de6c7:	mov    %rdx,0x8(%rsp)
   de6cc:	mov    %rdx,0x1b8(%rsp)
   de6d4:	test   %r13,%r13
   de6d7:	mov    0x238(%rsp),%r8
   de6df:	je     de608 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2928>
   de6e5:	mov    %rbx,0x240(%rsp)
   de6ed:	mov    %r13,%rdx
   de6f0:	mov    %r12,0x10(%rsp)
   de6f5:	xor    %r15d,%r15d
   de6f8:	nopl   0x0(%rax,%rax,1)
   de700:	mov    %r8,%rcx
   de703:	sub    %r15,%rcx
   de706:	mov    $0x0,%esi
   de70b:	cmovae %rcx,%rsi
   de70f:	cmp    $0x1,%rsi
   de713:	adc    $0x0,%rsi
   de717:	add    $0x7f,%esi
   de71a:	and    $0xfffff80,%esi
   de720:	mov    (%rbx),%rdi
   de723:	mov    %r12,%rcx
   de726:	not    %rcx
   de729:	add    %rdi,%rcx
   de72c:	cmp    %rsi,%rcx
   de72f:	cmovae %rsi,%rcx
   de733:	add    %rcx,%r12
   de736:	dec    %rdi
   de739:	cmp    %rdi,%r12
   de73c:	jne    de748 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a68>
   de73e:	dec    %r13
   de741:	add    $0x20,%rbx
   de745:	xor    %r12d,%r12d
   de748:	add    %rcx,%r15
   de74b:	cmp    %r8,%r15
   de74e:	jae    de755 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a75>
   de750:	test   %r13,%r13
   de753:	jne    de700 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a20>
   de755:	mov    %rdx,%rbp
   de758:	sub    %r13,%rbp
   de75b:	cmp    $0x1,%r12
   de75f:	sbb    $0xffffffffffffffff,%rbp
   de763:	test   %r15,%r15
   de766:	je     de600 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2920>
   de76c:	cmp    %rdx,%rbp
   de76f:	ja     dec75 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f95>
   de775:	mov    0x1a0(%rsp),%rcx
   de77d:	movq   0x20(%rcx),%xmm0
   de782:	movdqa %xmm0,0x250(%rsp)
   de78b:	mov    0x28(%rcx),%ecx
   de78e:	mov    %ecx,0x118(%rsp)
   de795:	mov    0x8(%rsp),%rcx
   de79a:	cmp    0x1a8(%rsp),%rcx
   de7a2:	jne    de668 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2988>
   de7a8:	lea    0x1a8(%rsp),%rdi
   de7b0:	call   e6b60 <<alloc::raw_vec::RawVec<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>>::grow_one>
   de7b5:	jmp    de660 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2980>
   de7ba:	mov    0x1a8(%rsp),%rax
   de7c2:	mov    0x1b0(%rsp),%r13
   de7ca:	movl   $0x0,0x270(%rsp)
   de7d5:	movb   $0x0,0x274(%rsp)
   de7dd:	movq   $0x0,0x278(%rsp)
   de7e9:	mov    0x8(%rsp),%r8
   de7ee:	mov    %r8,0xf0(%rsp)
   de7f6:	lea    0xf8(%rsp),%rcx
   de7fe:	lea    0x328(%rsp),%rdx
   de806:	mov    %rdx,0xf8(%rsp)
   de80e:	lea    0x1c0(%rsp),%rsi
   de816:	mov    %rsi,0x100(%rsp)
   de81e:	mov    0x20(%rsp),%rdi
   de823:	mov    %rdi,0x108(%rsp)
   de82b:	movq   $0x0,0xc8(%rsp)
   de837:	movq   $0x8,0xd0(%rsp)
   de843:	movq   $0x0,0xd8(%rsp)
   de84f:	movb   $0x0,0x1f(%rsp)
   de854:	mov    %rax,0x30(%rsp)
   de859:	mov    %r13,0x38(%rsp)
   de85e:	mov    %r8,0x40(%rsp)
   de863:	mov    %rdx,0x48(%rsp)
   de868:	mov    %rsi,0x50(%rsp)
   de86d:	mov    %rdi,0x58(%rsp)
   de872:	lea    0x60(%rsp),%rdx
   de877:	lea    0x270(%rsp),%rsi
   de87f:	mov    %rsi,0x60(%rsp)
   de884:	mov    %rax,0xe0(%rsp)
   de88c:	mov    %r13,0xe8(%rsp)
   de894:	lea    0x110(%rsp),%rsi
   de89c:	mov    %rax,0xb0(%rsp)
   de8a4:	mov    %r13,0xb8(%rsp)
   de8ac:	mov    %r8,0x180(%rsp)
   de8b4:	lea    0x1f(%rsp),%rdi
   de8b9:	mov    %rdi,0x160(%rsp)
   de8c1:	mov    %rdx,0x168(%rsp)
   de8c9:	mov    %rsi,0x170(%rsp)
   de8d1:	mov    %rcx,0x178(%rsp)
   de8d9:	lea    0xb0(%rsp),%rcx
   de8e1:	mov    %rcx,0x140(%rsp)
   de8e9:	movq   $0x0,0x148(%rsp)
   de8f5:	mov    %r8,0x150(%rsp)
   de8fd:	mov    %r8,0x158(%rsp)
   de905:	movq   $0x0,0xc0(%rsp)
   de911:	cmp    %r8,%rax
   de914:	jb     e0879 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b99>
   de91a:	mov    %fs:0xffffffffffffffd0,%rax
   de923:	test   %rax,%rax
   de926:	je     dea6d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d8d>
   de92c:	add    $0x110,%rax
   de932:	jmp    dea72 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d92>
   de937:	mov    0x150(%rsp),%r14
   de93f:	test   %r14,%r14
   de942:	je     de97c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c9c>
   de944:	lea    0x18(%r13),%r15
   de948:	mov    0x28f59(%rip),%r12        # 1078a8 <free@GLIBC_2.2.5>
   de94f:	jmp    de96d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c8d>
   de951:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de960:	add    $0x20,%r15
   de964:	dec    %r14
   de967:	je     ddfea <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x230a>
   de96d:	cmpq   $0x11,(%r15)
   de971:	jb     de960 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c80>
   de973:	mov    -0x10(%r15),%rdi
   de977:	call   *%r12
   de97a:	jmp    de960 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c80>
   de97c:	mov    %fs:0xffffffffffffffd0,%rax
   de985:	test   %rax,%rax
   de988:	je     de992 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cb2>
   de98a:	add    $0x110,%rax
   de990:	jmp    de99a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cba>
   de992:	xor    %r14d,%r14d
   de995:	call   7f660 <rayon_core::registry::global_registry>
   de99a:	mov    (%rax),%rax
   de99d:	mov    0x208(%rax),%rcx
   de9a4:	test   %rcx,%rcx
   de9a7:	mov    0x240(%rsp),%rax
   de9af:	je     e0aa1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4dc1>
   de9b5:	shl    $0x2,%rcx
   de9b9:	mov    %rax,%rdx
   de9bc:	or     %rcx,%rdx
   de9bf:	shr    $0x20,%rdx
   de9c3:	je     de9cc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cec>
   de9c5:	xor    %edx,%edx
   de9c7:	div    %rcx
   de9ca:	jmp    de9d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cf0>
   de9cc:	xor    %edx,%edx
   de9ce:	div    %ecx
   de9d0:	cmp    $0x1,%rdx
   de9d4:	sbb    $0xffffffffffffffff,%rax
   de9d8:	cmp    $0x1001,%rax
   de9de:	mov    $0x1000,%r12d
   de9e4:	cmovae %rax,%r12
   de9e8:	mov    0x1b8(%rsp),%r14
   de9f0:	test   %r14,%r14
   de9f3:	je     dea5e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d7e>
   de9f5:	mov    0x1b0(%rsp),%r15
   de9fd:	mov    %r14,%rax
   dea00:	or     %r12,%rax
   dea03:	shr    $0x20,%rax
   dea07:	je     dee73 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3193>
   dea0d:	mov    %r14,%rax
   dea10:	xor    %edx,%edx
   dea12:	div    %r12
   dea15:	mov    %rax,%rbp
   dea18:	cmp    $0x1,%rdx
   dea1c:	sbb    $0xffffffffffffffff,%rbp
   dea20:	je     dee8b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31ab>
   dea26:	mov    %rbp,%r13
   dea29:	shl    $0x5,%r13
   dea2d:	mov    %r13,%rdi
   dea30:	call   *0x28e8a(%rip)        # 1078c0 <malloc@GLIBC_2.2.5>
   dea36:	mov    %rax,%rsi
   dea39:	test   %rax,%rax
   dea3c:	mov    0x250(%rsp),%edx
   dea43:	jne    dee97 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31b7>
   dea49:	xor    %r14d,%r14d
   dea4c:	mov    $0x8,%edi
   dea51:	mov    %r13,%rsi
   dea54:	call   690eb <alloc::raw_vec::handle_error>
   dea59:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   dea5e:	mov    $0x8,%esi
   dea63:	xor    %ebp,%ebp
   dea65:	xor    %r13d,%r13d
   dea68:	jmp    deeca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31ea>
   dea6d:	call   7f660 <rayon_core::registry::global_registry>
   dea72:	mov    (%rax),%rax
   dea75:	mov    0x208(%rax),%rcx
   dea7c:	lea    0x160(%rsp),%rax
   dea84:	lea    0x120(%rsp),%rdi
   dea8c:	mov    $0x1,%r8d
   dea92:	mov    0x8(%rsp),%rbx
   dea97:	mov    %rbx,%rsi
   dea9a:	xor    %edx,%edx
   dea9c:	mov    %r13,%r9
   dea9f:	push   %rax
   deaa0:	push   %rbx
   deaa1:	call   e4950 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#4}>>>
   deaa6:	add    $0x10,%rsp
   deaaa:	test   %rbx,%rbx
   deaad:	je     deab9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2dd9>
   deaaf:	cmp    %rbx,0xc0(%rsp)
   deab7:	jne    deac5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2de5>
   deab9:	movq   $0x0,0xc0(%rsp)
   deac5:	cmpq   $0x0,0xb0(%rsp)
   deace:	je     deade <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2dfe>
   dead0:	mov    0xb8(%rsp),%rdi
   dead8:	call   *0x28dca(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   deade:	mov    0x130(%rsp),%rax
   deae6:	test   %rax,%rax
   deae9:	je     deb1b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e3b>
   deaeb:	mov    0x120(%rsp),%rcx
   deaf3:	test   %rcx,%rcx
   deaf6:	je     deb1b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e3b>
   deaf8:	xor    %edx,%edx
   deafa:	nopw   0x0(%rax,%rax,1)
   deb00:	add    0x10(%rcx),%rdx
   deb04:	dec    %rax
   deb07:	je     deb12 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e32>
   deb09:	mov    0x18(%rcx),%rcx
   deb0d:	test   %rcx,%rcx
   deb10:	jne    deb00 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e20>
   deb12:	test   %rdx,%rdx
   deb15:	jne    e075c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a7c>
   deb1b:	movdqu 0x120(%rsp),%xmm0
   deb24:	movdqa %xmm0,0xe0(%rsp)
   deb2d:	mov    0x130(%rsp),%rbx
   deb35:	mov    %rbx,0xf0(%rsp)
   deb3d:	mov    0xe0(%rsp),%rdi
   deb45:	mov    0xe8(%rsp),%rax
   deb4d:	test   %rdi,%rdi
   deb50:	je     dec70 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f90>
   deb56:	mov    %rax,0x10(%rsp)
   deb5b:	dec    %rbx
   deb5e:	mov    %rbx,%rax
   deb61:	jmp    deb85 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ea5>
   deb63:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   deb70:	mov    0x8(%rsp),%rbx
   deb75:	lea    -0x1(%rbx),%rax
   deb79:	mov    %r12,%rdi
   deb7c:	test   %r12,%r12
   deb7f:	je     dec8b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2fab>
   deb85:	mov    %rax,%rbx
   deb88:	mov    0x18(%rdi),%r12
   deb8c:	test   %r12,%r12
   deb8f:	je     deba0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ec0>
   deb91:	movq   $0x0,0x20(%r12)
   deb9a:	jmp    deba9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ec9>
   deb9c:	nopl   0x0(%rax)
   deba0:	movq   $0x0,0x10(%rsp)
   deba9:	mov    (%rdi),%r15
   debac:	movdqu 0x8(%rdi),%xmm0
   debb1:	movdqa %xmm0,0x160(%rsp)
   debba:	call   *0x28ce8(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   debc0:	cmp    $0xffffffffffffffff,%r15
   debc4:	je     dec8e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2fae>
   debca:	mov    %rbx,0x8(%rsp)
   debcf:	mov    %r15,0x30(%rsp)
   debd4:	movdqa 0x160(%rsp),%xmm0
   debdd:	lea    0x38(%rsp),%rax
   debe2:	movdqu %xmm0,(%rax)
   debe6:	mov    0x38(%rsp),%r13
   debeb:	mov    0x40(%rsp),%rbp
   debf0:	mov    0xc8(%rsp),%rax
   debf8:	mov    0xd8(%rsp),%rbx
   dec00:	sub    %rbx,%rax
   dec03:	cmp    %rax,%rbp
   dec06:	ja     dec48 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f68>
   dec08:	test   %rbp,%rbp
   dec0b:	je     dec26 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f46>
   dec0d:	imul   $0x38,%rbx,%rdi
   dec11:	add    0xd0(%rsp),%rdi
   dec19:	imul   $0x38,%rbp,%rdx
   dec1d:	mov    %r13,%rsi
   dec20:	call   *0x28c8a(%rip)        # 1078b0 <memcpy@GLIBC_2.14>
   dec26:	add    %rbp,%rbx
   dec29:	mov    %rbx,0xd8(%rsp)
   dec31:	test   %r15,%r15
   dec34:	je     deb70 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e90>
   dec3a:	mov    %r13,%rdi
   dec3d:	call   *0x28c65(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dec43:	jmp    deb70 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e90>
   dec48:	mov    $0x8,%ecx
   dec4d:	mov    $0x38,%r8d
   dec53:	lea    0xc8(%rsp),%rdi
   dec5b:	mov    %rbx,%rsi
   dec5e:	mov    %rbp,%rdx
   dec61:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   dec66:	mov    0xd8(%rsp),%rbx
   dec6e:	jmp    dec0d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f2d>
   dec70:	xor    %r12d,%r12d
   dec73:	jmp    dec93 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2fb3>
   dec75:	lea    0x27e34(%rip),%rcx        # 106ab0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x35b8>
   dec7c:	xor    %edi,%edi
   dec7e:	mov    %rbp,%rsi
   dec81:	call   70c30 <core::slice::index::slice_index_fail>
   dec86:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   dec8b:	xor    %r12d,%r12d
   dec8e:	mov    0x10(%rsp),%rax
   dec93:	mov    %r12,0xe0(%rsp)
   dec9b:	mov    %rax,0xe8(%rsp)
   deca3:	mov    %rbx,0xf0(%rsp)
   decab:	lea    0xe0(%rsp),%rdi
   decb3:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   decb8:	movdqu 0xc8(%rsp),%xmm0
   decc1:	movdqa %xmm0,0x30(%rsp)
   decc7:	mov    0xd8(%rsp),%rax
   deccf:	mov    %rax,0x40(%rsp)
   decd4:	mov    0x278(%rsp),%r13
   decdc:	mov    0x280(%rsp),%rax
   dece4:	cmpb   $0x0,0x274(%rsp)
   decec:	jne    e0896 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bb6>
   decf2:	test   %r13,%r13
   decf5:	je     ded5f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x307f>
   decf7:	mov    %rax,0x8(%rsp)
   decfc:	mov    0x38(%rsp),%rbx
   ded01:	mov    0x40(%rsp),%r12
   ded06:	test   %r12,%r12
   ded09:	je     ded40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3060>
   ded0b:	mov    0x28b96(%rip),%rbp        # 1078a8 <free@GLIBC_2.2.5>
   ded12:	mov    %rbx,%r15
   ded15:	jmp    ded31 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3051>
   ded17:	nopw   0x0(%rax,%rax,1)
   ded20:	mov    %r15,%rdi
   ded23:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   ded28:	add    $0x38,%r15
   ded2c:	dec    %r12
   ded2f:	je     ded40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3060>
   ded31:	cmpq   $0x3,0x28(%r15)
   ded36:	jb     ded20 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3040>
   ded38:	mov    0x18(%r15),%rdi
   ded3c:	call   *%rbp
   ded3e:	jmp    ded20 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3040>
   ded40:	mov    $0xffffffffffffffff,%r15
   ded47:	cmpq   $0x0,0x30(%rsp)
   ded4d:	je     ded58 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3078>
   ded4f:	mov    %rbx,%rdi
   ded52:	call   *0x28b50(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   ded58:	mov    0x8(%rsp),%rax
   ded5d:	jmp    ded6e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x308e>
   ded5f:	mov    0x30(%rsp),%r15
   ded64:	mov    0x38(%rsp),%r13
   ded69:	mov    0x40(%rsp),%rax
   ded6e:	mov    %r15,0x10(%rsp)
   ded73:	mov    %rax,0x8(%rsp)
   ded78:	mov    %r13,0x110(%rsp)
   ded80:	mov    0x1c8(%rsp),%rcx
   ded88:	test   %rcx,%rcx
   ded8b:	je     dedb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30d7>
   ded8d:	lea    (%rcx,%rcx,2),%rax
   ded91:	lea    0x1b(,%rax,4),%rax
   ded99:	and    $0xfffffffffffffff0,%rax
   ded9d:	add    %rax,%rcx
   deda0:	cmp    $0xffffffffffffffef,%rcx
   deda4:	je     dedb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30d7>
   deda6:	mov    0x1c0(%rsp),%rdi
   dedae:	sub    %rax,%rdi
   dedb1:	call   *0x28af1(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dedb7:	xor    %r12d,%r12d
   dedba:	mov    0x28ae7(%rip),%rbp        # 1078a8 <free@GLIBC_2.2.5>
   dedc1:	jmp    dede2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3102>
   dedc3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dedd0:	mov    %rbx,%rdi
   dedd3:	call   *%rbp
   dedd5:	inc    %r12
   dedd8:	cmp    0xa8(%rsp),%r12
   dede0:	je     dee48 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3168>
   dede2:	lea    (%r12,%r12,2),%rcx
   dede6:	shl    $0x4,%rcx
   dedea:	lea    (%r14,%rcx,1),%rax
   dedee:	cmpq   $0x0,(%r14,%rcx,1)
   dedf3:	je     dee10 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3130>
   dedf5:	cmpq   $0x11,0x18(%rax)
   dedfa:	jb     dedd5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30f5>
   dedfc:	mov    0x8(%rax),%rbx
   dee00:	jmp    dedd0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30f0>
   dee02:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dee10:	mov    0x10(%rax),%r15
   dee14:	test   %r15,%r15
   dee17:	je     dedd5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30f5>
   dee19:	mov    0x8(%rax),%rbx
   dee1d:	lea    0x18(%rbx),%r13
   dee21:	jmp    dee39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3159>
   dee23:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dee30:	add    $0x20,%r13
   dee34:	dec    %r15
   dee37:	je     dedd0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x30f0>
   dee39:	cmpq   $0x11,0x0(%r13)
   dee3e:	jb     dee30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3150>
   dee40:	mov    -0x10(%r13),%rdi
   dee44:	call   *%rbp
   dee46:	jmp    dee30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3150>
   dee48:	cmpq   $0x0,0x290(%rsp)
   dee51:	je     dee5c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x317c>
   dee53:	mov    %r14,%rdi
   dee56:	call   *0x28a4c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dee5c:	mov    0x110(%rsp),%r13
   dee64:	mov    0x8(%rsp),%rbp
   dee69:	mov    0x10(%rsp),%r15
   dee6e:	jmp    df302 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3622>
   dee73:	mov    %r14d,%eax
   dee76:	xor    %edx,%edx
   dee78:	div    %r12d
   dee7b:	mov    %eax,%ebp
   dee7d:	cmp    $0x1,%rdx
   dee81:	sbb    $0xffffffffffffffff,%rbp
   dee85:	jne    dea26 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d46>
   dee8b:	mov    $0x8,%esi
   dee90:	mov    0x250(%rsp),%edx
   dee97:	xor    %r13d,%r13d
   dee9a:	mov    %rsi,%rax
   dee9d:	nopl   (%rax)
   deea0:	cmp    %r14,%r12
   deea3:	mov    %r14,%rcx
   deea6:	cmovb  %r12,%rcx
   deeaa:	mov    %ebx,(%rax)
   deeac:	mov    %ebx,0x4(%rax)
   deeaf:	mov    %edx,0x8(%rax)
   deeb2:	mov    %r15,0x10(%rax)
   deeb6:	lea    (%r15,%rcx,8),%r15
   deeba:	inc    %r13
   deebd:	mov    %r15,0x18(%rax)
   deec1:	add    $0x20,%rax
   deec5:	sub    %rcx,%r14
   deec8:	jne    deea0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31c0>
   deeca:	movl   $0x0,0x270(%rsp)
   deed5:	movb   $0x0,0x274(%rsp)
   deedd:	movq   $0x0,0x278(%rsp)
   deee9:	mov    %r13,0xf0(%rsp)
   deef1:	lea    0xf8(%rsp),%rax
   deef9:	lea    0x328(%rsp),%rcx
   def01:	mov    %rcx,0xf8(%rsp)
   def09:	lea    0x1a8(%rsp),%rdx
   def11:	mov    %rdx,0x100(%rsp)
   def19:	movq   $0x0,0xc8(%rsp)
   def25:	movq   $0x8,0xd0(%rsp)
   def31:	movq   $0x0,0xd8(%rsp)
   def3d:	movb   $0x0,0x1f(%rsp)
   def42:	mov    %rbp,0x30(%rsp)
   def47:	mov    %rsi,0x38(%rsp)
   def4c:	mov    %r13,0x40(%rsp)
   def51:	mov    %rcx,0x48(%rsp)
   def56:	mov    %rdx,0x50(%rsp)
   def5b:	lea    0x58(%rsp),%rcx
   def60:	lea    0x270(%rsp),%rdx
   def68:	mov    %rdx,0x58(%rsp)
   def6d:	mov    %rbp,0xe0(%rsp)
   def75:	mov    %rsi,0xe8(%rsp)
   def7d:	lea    0x108(%rsp),%rdx
   def85:	mov    %rbp,0xb0(%rsp)
   def8d:	mov    %rsi,%rbx
   def90:	mov    %rsi,0xb8(%rsp)
   def98:	mov    %r13,0x1e0(%rsp)
   defa0:	lea    0x1f(%rsp),%rsi
   defa5:	mov    %rsi,0x1c0(%rsp)
   defad:	mov    %rcx,0x1c8(%rsp)
   defb5:	mov    %rdx,0x1d0(%rsp)
   defbd:	mov    %rax,0x1d8(%rsp)
   defc5:	lea    0xb0(%rsp),%rax
   defcd:	mov    %rax,0x160(%rsp)
   defd5:	movq   $0x0,0x168(%rsp)
   defe1:	mov    %r13,0x170(%rsp)
   defe9:	mov    %r13,0x178(%rsp)
   deff1:	movq   $0x0,0xc0(%rsp)
   deffd:	cmp    %r13,%rbp
   df000:	jb     e077b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a9b>
   df006:	mov    %fs:0xffffffffffffffd0,%rax
   df00f:	test   %rax,%rax
   df012:	je     df01c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x333c>
   df014:	add    $0x110,%rax
   df01a:	jmp    df021 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3341>
   df01c:	call   7f660 <rayon_core::registry::global_registry>
   df021:	mov    (%rax),%rax
   df024:	mov    0x208(%rax),%rcx
   df02b:	lea    0x1c0(%rsp),%rax
   df033:	lea    0x120(%rsp),%rdi
   df03b:	mov    $0x1,%r8d
   df041:	mov    %r13,%rsi
   df044:	xor    %edx,%edx
   df046:	mov    %rbx,%r9
   df049:	push   %rax
   df04a:	push   %r13
   df04c:	call   e2940 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::corpus::merging::PreparedChunk, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::corpus::merging::NeighborScratch as core::default::Default>::default, <tk_train::trainers::bpe::corpus::Corpus>::prepare_fresh::{closure#1}>>>
   df051:	add    $0x10,%rsp
   df055:	test   %r13,%r13
   df058:	je     df064 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3384>
   df05a:	cmp    %r13,0xc0(%rsp)
   df062:	jne    df070 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3390>
   df064:	movq   $0x0,0xc0(%rsp)
   df070:	cmpq   $0x0,0xb0(%rsp)
   df079:	je     df089 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33a9>
   df07b:	mov    0xb8(%rsp),%rdi
   df083:	call   *0x2881f(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df089:	mov    0x130(%rsp),%rax
   df091:	test   %rax,%rax
   df094:	je     df0cb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33eb>
   df096:	mov    0x120(%rsp),%rcx
   df09e:	test   %rcx,%rcx
   df0a1:	je     df0cb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33eb>
   df0a3:	xor    %edx,%edx
   df0a5:	data16 cs nopw 0x0(%rax,%rax,1)
   df0b0:	add    0x10(%rcx),%rdx
   df0b4:	dec    %rax
   df0b7:	je     df0c2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33e2>
   df0b9:	mov    0x18(%rcx),%rcx
   df0bd:	test   %rcx,%rcx
   df0c0:	jne    df0b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33d0>
   df0c2:	test   %rdx,%rdx
   df0c5:	jne    e0714 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a34>
   df0cb:	movdqu 0x120(%rsp),%xmm0
   df0d4:	movdqa %xmm0,0x1c0(%rsp)
   df0dd:	mov    0x130(%rsp),%r15
   df0e5:	mov    %r15,0x1d0(%rsp)
   df0ed:	mov    0x1c0(%rsp),%rdi
   df0f5:	mov    0x1c8(%rsp),%rax
   df0fd:	test   %rdi,%rdi
   df100:	je     df219 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3539>
   df106:	mov    %rax,0x8(%rsp)
   df10b:	dec    %r15
   df10e:	mov    %r15,%rax
   df111:	jmp    df130 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3450>
   df113:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   df120:	lea    -0x1(%r15),%rax
   df124:	mov    %rbx,%rdi
   df127:	test   %rbx,%rbx
   df12a:	je     df21d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x353d>
   df130:	mov    %rax,%r15
   df133:	mov    0x18(%rdi),%rbx
   df137:	test   %rbx,%rbx
   df13a:	je     df150 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3470>
   df13c:	movq   $0x0,0x20(%rbx)
   df144:	jmp    df159 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3479>
   df146:	cs nopw 0x0(%rax,%rax,1)
   df150:	movq   $0x0,0x8(%rsp)
   df159:	mov    (%rdi),%r12
   df15c:	movups 0x8(%rdi),%xmm0
   df160:	movaps %xmm0,0xe0(%rsp)
   df168:	call   *0x2873a(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df16e:	cmp    $0xffffffffffffffff,%r12
   df172:	je     df21f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x353f>
   df178:	mov    %r12,0x30(%rsp)
   df17d:	movdqa 0xe0(%rsp),%xmm0
   df186:	lea    0x38(%rsp),%rax
   df18b:	movdqu %xmm0,(%rax)
   df18f:	mov    0x38(%rsp),%r14
   df194:	mov    0x40(%rsp),%r13
   df199:	mov    0xc8(%rsp),%rax
   df1a1:	mov    0xd8(%rsp),%rbp
   df1a9:	sub    %rbp,%rax
   df1ac:	cmp    %rax,%r13
   df1af:	ja     df1f1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3511>
   df1b1:	test   %r13,%r13
   df1b4:	je     df1cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34ef>
   df1b6:	imul   $0x38,%rbp,%rdi
   df1ba:	add    0xd0(%rsp),%rdi
   df1c2:	imul   $0x38,%r13,%rdx
   df1c6:	mov    %r14,%rsi
   df1c9:	call   *0x286e1(%rip)        # 1078b0 <memcpy@GLIBC_2.14>
   df1cf:	add    %r13,%rbp
   df1d2:	mov    %rbp,0xd8(%rsp)
   df1da:	test   %r12,%r12
   df1dd:	je     df120 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3440>
   df1e3:	mov    %r14,%rdi
   df1e6:	call   *0x286bc(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df1ec:	jmp    df120 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3440>
   df1f1:	mov    $0x8,%ecx
   df1f6:	mov    $0x38,%r8d
   df1fc:	lea    0xc8(%rsp),%rdi
   df204:	mov    %rbp,%rsi
   df207:	mov    %r13,%rdx
   df20a:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   df20f:	mov    0xd8(%rsp),%rbp
   df217:	jmp    df1b6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34d6>
   df219:	xor    %ebx,%ebx
   df21b:	jmp    df224 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3544>
   df21d:	xor    %ebx,%ebx
   df21f:	mov    0x8(%rsp),%rax
   df224:	mov    %rbx,0x1c0(%rsp)
   df22c:	mov    %rax,0x1c8(%rsp)
   df234:	mov    %r15,0x1d0(%rsp)
   df23c:	lea    0x1c0(%rsp),%rdi
   df244:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   df249:	movdqu 0xc8(%rsp),%xmm0
   df252:	movdqa %xmm0,0x30(%rsp)
   df258:	mov    0xd8(%rsp),%rax
   df260:	mov    %rax,0x40(%rsp)
   df265:	mov    0x278(%rsp),%r13
   df26d:	mov    0x280(%rsp),%rbp
   df275:	cmpb   $0x0,0x274(%rsp)
   df27d:	jne    e0798 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ab8>
   df283:	test   %r13,%r13
   df286:	je     df604 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3924>
   df28c:	mov    0x38(%rsp),%rbx
   df291:	mov    0x40(%rsp),%r15
   df296:	test   %r15,%r15
   df299:	je     df2d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x35f1>
   df29b:	mov    0x28606(%rip),%r12        # 1078a8 <free@GLIBC_2.2.5>
   df2a2:	mov    %rbx,%r14
   df2a5:	jmp    df2c1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x35e1>
   df2a7:	nopw   0x0(%rax,%rax,1)
   df2b0:	mov    %r14,%rdi
   df2b3:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   df2b8:	add    $0x38,%r14
   df2bc:	dec    %r15
   df2bf:	je     df2d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x35f1>
   df2c1:	cmpq   $0x3,0x28(%r14)
   df2c6:	jb     df2b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x35d0>
   df2c8:	mov    0x18(%r14),%rdi
   df2cc:	call   *%r12
   df2cf:	jmp    df2b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x35d0>
   df2d1:	mov    $0xffffffffffffffff,%r15
   df2d8:	cmpq   $0x0,0x30(%rsp)
   df2de:	je     df2e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3609>
   df2e0:	mov    %rbx,%rdi
   df2e3:	call   *0x285bf(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df2e9:	cmpq   $0x0,0x1a8(%rsp)
   df2f2:	je     df302 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3622>
   df2f4:	mov    0x1b0(%rsp),%rdi
   df2fc:	call   *0x285a6(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df302:	cmp    $0xffffffffffffffff,%r15
   df306:	je     e03a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46c0>
   df30c:	mov    %r15,0x160(%rsp)
   df314:	mov    %r13,0x168(%rsp)
   df31c:	mov    %rbp,0x170(%rsp)
   df324:	mov    0x20(%rsp),%rax
   df329:	mov    %rax,0x178(%rsp)
   df331:	movq   $0x0,0xb0(%rsp)
   df33d:	movq   $0x8,0xb8(%rsp)
   df349:	movq   $0x0,0xc0(%rsp)
   df355:	mov    %rbp,0xc8(%rsp)
   df35d:	test   %rbp,%rbp
   df360:	jne    e0614 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4934>
   df366:	mov    $0x8,%r12d
   df36c:	xor    %r13d,%r13d
   df36f:	xor    %r15d,%r15d
   df372:	lea    (%r15,%r15,2),%rax
   df376:	lea    (%r12,%rax,8),%rax
   df37a:	movdqu 0x160(%rsp),%xmm0
   df383:	movdqu 0x170(%rsp),%xmm1
   df38c:	movdqa %xmm1,0xf0(%rsp)
   df395:	movdqa %xmm0,0xe0(%rsp)
   df39e:	lea    0xf8(%rsp),%rcx
   df3a6:	movdqa %xmm0,0x140(%rsp)
   df3af:	mov    0xf0(%rsp),%rbx
   df3b7:	mov    %rbx,0x1d8(%rsp)
   df3bf:	mov    %rcx,0x1c0(%rsp)
   df3c7:	mov    %rax,0x1c8(%rsp)
   df3cf:	mov    %rbp,0x1d0(%rsp)
   df3d7:	lea    0x140(%rsp),%rax
   df3df:	mov    %rax,0x30(%rsp)
   df3e4:	movq   $0x0,0x38(%rsp)
   df3ed:	mov    %rbx,0x40(%rsp)
   df3f2:	mov    %rbx,0x48(%rsp)
   df3f7:	movq   $0x0,0x150(%rsp)
   df403:	cmp    %rbx,0x140(%rsp)
   df40b:	jb     e066d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x498d>
   df411:	mov    0x148(%rsp),%r14
   df419:	mov    %r14,0x120(%rsp)
   df421:	mov    %rbx,0x128(%rsp)
   df429:	mov    %fs:0xffffffffffffffd0,%rax
   df432:	test   %rax,%rax
   df435:	je     df43f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x375f>
   df437:	add    $0x110,%rax
   df43d:	jmp    df444 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3764>
   df43f:	call   7f660 <rayon_core::registry::global_registry>
   df444:	mov    %r13,0x288(%rsp)
   df44c:	mov    (%rax),%rax
   df44f:	mov    0x208(%rax),%rcx
   df456:	lea    0x1c0(%rsp),%rax
   df45e:	lea    0x2c0(%rsp),%rdi
   df466:	mov    $0x1,%r8d
   df46c:	mov    %rbx,%rsi
   df46f:	xor    %edx,%edx
   df471:	mov    %r14,%r9
   df474:	mov    %rax,0x340(%rsp)
   df47c:	push   %rax
   df47d:	push   %rbx
   df47e:	call   e9460 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<tk_train::trainers::bpe::corpus::merging::PreparedChunk>, rayon::iter::map::MapConsumer<rayon::iter::collect::consumer::CollectConsumer<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>, <tk_train::trainers::bpe::corpus::Corpus>::merge::{closure#0}>>>
   df483:	add    $0x10,%rsp
   df487:	mov    %rbp,%r13
   df48a:	mov    %r12,0x110(%rsp)
   df492:	lea    0x30(%rsp),%rdi
   df497:	call   e92d0 <core::ptr::drop_glue::<rayon::vec::Drain<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   df49c:	mov    0x148(%rsp),%rbx
   df4a4:	mov    0x150(%rsp),%r12
   df4ac:	test   %r12,%r12
   df4af:	je     df4e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3800>
   df4b1:	mov    0x283f0(%rip),%rbp        # 1078a8 <free@GLIBC_2.2.5>
   df4b8:	mov    %rbx,%r14
   df4bb:	jmp    df4d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37f1>
   df4bd:	nopl   (%rax)
   df4c0:	mov    %r14,%rdi
   df4c3:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   df4c8:	add    $0x38,%r14
   df4cc:	dec    %r12
   df4cf:	je     df4e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3800>
   df4d1:	cmpq   $0x3,0x28(%r14)
   df4d6:	jb     df4c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37e0>
   df4d8:	mov    0x18(%r14),%rdi
   df4dc:	call   *%rbp
   df4de:	jmp    df4c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37e0>
   df4e0:	cmpq   $0x0,0x140(%rsp)
   df4e9:	je     df4f4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3814>
   df4eb:	mov    %rbx,%rdi
   df4ee:	call   *0x283b4(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df4f4:	mov    0x2d0(%rsp),%r14
   df4fc:	mov    %r14,0x1c0(%rsp)
   df504:	mov    %r13,%rdx
   df507:	cmp    %r13,%r14
   df50a:	mov    0x110(%rsp),%r13
   df512:	mov    0x288(%rsp),%r12
   df51a:	jne    e068a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x49aa>
   df520:	lea    (%r15,%rdx,1),%rbp
   df524:	cmp    $0xffffffffffffffff,%r12
   df528:	je     e03a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46c0>
   df52e:	mov    0xa0(%rsp),%rax
   df536:	mov    0x18(%rax),%rcx
   df53a:	mov    %rcx,0x2f0(%rsp)
   df542:	mov    0x20(%rax),%rax
   df546:	mov    %rax,0x190(%rsp)
   df54e:	mov    $0x8,%eax
   df553:	test   %rbp,%rbp
   df556:	je     df64b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x396b>
   df55c:	lea    (%r15,%rdx,1),%rcx
   df560:	dec    %rcx
   df563:	mov    %ebp,%edi
   df565:	and    $0x3,%edi
   df568:	cmp    $0x3,%rcx
   df56c:	jae    df575 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3895>
   df56e:	xor    %ecx,%ecx
   df570:	xor    %r15d,%r15d
   df573:	jmp    df5b1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38d1>
   df575:	mov    %rbp,%rdx
   df578:	and    $0xfffffffffffffffc,%rdx
   df57c:	lea    0x58(%r13),%rsi
   df580:	xor    %ecx,%ecx
   df582:	xor    %r15d,%r15d
   df585:	data16 cs nopw 0x0(%rax,%rax,1)
   df590:	add    -0x48(%rsi),%r15
   df594:	add    -0x30(%rsi),%r15
   df598:	add    -0x18(%rsi),%r15
   df59c:	add    (%rsi),%r15
   df59f:	add    $0x4,%rcx
   df5a3:	add    $0x60,%rsi
   df5a7:	cmp    %rcx,%rdx
   df5aa:	jne    df590 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38b0>
   df5ac:	test   %rdi,%rdi
   df5af:	je     df5dd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38fd>
   df5b1:	lea    (%rcx,%rcx,2),%rcx
   df5b5:	lea    0x10(,%rcx,8),%rcx
   df5bd:	add    %r13,%rcx
   df5c0:	shl    $0x3,%edi
   df5c3:	lea    (%rdi,%rdi,2),%rsi
   df5c7:	xor    %edx,%edx
   df5c9:	nopl   0x0(%rax)
   df5d0:	add    (%rcx,%rdx,1),%r15
   df5d4:	add    $0x18,%rdx
   df5d8:	cmp    %rdx,%rsi
   df5db:	jne    df5d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38f0>
   df5dd:	lea    0x0(,%r15,8),%rcx
   df5e5:	lea    (%rcx,%rcx,4),%rsi
   df5e9:	movabs $0x333333333333333,%rcx
   df5f3:	cmp    %rcx,%r15
   df5f6:	jbe    df627 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3947>
   df5f8:	xor    %edi,%edi
   df5fa:	call   690eb <alloc::raw_vec::handle_error>
   df5ff:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   df604:	mov    0x30(%rsp),%r15
   df609:	mov    0x38(%rsp),%r13
   df60e:	mov    0x40(%rsp),%rbp
   df613:	cmpq   $0x0,0x1a8(%rsp)
   df61c:	jne    df2f4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3614>
   df622:	jmp    df302 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3622>
   df627:	test   %rsi,%rsi
   df62a:	je     df64b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x396b>
   df62c:	mov    %rsi,%rbx
   df62f:	mov    $0x8,%esi
   df634:	mov    %rbx,%rdi
   df637:	call   68f90 <__rustc::__rust_alloc>
   df63c:	mov    $0x8,%edi
   df641:	mov    %rbx,%rsi
   df644:	test   %rax,%rax
   df647:	jne    df64e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x396e>
   df649:	jmp    df5fa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x391a>
   df64b:	xor    %r15d,%r15d
   df64e:	shl    $0x3,%rbp
   df652:	lea    0x0(,%rbp,2),%rcx
   df65a:	add    %rbp,%rcx
   df65d:	lea    0x0(%r13,%rcx,1),%rdx
   df662:	mov    %r15,0x160(%rsp)
   df66a:	mov    %rax,0x338(%rsp)
   df672:	mov    %rax,0x168(%rsp)
   df67a:	movq   $0x0,0x170(%rsp)
   df686:	mov    %r13,0x30(%rsp)
   df68b:	mov    %r13,0x38(%rsp)
   df690:	mov    %r12,0x40(%rsp)
   df695:	mov    %rdx,0x20(%rsp)
   df69a:	mov    %rdx,0x48(%rsp)
   df69f:	movq   $0x0,0x50(%rsp)
   df6a8:	movq   $0x0,0x70(%rsp)
   df6b1:	lea    (%rcx,%r13,1),%rbx
   df6b5:	add    $0xffffffffffffffe8,%rbx
   df6b9:	xor    %r14d,%r14d
   df6bc:	lea    0x27a65(%rip),%rax        # 107128 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c30>
   df6c3:	mov    %rax,0x320(%rsp)
   df6cb:	xor    %ecx,%ecx
   df6cd:	mov    %r13,%r15
   df6d0:	mov    %rbx,0x290(%rsp)
   df6d8:	mov    %r14,0x298(%rsp)
   df6e0:	test   %rcx,%rcx
   df6e3:	sete   %al
   df6e6:	mov    %rbx,%rdx
   df6e9:	sub    %r15,%rdx
   df6ec:	mov    %rdx,0x138(%rsp)
   df6f4:	lea    0x18(%r15),%rdx
   df6f8:	mov    %rdx,0x10(%rsp)
   df6fd:	mov    %rcx,%r14
   df700:	mov    0x1a0(%rsp),%rcx
   df708:	jmp    df758 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a78>
   df70a:	nopw   0x0(%rax,%rax,1)
   df710:	mov    0x118(%rsp),%rax
   df718:	lea    (%rax,%rax,8),%rax
   df71c:	lea    (%r14,%rax,8),%r12
   df720:	mov    %r14,0x50(%rsp)
   df725:	mov    %r14,0x58(%rsp)
   df72a:	mov    0x250(%rsp),%rcx
   df732:	mov    %rcx,0x60(%rsp)
   df737:	mov    %r12,0x68(%rsp)
   df73c:	addq   $0xffffffffffffffe8,0x138(%rsp)
   df745:	addq   $0x18,0x10(%rsp)
   df74b:	xor    %eax,%eax
   df74d:	mov    %r14,%rbp
   df750:	mov    0x240(%rsp),%r15
   df758:	test   $0x1,%al
   df75a:	je     df770 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a90>
   df75c:	mov    %r14,0x8(%rsp)
   df761:	cmp    0x20(%rsp),%r15
   df766:	jne    df7af <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3acf>
   df768:	jmp    dfd9b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40bb>
   df76d:	nopl   (%rax)
   df770:	cmp    %r12,%rbp
   df773:	jne    df8cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bef>
   df779:	test   %r14,%r14
   df77c:	je     df792 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ab2>
   df77e:	test   %rcx,%rcx
   df781:	je     df792 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ab2>
   df783:	mov    %r14,%rdi
   df786:	mov    %rcx,%r14
   df789:	call   *0x28119(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df78f:	mov    %r14,%rcx
   df792:	movq   $0x0,0x50(%rsp)
   df79b:	movq   $0x0,0x8(%rsp)
   df7a4:	cmp    0x20(%rsp),%r15
   df7a9:	je     dfd9b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40bb>
   df7af:	lea    0x18(%r15),%rax
   df7b3:	mov    %rax,0x38(%rsp)
   df7b8:	mov    (%r15),%rdx
   df7bb:	cmp    $0xffffffffffffffff,%rdx
   df7bf:	je     dfd5b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x407b>
   df7c5:	mov    %rdx,0x250(%rsp)
   df7cd:	mov    %rax,0x240(%rsp)
   df7d5:	mov    0x8(%r15),%r14
   df7d9:	mov    0x10(%r15),%rax
   df7dd:	mov    %rax,0x118(%rsp)
   df7e5:	cmpq   $0x0,0x8(%rsp)
   df7eb:	je     df710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a30>
   df7f1:	mov    %rcx,0x1a0(%rsp)
   df7f9:	mov    %r14,0x238(%rsp)
   df801:	sub    %rbp,%r12
   df804:	je     df8a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bc0>
   df80a:	shr    $0x3,%r12
   df80e:	movabs $0x8e38e38e38e38e39,%rax
   df818:	imul   %rax,%r12
   df81c:	xor    %ebx,%ebx
   df81e:	jmp    df834 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b54>
   df820:	mov    %r13,%r12
   df823:	mov    %r15,%rdi
   df826:	call   *0x2807c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df82c:	inc    %rbx
   df82f:	cmp    %r12,%rbx
   df832:	je     df8a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bc0>
   df834:	lea    (%rbx,%rbx,8),%rcx
   df838:	lea    0x0(,%rcx,8),%rax
   df840:	add    %rbp,%rax
   df843:	cmpq   $0x0,0x0(%rbp,%rcx,8)
   df849:	je     df860 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b80>
   df84b:	cmpq   $0x11,0x18(%rax)
   df850:	jb     df82c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b4c>
   df852:	mov    0x8(%rax),%r15
   df856:	jmp    df823 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b43>
   df858:	nopl   0x0(%rax,%rax,1)
   df860:	mov    0x10(%rax),%r14
   df864:	test   %r14,%r14
   df867:	je     df82c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b4c>
   df869:	mov    %r12,%r13
   df86c:	mov    0x8(%rax),%r15
   df870:	lea    0x18(%r15),%r12
   df874:	jmp    df889 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ba9>
   df876:	cs nopw 0x0(%rax,%rax,1)
   df880:	add    $0x20,%r12
   df884:	dec    %r14
   df887:	je     df820 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3b40>
   df889:	cmpq   $0x11,(%r12)
   df88e:	jb     df880 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ba0>
   df890:	mov    -0x10(%r12),%rdi
   df895:	call   *0x2800d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df89b:	jmp    df880 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ba0>
   df89d:	nopl   (%rax)
   df8a0:	cmpq   $0x0,0x1a0(%rsp)
   df8a9:	mov    0x290(%rsp),%rbx
   df8b1:	mov    0x238(%rsp),%r14
   df8b9:	je     df710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a30>
   df8bf:	mov    0x8(%rsp),%rdi
   df8c4:	call   *0x27fde(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   df8ca:	jmp    df710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a30>
   df8cf:	mov    %rcx,0x1a0(%rsp)
   df8d7:	lea    0x48(%rbp),%rax
   df8db:	mov    %rax,0x118(%rsp)
   df8e3:	mov    %rax,0x58(%rsp)
   df8e8:	mov    0x0(%rbp),%rsi
   df8ec:	mov    0x8(%rbp),%rdx
   df8f0:	mov    0x10(%rbp),%r13
   df8f4:	mov    0x18(%rbp),%rdi
   df8f8:	movdqu 0x20(%rbp),%xmm0
   df8fd:	movdqu 0x30(%rbp),%xmm1
   df902:	mov    0x40(%rbp),%rax
   df906:	mov    %rax,0x100(%rsp)
   df90e:	movdqa %xmm1,0xf0(%rsp)
   df917:	movdqa %xmm0,0xe0(%rsp)
   df920:	lea    0x1e0(%rsp),%rcx
   df928:	mov    %rax,0x20(%rcx)
   df92c:	movdqu %xmm1,0x10(%rcx)
   df931:	movdqu %xmm0,(%rcx)
   df935:	mov    %rsi,0x1c0(%rsp)
   df93d:	mov    %rdx,0x308(%rsp)
   df945:	mov    %rdx,0x1c8(%rsp)
   df94d:	mov    %r13,0x1d0(%rsp)
   df955:	mov    %rdi,0x1d8(%rsp)
   df95d:	mov    0x1f0(%rsp),%r8
   df965:	test   %r8,%r8
   df968:	mov    %r12,0x318(%rsp)
   df970:	mov    %r14,0x238(%rsp)
   df978:	mov    %rdi,0x138(%rsp)
   df980:	mov    %r8,0x8(%rsp)
   df985:	je     dfa01 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d21>
   df987:	cmpq   $0x0,0x190(%rsp)
   df990:	je     e0b4e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e6e>
   df996:	mov    0x1e0(%rsp),%eax
   df99d:	mov    0x1e4(%rsp),%ecx
   df9a4:	movabs $0xbe5466cf34e90c6c,%rdx
   df9ae:	xor    %rdx,%rax
   df9b1:	movabs $0x5851f42d4c957f2d,%rdi
   df9bb:	mul    %rdi
   df9be:	xor    %rdx,%rcx
   df9c1:	xor    %rcx,%rax
   df9c4:	mul    %rdi
   df9c7:	mov    %rax,%rcx
   df9ca:	xor    %rdx,%rcx
   df9cd:	mov    %rcx,%rax
   df9d0:	movabs $0x452821e638d01377,%rdx
   df9da:	mul    %rdx
   df9dd:	xor    %rax,%rdx
   df9e0:	rol    %cl,%rdx
   df9e3:	mov    %rdx,%rax
   df9e6:	mov    0x190(%rsp),%rcx
   df9ee:	or     %rcx,%rax
   df9f1:	shr    $0x20,%rax
   df9f5:	je     dfa27 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d47>
   df9f7:	mov    %rdx,%rax
   df9fa:	xor    %edx,%edx
   df9fc:	div    %rcx
   df9ff:	jmp    dfa2d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d4d>
   dfa01:	sub    $0x1,%rsi
   dfa05:	setb   0x2f(%rsp)
   dfa0a:	cmovb  %rdi,%rsi
   dfa0e:	test   %rsi,%rsi
   dfa11:	je     dfabf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ddf>
   dfa17:	cmpq   $0x0,0x190(%rsp)
   dfa20:	jne    dfa4c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d6c>
   dfa22:	jmp    e0b4e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e6e>
   dfa27:	mov    %edx,%eax
   dfa29:	xor    %edx,%edx
   dfa2b:	div    %ecx
   dfa2d:	mov    %rdx,0x2e8(%rsp)
   dfa35:	sub    $0x1,%rsi
   dfa39:	setb   0x2f(%rsp)
   dfa3e:	cmovb  0x138(%rsp),%rsi
   dfa47:	test   %rsi,%rsi
   dfa4a:	je     dfabf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ddf>
   dfa4c:	mov    0x1e8(%rsp),%eax
   dfa53:	mov    0x1ec(%rsp),%ebp
   dfa5a:	mov    %rax,0x10(%rsp)
   dfa5f:	movabs $0xbe5466cf34e90c6c,%rcx
   dfa69:	xor    %rcx,%rax
   dfa6c:	movabs $0x5851f42d4c957f2d,%rsi
   dfa76:	mul    %rsi
   dfa79:	mov    %rbp,%rcx
   dfa7c:	xor    %rdx,%rcx
   dfa7f:	xor    %rcx,%rax
   dfa82:	mul    %rsi
   dfa85:	mov    %rax,%rcx
   dfa88:	xor    %rdx,%rcx
   dfa8b:	mov    %rcx,%rax
   dfa8e:	movabs $0x452821e638d01377,%rdx
   dfa98:	mul    %rdx
   dfa9b:	xor    %rax,%rdx
   dfa9e:	rol    %cl,%rdx
   dfaa1:	mov    %rdx,%rax
   dfaa4:	mov    0x190(%rsp),%rcx
   dfaac:	or     %rcx,%rax
   dfaaf:	shr    $0x20,%rax
   dfab3:	je     dfad7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3df7>
   dfab5:	mov    %rdx,%rax
   dfab8:	xor    %edx,%edx
   dfaba:	div    %rcx
   dfabd:	jmp    dfadd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3dfd>
   dfabf:	mov    0x1e8(%rsp),%eax
   dfac6:	mov    %rax,0x10(%rsp)
   dfacb:	mov    0x1ec(%rsp),%ebp
   dfad2:	xor    %r12d,%r12d
   dfad5:	jmp    dfae8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e08>
   dfad7:	mov    %edx,%eax
   dfad9:	xor    %edx,%edx
   dfadb:	div    %ecx
   dfadd:	mov    %rdx,0x310(%rsp)
   dfae5:	mov    $0x1,%r12b
   dfae8:	movsd  0x1e0(%rsp),%xmm0
   dfaf1:	movaps %xmm0,0x250(%rsp)
   dfaf9:	movdqu 0x1f8(%rsp),%xmm0
   dfb02:	movdqa %xmm0,0x240(%rsp)
   dfb0b:	mov    0x298(%rsp),%rax
   dfb13:	cmp    0x160(%rsp),%rax
   dfb1b:	jne    dfb3a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e5a>
   dfb1d:	lea    0x160(%rsp),%rdi
   dfb25:	call   ea260 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::corpus::PairChange<()>>>::grow_one>
   dfb2a:	mov    0x168(%rsp),%rax
   dfb32:	mov    %rax,0x338(%rsp)
   dfb3a:	mov    0x298(%rsp),%rax
   dfb42:	lea    0x1(%rax),%r14
   dfb46:	lea    (%rax,%rax,4),%rax
   dfb4a:	mov    0x338(%rsp),%rcx
   dfb52:	movaps 0x250(%rsp),%xmm0
   dfb5a:	movlps %xmm0,(%rcx,%rax,8)
   dfb5e:	mov    0x10(%rsp),%rdx
   dfb63:	mov    %edx,0x8(%rcx,%rax,8)
   dfb67:	mov    %ebp,0xc(%rcx,%rax,8)
   dfb6b:	mov    0x8(%rsp),%rdx
   dfb70:	mov    %rdx,0x10(%rcx,%rax,8)
   dfb75:	movdqa 0x240(%rsp),%xmm0
   dfb7e:	movdqu %xmm0,0x18(%rcx,%rax,8)
   dfb84:	mov    %r14,0x170(%rsp)
   dfb8c:	test   %rdx,%rdx
   dfb8f:	je     dfc5d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f7d>
   dfb95:	mov    0x2e8(%rsp),%rcx
   dfb9d:	cmp    0x310(%rsp),%rcx
   dfba5:	sete   %al
   dfba8:	test   %al,%r12b
   dfbab:	je     dfc1d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f3d>
   dfbad:	cmp    0x190(%rsp),%rcx
   dfbb5:	jae    e0b6d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e8d>
   dfbbb:	imul   $0x70,%rcx,%rdi
   dfbbf:	add    0x2f0(%rsp),%rdi
   dfbc7:	movdqu 0x1c0(%rsp),%xmm0
   dfbd0:	movdqu 0x1d0(%rsp),%xmm1
   dfbd9:	lea    0xe8(%rsp),%rax
   dfbe1:	movdqu %xmm1,0x10(%rax)
   dfbe6:	movdqu %xmm0,(%rax)
   dfbea:	movq   $0x1,0xe0(%rsp)
   dfbf6:	mov    0x298(%rsp),%rsi
   dfbfe:	mov    $0x1,%edx
   dfc03:	lea    0xe0(%rsp),%rcx
   dfc0b:	call   ea2d0 <<tk_train::trainers::bpe::index::Shard>::push>
   dfc10:	mov    0x318(%rsp),%r12
   dfc18:	jmp    dfd46 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4066>
   dfc1d:	cmp    0x190(%rsp),%rcx
   dfc25:	jae    e0b7c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e9c>
   dfc2b:	imul   $0x70,%rcx,%rdi
   dfc2f:	add    0x2f0(%rsp),%rdi
   dfc37:	movq   $0x0,0xe0(%rsp)
   dfc43:	mov    0x298(%rsp),%rsi
   dfc4b:	mov    $0x1,%edx
   dfc50:	lea    0xe0(%rsp),%rcx
   dfc58:	call   ea2d0 <<tk_train::trainers::bpe::index::Shard>::push>
   dfc5d:	test   %r12b,%r12b
   dfc60:	je     dfcd4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ff4>
   dfc62:	mov    0x310(%rsp),%rax
   dfc6a:	cmp    0x190(%rsp),%rax
   dfc72:	mov    0x318(%rsp),%r12
   dfc7a:	jae    e0b5c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e7c>
   dfc80:	imul   $0x70,%rax,%rdi
   dfc84:	add    0x2f0(%rsp),%rdi
   dfc8c:	movdqu 0x1c0(%rsp),%xmm0
   dfc95:	movdqu 0x1d0(%rsp),%xmm1
   dfc9e:	lea    0xe8(%rsp),%rax
   dfca6:	movdqu %xmm1,0x10(%rax)
   dfcab:	movdqu %xmm0,(%rax)
   dfcaf:	movq   $0x1,0xe0(%rsp)
   dfcbb:	mov    0x298(%rsp),%rsi
   dfcc3:	xor    %edx,%edx
   dfcc5:	lea    0xe0(%rsp),%rcx
   dfccd:	call   ea2d0 <<tk_train::trainers::bpe::index::Shard>::push>
   dfcd2:	jmp    dfd46 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4066>
   dfcd4:	cmpb   $0x0,0x2f(%rsp)
   dfcd9:	mov    0x318(%rsp),%r12
   dfce1:	je     dfd17 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4037>
   dfce3:	test   %r13,%r13
   dfce6:	je     dfd46 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4066>
   dfce8:	mov    %r12,%rbp
   dfceb:	mov    %rbx,%r12
   dfcee:	mov    0x308(%rsp),%rax
   dfcf6:	lea    0x18(%rax),%rbx
   dfcfa:	jmp    dfd05 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4025>
   dfcfc:	add    $0x20,%rbx
   dfd00:	dec    %r13
   dfd03:	je     dfd32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4052>
   dfd05:	cmpq   $0x11,(%rbx)
   dfd09:	jb     dfcfc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x401c>
   dfd0b:	mov    -0x10(%rbx),%rdi
   dfd0f:	call   *0x27b93(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dfd15:	jmp    dfcfc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x401c>
   dfd17:	cmpq   $0x11,0x138(%rsp)
   dfd20:	jb     dfd46 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4066>
   dfd22:	mov    0x308(%rsp),%rdi
   dfd2a:	call   *0x27b78(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dfd30:	jmp    dfd46 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4066>
   dfd32:	mov    0x308(%rsp),%rdi
   dfd3a:	call   *0x27b68(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dfd40:	mov    %r12,%rbx
   dfd43:	mov    %rbp,%r12
   dfd46:	mov    0x238(%rsp),%rcx
   dfd4e:	mov    0x118(%rsp),%rbp
   dfd56:	jmp    df6d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39f8>
   dfd5b:	cmp    %rax,0x20(%rsp)
   dfd60:	je     dfd9b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40bb>
   dfd62:	movabs $0xaaaaaaaaaaaaaaab,%rcx
   dfd6c:	mov    0x138(%rsp),%rax
   dfd74:	mul    %rcx
   dfd77:	mov    %rdx,%rbx
   dfd7a:	shr    $0x4,%rbx
   dfd7e:	xchg   %ax,%ax
   dfd80:	mov    0x10(%rsp),%r14
   dfd85:	mov    %r14,%rdi
   dfd88:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   dfd8d:	add    $0x18,%r14
   dfd91:	mov    %r14,0x10(%rsp)
   dfd96:	dec    %rbx
   dfd99:	jne    dfd80 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40a0>
   dfd9b:	cmpq   $0x0,0x288(%rsp)
   dfda4:	je     dfdb4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40d4>
   dfda6:	mov    0x110(%rsp),%rdi
   dfdae:	call   *0x27af4(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   dfdb4:	lea    0x50(%rsp),%rdi
   dfdb9:	call   ea3e0 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>>
   dfdbe:	lea    0x70(%rsp),%rdi
   dfdc3:	call   ea3e0 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>>
   dfdc8:	movdqu 0x160(%rsp),%xmm0
   dfdd1:	movdqa %xmm0,0x120(%rsp)
   dfdda:	mov    0x170(%rsp),%rax
   dfde2:	mov    %rax,0x130(%rsp)
   dfdea:	mov    0xa0(%rsp),%rcx
   dfdf2:	movzbl 0x30(%rcx),%eax
   dfdf6:	mov    %al,0x328(%rsp)
   dfdfd:	mov    0x28(%rcx),%rax
   dfe01:	mov    %rax,0x270(%rsp)
   dfe09:	movl   $0x0,0xb0(%rsp)
   dfe14:	movb   $0x0,0xb4(%rsp)
   dfe1c:	movq   $0x0,0xb8(%rsp)
   dfe28:	movq   $0x0,0x140(%rsp)
   dfe34:	movq   $0x8,0x148(%rsp)
   dfe40:	movq   $0x0,0x150(%rsp)
   dfe4c:	movb   $0x0,0x1a8(%rsp)
   dfe54:	mov    0x2f0(%rsp),%r15
   dfe5c:	mov    %r15,0x30(%rsp)
   dfe61:	mov    0x190(%rsp),%r14
   dfe69:	mov    %r14,0x38(%rsp)
   dfe6e:	lea    0x120(%rsp),%rax
   dfe76:	mov    %rax,0x40(%rsp)
   dfe7b:	lea    0x328(%rsp),%rcx
   dfe83:	mov    %rcx,0x48(%rsp)
   dfe88:	lea    0x270(%rsp),%rdx
   dfe90:	mov    %rdx,0x50(%rsp)
   dfe95:	lea    0x58(%rsp),%rsi
   dfe9a:	lea    0xb0(%rsp),%rdi
   dfea2:	mov    %rdi,0x58(%rsp)
   dfea7:	mov    %r15,0xe0(%rsp)
   dfeaf:	mov    %r14,0xe8(%rsp)
   dfeb7:	lea    0xf0(%rsp),%rdi
   dfebf:	mov    %rax,0xf0(%rsp)
   dfec7:	mov    %rcx,0xf8(%rsp)
   dfecf:	mov    %rdx,0x100(%rsp)
   dfed7:	mov    %r15,0xc8(%rsp)
   dfedf:	mov    %r14,0xd0(%rsp)
   dfee7:	lea    0xd8(%rsp),%rax
   dfeef:	mov    %r14,0x1e0(%rsp)
   dfef7:	lea    0x1a8(%rsp),%rcx
   dfeff:	mov    %rcx,0x1c0(%rsp)
   dff07:	mov    %rsi,0x1c8(%rsp)
   dff0f:	mov    %rdi,0x1d0(%rsp)
   dff17:	mov    %rax,0x1d8(%rsp)
   dff1f:	mov    %fs:0xffffffffffffffd0,%rax
   dff28:	test   %rax,%rax
   dff2b:	je     dff35 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4255>
   dff2d:	add    $0x110,%rax
   dff33:	jmp    dff3a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x425a>
   dff35:	call   7f660 <rayon_core::registry::global_registry>
   dff3a:	mov    (%rax),%rax
   dff3d:	mov    0x208(%rax),%rcx
   dff44:	xor    %eax,%eax
   dff46:	cmp    $0xffffffffffffffff,%r14
   dff4a:	sete   %al
   dff4d:	cmp    %rcx,%rax
   dff50:	cmova  %rax,%rcx
   dff54:	lea    0x160(%rsp),%rbx
   dff5c:	mov    $0x1,%r8d
   dff62:	mov    %rbx,%rdi
   dff65:	mov    %r14,%rsi
   dff68:	xor    %edx,%edx
   dff6a:	mov    %r15,%r9
   dff6d:	lea    0x1c0(%rsp),%rax
   dff75:	push   %rax
   dff76:	push   %r14
   dff78:	call   ea4f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>>
   dff7d:	add    $0x10,%rsp
   dff81:	mov    0x170(%rsp),%rax
   dff89:	test   %rax,%rax
   dff8c:	je     dffbb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42db>
   dff8e:	mov    0x160(%rsp),%rcx
   dff96:	test   %rcx,%rcx
   dff99:	je     dffbb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42db>
   dff9b:	xor    %edx,%edx
   dff9d:	nopl   (%rax)
   dffa0:	add    0x10(%rcx),%rdx
   dffa4:	dec    %rax
   dffa7:	je     dffb2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42d2>
   dffa9:	mov    0x18(%rcx),%rcx
   dffad:	test   %rcx,%rcx
   dffb0:	jne    dffa0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42c0>
   dffb2:	test   %rdx,%rdx
   dffb5:	jne    e07d4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4af4>
   dffbb:	movdqu 0x160(%rsp),%xmm0
   dffc4:	movdqa %xmm0,0x1c0(%rsp)
   dffcd:	mov    0x170(%rsp),%rbx
   dffd5:	mov    %rbx,0x1d0(%rsp)
   dffdd:	mov    0x1c0(%rsp),%rdi
   dffe5:	mov    0x1c8(%rsp),%rax
   dffed:	test   %rdi,%rdi
   dfff0:	je     e011e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x443e>
   dfff6:	mov    %rax,0x8(%rsp)
   dfffb:	dec    %rbx
   dfffe:	mov    %rbx,%rax
   e0001:	jmp    e0020 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4340>
   e0003:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e0010:	lea    -0x1(%rbx),%rax
   e0014:	mov    %r12,%rdi
   e0017:	test   %r12,%r12
   e001a:	je     e0123 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4443>
   e0020:	mov    %rax,%rbx
   e0023:	mov    0x18(%rdi),%r12
   e0027:	test   %r12,%r12
   e002a:	je     e0040 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4360>
   e002c:	movq   $0x0,0x20(%r12)
   e0035:	jmp    e0049 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4369>
   e0037:	nopw   0x0(%rax,%rax,1)
   e0040:	movq   $0x0,0x8(%rsp)
   e0049:	mov    (%rdi),%r15
   e004c:	movdqu 0x8(%rdi),%xmm0
   e0051:	movdqa %xmm0,0xe0(%rsp)
   e005a:	call   *0x27848(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0060:	cmp    $0xffffffffffffffff,%r15
   e0064:	je     e0126 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4446>
   e006a:	mov    %r15,0x30(%rsp)
   e006f:	movdqa 0xe0(%rsp),%xmm0
   e0078:	lea    0x38(%rsp),%rax
   e007d:	movdqu %xmm0,(%rax)
   e0081:	mov    0x38(%rsp),%r14
   e0086:	mov    0x40(%rsp),%r13
   e008b:	mov    0x140(%rsp),%rax
   e0093:	mov    0x150(%rsp),%rbp
   e009b:	sub    %rbp,%rax
   e009e:	cmp    %rax,%r13
   e00a1:	ja     e00f6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4416>
   e00a3:	test   %r13,%r13
   e00a6:	je     e00d4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x43f4>
   e00a8:	lea    0x0(,%rbp,2),%rdi
   e00b0:	add    %rbp,%rdi
   e00b3:	shl    $0x3,%rdi
   e00b7:	add    0x148(%rsp),%rdi
   e00bf:	lea    0x0(,%r13,8),%rax
   e00c7:	lea    (%rax,%rax,2),%rdx
   e00cb:	mov    %r14,%rsi
   e00ce:	call   *0x277dc(%rip)        # 1078b0 <memcpy@GLIBC_2.14>
   e00d4:	add    %r13,%rbp
   e00d7:	mov    %rbp,0x150(%rsp)
   e00df:	test   %r15,%r15
   e00e2:	je     e0010 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4330>
   e00e8:	mov    %r14,%rdi
   e00eb:	call   *0x277b7(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e00f1:	jmp    e0010 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4330>
   e00f6:	mov    $0x8,%ecx
   e00fb:	mov    $0x18,%r8d
   e0101:	lea    0x140(%rsp),%rdi
   e0109:	mov    %rbp,%rsi
   e010c:	mov    %r13,%rdx
   e010f:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e0114:	mov    0x150(%rsp),%rbp
   e011c:	jmp    e00a8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x43c8>
   e011e:	xor    %r12d,%r12d
   e0121:	jmp    e012b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x444b>
   e0123:	xor    %r12d,%r12d
   e0126:	mov    0x8(%rsp),%rax
   e012b:	mov    %r12,0x1c0(%rsp)
   e0133:	mov    %rax,0x1c8(%rsp)
   e013b:	mov    %rbx,0x1d0(%rsp)
   e0143:	lea    0x1c0(%rsp),%rdi
   e014b:	call   eda60 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0150:	movdqu 0x140(%rsp),%xmm0
   e0159:	movdqa %xmm0,0x2c0(%rsp)
   e0162:	mov    0x150(%rsp),%rax
   e016a:	mov    %rax,0x2d0(%rsp)
   e0172:	mov    0xb8(%rsp),%r13
   e017a:	mov    0xc0(%rsp),%rbp
   e0182:	cmpb   $0x0,0xb4(%rsp)
   e018a:	jne    e08d2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bf2>
   e0190:	test   %r13,%r13
   e0193:	je     e01de <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44fe>
   e0195:	mov    0x2c8(%rsp),%rbx
   e019d:	mov    0x2d0(%rsp),%r15
   e01a5:	test   %r15,%r15
   e01a8:	je     e01c1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44e1>
   e01aa:	mov    %rbx,%r14
   e01ad:	nopl   (%rax)
   e01b0:	mov    %r14,%rdi
   e01b3:	call   dae20 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   e01b8:	add    $0x18,%r14
   e01bc:	dec    %r15
   e01bf:	jne    e01b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44d0>
   e01c1:	cmpq   $0x0,0x2c0(%rsp)
   e01ca:	je     e0387 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46a7>
   e01d0:	mov    %rbx,%rdi
   e01d3:	call   *0x276cf(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e01d9:	jmp    e0387 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46a7>
   e01de:	mov    0x2c0(%rsp),%rax
   e01e6:	mov    0x2c8(%rsp),%r13
   e01ee:	mov    0x2d0(%rsp),%rbp
   e01f6:	cmp    $0xffffffffffffffff,%rax
   e01fa:	je     e0387 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46a7>
   e0200:	lea    0x0(,%rbp,2),%rcx
   e0208:	add    %rbp,%rcx
   e020b:	lea    0x0(%r13,%rcx,8),%rcx
   e0210:	mov    %r13,0x30(%rsp)
   e0215:	mov    %r13,0x38(%rsp)
   e021a:	mov    %rax,0x40(%rsp)
   e021f:	mov    %rcx,0x48(%rsp)
   e0224:	lea    0x50(%rsp),%r14
   e0229:	movq   $0x0,0x50(%rsp)
   e0232:	movq   $0x0,0x70(%rsp)
   e023b:	lea    0x1c8(%rsp),%r15
   e0243:	lea    0x1c0(%rsp),%r13
   e024b:	lea    0x30(%rsp),%rbp
   e0250:	mov    0xa0(%rsp),%rbx
   e0258:	jmp    e028e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45ae>
   e025a:	nopw   0x0(%rax,%rax,1)
   e0260:	xor    %r12d,%r12d
   e0263:	lea    (%r12,%r12,2),%rdi
   e0267:	shl    $0x4,%rdi
   e026b:	movdqu (%r15),%xmm0
   e0270:	movdqu 0x10(%r15),%xmm1
   e0276:	movdqu %xmm1,0x10(%rax,%rdi,1)
   e027c:	movdqu %xmm0,(%rax,%rdi,1)
   e0281:	mov    %rcx,0x20(%rax,%rdi,1)
   e0286:	mov    %edx,0x28(%rax,%rdi,1)
   e028a:	mov    %esi,0x2c(%rax,%rdi,1)
   e028e:	mov    %r13,%rdi
   e0291:	mov    %rbp,%rsi
   e0294:	call   dac50 <<core::iter::adapters::flatten::FlattenCompat<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>, alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>> as core::iter::traits::iterator::Iterator>::next>
   e0299:	cmpl   $0x1,0x1c0(%rsp)
   e02a1:	jne    e03d7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46f7>
   e02a7:	mov    0x10(%rbx),%r12
   e02ab:	cmp    (%rbx),%r12
   e02ae:	jne    e02b8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45d8>
   e02b0:	mov    %rbx,%rdi
   e02b3:	call   e1700 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   e02b8:	mov    0x8(%rbx),%rax
   e02bc:	lea    (%r12,%r12,2),%rsi
   e02c0:	shl    $0x4,%rsi
   e02c4:	movdqu (%r15),%xmm0
   e02c9:	movdqu 0x10(%r15),%xmm1
   e02cf:	movdqu 0x20(%r15),%xmm2
   e02d5:	movdqu %xmm2,0x20(%rax,%rsi,1)
   e02db:	movdqu %xmm1,0x10(%rax,%rsi,1)
   e02e1:	movdqu %xmm0,(%rax,%rsi,1)
   e02e6:	lea    0x1(%r12),%rcx
   e02eb:	mov    %rcx,0x10(%rbx)
   e02ef:	mov    0x20(%rax,%rsi,1),%rcx
   e02f4:	mov    0x28(%rax,%rsi,1),%edx
   e02f8:	mov    0x2c(%rax,%rsi,1),%esi
   e02fc:	test   %r12,%r12
   e02ff:	jne    e0352 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4672>
   e0301:	jmp    e0260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4580>
   e0306:	cs nopw 0x0(%rax,%rax,1)
   e0310:	test   %r9b,%r9b
   e0313:	jne    e0263 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4583>
   e0319:	lea    (%r12,%r12,2),%r9
   e031d:	shl    $0x4,%r9
   e0321:	movdqu (%r8),%xmm0
   e0326:	movdqu 0x10(%r8),%xmm1
   e032c:	movdqu 0x20(%r8),%xmm2
   e0332:	movdqu %xmm2,0x20(%rax,%r9,1)
   e0339:	movdqu %xmm1,0x10(%rax,%r9,1)
   e0340:	movdqu %xmm0,(%rax,%r9,1)
   e0346:	mov    %rdi,%r12
   e0349:	test   %rdi,%rdi
   e034c:	je     e0260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4580>
   e0352:	lea    -0x1(%r12),%rdi
   e0357:	shr    $0x3,%rdi
   e035b:	lea    (%rdi,%rdi,2),%r9
   e035f:	shl    $0x4,%r9
   e0363:	lea    (%rax,%r9,1),%r8
   e0367:	cmp    0x20(%rax,%r9,1),%rcx
   e036c:	setbe  %r9b
   e0370:	jne    e0310 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4630>
   e0372:	cmp    %edx,0x28(%r8)
   e0376:	setbe  %r9b
   e037a:	jne    e0310 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4630>
   e037c:	cmp    %esi,0x2c(%r8)
   e0380:	ja     e0319 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4639>
   e0382:	jmp    e0263 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4583>
   e0387:	cmpq   $0x0,0x120(%rsp)
   e0390:	je     e03a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46c0>
   e0392:	mov    0x128(%rsp),%rdi
   e039a:	call   *0x27508(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e03a0:	mov    0x2a0(%rsp),%rax
   e03a8:	mov    %r13,0x8(%rax)
   e03ac:	mov    %rbp,0x10(%rax)
   e03b0:	movq   $0x1,(%rax)
   e03b7:	mov    0x198(%rsp),%rdi
   e03bf:	call   *0x274e3(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e03c5:	add    $0x3e8,%rsp
   e03cc:	pop    %rbx
   e03cd:	pop    %r12
   e03cf:	pop    %r13
   e03d1:	pop    %r14
   e03d3:	pop    %r15
   e03d5:	pop    %rbp
   e03d6:	ret
   e03d7:	mov    0x30(%rsp),%r15
   e03dc:	test   %r15,%r15
   e03df:	je     e0432 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4752>
   e03e1:	mov    0x38(%rsp),%r13
   e03e6:	mov    0x48(%rsp),%rcx
   e03eb:	mov    %rcx,%rax
   e03ee:	sub    %r13,%rax
   e03f1:	movabs $0xaaaaaaaaaaaaaaab,%rdx
   e03fb:	mul    %rdx
   e03fe:	cmp    %r13,%rcx
   e0401:	je     e0421 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4741>
   e0403:	mov    %rdx,%rbp
   e0406:	shr    $0x4,%rbp
   e040a:	nopw   0x0(%rax,%rax,1)
   e0410:	mov    %r13,%rdi
   e0413:	call   dae20 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   e0418:	add    $0x18,%r13
   e041c:	dec    %rbp
   e041f:	jne    e0410 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4730>
   e0421:	cmpq   $0x0,0x40(%rsp)
   e0427:	je     e0432 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4752>
   e0429:	mov    %r15,%rdi
   e042c:	call   *0x27476(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0432:	mov    %r14,%rdi
   e0435:	call   daf10 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>>>>
   e043a:	lea    0x70(%rsp),%rdi
   e043f:	call   daf10 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>>>>
   e0444:	cmpq   $0x0,0x120(%rsp)
   e044d:	je     e045d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x477d>
   e044f:	mov    0x128(%rsp),%rdi
   e0457:	call   *0x2744b(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e045d:	mov    0x300(%rsp),%r14
   e0465:	mov    0x58(%r14),%rax
   e0469:	mov    0x68(%r14),%rbx
   e046d:	sub    %rbx,%rax
   e0470:	mov    0xa8(%rsp),%r15
   e0478:	cmp    %rax,%r15
   e047b:	ja     e0980 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ca0>
   e0481:	lea    0x0(,%rbx,8),%rdi
   e0489:	add    0x60(%r14),%rdi
   e048d:	mov    0x198(%rsp),%r12
   e0495:	mov    %r12,%rsi
   e0498:	mov    0x2f8(%rsp),%rdx
   e04a0:	call   *0x2740a(%rip)        # 1078b0 <memcpy@GLIBC_2.14>
   e04a6:	add    %r15,%rbx
   e04a9:	mov    %rbx,0x68(%r14)
   e04ad:	mov    %r12,%rdi
   e04b0:	call   *0x273f2(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e04b6:	mov    0x2a0(%rsp),%rax
   e04be:	movq   $0x2,0x8(%rax)
   e04c6:	mov    %r15,0x10(%rax)
   e04ca:	movq   $0x0,(%rax)
   e04d1:	jmp    e03c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e5>
   e04d6:	xor    %edi,%edi
   e04d8:	mov    %rdx,%rsi
   e04db:	call   690eb <alloc::raw_vec::handle_error>
   e04e0:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e04e5:	mov    0x38(%rsp),%rcx
   e04ea:	mov    0x2a0(%rsp),%rdx
   e04f2:	mov    %rax,0x8(%rdx)
   e04f6:	mov    %rcx,0x10(%rdx)
   e04fa:	mov    $0x1,%eax
   e04ff:	mov    %rax,(%rdx)
   e0502:	mov    0x3a8(%rsp),%rcx
   e050a:	test   %rcx,%rcx
   e050d:	jne    dd0c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13e0>
   e0513:	jmp    dd0e6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1406>
   e0518:	mov    $0x19,%esi
   e051d:	mov    0x198(%rsp),%rdi
   e0525:	mov    0x138(%rsp),%rdx
   e052d:	call   71e10 <core::option::expect_failed>
   e0532:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0537:	mov    %rsi,%rdx
   e053a:	lea    0x26a07(%rip),%rcx        # 106f48 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a50>
   e0541:	jmp    e05e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4900>
   e0546:	mov    $0x1,%r14b
   e0549:	lea    -0xc225c(%rip),%rdi        # 1e2f4 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x5a7>
   e0550:	lea    0x26d81(%rip),%rdx        # 1072d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3de0>
   e0557:	mov    $0x53,%esi
   e055c:	call   6ed60 <core::panicking::panic_fmt>
   e0561:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0566:	mov    %rdx,%rsi
   e0569:	lea    0x27008(%rip),%rcx        # 107578 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4080>
   e0570:	jmp    e05e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4900>
   e0572:	mov    0x2a0(%rsp),%rdx
   e057a:	movq   $0x1,0x8(%rdx)
   e0582:	jmp    dd0ae <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13ce>
   e0587:	mov    $0x8,%esi
   e058c:	xor    %edi,%edi
   e058e:	lea    0x26e4b(%rip),%rcx        # 1073e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ee8>
   e0595:	jmp    e05e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4900>
   e0597:	call   c0670 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e059c:	jmp    dbd63 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x83>
   e05a1:	call   c0510 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e05a6:	mov    %rax,%rbx
   e05a9:	jmp    dbd7a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9a>
   e05ae:	call   c0670 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e05b3:	jmp    dbe94 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b4>
   e05b8:	call   c0510 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e05bd:	mov    %rax,%rbx
   e05c0:	jmp    dbeab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cb>
   e05c5:	mov    %r11,%rdi
   e05c8:	mov    %rdx,%rsi
   e05cb:	lea    0x269be(%rip),%rcx        # 106f90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   e05d2:	jmp    e05e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4900>
   e05d4:	mov    $0x8,%edi
   e05d9:	lea    0x269c8(%rip),%rcx        # 106fa8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ab0>
   e05e0:	mov    $0x1,%r14b
   e05e3:	call   70c30 <core::slice::index::slice_index_fail>
   e05e8:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e05ed:	mov    %rdx,%rsi
   e05f0:	jmp    e053a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x485a>
   e05f5:	lea    0x120(%rsp),%rdi
   e05fd:	mov    $0x8,%ecx
   e0602:	mov    $0x38,%r8d
   e0608:	xor    %esi,%esi
   e060a:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e060f:	jmp    de20b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x252b>
   e0614:	lea    0xb0(%rsp),%rdi
   e061c:	mov    $0x8,%ecx
   e0621:	mov    $0x18,%r8d
   e0627:	xor    %esi,%esi
   e0629:	mov    %rbp,%rdx
   e062c:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e0631:	mov    0xb0(%rsp),%r13
   e0639:	mov    0xc0(%rsp),%r15
   e0641:	mov    %r13,%rax
   e0644:	sub    %r15,%rax
   e0647:	cmp    %rbp,%rax
   e064a:	jae    e0b9b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ebb>
   e0650:	lea    -0xc2551(%rip),%rdi        # 1e106 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x3b9>
   e0657:	lea    0x26662(%rip),%rdx        # 106cc0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x37c8>
   e065e:	mov    $0x2f,%esi
   e0663:	call   70d00 <core::panicking::panic>
   e0668:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e066d:	lea    -0xc256e(%rip),%rdi        # 1e106 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x3b9>
   e0674:	lea    0x26b6d(%rip),%rdx        # 1071e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3cf0>
   e067b:	mov    $0x2f,%esi
   e0680:	call   70d00 <core::panicking::panic>
   e0685:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e068a:	lea    0xc8(%rsp),%rax
   e0692:	mov    %rax,0x30(%rsp)
   e0697:	lea    -0x6c9be(%rip),%rax        # 73ce0 <<u64 as core::fmt::Display>::fmt>
   e069e:	mov    %rax,0x38(%rsp)
   e06a3:	lea    0x1c0(%rsp),%rcx
   e06ab:	mov    %rcx,0x40(%rsp)
   e06b0:	mov    %rax,0x48(%rsp)
   e06b5:	lea    -0xce905(%rip),%rdi        # 11db7 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x5e37>
   e06bc:	lea    0x26225(%rip),%rdx        # 1068e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33f0>
   e06c3:	lea    0x30(%rsp),%rsi
   e06c8:	call   6ed60 <core::panicking::panic_fmt>
   e06cd:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e06d2:	mov    %r13,0x1c0(%rsp)
   e06da:	mov    %rbp,0x1c8(%rsp)
   e06e2:	lea    -0xc26a3(%rip),%rdi        # 1e046 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e06e9:	lea    0x264e0(%rip),%rcx        # 106bd0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36d8>
   e06f0:	lea    0x262f9(%rip),%r8        # 1069f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e06f7:	lea    0x1c0(%rsp),%rdx
   e06ff:	mov    $0x2b,%esi
   e0704:	movzbl 0x10(%rsp),%r14d
   e070a:	call   71e50 <core::result::unwrap_failed>
   e070f:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0714:	lea    0xc8(%rsp),%rdi
   e071c:	mov    $0x8,%ecx
   e0721:	mov    $0x38,%r8d
   e0727:	xor    %esi,%esi
   e0729:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e072e:	jmp    df0cb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33eb>
   e0733:	call   c0670 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e0738:	jmp    ddb91 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1eb1>
   e073d:	call   c0510 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e0742:	mov    %rax,%rbx
   e0745:	jmp    ddba8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ec8>
   e074a:	lea    0x30(%rsp),%rdi
   e074f:	mov    %r12,%rdx
   e0752:	call   c4660 <<hashbrown::raw::RawTable<((u32, u32), u32)>>::reserve_rehash::<hashbrown::map::make_hasher<(u32, u32), u32, ahash::random_state::RandomState>::{closure#0}>>
   e0757:	jmp    ddcd3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ff3>
   e075c:	lea    0xc8(%rsp),%rdi
   e0764:	mov    $0x8,%ecx
   e0769:	mov    $0x38,%r8d
   e076f:	xor    %esi,%esi
   e0771:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e0776:	jmp    deb1b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e3b>
   e077b:	lea    -0xc267c(%rip),%rdi        # 1e106 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x3b9>
   e0782:	lea    0x26a5f(%rip),%rdx        # 1071e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3cf0>
   e0789:	mov    $0x2f,%esi
   e078e:	call   70d00 <core::panicking::panic>
   e0793:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0798:	mov    %r13,0x1c0(%rsp)
   e07a0:	mov    %rbp,0x1c8(%rsp)
   e07a8:	lea    -0xc2769(%rip),%rdi        # 1e046 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e07af:	lea    0x2641a(%rip),%rcx        # 106bd0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36d8>
   e07b6:	lea    0x26233(%rip),%r8        # 1069f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e07bd:	lea    0x1c0(%rsp),%rdx
   e07c5:	mov    $0x2b,%esi
   e07ca:	call   71e50 <core::result::unwrap_failed>
   e07cf:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e07d4:	lea    0x140(%rsp),%rdi
   e07dc:	mov    $0x8,%ecx
   e07e1:	mov    $0x18,%r8d
   e07e7:	xor    %esi,%esi
   e07e9:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e07ee:	jmp    dffbb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42db>
   e07f3:	lea    0x1c0(%rsp),%rax
   e07fb:	mov    %rax,0x30(%rsp)
   e0800:	lea    0x26621(%rip),%rdx        # 106e28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3930>
   e0807:	jmp    e081d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b3d>
   e0809:	lea    0x1c0(%rsp),%rax
   e0811:	mov    %rax,0x30(%rsp)
   e0816:	lea    0x26623(%rip),%rdx        # 106e40 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3948>
   e081d:	lea    0xe0(%rsp),%rax
   e0825:	lea    -0x6cb4c(%rip),%rcx        # 73ce0 <<u64 as core::fmt::Display>::fmt>
   e082c:	mov    0x290(%rsp),%rsi
   e0834:	mov    %rcx,0x8(%rsi)
   e0838:	mov    %rax,0x10(%rsi)
   e083c:	mov    %rcx,0x18(%rsi)
   e0840:	lea    -0xd02a1(%rip),%rdi        # 105a6 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x4626>
   e0847:	call   6ed60 <core::panicking::panic_fmt>
   e084c:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0851:	mov    $0x1a,%esi
   e0856:	lea    -0xc25f3(%rip),%rax        # 1e26a <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x51d>
   e085d:	mov    %rax,0x198(%rsp)
   e0865:	lea    0x26874(%rip),%rax        # 1070e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3be8>
   e086c:	mov    %rax,0x138(%rsp)
   e0874:	jmp    e051d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x483d>
   e0879:	lea    -0xc277a(%rip),%rdi        # 1e106 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x3b9>
   e0880:	lea    0x26961(%rip),%rdx        # 1071e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3cf0>
   e0887:	mov    $0x2f,%esi
   e088c:	call   70d00 <core::panicking::panic>
   e0891:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0896:	mov    %r13,0xe0(%rsp)
   e089e:	mov    %rax,0xe8(%rsp)
   e08a6:	lea    -0xc2867(%rip),%rdi        # 1e046 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e08ad:	lea    0x2631c(%rip),%rcx        # 106bd0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36d8>
   e08b4:	lea    0x26135(%rip),%r8        # 1069f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e08bb:	lea    0xe0(%rsp),%rdx
   e08c3:	mov    $0x2b,%esi
   e08c8:	call   71e50 <core::result::unwrap_failed>
   e08cd:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e08d2:	mov    %r13,0x30(%rsp)
   e08d7:	mov    %rbp,0x38(%rsp)
   e08dc:	lea    -0xc289d(%rip),%rdi        # 1e046 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e08e3:	lea    0x262e6(%rip),%rcx        # 106bd0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36d8>
   e08ea:	lea    0x260ff(%rip),%r8        # 1069f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e08f1:	lea    0x30(%rsp),%rdx
   e08f6:	mov    $0x2b,%esi
   e08fb:	call   71e50 <core::result::unwrap_failed>
   e0900:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0905:	lea    0x2654c(%rip),%rdi        # 106e58 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3960>
   e090c:	call   6e8d0 <compact_str::unwrap_with_msg_fail::<compact_str::ReserveError>>
   e0911:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0916:	xor    %eax,%eax
   e0918:	jmp    e091c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c3c>
   e091a:	mov    $0x1,%al
   e091c:	mov    %al,0x1c0(%rsp)
   e0923:	mov    $0x1,%r14b
   e0926:	lea    -0xc23ad(%rip),%rdi        # 1e580 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x833>
   e092d:	lea    0x2633c(%rip),%rcx        # 106c70 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3778>
   e0934:	lea    0x26c25(%rip),%r8        # 107560 <bitcannon::classify::atom_tables::ATOM_TABLES+0x4068>
   e093b:	lea    0x1c0(%rsp),%rdx
   e0943:	mov    $0x13,%esi
   e0948:	call   71e50 <core::result::unwrap_failed>
   e094d:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0952:	lea    -0xc26d5(%rip),%rdi        # 1e284 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x537>
   e0959:	lea    0x26798(%rip),%rdx        # 1070f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c00>
   e0960:	mov    $0x15,%esi
   e0965:	call   71e10 <core::option::expect_failed>
   e096a:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e096f:	lea    0x26a3a(%rip),%rdi        # 1073b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3eb8>
   e0976:	call   72180 <core::panicking::panic_const::panic_const_rem_by_zero>
   e097b:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0980:	lea    0x58(%r14),%rdi
   e0984:	mov    $0x4,%ecx
   e0989:	mov    $0x8,%r8d
   e098f:	mov    %rbx,%rsi
   e0992:	mov    %r15,%rdx
   e0995:	call   cb8e0 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e099a:	mov    0x300(%rsp),%r14
   e09a2:	mov    0x68(%r14),%rbx
   e09a6:	mov    0xa8(%rsp),%r15
   e09ae:	jmp    e0481 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47a1>
   e09b3:	mov    %rbp,%r12
   e09b6:	lea    0x26873(%rip),%rcx        # 107230 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d38>
   e09bd:	mov    %rcx,0x110(%rsp)
   e09c5:	mov    %rax,%rdi
   e09c8:	mov    %r12,%rsi
   e09cb:	mov    0x110(%rsp),%rdx
   e09d3:	call   6fcfb <core::panicking::panic_bounds_check>
   e09d8:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e09dd:	lea    0x264a4(%rip),%rax        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e09e4:	mov    %rax,0x288(%rsp)
   e09ec:	mov    %rbp,%rsi
   e09ef:	mov    0x288(%rsp),%rdx
   e09f7:	call   6fcfb <core::panicking::panic_bounds_check>
   e09fc:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0a01:	mov    $0x4,%edi
   e0a06:	mov    %rbx,%rsi
   e0a09:	call   690eb <alloc::raw_vec::handle_error>
   e0a0e:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0a13:	shr    $0x7,%r10
   e0a17:	xor    %eax,%eax
   e0a19:	mov    %r10,%rbx
   e0a1c:	jmp    e0a75 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d95>
   e0a1e:	mov    %rdx,%rbx
   e0a21:	jmp    e0a75 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d95>
   e0a23:	mov    %rax,%rbx
   e0a26:	mov    %r14,%rax
   e0a29:	lea    0x26458(%rip),%rcx        # 106e88 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3990>
   e0a30:	jmp    e0a6d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d8d>
   e0a32:	mov    %rax,%rbx
   e0a35:	mov    %r14,%rax
   e0a38:	lea    0x26431(%rip),%rcx        # 106e70 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3978>
   e0a3f:	jmp    e0a6d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d8d>
   e0a41:	lea    0x262c0(%rip),%rdi        # 106d08 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3810>
   e0a48:	call   72160 <core::panicking::panic_const::panic_const_div_by_zero>
   e0a4d:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0a52:	mov    $0x8,%edi
   e0a57:	mov    $0x20,%esi
   e0a5c:	call   690eb <alloc::raw_vec::handle_error>
   e0a61:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0a66:	lea    0x25d23(%rip),%rcx        # 106790 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3298>
   e0a6d:	mov    %rcx,0x118(%rsp)
   e0a75:	mov    $0x1,%r14b
   e0a78:	mov    %rbx,%rdi
   e0a7b:	mov    %rax,%rsi
   e0a7e:	mov    0x118(%rsp),%rdx
   e0a86:	call   6fcfb <core::panicking::panic_bounds_check>
   e0a8b:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0a90:	lea    0x26259(%rip),%rdi        # 106cf0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x37f8>
   e0a97:	call   72160 <core::panicking::panic_const::panic_const_div_by_zero>
   e0a9c:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0aa1:	xor    %r14d,%r14d
   e0aa4:	lea    0x2622d(%rip),%rdi        # 106cd8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x37e0>
   e0aab:	call   72160 <core::panicking::panic_const::panic_const_div_by_zero>
   e0ab0:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0ab5:	mov    %rbx,%r14
   e0ab8:	lea    0x26381(%rip),%rax        # 106e40 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3948>
   e0abf:	mov    %rax,0x2f8(%rsp)
   e0ac7:	mov    %r14,%rdi
   e0aca:	mov    0x2f8(%rsp),%rdx
   e0ad2:	call   6fcfb <core::panicking::panic_bounds_check>
   e0ad7:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0adc:	lea    -0xc67a7(%rip),%rdi        # 1a33c <bitcannon::classify::atom_tables::BMP_RLE+0x2690>
   e0ae3:	lea    0x239ae(%rip),%rcx        # 104498 <bitcannon::classify::atom_tables::ATOM_TABLES+0xfa0>
   e0aea:	lea    0x239df(%rip),%r8        # 1044d0 <bitcannon::classify::atom_tables::ATOM_TABLES+0xfd8>
   e0af1:	lea    0x26f(%rsp),%rdx
   e0af9:	mov    $0xe,%esi
   e0afe:	call   71e50 <core::result::unwrap_failed>
   e0b03:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0b08:	mov    %r15,%r14
   e0b0b:	lea    0x26796(%rip),%rax        # 1072a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3db0>
   e0b12:	jmp    e0b23 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e43>
   e0b14:	mov    %rbx,%r14
   e0b17:	jmp    e0b2b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e4b>
   e0b19:	mov    %r15,%r14
   e0b1c:	lea    0x2679d(%rip),%rax        # 1072c0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3dc8>
   e0b23:	mov    %rax,0x110(%rsp)
   e0b2b:	mov    %r14,%rdi
   e0b2e:	mov    %r12,%rsi
   e0b31:	mov    0x110(%rsp),%rdx
   e0b39:	call   6fcfb <core::panicking::panic_bounds_check>
   e0b3e:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0b40:	lea    0x26869(%rip),%rdi        # 1073b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3eb8>
   e0b47:	call   72180 <core::panicking::panic_const::panic_const_rem_by_zero>
   e0b4c:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0b4e:	lea    0x2685b(%rip),%rdi        # 1073b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3eb8>
   e0b55:	call   72180 <core::panicking::panic_const::panic_const_rem_by_zero>
   e0b5a:	jmp    e0b99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4eb9>
   e0b5c:	mov    %rax,0x2e8(%rsp)
   e0b64:	lea    0x265d5(%rip),%rax        # 107140 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c48>
   e0b6b:	jmp    e0b74 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e94>
   e0b6d:	lea    0x2659c(%rip),%rax        # 107110 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c18>
   e0b74:	mov    %rax,0x320(%rsp)
   e0b7c:	mov    0x2e8(%rsp),%rdi
   e0b84:	mov    0x190(%rsp),%rsi
   e0b8c:	mov    0x320(%rsp),%rdx
   e0b94:	call   6fcfb <core::panicking::panic_bounds_check>
   e0b99:	ud2
   e0b9b:	mov    0xb8(%rsp),%r12
   e0ba3:	jmp    df372 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3692>
   e0ba8:	mov    %rax,%r15
   e0bab:	mov    0x198(%rsp),%rdi
   e0bb3:	call   *0x26cef(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0bb9:	mov    %r15,%rdi
   e0bbc:	call   1019a0 <_Unwind_Resume@plt>
   e0bc1:	mov    %rax,%r15
   e0bc4:	mov    %rbx,0x340(%rsp)
   e0bcc:	jmp    e0c32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4f52>
   e0bce:	mov    %rax,%rbx
   e0bd1:	lea    0x120(%rsp),%rdi
   e0bd9:	jmp    e0c7b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4f9b>
   e0bde:	jmp    e0cfd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x501d>
   e0be3:	mov    %rax,%r12
   e0be6:	lea    0x120(%rsp),%rdi
   e0bee:	jmp    e0cb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4fd7>
   e0bf3:	jmp    e10c3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x53e3>
   e0bf8:	mov    %rax,%rbx
   e0bfb:	mov    %r13,0x250(%rsp)
   e0c03:	jmp    e0ceb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x500b>
   e0c08:	mov    %rax,%r15
   e0c0b:	mov    %r12,0x1c0(%rsp)
   e0c13:	mov    0x8(%rsp),%rax
   e0c18:	mov    %rax,0x1c8(%rsp)
   e0c20:	mov    %rbx,0x1d0(%rsp)
   e0c28:	lea    0x30(%rsp),%rdi
   e0c2d:	call   dabf0 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0c32:	mov    0x340(%rsp),%rdi
   e0c3a:	call   eda60 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0c3f:	jmp    e0d8c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x50ac>
   e0c44:	mov    %rax,%rbx
   e0c47:	mov    %r12,0xe0(%rsp)
   e0c4f:	mov    0x10(%rsp),%rax
   e0c54:	mov    %rax,0xe8(%rsp)
   e0c5c:	mov    0x8(%rsp),%rax
   e0c61:	mov    %rax,0xf0(%rsp)
   e0c69:	lea    0x30(%rsp),%rdi
   e0c6e:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0c73:	lea    0xe0(%rsp),%rdi
   e0c7b:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e0c80:	jmp    e0f86 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52a6>
   e0c85:	mov    %rax,%r12
   e0c88:	mov    %rbx,0x1c0(%rsp)
   e0c90:	mov    0x8(%rsp),%rax
   e0c95:	mov    %rax,0x1c8(%rsp)
   e0c9d:	mov    %r15,0x1d0(%rsp)
   e0ca5:	lea    0x30(%rsp),%rdi
   e0caa:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0caf:	lea    0x1c0(%rsp),%rdi
   e0cb7:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e0cbc:	jmp    e1007 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5327>
   e0cc1:	mov    %rax,%rbx
   e0cc4:	mov    %r14,0x1c0(%rsp)
   e0ccc:	mov    0x8(%rsp),%rax
   e0cd1:	mov    %rax,0x1c8(%rsp)
   e0cd9:	mov    %r15,0x1d0(%rsp)
   e0ce1:	lea    0x30(%rsp),%rdi
   e0ce6:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0ceb:	mov    0x250(%rsp),%rdi
   e0cf3:	call   e44f0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>>
   e0cf8:	jmp    e0e25 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5145>
   e0cfd:	mov    %rax,0x8(%rsp)
   e0d02:	mov    0x30(%rsp),%rdi
   e0d07:	mov    0x38(%rsp),%rsi
   e0d0c:	call   e2740 <core::ptr::drop_glue::<std::collections::hash::map::HashMap<(u32, u32), u32, ahash::random_state::RandomState>>>
   e0d11:	jmp    e120b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x552b>
   e0d16:	mov    %rax,%r15
   e0d19:	jmp    e0d32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5052>
   e0d1b:	jmp    e0d22 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5042>
   e0d1d:	jmp    e1164 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5484>
   e0d22:	mov    %rax,%r15
   e0d25:	lea    0x1c0(%rsp),%rdi
   e0d2d:	call   e25c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   e0d32:	lea    0x30(%rsp),%rdi
   e0d37:	call   ea1d0 <core::ptr::drop_glue::<core::iter::adapters::flatten::Flatten<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>>>
   e0d3c:	cmpq   $0x0,0x160(%rsp)
   e0d45:	je     e12ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x560f>
   e0d4b:	mov    0x168(%rsp),%rdi
   e0d53:	jmp    e0f18 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5238>
   e0d58:	mov    %rax,%r15
   e0d5b:	lea    0x30(%rsp),%rdi
   e0d60:	call   db020 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e0d65:	jmp    e1174 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5494>
   e0d6a:	jmp    e1117 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5437>
   e0d6f:	jmp    e1117 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5437>
   e0d74:	mov    %rax,%r15
   e0d77:	lea    0x120(%rsp),%rdi
   e0d7f:	call   e9400 <core::ptr::drop_glue::<rayon::vec::DrainProducer<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0d84:	jmp    e10d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x53f8>
   e0d89:	mov    %rax,%r15
   e0d8c:	lea    0x140(%rsp),%rdi
   e0d94:	call   dabf0 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0d99:	mov    0xb8(%rsp),%rdi
   e0da1:	mov    0xc0(%rsp),%rsi
   e0da9:	call   d8110 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0dae:	jmp    e0f01 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5221>
   e0db3:	call   71f13 <core::panicking::panic_in_cleanup>
   e0db8:	mov    %rax,0x8(%rsp)
   e0dbd:	jmp    e120b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x552b>
   e0dc2:	jmp    e1117 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5437>
   e0dc7:	mov    %rax,%rbx
   e0dca:	jmp    e11cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54ef>
   e0dcf:	mov    %rax,%r15
   e0dd2:	cmpb   $0xd8,0x35f(%rsp)
   e0dda:	jne    e1174 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5494>
   e0de0:	mov    0x348(%rsp),%rdi
   e0de8:	call   *0x26aba(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0dee:	jmp    e1174 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5494>
   e0df3:	mov    %rax,%r15
   e0df6:	jmp    e10e2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5402>
   e0dfb:	mov    %rax,%r15
   e0dfe:	lea    0x1c8(%rsp),%rdi
   e0e06:	call   db020 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e0e0b:	lea    0x30(%rsp),%rdi
   e0e10:	call   db0a0 <core::ptr::drop_glue::<core::iter::adapters::flatten::Flatten<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0e15:	jmp    e0f01 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5221>
   e0e1a:	mov    %rax,%r15
   e0e1d:	jmp    e1058 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5378>
   e0e22:	mov    %rax,%rbx
   e0e25:	lea    0x120(%rsp),%rdi
   e0e2d:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0e32:	mov    0xd0(%rsp),%rdi
   e0e3a:	mov    0xd8(%rsp),%rsi
   e0e42:	call   d8110 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0e47:	mov    %rbx,%r15
   e0e4a:	movzbl 0x10(%rsp),%r14d
   e0e50:	jmp    e1058 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5378>
   e0e55:	call   71f13 <core::panicking::panic_in_cleanup>
   e0e5a:	mov    %rax,%r15
   e0e5d:	jmp    e106b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x538b>
   e0e62:	jmp    e0e6c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x518c>
   e0e64:	mov    %rax,%r15
   e0e67:	jmp    e1189 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54a9>
   e0e6c:	mov    %rax,%r15
   e0e6f:	lea    0x2a8(%rsp),%rdi
   e0e77:	call   e47c0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::SelectedMerge>>>
   e0e7c:	mov    %r15,%rdi
   e0e7f:	call   1019a0 <_Unwind_Resume@plt>
   e0e84:	mov    %rax,%r15
   e0e87:	mov    $0x1,%r14b
   e0e8a:	jmp    e111f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x543f>
   e0e8f:	jmp    e0eb5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51d5>
   e0e91:	mov    %rax,%r15
   e0e94:	cmpq   $0x0,0x160(%rsp)
   e0e9d:	je     e106b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x538b>
   e0ea3:	mov    0x168(%rsp),%rdi
   e0eab:	jmp    e1065 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5385>
   e0eb0:	jmp    e11b3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54d3>
   e0eb5:	mov    %rax,%r15
   e0eb8:	lea    0x1c0(%rsp),%rdi
   e0ec0:	call   db020 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e0ec5:	jmp    e1174 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5494>
   e0eca:	mov    %rax,%r15
   e0ecd:	cmpq   $0x0,0x10(%rsp)
   e0ed3:	je     e1167 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5487>
   e0ed9:	mov    %r13,%rdi
   e0edc:	call   *0x269c6(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0ee2:	jmp    e1167 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5487>
   e0ee7:	mov    %rax,%r15
   e0eea:	lea    0x30(%rsp),%rdi
   e0eef:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0ef4:	lea    0x2c0(%rsp),%rdi
   e0efc:	call   dabf0 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0f01:	cmpq   $0x0,0x120(%rsp)
   e0f0a:	je     e12ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x560f>
   e0f10:	mov    0x128(%rsp),%rdi
   e0f18:	call   *0x2698a(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0f1e:	mov    0x198(%rsp),%rdi
   e0f26:	call   *0x2697c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0f2c:	mov    %r15,%rdi
   e0f2f:	call   1019a0 <_Unwind_Resume@plt>
   e0f34:	call   71f13 <core::panicking::panic_in_cleanup>
   e0f39:	mov    %rax,%rbx
   e0f3c:	lea    0xe0(%rsp),%rdi
   e0f44:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0f49:	lea    0x30(%rsp),%rdi
   e0f4e:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0f53:	jmp    e11cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54ef>
   e0f58:	call   71f13 <core::panicking::panic_in_cleanup>
   e0f5d:	mov    %rax,%rbx
   e0f60:	lea    0x140(%rsp),%rdi
   e0f68:	call   e48b0 <core::ptr::drop_glue::<rayon::vec::Drain<((u32, u32), u32, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>)>>>
   e0f6d:	cmpq   $0x0,0xb0(%rsp)
   e0f76:	je     e0f86 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52a6>
   e0f78:	mov    0xb8(%rsp),%rdi
   e0f80:	call   *0x26922(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e0f86:	lea    0xc8(%rsp),%rdi
   e0f8e:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0f93:	mov    0x278(%rsp),%rdi
   e0f9b:	mov    0x280(%rsp),%rsi
   e0fa3:	call   d8110 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0fa8:	jmp    e11cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54ef>
   e0fad:	call   71f13 <core::panicking::panic_in_cleanup>
   e0fb2:	call   71f13 <core::panicking::panic_in_cleanup>
   e0fb7:	mov    %rax,%r15
   e0fba:	lea    0x1c0(%rsp),%rdi
   e0fc2:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0fc7:	lea    0x30(%rsp),%rdi
   e0fcc:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e0fd1:	xor    %r14d,%r14d
   e0fd4:	jmp    e111f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x543f>
   e0fd9:	call   71f13 <core::panicking::panic_in_cleanup>
   e0fde:	mov    %rax,%r12
   e0fe1:	lea    0x160(%rsp),%rdi
   e0fe9:	call   e28a0 <core::ptr::drop_glue::<rayon::vec::Drain<((u32, u32), u32, core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>)>>>
   e0fee:	cmpq   $0x0,0xb0(%rsp)
   e0ff7:	je     e1007 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5327>
   e0ff9:	mov    0xb8(%rsp),%rdi
   e1001:	call   *0x268a1(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e1007:	lea    0xc8(%rsp),%rdi
   e100f:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e1014:	mov    0x278(%rsp),%rdi
   e101c:	mov    0x280(%rsp),%rsi
   e1024:	call   d8110 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e1029:	xor    %r14d,%r14d
   e102c:	mov    %r12,%r15
   e102f:	jmp    e111f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x543f>
   e1034:	call   71f13 <core::panicking::panic_in_cleanup>
   e1039:	call   71f13 <core::panicking::panic_in_cleanup>
   e103e:	mov    %rax,%r15
   e1041:	lea    0x1c0(%rsp),%rdi
   e1049:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e104e:	lea    0x30(%rsp),%rdi
   e1053:	call   e4470 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e1058:	test   %r14b,%r14b
   e105b:	jne    e106b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x538b>
   e105d:	mov    0x240(%rsp),%rdi
   e1065:	call   *0x2683d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e106b:	lea    0xe0(%rsp),%rdi
   e1073:	call   db020 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e1078:	lea    0x140(%rsp),%rdi
   e1080:	call   e2640 <core::ptr::drop_glue::<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::corpus::SelectedMerge>>>
   e1085:	mov    0x198(%rsp),%rdi
   e108d:	call   *0x26815(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e1093:	mov    %r15,%rdi
   e1096:	call   1019a0 <_Unwind_Resume@plt>
   e109b:	call   71f13 <core::panicking::panic_in_cleanup>
   e10a0:	mov    %rax,%r15
   e10a3:	test   %r14,%r14
   e10a6:	je     e10ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x540f>
   e10a8:	mov    0x2c0(%rsp),%rbx
   e10b0:	mov    %rbx,%rdi
   e10b3:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e10b8:	add    $0x18,%rbx
   e10bc:	dec    %r14
   e10bf:	jne    e10b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x53d0>
   e10c1:	jmp    e10ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x540f>
   e10c3:	mov    %rax,%r15
   e10c6:	lea    0x160(%rsp),%rdi
   e10ce:	call   ea0f0 <core::ptr::drop_glue::<rayon::iter::collect::special_extend<rayon::iter::map::Map<rayon::vec::IntoIter<tk_train::trainers::bpe::corpus::merging::PreparedChunk>, <tk_train::trainers::bpe::corpus::Corpus>::merge::{closure#0}>, alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>::{closure#0}>>
   e10d3:	jmp    e10ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x540f>
   e10d5:	mov    %rax,%r15
   e10d8:	lea    0x30(%rsp),%rdi
   e10dd:	call   e92d0 <core::ptr::drop_glue::<rayon::vec::Drain<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e10e2:	lea    0x140(%rsp),%rdi
   e10ea:	call   e9250 <core::ptr::drop_glue::<rayon::vec::IntoIter<tk_train::trainers::bpe::corpus::merging::PreparedChunk>>>
   e10ef:	lea    0xb0(%rsp),%rdi
   e10f7:	call   ea170 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>>
   e10fc:	mov    0x198(%rsp),%rdi
   e1104:	call   *0x2679e(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e110a:	mov    %r15,%rdi
   e110d:	call   1019a0 <_Unwind_Resume@plt>
   e1112:	call   71f13 <core::panicking::panic_in_cleanup>
   e1117:	mov    %rax,%r15
   e111a:	jmp    e1174 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5494>
   e111c:	mov    %rax,%r15
   e111f:	cmpq   $0x0,0x1a8(%rsp)
   e1128:	je     e1138 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5458>
   e112a:	mov    0x1b0(%rsp),%rdi
   e1132:	call   *0x26770(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e1138:	test   %r14b,%r14b
   e113b:	je     e12ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x560f>
   e1141:	lea    0x140(%rsp),%rdi
   e1149:	call   e25c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   e114e:	mov    0x198(%rsp),%rdi
   e1156:	call   *0x2674c(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e115c:	mov    %r15,%rdi
   e115f:	call   1019a0 <_Unwind_Resume@plt>
   e1164:	mov    %rax,%r15
   e1167:	lea    0xe0(%rsp),%rdi
   e116f:	call   e25c0 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   e1174:	mov    0x3a0(%rsp),%rdi
   e117c:	mov    0x3a8(%rsp),%rsi
   e1184:	call   c8d00 <core::ptr::drop_glue::<ahash::hash_set::AHashSet<char>>>
   e1189:	mov    0x360(%rsp),%rdi
   e1191:	mov    0x368(%rsp),%rsi
   e1199:	call   c8d00 <core::ptr::drop_glue::<ahash::hash_set::AHashSet<char>>>
   e119e:	lea    0x2a8(%rsp),%rdi
   e11a6:	call   e47c0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::SelectedMerge>>>
   e11ab:	mov    %r15,%rdi
   e11ae:	call   1019a0 <_Unwind_Resume@plt>
   e11b3:	mov    %rax,%rbx
   e11b6:	cmpq   $0x0,0x1a8(%rsp)
   e11bf:	je     e11cf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x54ef>
   e11c1:	mov    0x1b0(%rsp),%rdi
   e11c9:	call   *0x266d9(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e11cf:	mov    0x1c8(%rsp),%rcx
   e11d7:	test   %rcx,%rcx
   e11da:	mov    %rbx,0x8(%rsp)
   e11df:	je     e120b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x552b>
   e11e1:	lea    (%rcx,%rcx,2),%rax
   e11e5:	lea    0x1b(,%rax,4),%rax
   e11ed:	and    $0xfffffffffffffff0,%rax
   e11f1:	add    %rax,%rcx
   e11f4:	cmp    $0xffffffffffffffef,%rcx
   e11f8:	je     e120b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x552b>
   e11fa:	mov    0x1c0(%rsp),%rdi
   e1202:	sub    %rax,%rdi
   e1205:	call   *0x2669d(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e120b:	xor    %r15d,%r15d
   e120e:	mov    0x26693(%rip),%r12        # 1078a8 <free@GLIBC_2.2.5>
   e1215:	jmp    e1233 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5553>
   e1217:	nopw   0x0(%rax,%rax,1)
   e1220:	mov    %rbx,%rdi
   e1223:	call   *%r12
   e1226:	inc    %r15
   e1229:	cmp    0xa8(%rsp),%r15
   e1231:	je     e1299 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55b9>
   e1233:	lea    (%r15,%r15,2),%rcx
   e1237:	shl    $0x4,%rcx
   e123b:	lea    (%r14,%rcx,1),%rax
   e123f:	cmpq   $0x0,(%r14,%rcx,1)
   e1244:	je     e1260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5580>
   e1246:	cmpq   $0x11,0x18(%rax)
   e124b:	jb     e1226 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5546>
   e124d:	mov    0x8(%rax),%rbx
   e1251:	jmp    e1220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5540>
   e1253:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e1260:	mov    0x10(%rax),%r13
   e1264:	test   %r13,%r13
   e1267:	je     e1226 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5546>
   e1269:	mov    0x8(%rax),%rbx
   e126d:	lea    0x18(%rbx),%rbp
   e1271:	jmp    e1289 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55a9>
   e1273:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e1280:	add    $0x20,%rbp
   e1284:	dec    %r13
   e1287:	je     e1220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5540>
   e1289:	cmpq   $0x11,0x0(%rbp)
   e128e:	jb     e1280 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55a0>
   e1290:	mov    -0x10(%rbp),%rdi
   e1294:	call   *%r12
   e1297:	jmp    e1280 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55a0>
   e1299:	cmpq   $0x0,0x290(%rsp)
   e12a2:	je     e12ad <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55cd>
   e12a4:	mov    %r14,%rdi
   e12a7:	call   *0x265fb(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e12ad:	mov    0x8(%rsp),%r15
   e12b2:	mov    0x198(%rsp),%rdi
   e12ba:	call   *0x265e8(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e12c0:	mov    %r15,%rdi
   e12c3:	call   1019a0 <_Unwind_Resume@plt>
   e12c8:	mov    %rax,%r15
   e12cb:	mov    %r13,%rbx
   e12ce:	xchg   %ax,%ax
   e12d0:	mov    %rbx,%rdi
   e12d3:	call   e46d0 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::corpus::PairChange<tk_train::trainers::bpe::positions::Positions>>>>
   e12d8:	add    $0x18,%rbx
   e12dc:	dec    %rbp
   e12df:	jne    e12d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x55f0>
   e12e1:	test   %r12,%r12
   e12e4:	je     e12ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x560f>
   e12e6:	mov    %r13,%rdi
   e12e9:	call   *0x265b9(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e12ef:	mov    0x198(%rsp),%rdi
   e12f7:	call   *0x265ab(%rip)        # 1078a8 <free@GLIBC_2.2.5>
   e12fd:	mov    %r15,%rdi
   e1300:	call   1019a0 <_Unwind_Resume@plt>
