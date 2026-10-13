
/tmp/bpe-corpus-binaries-20261013/final-baseline-production:     file format elf64-x86-64


Disassembly of section .text:

00000000000dbd30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next>:
   dbd30:	push   %rbp
   dbd31:	push   %r15
   dbd33:	push   %r14
   dbd35:	push   %r13
   dbd37:	push   %r12
   dbd39:	push   %rbx
   dbd3a:	sub    $0x3e8,%rsp
   dbd41:	mov    %rcx,0x20(%rsp)
   dbd46:	mov    %rdx,0x1c8(%rsp)
   dbd4e:	mov    %rsi,%rbp
   dbd51:	mov    %rdi,0x2c0(%rsp)
   dbd59:	mov    %r8,0x30(%rsp)
   dbd5e:	movzbl 0x30(%r8),%eax
   dbd63:	movq   $0x0,0x2e0(%rsp)
   dbd6f:	movq   $0x8,0x2e8(%rsp)
   dbd7b:	movq   $0x0,0x2f0(%rsp)
   dbd87:	mov    %al,0x2f8(%rsp)
   dbd8e:	mov    $0x1,%ecx
   dbd93:	mov    %rcx,0x130(%rsp)
   dbd9b:	mov    %al,0x38(%rsp)
   dbd9f:	test   %al,%al
   dbda1:	jne    dbdcd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9d>
   dbda3:	mov    0xc8(%rbp),%rax
   dbdaa:	mov    0x1c8(%rsp),%rcx
   dbdb2:	sub    0x30(%rcx),%rax
   dbdb6:	cmp    $0x100,%rax
   dbdbc:	mov    $0x100,%ecx
   dbdc1:	cmovb  %rax,%rcx
   dbdc5:	mov    %rcx,0x130(%rsp)
   dbdcd:	mov    0x2a4bc(%rip),%rax        # 106290 <ahash::random_state::RAND_SOURCE>
   dbdd4:	test   %rax,%rax
   dbdd7:	je     e0476 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4746>
   dbddd:	mov    (%rax),%r14
   dbde0:	mov    0x8(%rax),%r15
   dbde4:	mov    0x2a4ad(%rip),%rbx        # 106298 <ahash::random_state::get_fixed_seeds::SEEDS>
   dbdeb:	test   %rbx,%rbx
   dbdee:	je     e0480 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4750>
   dbdf4:	mov    %r14,%rdi
   dbdf7:	call   *0x18(%r15)
   dbdfb:	xor    0x8(%rbx),%rax
   dbdff:	movabs $0x5851f42d4c957f2d,%r15
   dbe09:	mul    %r15
   dbe0c:	mov    %rax,%rdi
   dbe0f:	xor    %rdx,%rdi
   dbe12:	mov    (%rbx),%r10
   dbe15:	mov    0x20(%rbx),%r11
   dbe19:	mov    0x30(%rbx),%r9
   dbe1d:	mov    %r11,%rax
   dbe20:	xor    %rdi,%rax
   dbe23:	mul    %r15
   dbe26:	xor    %r9,%rdx
   dbe29:	xor    %rdx,%rax
   dbe2c:	mul    %r15
   dbe2f:	mov    %rax,%rcx
   dbe32:	xor    %rdx,%rcx
   dbe35:	mov    %rcx,%rax
   dbe38:	mul    %r10
   dbe3b:	mov    %rdx,%rsi
   dbe3e:	xor    %rax,%rsi
   dbe41:	rol    %cl,%rsi
   dbe44:	mov    0x28(%rbx),%r14
   dbe48:	mov    0x38(%rbx),%rbx
   dbe4c:	mov    %r14,%rax
   dbe4f:	xor    %rdi,%rax
   dbe52:	mul    %r15
   dbe55:	xor    %rbx,%rdx
   dbe58:	xor    %rdx,%rax
   dbe5b:	mul    %r15
   dbe5e:	mov    %rax,%rcx
   dbe61:	xor    %rdx,%rcx
   dbe64:	mov    %rcx,%rax
   dbe67:	mul    %r10
   dbe6a:	mov    %rdx,%r8
   dbe6d:	xor    %rax,%r8
   dbe70:	rol    %cl,%r8
   dbe73:	xor    %rdi,%r9
   dbe76:	mov    %r9,%rax
   dbe79:	mul    %r15
   dbe7c:	xor    %rdx,%r14
   dbe7f:	xor    %r14,%rax
   dbe82:	mul    %r15
   dbe85:	mov    %rax,%rcx
   dbe88:	xor    %rdx,%rcx
   dbe8b:	mov    %rcx,%rax
   dbe8e:	mul    %r10
   dbe91:	mov    %rdx,%r9
   dbe94:	xor    %rax,%r9
   dbe97:	rol    %cl,%r9
   dbe9a:	xor    %rbx,%rdi
   dbe9d:	mov    %rdi,%rax
   dbea0:	mul    %r15
   dbea3:	xor    %rdx,%r11
   dbea6:	xor    %r11,%rax
   dbea9:	mul    %r15
   dbeac:	mov    %rax,%rcx
   dbeaf:	xor    %rdx,%rcx
   dbeb2:	mov    %rcx,%rax
   dbeb5:	mul    %r10
   dbeb8:	xor    %rax,%rdx
   dbebb:	rol    %cl,%rdx
   dbebe:	movaps 0x26fcb(%rip),%xmm0        # 102e90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   dbec5:	movaps %xmm0,0x350(%rsp)
   dbecd:	movdqa 0x26fab(%rip),%xmm0        # 102e80 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   dbed5:	movdqa %xmm0,0x340(%rsp)
   dbede:	mov    %rsi,0x360(%rsp)
   dbee6:	mov    %r8,0x368(%rsp)
   dbeee:	mov    %r9,0x370(%rsp)
   dbef6:	mov    %rdx,0x378(%rsp)
   dbefe:	mov    0x2a38b(%rip),%rax        # 106290 <ahash::random_state::RAND_SOURCE>
   dbf05:	test   %rax,%rax
   dbf08:	je     e048d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x475d>
   dbf0e:	mov    (%rax),%r14
   dbf11:	mov    0x8(%rax),%r15
   dbf15:	mov    0x2a37c(%rip),%rbx        # 106298 <ahash::random_state::get_fixed_seeds::SEEDS>
   dbf1c:	test   %rbx,%rbx
   dbf1f:	je     e0497 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4767>
   dbf25:	mov    %r14,%rdi
   dbf28:	call   *0x18(%r15)
   dbf2c:	xor    0x8(%rbx),%rax
   dbf30:	movabs $0x5851f42d4c957f2d,%r15
   dbf3a:	mul    %r15
   dbf3d:	mov    %rax,%rdi
   dbf40:	xor    %rdx,%rdi
   dbf43:	mov    (%rbx),%r10
   dbf46:	mov    0x20(%rbx),%r11
   dbf4a:	mov    0x30(%rbx),%r9
   dbf4e:	mov    %r11,%rax
   dbf51:	xor    %rdi,%rax
   dbf54:	mul    %r15
   dbf57:	xor    %r9,%rdx
   dbf5a:	xor    %rdx,%rax
   dbf5d:	mul    %r15
   dbf60:	mov    %rax,%rcx
   dbf63:	xor    %rdx,%rcx
   dbf66:	mov    %rcx,%rax
   dbf69:	mul    %r10
   dbf6c:	mov    %rdx,%rsi
   dbf6f:	xor    %rax,%rsi
   dbf72:	rol    %cl,%rsi
   dbf75:	mov    0x28(%rbx),%r14
   dbf79:	mov    0x38(%rbx),%rbx
   dbf7d:	mov    %r14,%rax
   dbf80:	xor    %rdi,%rax
   dbf83:	mul    %r15
   dbf86:	xor    %rbx,%rdx
   dbf89:	xor    %rdx,%rax
   dbf8c:	mul    %r15
   dbf8f:	mov    %rax,%rcx
   dbf92:	xor    %rdx,%rcx
   dbf95:	mov    %rcx,%rax
   dbf98:	mul    %r10
   dbf9b:	mov    %rdx,%r8
   dbf9e:	xor    %rax,%r8
   dbfa1:	rol    %cl,%r8
   dbfa4:	xor    %rdi,%r9
   dbfa7:	mov    %r9,%rax
   dbfaa:	mul    %r15
   dbfad:	xor    %rdx,%r14
   dbfb0:	xor    %r14,%rax
   dbfb3:	mul    %r15
   dbfb6:	mov    %rax,%rcx
   dbfb9:	xor    %rdx,%rcx
   dbfbc:	mov    %rcx,%rax
   dbfbf:	mul    %r10
   dbfc2:	mov    %rdx,%r9
   dbfc5:	xor    %rax,%r9
   dbfc8:	rol    %cl,%r9
   dbfcb:	xor    %rbx,%rdi
   dbfce:	mov    %rdi,%rax
   dbfd1:	mul    %r15
   dbfd4:	xor    %rdx,%r11
   dbfd7:	xor    %r11,%rax
   dbfda:	mul    %r15
   dbfdd:	mov    %rax,%rcx
   dbfe0:	xor    %rdx,%rcx
   dbfe3:	mov    %rcx,%rax
   dbfe6:	mul    %r10
   dbfe9:	xor    %rax,%rdx
   dbfec:	rol    %cl,%rdx
   dbfef:	movaps 0x26e9a(%rip),%xmm0        # 102e90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   dbff6:	movaps %xmm0,0x390(%rsp)
   dbffe:	movdqa 0x26e7a(%rip),%xmm0        # 102e80 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   dc006:	movdqa %xmm0,0x380(%rsp)
   dc00f:	mov    %rsi,0x3a0(%rsp)
   dc017:	mov    %r8,0x3a8(%rsp)
   dc01f:	mov    %r9,0x3b0(%rsp)
   dc027:	mov    %rdx,0x3b8(%rsp)
   dc02f:	mov    0x2f0(%rsp),%r14
   dc037:	cmp    0x130(%rsp),%r14
   dc03f:	jae    dd031 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1301>
   dc045:	movabs $0x7ffffffffffffff1,%rcx
   dc04f:	mov    0x20(%rsp),%rax
   dc054:	add    $0x30,%rax
   dc058:	mov    %rax,0x198(%rsp)
   dc060:	lea    0x27349(%rip),%rax        # 1033b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x38b8>
   dc067:	mov    %rax,0x120(%rsp)
   dc06f:	lea    0xb0(%rsp),%rax
   dc077:	mov    %rax,0x330(%rsp)
   dc07f:	add    $0x7,%rcx
   dc083:	mov    %rcx,0x190(%rsp)
   dc08b:	lea    0x27066(%rip),%rax        # 1030f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3600>
   dc092:	mov    %rax,0x128(%rsp)
   dc09a:	lea    0x27687(%rip),%rax        # 103728 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c30>
   dc0a1:	mov    %rax,0x1c0(%rsp)
   dc0a9:	lea    0x27798(%rip),%rax        # 103848 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d50>
   dc0b0:	mov    %rax,0x188(%rsp)
   dc0b8:	lea    -0xbe2d6(%rip),%rax        # 1dde9 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x4cc>
   dc0bf:	mov    %rax,0x180(%rsp)
   dc0c7:	mov    %rbp,0x58(%rsp)
   dc0cc:	mov    %r14,0x298(%rsp)
   dc0d4:	mov    0x30(%rsp),%rax
   dc0d9:	mov    0x10(%rax),%r15
   dc0dd:	test   %r15,%r15
   dc0e0:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc0e6:	mov    0x20(%rax),%rdi
   dc0ea:	test   %rdi,%rdi
   dc0ed:	je     e07f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ac0>
   dc0f3:	mov    0x30(%rsp),%rax
   dc0f8:	mov    0x8(%rax),%r14
   dc0fc:	jmp    dc125 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f5>
   dc0fe:	xchg   %ax,%ax
   dc100:	mov    %r12,%rdi
   dc103:	call   *%r15
   dc106:	mov    0x30(%rsp),%rax
   dc10b:	mov    0x10(%rax),%r15
   dc10f:	test   %r15,%r15
   dc112:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc118:	mov    0x20(%rax),%rdi
   dc11c:	test   %rdi,%rdi
   dc11f:	je     e07f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ac0>
   dc125:	mov    0x28(%r14),%r13d
   dc129:	mov    0x2c(%r14),%ebx
   dc12d:	mov    %r13,%rax
   dc130:	movabs $0xbe5466cf34e90c6c,%rcx
   dc13a:	xor    %rcx,%rax
   dc13d:	movabs $0x5851f42d4c957f2d,%rcx
   dc147:	mul    %rcx
   dc14a:	xor    %rbx,%rdx
   dc14d:	xor    %rdx,%rax
   dc150:	mul    %rcx
   dc153:	mov    %rax,%rcx
   dc156:	xor    %rdx,%rcx
   dc159:	mov    %rcx,%rax
   dc15c:	movabs $0x452821e638d01377,%rdx
   dc166:	mul    %rdx
   dc169:	mov    %rdx,%rbp
   dc16c:	xor    %rax,%rbp
   dc16f:	rol    %cl,%rbp
   dc172:	mov    0x20(%r14),%rsi
   dc176:	mov    %rbp,%rax
   dc179:	or     %rdi,%rax
   dc17c:	shr    $0x20,%rax
   dc180:	je     dc190 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x460>
   dc182:	mov    %rbp,%rax
   dc185:	xor    %edx,%edx
   dc187:	div    %rdi
   dc18a:	jmp    dc196 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x466>
   dc18c:	nopl   0x0(%rax)
   dc190:	mov    %ebp,%eax
   dc192:	xor    %edx,%edx
   dc194:	div    %edi
   dc196:	mov    0x30(%rsp),%rax
   dc19b:	mov    0x18(%rax),%rdi
   dc19f:	imul   $0x70,%rdx,%rax
   dc1a3:	cmpq   $0x0,0x48(%rdi,%rax,1)
   dc1a9:	je     dc280 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x550>
   dc1af:	add    %rax,%rdi
   dc1b2:	mov    0x58(%rdi),%rax
   dc1b6:	xor    %r13,%rax
   dc1b9:	movabs $0x5851f42d4c957f2d,%rcx
   dc1c3:	mul    %rcx
   dc1c6:	xor    %rbx,%rdx
   dc1c9:	xor    %rdx,%rax
   dc1cc:	mul    %rcx
   dc1cf:	mov    %rax,%rcx
   dc1d2:	xor    %rdx,%rcx
   dc1d5:	mov    %rcx,%rax
   dc1d8:	mulq   0x50(%rdi)
   dc1dc:	xor    %rax,%rdx
   dc1df:	rol    %cl,%rdx
   dc1e2:	mov    %rdx,%r8
   dc1e5:	shr    $0x39,%r8
   dc1e9:	mov    0x30(%rdi),%rax
   dc1ed:	mov    0x38(%rdi),%rcx
   dc1f1:	movd   %r8d,%xmm0
   dc1f6:	punpcklbw %xmm0,%xmm0
   dc1fa:	pshuflw $0x0,%xmm0,%xmm0
   dc1ff:	pshufd $0x44,%xmm0,%xmm0
   dc204:	xor    %edi,%edi
   dc206:	and    %rcx,%rdx
   dc209:	movdqu (%rax,%rdx,1),%xmm1
   dc20e:	movdqa %xmm1,%xmm2
   dc212:	pcmpeqb %xmm0,%xmm2
   dc216:	pmovmskb %xmm2,%r8d
   dc21b:	test   %r8d,%r8d
   dc21e:	je     dc260 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x530>
   dc220:	tzcnt  %r8d,%r10d
   dc225:	add    %rdx,%r10
   dc228:	and    %rcx,%r10
   dc22b:	shl    $0x4,%r10
   dc22f:	mov    %rax,%r9
   dc232:	sub    %r10,%r9
   dc235:	cmp    -0x10(%r9),%r13d
   dc239:	jne    dc245 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x515>
   dc23b:	cmp    -0xc(%r9),%ebx
   dc23f:	je     dc430 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x700>
   dc245:	lea    -0x1(%r8),%r9d
   dc249:	and    %r8w,%r9w
   dc24d:	mov    %r9d,%r8d
   dc250:	jne    dc220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4f0>
   dc252:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dc260:	pcmpeqd %xmm2,%xmm2
   dc264:	pcmpeqb %xmm2,%xmm1
   dc268:	pmovmskb %xmm1,%r8d
   dc26d:	test   %r8d,%r8d
   dc270:	jne    dc280 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x550>
   dc272:	add    %rdi,%rdx
   dc275:	add    $0x10,%rdx
   dc279:	add    $0x10,%rdi
   dc27d:	jmp    dc206 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d6>
   dc27f:	nop
   dc280:	lea    -0x1(%r15),%rbx
   dc284:	mov    0x30(%rsp),%rax
   dc289:	mov    %rbx,0x10(%rax)
   dc28d:	lea    (%rbx,%rbx,2),%rsi
   dc291:	shl    $0x4,%rsi
   dc295:	mov    (%r14,%rsi,1),%rdi
   dc299:	mov    0x8(%r14,%rsi,1),%r12
   dc29e:	mov    0x10(%r14,%rsi,1),%rcx
   dc2a3:	mov    0x18(%r14,%rsi,1),%rax
   dc2a8:	test   %rbx,%rbx
   dc2ab:	je     dc5d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8a0>
   dc2b1:	add    %r14,%rsi
   dc2b4:	mov    (%r14),%r8
   dc2b7:	mov    %r8,0x18(%rsp)
   dc2bc:	mov    0x8(%r14),%r8
   dc2c0:	mov    %r8,0x8(%rsp)
   dc2c5:	mov    0x10(%r14),%r13
   dc2c9:	mov    0x18(%r14),%r8
   dc2cd:	mov    %r8,0x40(%rsp)
   dc2d2:	movups 0x20(%rsi),%xmm0
   dc2d6:	mov    %rdi,(%r14)
   dc2d9:	mov    %r12,0x8(%r14)
   dc2dd:	mov    %rcx,0x10(%r14)
   dc2e1:	mov    %rax,0x18(%r14)
   dc2e5:	movups %xmm0,0x20(%r14)
   dc2ea:	movdqu (%r14),%xmm0
   dc2ef:	movdqu 0x10(%r14),%xmm1
   dc2f5:	movdqu 0x20(%r14),%xmm2
   dc2fb:	movdqa %xmm2,0xd0(%rsp)
   dc304:	movdqa %xmm1,0xc0(%rsp)
   dc30d:	mov    %rbx,%r12
   dc310:	sub    $0x8,%r12
   dc314:	mov    $0x0,%eax
   dc319:	cmovb  %rax,%r12
   dc31d:	movdqa %xmm0,0xb0(%rsp)
   dc326:	mov    $0x1,%edx
   dc32b:	mov    $0x0,%eax
   dc330:	cmp    $0xa,%r15
   dc334:	jb     dc3a1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x671>
   dc336:	xor    %eax,%eax
   dc338:	nopl   0x0(%rax,%rax,1)
   dc340:	mov    %rax,%rbp
   dc343:	mov    %r14,%rdi
   dc346:	mov    %rbx,%rsi
   dc349:	call   db190 <<dary_heap::Hole<tk_train::trainers::bpe::index::Candidate>>::max_sibling::<8>>
   dc34e:	lea    (%rax,%rax,2),%rcx
   dc352:	shl    $0x4,%rcx
   dc356:	lea    0x0(,%rbp,2),%rdx
   dc35e:	add    %rbp,%rdx
   dc361:	shl    $0x4,%rdx
   dc365:	movdqu (%r14,%rcx,1),%xmm0
   dc36b:	movdqu 0x10(%r14,%rcx,1),%xmm1
   dc372:	movdqu 0x20(%r14,%rcx,1),%xmm2
   dc379:	movdqu %xmm0,(%r14,%rdx,1)
   dc37f:	movdqu %xmm1,0x10(%r14,%rdx,1)
   dc386:	movdqu %xmm2,0x20(%r14,%rdx,1)
   dc38d:	lea    0x1(,%rax,8),%rdx
   dc395:	mov    %rax,%rcx
   dc398:	shl    $0x3,%rcx
   dc39c:	cmp    %r12,%rcx
   dc39f:	jb     dc340 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x610>
   dc3a1:	lea    0x1(%rdx),%rsi
   dc3a5:	cmp    %rbx,%rsi
   dc3a8:	jae    dc61c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8ec>
   dc3ae:	lea    (%rdx,%rdx,2),%rcx
   dc3b2:	shl    $0x4,%rcx
   dc3b6:	lea    (%r14,%rcx,1),%rdi
   dc3ba:	add    $0x5c,%rdi
   dc3be:	mov    $0x2,%r8d
   dc3c4:	sub    %r15,%r8
   dc3c7:	mov    0x8(%rsp),%r12
   dc3cc:	jmp    dc3eb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6bb>
   dc3ce:	xchg   %ax,%ax
   dc3d0:	add    $0x30,%rdi
   dc3d4:	lea    (%r8,%rsi,1),%r9
   dc3d8:	inc    %r9
   dc3db:	inc    %rsi
   dc3de:	mov    %rcx,%rdx
   dc3e1:	cmp    $0x1,%r9
   dc3e5:	je     dc624 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8f4>
   dc3eb:	lea    (%rdx,%rdx,2),%rcx
   dc3ef:	shl    $0x4,%rcx
   dc3f3:	mov    0x20(%r14,%rcx,1),%r9
   dc3f8:	cmp    -0xc(%rdi),%r9
   dc3fc:	setbe  %r9b
   dc400:	jne    dc41e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6ee>
   dc402:	add    %r14,%rcx
   dc405:	mov    -0x4(%rdi),%r9d
   dc409:	cmp    0x28(%rcx),%r9d
   dc40d:	setbe  %r9b
   dc411:	jne    dc41e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6ee>
   dc413:	mov    (%rdi),%r9d
   dc416:	cmp    0x2c(%rcx),%r9d
   dc41a:	setbe  %r9b
   dc41e:	mov    %rsi,%rcx
   dc421:	test   %r9b,%r9b
   dc424:	jne    dc3d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6a0>
   dc426:	mov    %rdx,%rcx
   dc429:	jmp    dc3d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x6a0>
   dc42b:	nopl   0x0(%rax,%rax,1)
   dc430:	mov    -0x8(%r9),%r14
   dc434:	cmp    %r14,%rsi
   dc437:	je     dc7a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xa70>
   dc43d:	lea    0xb0(%rsp),%rdi
   dc445:	mov    0x30(%rsp),%r12
   dc44a:	mov    %r12,%rsi
   dc44d:	call   e1100 <<dary_heap::DaryHeap<tk_train::trainers::bpe::index::Candidate, 8>>::pop>
   dc452:	cmpb   $0x0,0xb0(%rsp)
   dc45a:	je     e03fa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46ca>
   dc460:	lea    0xb8(%rsp),%rax
   dc468:	movdqu (%rax),%xmm0
   dc46c:	movdqu 0x10(%rax),%xmm1
   dc471:	movdqu 0x20(%rax),%xmm2
   dc476:	movdqa %xmm2,0x1f0(%rsp)
   dc47f:	movdqa %xmm1,0x1e0(%rsp)
   dc488:	movdqa %xmm0,0x1d0(%rsp)
   dc491:	mov    %r14,0x1f0(%rsp)
   dc499:	mov    0x10(%r12),%rbx
   dc49e:	cmp    (%r12),%rbx
   dc4a2:	jne    dc4ac <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x77c>
   dc4a4:	mov    %r12,%rdi
   dc4a7:	call   e1480 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   dc4ac:	mov    0x8(%r12),%r14
   dc4b1:	lea    (%rbx,%rbx,2),%rdx
   dc4b5:	shl    $0x4,%rdx
   dc4b9:	movdqa 0x1d0(%rsp),%xmm0
   dc4c2:	movdqa 0x1e0(%rsp),%xmm1
   dc4cb:	movdqa 0x1f0(%rsp),%xmm2
   dc4d4:	movdqu %xmm2,0x20(%r14,%rdx,1)
   dc4db:	movdqu %xmm1,0x10(%r14,%rdx,1)
   dc4e2:	movdqu %xmm0,(%r14,%rdx,1)
   dc4e8:	lea    0x1(%rbx),%r15
   dc4ec:	mov    %r15,0x10(%r12)
   dc4f1:	mov    0x20(%r14,%rdx,1),%rax
   dc4f6:	mov    0x28(%r14,%rdx,1),%ecx
   dc4fb:	mov    0x2c(%r14,%rdx,1),%edx
   dc500:	test   %rbx,%rbx
   dc503:	jne    dc59b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x86b>
   dc509:	xor    %ebx,%ebx
   dc50b:	lea    (%rbx,%rbx,2),%rsi
   dc50f:	shl    $0x4,%rsi
   dc513:	movdqa 0x1d0(%rsp),%xmm0
   dc51c:	movdqa 0x1e0(%rsp),%xmm1
   dc525:	movdqu %xmm1,0x10(%r14,%rsi,1)
   dc52c:	movdqu %xmm0,(%r14,%rsi,1)
   dc532:	mov    %rax,0x20(%r14,%rsi,1)
   dc537:	mov    %ecx,0x28(%r14,%rsi,1)
   dc53c:	mov    %edx,0x2c(%r14,%rsi,1)
   dc541:	mov    0x30(%rsp),%rax
   dc546:	mov    0x20(%rax),%rdi
   dc54a:	test   %rdi,%rdi
   dc54d:	jne    dc125 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f5>
   dc553:	jmp    e07f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ac0>
   dc558:	nopl   0x0(%rax,%rax,1)
   dc560:	test   %r8b,%r8b
   dc563:	jne    dc50b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7db>
   dc565:	lea    (%rbx,%rbx,2),%r8
   dc569:	shl    $0x4,%r8
   dc56d:	movdqu (%rdi),%xmm0
   dc571:	movdqu 0x10(%rdi),%xmm1
   dc576:	movdqu 0x20(%rdi),%xmm2
   dc57b:	movdqu %xmm2,0x20(%r14,%r8,1)
   dc582:	movdqu %xmm1,0x10(%r14,%r8,1)
   dc589:	movdqu %xmm0,(%r14,%r8,1)
   dc58f:	mov    %rsi,%rbx
   dc592:	test   %rsi,%rsi
   dc595:	je     dc509 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7d9>
   dc59b:	lea    -0x1(%rbx),%rsi
   dc59f:	shr    $0x3,%rsi
   dc5a3:	lea    (%rsi,%rsi,2),%r8
   dc5a7:	shl    $0x4,%r8
   dc5ab:	lea    (%r14,%r8,1),%rdi
   dc5af:	cmp    0x20(%r14,%r8,1),%rax
   dc5b4:	setbe  %r8b
   dc5b8:	jne    dc560 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x830>
   dc5ba:	cmp    %ecx,0x28(%rdi)
   dc5bd:	setbe  %r8b
   dc5c1:	jne    dc560 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x830>
   dc5c3:	cmp    %edx,0x2c(%rdi)
   dc5c6:	ja     dc565 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x835>
   dc5c8:	jmp    dc50b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x7db>
   dc5cd:	nopl   (%rax)
   dc5d0:	mov    %rdi,%r10
   dc5d3:	mov    %rcx,%r13
   dc5d6:	mov    %rax,%r11
   dc5d9:	mov    0x27910(%rip),%r15        # 103ef0 <free@GLIBC_2.2.5>
   dc5e0:	test   %r10,%r10
   dc5e3:	jne    dc707 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9d7>
   dc5e9:	test   %r13,%r13
   dc5ec:	je     dc106 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d6>
   dc5f2:	lea    0x18(%r12),%rbx
   dc5f7:	jmp    dc60d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8dd>
   dc5f9:	nopl   0x0(%rax)
   dc600:	add    $0x20,%rbx
   dc604:	dec    %r13
   dc607:	je     dc100 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d0>
   dc60d:	cmpq   $0x11,(%rbx)
   dc611:	jb     dc600 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8d0>
   dc613:	mov    -0x10(%rbx),%rdi
   dc617:	call   *%r15
   dc61a:	jmp    dc600 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8d0>
   dc61c:	mov    %rdx,%rcx
   dc61f:	mov    0x8(%rsp),%r12
   dc624:	cmp    %rbx,%rcx
   dc627:	jae    dc663 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x933>
   dc629:	lea    (%rcx,%rcx,2),%rdx
   dc62d:	shl    $0x4,%rdx
   dc631:	lea    (%rax,%rax,2),%rax
   dc635:	shl    $0x4,%rax
   dc639:	movdqu (%r14,%rdx,1),%xmm0
   dc63f:	movdqu 0x10(%r14,%rdx,1),%xmm1
   dc646:	movdqu 0x20(%r14,%rdx,1),%xmm2
   dc64d:	movdqu %xmm2,0x20(%r14,%rax,1)
   dc654:	movdqu %xmm1,0x10(%r14,%rax,1)
   dc65b:	movdqu %xmm0,(%r14,%rax,1)
   dc661:	jmp    dc666 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x936>
   dc663:	mov    %rax,%rcx
   dc666:	mov    0x27883(%rip),%r15        # 103ef0 <free@GLIBC_2.2.5>
   dc66d:	mov    0x18(%rsp),%r10
   dc672:	mov    0x40(%rsp),%r11
   dc677:	lea    (%rcx,%rcx,2),%rsi
   dc67b:	shl    $0x4,%rsi
   dc67f:	movdqa 0xb0(%rsp),%xmm0
   dc688:	movdqa 0xc0(%rsp),%xmm1
   dc691:	movdqa 0xd0(%rsp),%xmm2
   dc69a:	movdqu %xmm2,0x20(%r14,%rsi,1)
   dc6a1:	movdqu %xmm1,0x10(%r14,%rsi,1)
   dc6a8:	movdqu %xmm0,(%r14,%rsi,1)
   dc6ae:	mov    0x20(%r14,%rsi,1),%rax
   dc6b3:	mov    0x28(%r14,%rsi,1),%edx
   dc6b8:	mov    0x2c(%r14,%rsi,1),%esi
   dc6bd:	test   %rcx,%rcx
   dc6c0:	jne    dc75e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xa2e>
   dc6c6:	xor    %ecx,%ecx
   dc6c8:	lea    (%rcx,%rcx,2),%rcx
   dc6cc:	shl    $0x4,%rcx
   dc6d0:	movdqa 0xb0(%rsp),%xmm0
   dc6d9:	movdqa 0xc0(%rsp),%xmm1
   dc6e2:	movdqu %xmm1,0x10(%r14,%rcx,1)
   dc6e9:	movdqu %xmm0,(%r14,%rcx,1)
   dc6ef:	mov    %rax,0x20(%r14,%rcx,1)
   dc6f4:	mov    %edx,0x28(%r14,%rcx,1)
   dc6f9:	mov    %esi,0x2c(%r14,%rcx,1)
   dc6fe:	test   %r10,%r10
   dc701:	je     dc5e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x8b9>
   dc707:	cmp    $0x11,%r11
   dc70b:	jae    dc100 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d0>
   dc711:	jmp    dc106 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3d6>
   dc716:	cs nopw 0x0(%rax,%rax,1)
   dc720:	test   %r9b,%r9b
   dc723:	jne    dc6c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x998>
   dc725:	lea    (%rcx,%rcx,2),%rcx
   dc729:	shl    $0x4,%rcx
   dc72d:	movdqu (%r8),%xmm0
   dc732:	movdqu 0x10(%r8),%xmm1
   dc738:	movdqu 0x20(%r8),%xmm2
   dc73e:	movdqu %xmm2,0x20(%r14,%rcx,1)
   dc745:	movdqu %xmm1,0x10(%r14,%rcx,1)
   dc74c:	movdqu %xmm0,(%r14,%rcx,1)
   dc752:	mov    %rdi,%rcx
   dc755:	test   %rdi,%rdi
   dc758:	je     dc6c6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x996>
   dc75e:	lea    -0x1(%rcx),%rdi
   dc762:	shr    $0x3,%rdi
   dc766:	lea    (%rdi,%rdi,2),%r9
   dc76a:	shl    $0x4,%r9
   dc76e:	lea    (%r14,%r9,1),%r8
   dc772:	cmp    0x20(%r14,%r9,1),%rax
   dc777:	setbe  %r9b
   dc77b:	jne    dc720 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9f0>
   dc77d:	cmp    %edx,0x28(%r8)
   dc781:	setbe  %r9b
   dc785:	jne    dc720 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9f0>
   dc787:	cmp    %esi,0x2c(%r8)
   dc78b:	ja     dc725 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x9f5>
   dc78d:	jmp    dc6c8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x998>
   dc792:	data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dc7a0:	mov    0x30(%rsp),%rax
   dc7a5:	cmp    0x28(%rax),%rsi
   dc7a9:	jb     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc7af:	cmpq   $0x0,0x298(%rsp)
   dc7b8:	je     dc95b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xc2b>
   dc7be:	cmp    %ebx,%r13d
   dc7c1:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc7c7:	cmpq   $0x0,0x398(%rsp)
   dc7d0:	je     dc891 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb61>
   dc7d6:	mov    0x3a8(%rsp),%rax
   dc7de:	xor    %r13,%rax
   dc7e1:	movabs $0x5851f42d4c957f2d,%rcx
   dc7eb:	mul    %rcx
   dc7ee:	mov    %rax,%rcx
   dc7f1:	xor    %rdx,%rcx
   dc7f4:	mov    %rcx,%rax
   dc7f7:	mulq   0x3a0(%rsp)
   dc7ff:	xor    %rax,%rdx
   dc802:	rol    %cl,%rdx
   dc805:	mov    %rdx,%rsi
   dc808:	shr    $0x39,%rsi
   dc80c:	mov    0x380(%rsp),%rax
   dc814:	mov    0x388(%rsp),%rcx
   dc81c:	movd   %esi,%xmm0
   dc820:	punpcklbw %xmm0,%xmm0
   dc824:	pshuflw $0x0,%xmm0,%xmm0
   dc829:	pshufd $0x44,%xmm0,%xmm0
   dc82e:	xor    %esi,%esi
   dc830:	and    %rcx,%rdx
   dc833:	movdqu (%rax,%rdx,1),%xmm1
   dc838:	movdqa %xmm1,%xmm2
   dc83c:	pcmpeqb %xmm0,%xmm2
   dc840:	pmovmskb %xmm2,%edi
   dc844:	test   %edi,%edi
   dc846:	je     dc874 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb44>
   dc848:	tzcnt  %edi,%r8d
   dc84d:	add    %rdx,%r8
   dc850:	and    %rcx,%r8
   dc853:	shl    $0x2,%r8
   dc857:	mov    %rax,%r9
   dc85a:	sub    %r8,%r9
   dc85d:	cmp    -0x4(%r9),%r13d
   dc861:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc867:	lea    -0x1(%rdi),%r8d
   dc86b:	and    %di,%r8w
   dc86f:	mov    %r8d,%edi
   dc872:	jne    dc848 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb18>
   dc874:	pcmpeqd %xmm2,%xmm2
   dc878:	pcmpeqb %xmm2,%xmm1
   dc87c:	pmovmskb %xmm1,%edi
   dc880:	test   %edi,%edi
   dc882:	jne    dc891 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb61>
   dc884:	add    %rsi,%rdx
   dc887:	add    $0x10,%rdx
   dc88b:	add    $0x10,%rsi
   dc88f:	jmp    dc830 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xb00>
   dc891:	cmpq   $0x0,0x358(%rsp)
   dc89a:	je     dc95b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xc2b>
   dc8a0:	mov    0x368(%rsp),%rax
   dc8a8:	xor    %rbx,%rax
   dc8ab:	movabs $0x5851f42d4c957f2d,%rcx
   dc8b5:	mul    %rcx
   dc8b8:	mov    %rax,%rcx
   dc8bb:	xor    %rdx,%rcx
   dc8be:	mov    %rcx,%rax
   dc8c1:	mulq   0x360(%rsp)
   dc8c9:	xor    %rax,%rdx
   dc8cc:	rol    %cl,%rdx
   dc8cf:	mov    %rdx,%rsi
   dc8d2:	shr    $0x39,%rsi
   dc8d6:	mov    0x340(%rsp),%rax
   dc8de:	mov    0x348(%rsp),%rcx
   dc8e6:	movd   %esi,%xmm0
   dc8ea:	punpcklbw %xmm0,%xmm0
   dc8ee:	pshuflw $0x0,%xmm0,%xmm0
   dc8f3:	pshufd $0x44,%xmm0,%xmm0
   dc8f8:	xor    %esi,%esi
   dc8fa:	and    %rcx,%rdx
   dc8fd:	movdqu (%rax,%rdx,1),%xmm1
   dc902:	movdqa %xmm1,%xmm2
   dc906:	pcmpeqb %xmm0,%xmm2
   dc90a:	pmovmskb %xmm2,%edi
   dc90e:	test   %edi,%edi
   dc910:	je     dc93e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xc0e>
   dc912:	tzcnt  %edi,%r8d
   dc917:	add    %rdx,%r8
   dc91a:	and    %rcx,%r8
   dc91d:	shl    $0x2,%r8
   dc921:	mov    %rax,%r9
   dc924:	sub    %r8,%r9
   dc927:	cmp    -0x4(%r9),%ebx
   dc92b:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dc931:	lea    -0x1(%rdi),%r8d
   dc935:	and    %di,%r8w
   dc939:	mov    %r8d,%edi
   dc93c:	jne    dc912 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbe2>
   dc93e:	pcmpeqd %xmm2,%xmm2
   dc942:	pcmpeqb %xmm2,%xmm1
   dc946:	pmovmskb %xmm1,%edi
   dc94a:	test   %edi,%edi
   dc94c:	jne    dc95b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xc2b>
   dc94e:	add    %rsi,%rdx
   dc951:	add    $0x10,%rdx
   dc955:	add    $0x10,%rsi
   dc959:	jmp    dc8fa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xbca>
   dc95b:	mov    %r13,0x90(%rsp)
   dc963:	mov    0x1c8(%rsp),%rax
   dc96b:	mov    0x30(%rax),%rcx
   dc96f:	mov    %rcx,0x60(%rsp)
   dc974:	cmp    %r13,%rcx
   dc977:	jbe    e06a8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4978>
   dc97d:	mov    0x10(%rax),%rsi
   dc981:	cmp    %r13,%rsi
   dc984:	jbe    e0964 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c34>
   dc98a:	mov    0x8(%rax),%rax
   dc98e:	mov    %r13,%rdx
   dc991:	shl    $0x5,%rdx
   dc995:	lea    (%rax,%rdx,1),%r10
   dc999:	add    $0x8,%r10
   dc99d:	movzbl 0x1f(%rax,%rdx,1),%edi
   dc9a2:	lea    0x40(%rdi),%r8d
   dc9a6:	movzbl %r8b,%r8d
   dc9aa:	cmp    $0x18,%r8b
   dc9ae:	mov    $0x18,%r9d
   dc9b4:	cmovae %r9d,%r8d
   dc9b8:	cmp    $0xd8,%dil
   dc9bc:	cmovae 0x8(%rax,%rdx,1),%r10
   dc9c2:	movzbl %r8b,%r12d
   dc9c6:	cmovae 0x10(%rax,%rdx,1),%r12
   dc9cc:	mov    %rbx,0x90(%rsp)
   dc9d4:	mov    %rcx,0x60(%rsp)
   dc9d9:	cmp    %rbx,%rcx
   dc9dc:	jbe    e06be <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x498e>
   dc9e2:	cmp    %rbx,%rsi
   dc9e5:	jbe    e0952 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c22>
   dc9eb:	mov    %rbx,%rcx
   dc9ee:	shl    $0x5,%rcx
   dc9f2:	lea    (%rax,%rcx,1),%rsi
   dc9f6:	add    $0x8,%rsi
   dc9fa:	movzbl 0x1f(%rax,%rcx,1),%edx
   dc9ff:	lea    0x40(%rdx),%edi
   dca02:	movzbl %dil,%r8d
   dca06:	cmp    $0x18,%r8b
   dca0a:	mov    $0x18,%edi
   dca0f:	cmovae %edi,%r8d
   dca13:	cmp    $0xd8,%dl
   dca16:	cmovae 0x8(%rax,%rcx,1),%rsi
   dca1c:	movzbl %r8b,%r15d
   dca20:	cmovae 0x10(%rax,%rcx,1),%r15
   dca26:	mov    0x1c8(%rsp),%rax
   dca2e:	cmpq   $0xffffffffffffffff,0x70(%rax)
   dca33:	je     dca44 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd14>
   dca35:	mov    0x80(%rax),%r14
   dca3c:	mov    %r15,%rcx
   dca3f:	sub    %r14,%rcx
   dca42:	jae    dca49 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd19>
   dca44:	mov    %r15,%rax
   dca47:	jmp    dca8a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd5a>
   dca49:	mov    %rcx,0x18(%rsp)
   dca4e:	mov    %r10,0x40(%rsp)
   dca53:	mov    0x78(%rax),%rdi
   dca57:	mov    %r14,%rdx
   dca5a:	mov    %rsi,0x8(%rsp)
   dca5f:	call   *0x274e3(%rip)        # 103f48 <bcmp@GLIBC_2.2.5>
   dca65:	test   %eax,%eax
   dca67:	je     dca78 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd48>
   dca69:	mov    %r15,%rax
   dca6c:	mov    0x40(%rsp),%r10
   dca71:	mov    0x8(%rsp),%rsi
   dca76:	jmp    dca8a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd5a>
   dca78:	mov    0x8(%rsp),%rsi
   dca7d:	add    %r14,%rsi
   dca80:	mov    0x40(%rsp),%r10
   dca85:	mov    0x18(%rsp),%rax
   dca8a:	lea    (%rax,%r12,1),%r14
   dca8e:	cmp    $0x19,%r14
   dca92:	lea    0x3d0(%rsp),%r15
   dca9a:	mov    %rsi,0x8(%rsp)
   dca9f:	mov    %rax,0x18(%rsp)
   dcaa4:	jae    dcab4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xd84>
   dcaa6:	xor    %eax,%eax
   dcaa8:	movabs $0xc000000000000000,%r14
   dcab2:	jmp    dcaf3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xdc3>
   dcab4:	mov    %r10,0x40(%rsp)
   dcab9:	cmp    $0x21,%r14
   dcabd:	mov    $0x20,%eax
   dcac2:	cmovb  %rax,%r14
   dcac6:	test   %r14,%r14
   dcac9:	js     e0979 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c49>
   dcacf:	mov    %r14,%rdi
   dcad2:	call   *0x27430(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   dcad8:	test   %rax,%rax
   dcadb:	je     e0789 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a59>
   dcae1:	movabs $0xd800000000000000,%rcx
   dcaeb:	or     %rcx,%r14
   dcaee:	mov    0x40(%rsp),%r10
   dcaf3:	mov    %rax,0x3d0(%rsp)
   dcafb:	movq   $0x0,0x3d8(%rsp)
   dcb07:	mov    %r14,0x3e0(%rsp)
   dcb0f:	mov    %r15,%rdi
   dcb12:	mov    %r10,%rsi
   dcb15:	mov    %r12,%rdx
   dcb18:	call   e14f0 <<compact_str::repr::Repr>::push_str>
   dcb1d:	mov    %r15,%rdi
   dcb20:	mov    0x8(%rsp),%rsi
   dcb25:	mov    0x18(%rsp),%rdx
   dcb2a:	call   e14f0 <<compact_str::repr::Repr>::push_str>
   dcb2f:	lea    0x1d0(%rsp),%rdi
   dcb37:	mov    0x1c8(%rsp),%rsi
   dcb3f:	mov    %r15,%rdx
   dcb42:	call   e19b0 <<tk_train::trainers::bpe::vocabulary::Vocabulary>::insert_token>
   dcb47:	mov    0x1d0(%rsp),%rax
   dcb4f:	test   %rax,%rax
   dcb52:	jne    e03c4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4694>
   dcb58:	mov    0x1d8(%rsp),%r15d
   dcb60:	movzbl 0x1dc(%rsp),%ecx
   dcb68:	mov    %ecx,%eax
   dcb6a:	not    %al
   dcb6c:	or     0x38(%rsp),%al
   dcb70:	test   $0x1,%al
   dcb72:	je     dcb89 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe59>
   dcb74:	cmpq   $0x0,0x298(%rsp)
   dcb7d:	setne  %al
   dcb80:	test   %cl,%al
   dcb82:	je     dcbb2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe82>
   dcb84:	jmp    dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dcb89:	mov    0x20(%rsp),%rax
   dcb8e:	cmp    %r15,0x40(%rax)
   dcb92:	jbe    dcba3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xe73>
   dcb94:	mov    0x38(%rax),%rax
   dcb98:	cmpq   $0x0,(%rax,%r15,8)
   dcb9d:	jne    e0451 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4721>
   dcba3:	cmpq   $0x0,0x298(%rsp)
   dcbac:	jne    dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dcbb2:	mov    %cl,0x8(%rsp)
   dcbb6:	lea    0xb0(%rsp),%rdi
   dcbbe:	mov    0x30(%rsp),%r14
   dcbc3:	mov    %r14,%rsi
   dcbc6:	call   e1100 <<dary_heap::DaryHeap<tk_train::trainers::bpe::index::Candidate, 8>>::pop>
   dcbcb:	cmpl   $0x1,0xb0(%rsp)
   dcbd3:	jne    e0725 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x49f5>
   dcbd9:	lea    0xb8(%rsp),%rax
   dcbe1:	movdqu (%rax),%xmm0
   dcbe5:	movdqu 0x10(%rax),%xmm1
   dcbea:	movdqu 0x20(%rax),%xmm2
   dcbef:	movdqa %xmm2,0x1f0(%rsp)
   dcbf8:	movdqa %xmm1,0x1e0(%rsp)
   dcc01:	movdqa %xmm0,0x1d0(%rsp)
   dcc0a:	cmpb   $0x0,0x30(%r14)
   dcc0f:	jne    dcc67 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf37>
   dcc11:	mov    0x30(%rsp),%rax
   dcc16:	mov    0x20(%rax),%rcx
   dcc1a:	test   %rcx,%rcx
   dcc1d:	je     e09a5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c75>
   dcc23:	mov    %rbp,%rax
   dcc26:	or     %rcx,%rax
   dcc29:	shr    $0x20,%rax
   dcc2d:	je     dcc39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf09>
   dcc2f:	mov    %rbp,%rax
   dcc32:	xor    %edx,%edx
   dcc34:	div    %rcx
   dcc37:	jmp    dcc3f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf0f>
   dcc39:	mov    %ebp,%eax
   dcc3b:	xor    %edx,%edx
   dcc3d:	div    %ecx
   dcc3f:	mov    0x30(%rsp),%rax
   dcc44:	mov    0x18(%rax),%rax
   dcc48:	imul   $0x70,%rdx,%rcx
   dcc4c:	lea    (%rax,%rcx,1),%rdi
   dcc50:	add    $0x30,%rdi
   dcc54:	mov    %r13d,%esi
   dcc57:	mov    %ebx,%edx
   dcc59:	call   e2080 <<hashbrown::map::HashMap<(u32, u32), u64, ahash::random_state::RandomState>>::remove::<(u32, u32)>>
   dcc5e:	test   %rax,%rax
   dcc61:	je     e07d3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4aa3>
   dcc67:	movdqa 0x1d0(%rsp),%xmm0
   dcc70:	movdqa 0x1e0(%rsp),%xmm1
   dcc79:	movdqa 0x1f0(%rsp),%xmm2
   dcc82:	movdqa %xmm2,0x80(%rsp)
   dcc8b:	movdqa %xmm1,0x70(%rsp)
   dcc91:	movdqa %xmm0,0x60(%rsp)
   dcc97:	mov    0x20(%rsp),%rcx
   dcc9c:	mov    0x40(%rcx),%r12
   dcca0:	cmp    %r15,%r12
   dcca3:	jbe    dccb3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf83>
   dcca5:	mov    0x38(%rcx),%rax
   dcca9:	cmpq   $0x0,(%rax,%r15,8)
   dccae:	setne  %al
   dccb1:	jmp    dccb5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xf85>
   dccb3:	xor    %eax,%eax
   dccb5:	mov    0x58(%rsp),%rbp
   dccba:	or     %al,0x78(%rcx)
   dccbd:	mov    0x68(%rcx),%rax
   dccc1:	mov    $0x1,%r14d
   dccc7:	test   %rax,%rax
   dccca:	je     dcd05 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xfd5>
   dcccc:	cmp    %r15,%r12
   dcccf:	jne    dcd2d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xffd>
   dccd1:	mov    0x198(%rsp),%rax
   dccd9:	cmp    %r15,(%rax)
   dccdc:	jne    dcceb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xfbb>
   dccde:	mov    0x198(%rsp),%rdi
   dcce6:	call   75260 <<alloc::raw_vec::RawVec<usize>>::grow_one>
   dcceb:	mov    0x20(%rsp),%rcx
   dccf0:	mov    0x38(%rcx),%rax
   dccf4:	mov    %r14,(%rax,%r15,8)
   dccf8:	lea    0x1(%r15),%rax
   dccfc:	mov    %rax,0x40(%rcx)
   dcd00:	jmp    dcf48 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1218>
   dcd05:	cmp    %r13,%r12
   dcd08:	jbe    e09d9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ca9>
   dcd0e:	cmp    %rbx,%r12
   dcd11:	jbe    e09c2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c92>
   dcd17:	mov    0x20(%rsp),%rcx
   dcd1c:	mov    0x38(%rcx),%rcx
   dcd20:	mov    (%rcx,%rbx,8),%r14
   dcd24:	add    (%rcx,%r13,8),%r14
   dcd28:	cmp    %r15,%r12
   dcd2b:	je     dccd1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xfa1>
   dcd2d:	jbe    e09b6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c86>
   dcd33:	test   %rax,%rax
   dcd36:	jne    dcf32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1202>
   dcd3c:	mov    0x20(%rsp),%rax
   dcd41:	mov    0x38(%rax),%rax
   dcd45:	mov    (%rax,%r15,8),%rax
   dcd49:	test   %rax,%rax
   dcd4c:	je     dcf32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1202>
   dcd52:	cmp    %r14,%rax
   dcd55:	je     dcf32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1202>
   dcd5b:	mov    %r14,0x18(%rsp)
   dcd60:	mov    0x20(%rsp),%rax
   dcd65:	mov    0x10(%rax),%r14
   dcd69:	lea    0x0(,%r14,8),%rdx
   dcd71:	mov    %r14,%rax
   dcd74:	shr    $0x3d,%rax
   dcd78:	setne  %al
   dcd7b:	cmp    0x190(%rsp),%rdx
   dcd83:	seta   %cl
   dcd86:	or     %al,%cl
   dcd88:	jne    e039c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x466c>
   dcd8e:	test   %rdx,%rdx
   dcd91:	je     dcdbd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x108d>
   dcd93:	mov    $0x8,%esi
   dcd98:	mov    %rdx,%rdi
   dcd9b:	mov    %rdx,%rbp
   dcd9e:	call   68f50 <__rustc::__rust_alloc>
   dcda3:	mov    $0x8,%edi
   dcda8:	mov    %rbp,%rdx
   dcdab:	mov    %rax,%rbp
   dcdae:	mov    %r14,0x40(%rsp)
   dcdb3:	test   %rax,%rax
   dcdb6:	jne    dcdcb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x109b>
   dcdb8:	jmp    e039e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x466e>
   dcdbd:	mov    $0x8,%ebp
   dcdc2:	movq   $0x0,0x40(%rsp)
   dcdcb:	test   %r14,%r14
   dcdce:	je     dcddb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10ab>
   dcdd0:	mov    %rbp,%rdi
   dcdd3:	xor    %esi,%esi
   dcdd5:	call   *0x2713d(%rip)        # 103f18 <memset@GLIBC_2.2.5>
   dcddb:	mov    0x20(%rsp),%rax
   dcde0:	mov    0x50(%rax),%rax
   dcde4:	test   %rax,%rax
   dcde7:	je     dcefc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11cc>
   dcded:	mov    0x20(%rsp),%rsi
   dcdf2:	mov    0x48(%rsi),%rcx
   dcdf6:	lea    (%rcx,%rax,8),%rdx
   dcdfa:	cmpb   $0x0,(%rsi)
   dcdfd:	je     dce7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x114d>
   dcdff:	jmp    dce0a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10da>
   dce01:	cmp    %rdx,%rcx
   dce04:	je     dcefc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11cc>
   dce0a:	mov    (%rcx),%rdi
   dce0d:	cmp    %r14,%rdi
   dce10:	jae    e085b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b2b>
   dce16:	add    $0x8,%rcx
   dce1a:	mov    0x20(%rsp),%rax
   dce1f:	mov    0x8(%rax),%rsi
   dce23:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dce30:	mov    (%rsi,%rdi,4),%eax
   dce33:	cmp    $0xffffffff,%eax
   dce36:	je     dce01 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x10d1>
   dce38:	mov    (%rsi,%rdi,4),%eax
   dce3b:	cmp    %rax,%r12
   dce3e:	jbe    e0843 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b13>
   dce44:	mov    0x20(%rsp),%r8
   dce49:	mov    0x38(%r8),%r8
   dce4d:	mov    (%r8,%rax,8),%r8
   dce51:	mov    %r8,0x0(%rbp,%rdi,8)
   dce56:	lea    (%r8,%rdi,1),%rax
   dce5a:	dec    %rax
   dce5d:	cmp    %r14,%rax
   dce60:	jae    e0831 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b01>
   dce66:	add    %r8,%rdi
   dce69:	mov    %r8,-0x8(%rbp,%rdi,8)
   dce6e:	cmp    %r14,%rdi
   dce71:	jb     dce30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1100>
   dce73:	jmp    e085b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b2b>
   dce78:	cmp    %rdx,%rcx
   dce7b:	je     dcefc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11cc>
   dce7d:	mov    (%rcx),%rdi
   dce80:	cmp    %r14,%rdi
   dce83:	jae    e086a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b3a>
   dce89:	add    $0x8,%rcx
   dce8d:	mov    0x20(%rsp),%rax
   dce92:	mov    0x8(%rax),%rsi
   dce96:	cs nopw 0x0(%rax,%rax,1)
   dcea0:	movzwl (%rsi,%rdi,2),%eax
   dcea4:	cmp    $0xffff,%ax
   dcea8:	je     dce78 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1148>
   dceaa:	movzwl (%rsi,%rdi,2),%eax
   dceae:	cmp    $0xffff,%ax
   dceb2:	movzwl %ax,%eax
   dceb5:	mov    $0xffffffff,%r8d
   dcebb:	cmove  %r8,%rax
   dcebf:	cmp    %rax,%r12
   dcec2:	jbe    e0843 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b13>
   dcec8:	mov    0x20(%rsp),%r8
   dcecd:	mov    0x38(%r8),%r8
   dced1:	mov    (%r8,%rax,8),%r8
   dced5:	mov    %r8,0x0(%rbp,%rdi,8)
   dceda:	lea    (%r8,%rdi,1),%rax
   dcede:	dec    %rax
   dcee1:	cmp    %r14,%rax
   dcee4:	jae    e0831 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b01>
   dceea:	add    %r8,%rdi
   dceed:	mov    %r8,-0x8(%rbp,%rdi,8)
   dcef2:	cmp    %r14,%rdi
   dcef5:	jb     dcea0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1170>
   dcef7:	jmp    e086a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b3a>
   dcefc:	mov    0x20(%rsp),%r12
   dcf01:	mov    0x68(%r12),%rdi
   dcf06:	test   %rdi,%rdi
   dcf09:	je     dcf19 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11e9>
   dcf0b:	cmpq   $0x0,0x70(%r12)
   dcf11:	je     dcf19 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x11e9>
   dcf13:	call   *0x26fd7(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dcf19:	mov    %rbp,0x68(%r12)
   dcf1e:	mov    %r14,0x70(%r12)
   dcf23:	mov    0x40(%r12),%r12
   dcf28:	mov    0x58(%rsp),%rbp
   dcf2d:	mov    0x18(%rsp),%r14
   dcf32:	cmp    %r15,%r12
   dcf35:	jbe    e09c7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c97>
   dcf3b:	mov    0x20(%rsp),%rax
   dcf40:	mov    0x38(%rax),%rax
   dcf44:	mov    %r14,(%rax,%r15,8)
   dcf48:	movaps 0x60(%rsp),%xmm0
   dcf4d:	movaps 0x70(%rsp),%xmm1
   dcf52:	movaps 0x80(%rsp),%xmm2
   dcf5a:	movaps %xmm2,0xd0(%rsp)
   dcf62:	movaps %xmm1,0xc0(%rsp)
   dcf6a:	movaps %xmm0,0xb0(%rsp)
   dcf72:	mov    %r15d,0xe0(%rsp)
   dcf7a:	mov    0x2f0(%rsp),%r14
   dcf82:	cmp    0x2e0(%rsp),%r14
   dcf8a:	jne    dcf99 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1269>
   dcf8c:	lea    0x2e0(%rsp),%rdi
   dcf94:	call   e21c0 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::merge::Rule>>::grow_one>
   dcf99:	mov    0x2e8(%rsp),%rax
   dcfa1:	imul   $0x38,%r14,%rcx
   dcfa5:	mov    0xe0(%rsp),%rdx
   dcfad:	mov    %rdx,0x30(%rax,%rcx,1)
   dcfb2:	movdqa 0xb0(%rsp),%xmm0
   dcfbb:	movdqa 0xc0(%rsp),%xmm1
   dcfc4:	movdqa 0xd0(%rsp),%xmm2
   dcfcd:	movdqu %xmm2,0x20(%rax,%rcx,1)
   dcfd3:	movdqu %xmm1,0x10(%rax,%rcx,1)
   dcfd9:	movdqu %xmm0,(%rax,%rcx,1)
   dcfde:	inc    %r14
   dcfe1:	mov    %r14,0x2f0(%rsp)
   dcfe9:	cmp    %ebx,%r13d
   dcfec:	je     dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dcfee:	testb  $0x1,0x8(%rsp)
   dcff3:	jne    dd024 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x12f4>
   dcff5:	lea    0x340(%rsp),%rdi
   dcffd:	mov    %r13d,%esi
   dd000:	call   e22b0 <<hashbrown::map::HashMap<u32, (), ahash::random_state::RandomState>>::insert>
   dd005:	lea    0x380(%rsp),%rdi
   dd00d:	mov    %ebx,%esi
   dd00f:	call   e22b0 <<hashbrown::map::HashMap<u32, (), ahash::random_state::RandomState>>::insert>
   dd014:	cmp    0x130(%rsp),%r14
   dd01c:	jb     dc0cc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39c>
   dd022:	jmp    dd031 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1301>
   dd024:	mov    0x2f0(%rsp),%r14
   dd02c:	mov    0x58(%rsp),%rbp
   dd031:	test   %r14,%r14
   dd034:	je     dd0db <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13ab>
   dd03a:	mov    0x348(%rsp),%rcx
   dd042:	test   %rcx,%rcx
   dd045:	je     dd06d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x133d>
   dd047:	lea    0x13(,%rcx,4),%rax
   dd04f:	and    $0xfffffffffffffff0,%rax
   dd053:	add    %rax,%rcx
   dd056:	cmp    $0xffffffffffffffef,%rcx
   dd05a:	je     dd06d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x133d>
   dd05c:	mov    0x340(%rsp),%rdi
   dd064:	sub    %rax,%rdi
   dd067:	call   *0x26e83(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dd06d:	mov    0x388(%rsp),%rcx
   dd075:	test   %rcx,%rcx
   dd078:	je     dd0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1370>
   dd07a:	lea    0x13(,%rcx,4),%rax
   dd082:	and    $0xfffffffffffffff0,%rax
   dd086:	add    %rax,%rcx
   dd089:	cmp    $0xffffffffffffffef,%rcx
   dd08d:	je     dd0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1370>
   dd08f:	mov    0x380(%rsp),%rdi
   dd097:	sub    %rax,%rdi
   dd09a:	call   *0x26e50(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dd0a0:	mov    0x2e8(%rsp),%rbx
   dd0a8:	lea    0x0(,%r14,8),%r15
   dd0b0:	mov    %r15,%rdi
   dd0b3:	call   *0x26e4f(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   dd0b9:	mov    %rax,0x198(%rsp)
   dd0c1:	test   %rax,%rax
   dd0c4:	je     e087f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b4f>
   dd0ca:	cmp    $0x1,%r14
   dd0ce:	jne    dd1ff <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14cf>
   dd0d4:	xor    %eax,%eax
   dd0d6:	jmp    dd245 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1515>
   dd0db:	mov    0x2c0(%rsp),%rdx
   dd0e3:	movq   $0x0,0x8(%rdx)
   dd0eb:	xor    %eax,%eax
   dd0ed:	mov    %rax,(%rdx)
   dd0f0:	mov    0x388(%rsp),%rcx
   dd0f8:	test   %rcx,%rcx
   dd0fb:	je     dd123 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13f3>
   dd0fd:	lea    0x13(,%rcx,4),%rax
   dd105:	and    $0xfffffffffffffff0,%rax
   dd109:	add    %rax,%rcx
   dd10c:	cmp    $0xffffffffffffffef,%rcx
   dd110:	je     dd123 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13f3>
   dd112:	mov    0x380(%rsp),%rdi
   dd11a:	sub    %rax,%rdi
   dd11d:	call   *0x26dcd(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dd123:	mov    0x348(%rsp),%rcx
   dd12b:	test   %rcx,%rcx
   dd12e:	je     dd156 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1426>
   dd130:	lea    0x13(,%rcx,4),%rax
   dd138:	and    $0xfffffffffffffff0,%rax
   dd13c:	add    %rax,%rcx
   dd13f:	cmp    $0xffffffffffffffef,%rcx
   dd143:	je     dd156 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1426>
   dd145:	mov    0x340(%rsp),%rdi
   dd14d:	sub    %rax,%rdi
   dd150:	call   *0x26d9a(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dd156:	mov    0x2e8(%rsp),%rbx
   dd15e:	mov    0x2f0(%rsp),%r15
   dd166:	test   %r15,%r15
   dd169:	je     dd1e8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14b8>
   dd16b:	xor    %r12d,%r12d
   dd16e:	mov    0x26d7b(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   dd175:	mov    %r15,0x8(%rsp)
   dd17a:	jmp    dd193 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1463>
   dd17c:	nopl   0x0(%rax)
   dd180:	mov    0x8(%rsp),%r15
   dd185:	mov    %r14,%rdi
   dd188:	call   *%r13
   dd18b:	inc    %r12
   dd18e:	cmp    %r15,%r12
   dd191:	je     dd1e8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14b8>
   dd193:	imul   $0x38,%r12,%rcx
   dd197:	lea    (%rbx,%rcx,1),%rax
   dd19b:	cmpq   $0x0,(%rbx,%rcx,1)
   dd1a0:	je     dd1b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1480>
   dd1a2:	cmpq   $0x11,0x18(%rax)
   dd1a7:	jb     dd18b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x145b>
   dd1a9:	mov    0x8(%rax),%r14
   dd1ad:	jmp    dd185 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1455>
   dd1af:	nop
   dd1b0:	mov    0x10(%rax),%rbp
   dd1b4:	test   %rbp,%rbp
   dd1b7:	je     dd18b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x145b>
   dd1b9:	mov    0x8(%rax),%r14
   dd1bd:	lea    0x18(%r14),%r15
   dd1c1:	jmp    dd1d9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14a9>
   dd1c3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dd1d0:	add    $0x20,%r15
   dd1d4:	dec    %rbp
   dd1d7:	je     dd180 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1450>
   dd1d9:	cmpq   $0x11,(%r15)
   dd1dd:	jb     dd1d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14a0>
   dd1df:	mov    -0x10(%r15),%rdi
   dd1e3:	call   *%r13
   dd1e6:	jmp    dd1d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14a0>
   dd1e8:	cmpq   $0x0,0x2e0(%rsp)
   dd1f1:	je     df00d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32dd>
   dd1f7:	mov    %rbx,%rdi
   dd1fa:	jmp    df007 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32d7>
   dd1ff:	movabs $0x3fffffffffffffe,%rcx
   dd209:	and    %r14,%rcx
   dd20c:	lea    0x60(%rbx),%rdx
   dd210:	xor    %eax,%eax
   dd212:	mov    0x198(%rsp),%rsi
   dd21a:	nopw   0x0(%rax,%rax,1)
   dd220:	movq   (%rdx),%xmm0
   dd224:	movq   -0x38(%rdx),%xmm1
   dd229:	punpcklqdq %xmm0,%xmm1
   dd22d:	movdqu %xmm1,(%rsi,%rax,8)
   dd232:	add    $0x2,%rax
   dd236:	add    $0x70,%rdx
   dd23a:	cmp    %rax,%rcx
   dd23d:	jne    dd220 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x14f0>
   dd23f:	test   $0x1,%r14b
   dd243:	je     dd25a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x152a>
   dd245:	imul   $0x38,%rax,%rcx
   dd249:	mov    0x28(%rbx,%rcx,1),%rcx
   dd24e:	mov    0x198(%rsp),%rdx
   dd256:	mov    %rcx,(%rdx,%rax,8)
   dd25a:	movdqu 0x2e0(%rsp),%xmm0
   dd263:	movdqu 0x2f0(%rsp),%xmm1
   dd26c:	movdqa %xmm1,0x280(%rsp)
   dd275:	movdqa %xmm0,0x270(%rsp)
   dd27e:	cmpb   $0x0,0x10(%rbp)
   dd282:	mov    $0xffffffffffffffff,%rax
   dd289:	je     dd28f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x155f>
   dd28b:	mov    0x18(%rbp),%rax
   dd28f:	mov    %rax,0x58(%rsp)
   dd294:	movzbl 0x288(%rsp),%eax
   dd29c:	mov    %al,0x188(%rsp)
   dd2a3:	test   %al,%al
   dd2a5:	mov    %r14,0x298(%rsp)
   dd2ad:	mov    %r15,0x330(%rsp)
   dd2b5:	je     dd4a6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1776>
   dd2bb:	movdqu 0x2e0(%rsp),%xmm0
   dd2c4:	movdqu 0x2f0(%rsp),%xmm1
   dd2cd:	movdqa %xmm1,0xa0(%rsp)
   dd2d6:	movdqa %xmm0,0x90(%rsp)
   dd2df:	mov    0x58(%rsp),%rax
   dd2e4:	mov    %rax,0x300(%rsp)
   dd2ec:	cmpq   $0x0,0xa0(%rsp)
   dd2f5:	je     e08c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b95>
   dd2fb:	mov    0x98(%rsp),%r15
   dd303:	cmpq   $0x0,(%r15)
   dd307:	mov    $0x1,%eax
   dd30c:	mov    %r15,%rcx
   dd30f:	jne    dd319 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x15e9>
   dd311:	mov    0x10(%r15),%rax
   dd315:	mov    0x8(%r15),%rcx
   dd319:	shl    $0x5,%rax
   dd31d:	add    %rcx,%rax
   dd320:	mov    0x20(%rsp),%rdx
   dd325:	mov    %rdx,0x1e0(%rsp)
   dd32d:	movq   $0x0,0x1e8(%rsp)
   dd339:	movq   $0x0,0x208(%rsp)
   dd345:	mov    %rcx,0x228(%rsp)
   dd34d:	mov    %rax,0x230(%rsp)
   dd355:	movq   $0x0,0x238(%rsp)
   dd361:	movq   $0x2,0x1d0(%rsp)
   dd36d:	lea    0x1d0(%rsp),%rdi
   dd375:	mov    %rdi,0x18(%rsp)
   dd37a:	call   e2670 <<itertools::adaptors::coalesce::CoalesceBy<core::iter::adapters::map::Map<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>, <tk_train::trainers::bpe::merge::Batch>::prepare_cohort::{closure#0}>, itertools::adaptors::coalesce::DedupPred2CoalescePred<itertools::adaptors::coalesce::DedupEq>, itertools::adaptors::coalesce::NoCount> as core::iter::traits::iterator::Iterator>::next>
   dd37f:	cmp    $0x1,%rax
   dd383:	jne    ddec9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2199>
   dd389:	mov    %rdx,%r14
   dd38c:	mov    $0x20,%edi
   dd391:	call   *0x26b71(%rip)        # 103f08 <malloc@GLIBC_2.2.5>
   dd397:	test   %rax,%rax
   dd39a:	je     e091a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bea>
   dd3a0:	mov    %rax,%rbx
   dd3a3:	mov    %r14,(%rax)
   dd3a6:	movq   $0x4,0x60(%rsp)
   dd3af:	mov    %rax,0x68(%rsp)
   dd3b4:	movq   $0x1,0x70(%rsp)
   dd3bd:	movups 0x230(%rsp),%xmm0
   dd3c5:	movaps %xmm0,0x110(%rsp)
   dd3cd:	movups 0x220(%rsp),%xmm0
   dd3d5:	movaps %xmm0,0x100(%rsp)
   dd3dd:	movups 0x210(%rsp),%xmm0
   dd3e5:	movaps %xmm0,0xf0(%rsp)
   dd3ed:	movdqu 0x1d0(%rsp),%xmm0
   dd3f6:	movdqu 0x1e0(%rsp),%xmm1
   dd3ff:	movdqu 0x1f0(%rsp),%xmm2
   dd408:	movups 0x200(%rsp),%xmm3
   dd410:	movaps %xmm3,0xe0(%rsp)
   dd418:	movdqa %xmm2,0xd0(%rsp)
   dd421:	movdqa %xmm1,0xc0(%rsp)
   dd42a:	movdqa %xmm0,0xb0(%rsp)
   dd433:	mov    $0x1,%r14d
   dd439:	lea    0xb0(%rsp),%r13
   dd441:	lea    0x60(%rsp),%rbp
   dd446:	jmp    dd45c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x172c>
   dd448:	nopl   0x0(%rax,%rax,1)
   dd450:	mov    %r12,(%rbx,%r14,8)
   dd454:	inc    %r14
   dd457:	mov    %r14,0x70(%rsp)
   dd45c:	mov    %r13,%rdi
   dd45f:	call   e2670 <<itertools::adaptors::coalesce::CoalesceBy<core::iter::adapters::map::Map<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>, <tk_train::trainers::bpe::merge::Batch>::prepare_cohort::{closure#0}>, itertools::adaptors::coalesce::DedupPred2CoalescePred<itertools::adaptors::coalesce::DedupEq>, itertools::adaptors::coalesce::NoCount> as core::iter::traits::iterator::Iterator>::next>
   dd464:	cmp    $0x1,%rax
   dd468:	jne    ddf23 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21f3>
   dd46e:	mov    %rdx,%r12
   dd471:	cmp    0x60(%rsp),%r14
   dd476:	jne    dd450 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1720>
   dd478:	xor    %edx,%edx
   dd47a:	cmpq   $0x1,0xb0(%rsp)
   dd483:	sete   %dl
   dd486:	inc    %rdx
   dd489:	mov    $0x8,%ecx
   dd48e:	mov    $0x8,%r8d
   dd494:	mov    %rbp,%rdi
   dd497:	mov    %r14,%rsi
   dd49a:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   dd49f:	mov    0x68(%rsp),%rbx
   dd4a4:	jmp    dd450 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1720>
   dd4a6:	mov    0x280(%rsp),%rax
   dd4ae:	mov    %rax,0x18(%rsp)
   dd4b3:	test   %rax,%rax
   dd4b6:	je     e08da <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4baa>
   dd4bc:	mov    0x278(%rsp),%r15
   dd4c4:	mov    0x28(%r15),%edi
   dd4c8:	cmp    0x2c(%r15),%edi
   dd4cc:	jne    ddb81 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e51>
   dd4d2:	movq   $0x0,0x60(%rsp)
   dd4db:	movq   $0x8,0x68(%rsp)
   dd4e4:	movq   $0x0,0x70(%rsp)
   dd4ed:	mov    0x20(%rsp),%rcx
   dd4f2:	mov    0x40(%rcx),%rax
   dd4f6:	cmp    %rdi,%rax
   dd4f9:	jbe    e092e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bfe>
   dd4ff:	mov    0x38(%rcx),%rax
   dd503:	mov    (%rax,%rdi,8),%rax
   dd507:	mov    %rax,0x8(%rsp)
   dd50c:	cmpq   $0x0,(%r15)
   dd510:	mov    $0x1,%r8d
   dd516:	mov    %rdi,0x40(%rsp)
   dd51b:	jne    dd525 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x17f5>
   dd51d:	mov    0x10(%r15),%r8
   dd521:	mov    0x8(%r15),%r15
   dd525:	mov    0x8(%rsp),%rax
   dd52a:	add    %rax,%rax
   dd52d:	mov    %rax,0x130(%rsp)
   dd535:	shl    $0x5,%r8
   dd539:	add    %r15,%r8
   dd53c:	movq   $0x0,0xb0(%rsp)
   dd548:	movq   $0x0,0xd0(%rsp)
   dd554:	mov    %r15,0xf0(%rsp)
   dd55c:	mov    %r8,0xf8(%rsp)
   dd564:	movq   $0x0,0x100(%rsp)
   dd570:	mov    0x20(%rsp),%rax
   dd575:	mov    0x10(%rax),%r12
   dd579:	movzbl (%rax),%ecx
   dd57c:	mov    %cl,0x18(%rsp)
   dd580:	mov    0x8(%rax),%rbx
   dd584:	mov    $0x8,%eax
   dd589:	mov    %rax,0x190(%rsp)
   dd591:	lea    0x26028(%rip),%rax        # 1035c0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ac8>
   dd598:	mov    %rax,0x180(%rsp)
   dd5a0:	movq   $0x0,0x38(%rsp)
   dd5a9:	xor    %r9d,%r9d
   dd5ac:	xor    %eax,%eax
   dd5ae:	xor    %r14d,%r14d
   dd5b1:	jmp    dd5ee <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x18be>
   dd5b3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dd5c0:	mov    %r14,%r13
   dd5c3:	mov    0xb0(%rsp),%rax
   dd5cb:	mov    0xb8(%rsp),%rdx
   dd5d3:	mov    0xf0(%rsp),%r15
   dd5db:	mov    0x100(%rsp),%r9
   dd5e3:	mov    0xf8(%rsp),%r8
   dd5eb:	mov    %r13,%r14
   dd5ee:	test   %rax,%rax
   dd5f1:	je     dd610 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x18e0>
   dd5f3:	test   %rdx,%rdx
   dd5f6:	je     dd610 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x18e0>
   dd5f8:	mov    0xc8(%rsp),%rsi
   dd600:	jmp    dd91e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bee>
   dd605:	data16 cs nopw 0x0(%rax,%rax,1)
   dd610:	test   %r15,%r15
   dd613:	je     dda37 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d07>
   dd619:	cmp    %r8,%r15
   dd61c:	je     dda37 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d07>
   dd622:	mov    (%r15),%rdi
   dd625:	dec    %rdi
   dd628:	cmp    %rdi,%r9
   dd62b:	jne    dd695 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1965>
   dd62d:	movq   $0x0,0xc0(%rsp)
   dd639:	lea    0x20(%r15),%r10
   dd63d:	cmp    %r8,%r10
   dd640:	je     dd7d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1aa0>
   dd646:	mov    (%r10),%rax
   dd649:	dec    %rax
   dd64c:	je     dd6fe <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19ce>
   dd652:	mov    %eax,%ecx
   dd654:	and    $0x7f,%ecx
   dd657:	shr    $0x7,%rax
   dd65b:	cmp    $0x1,%rcx
   dd65f:	sbb    $0xffffffffffffffff,%rax
   dd663:	mov    $0x8,%edi
   dd668:	cmp    $0x2,%rax
   dd66c:	jae    dd7d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1aa8>
   dd672:	mov    0x38(%r15),%rax
   dd676:	mov    %rax,%rdx
   dd679:	cmp    $0x11,%rax
   dd67d:	jb     dd804 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ad4>
   dd683:	mov    0x30(%r15),%rdx
   dd687:	cmp    %rdi,%rdx
   dd68a:	jae    dd80d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1add>
   dd690:	jmp    e04c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4799>
   dd695:	mov    %rdi,%rcx
   dd698:	shr    $0x7,%rcx
   dd69c:	test   %r9,%r9
   dd69f:	je     dd6f9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19c9>
   dd6a1:	mov    %edi,%eax
   dd6a3:	and    $0x7f,%eax
   dd6a6:	cmp    $0x1,%rax
   dd6aa:	mov    %rcx,%rax
   dd6ad:	sbb    $0xffffffffffffffff,%rax
   dd6b1:	cmp    $0x2,%rax
   dd6b5:	jb     e0891 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b61>
   dd6bb:	mov    0x18(%r15),%r10
   dd6bf:	mov    %r10,%rdx
   dd6c2:	cmp    $0x11,%r10
   dd6c6:	jb     dd6cc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x199c>
   dd6c8:	mov    0x10(%r15),%rdx
   dd6cc:	lea    0x8(,%rax,8),%rsi
   dd6d4:	cmp    %rdx,%rsi
   dd6d7:	ja     e04b3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4783>
   dd6dd:	mov    %r9,%rdx
   dd6e0:	shr    $0x7,%rdx
   dd6e4:	cmp    %rax,%rdx
   dd6e7:	jae    e089f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b6f>
   dd6ed:	cmp    $0x11,%r10
   dd6f1:	jb     dd707 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19d7>
   dd6f3:	mov    0x8(%r15),%rax
   dd6f7:	jmp    dd70b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19db>
   dd6f9:	xor    %r10d,%r10d
   dd6fc:	jmp    dd710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x19e0>
   dd6fe:	add    $0x40,%r15
   dd702:	jmp    dd83b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b0b>
   dd707:	lea    0x8(%r15),%rax
   dd70b:	mov    0x8(%rax,%rdx,8),%r10
   dd710:	and    $0x7f,%edi
   dd713:	cmp    $0x1,%rdi
   dd717:	sbb    $0xffffffffffffffff,%rcx
   dd71b:	mov    $0x8,%edi
   dd720:	cmp    $0x2,%rcx
   dd724:	jae    dd744 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a14>
   dd726:	mov    0x18(%r15),%rax
   dd72a:	mov    %rax,%rsi
   dd72d:	cmp    $0x11,%rax
   dd731:	jb     dd76c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a3c>
   dd733:	mov    0x10(%r15),%rsi
   dd737:	mov    %rsi,%rdx
   dd73a:	sub    %rdi,%rdx
   dd73d:	jae    dd778 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a48>
   dd73f:	jmp    e0419 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e9>
   dd744:	dec    %rcx
   dd747:	bsr    %rcx,%rcx
   dd74b:	not    %ecx
   dd74d:	mov    $0xffffffffffffffff,%rax
   dd754:	shr    %cl,%rax
   dd757:	lea    0x10(,%rax,8),%rdi
   dd75f:	mov    0x18(%r15),%rax
   dd763:	mov    %rax,%rsi
   dd766:	cmp    $0x11,%rax
   dd76a:	jae    dd733 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a03>
   dd76c:	mov    %rsi,%rdx
   dd76f:	sub    %rdi,%rdx
   dd772:	jb     e0419 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e9>
   dd778:	cmp    %r10,%rdx
   dd77b:	jb     e04a4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4774>
   dd781:	cmp    $0x11,%rax
   dd785:	jb     dd7ab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a7b>
   dd787:	mov    0x8(%r15),%rax
   dd78b:	lea    0x20(%r15),%rcx
   dd78f:	movq   $0x0,0xc0(%rsp)
   dd79b:	sub    %r10,%rdx
   dd79e:	je     dd7c4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a94>
   dd7a0:	add    %r10,%rdi
   dd7a3:	add    %rdi,%rax
   dd7a6:	jmp    dd90d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bdd>
   dd7ab:	lea    0x8(%r15),%rax
   dd7af:	lea    0x20(%r15),%rcx
   dd7b3:	movq   $0x0,0xc0(%rsp)
   dd7bf:	sub    %r10,%rdx
   dd7c2:	jne    dd7a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1a70>
   dd7c4:	mov    %rcx,%r10
   dd7c7:	cmp    %r8,%r10
   dd7ca:	jne    dd646 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1916>
   dd7d0:	mov    %r10,%r15
   dd7d3:	jmp    dda2a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cfa>
   dd7d8:	dec    %rax
   dd7db:	bsr    %rax,%rcx
   dd7df:	not    %ecx
   dd7e1:	mov    $0xffffffffffffffff,%rax
   dd7e8:	shr    %cl,%rax
   dd7eb:	lea    0x10(,%rax,8),%rdi
   dd7f3:	mov    0x38(%r15),%rax
   dd7f7:	mov    %rax,%rdx
   dd7fa:	cmp    $0x11,%rax
   dd7fe:	jae    dd683 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1953>
   dd804:	cmp    %rdi,%rdx
   dd807:	jb     e04c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4799>
   dd80d:	cmp    $0x11,%rax
   dd811:	jb     dd82e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1afe>
   dd813:	mov    0x28(%r15),%rax
   dd817:	add    $0x40,%r15
   dd81b:	sub    %rdi,%rdx
   dd81e:	je     dd83b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b0b>
   dd820:	add    %rdi,%rax
   dd823:	xor    %r9d,%r9d
   dd826:	mov    %r15,%rcx
   dd829:	jmp    dd90d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bdd>
   dd82e:	lea    0x28(%r15),%rax
   dd832:	add    $0x40,%r15
   dd836:	sub    %rdi,%rdx
   dd839:	jne    dd820 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1af0>
   dd83b:	mov    %r15,%rcx
   dd83e:	jmp    dd857 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b27>
   dd840:	mov    $0x1,%eax
   dd845:	xor    %edx,%edx
   dd847:	lea    0x20(%r15),%rcx
   dd84b:	mov    %r15,%r10
   dd84e:	test   %rdx,%rdx
   dd851:	jne    dd90a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bda>
   dd857:	mov    %rcx,%r15
   dd85a:	cmp    %r8,%rcx
   dd85d:	je     dda27 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cf7>
   dd863:	mov    (%r15),%rax
   dd866:	dec    %rax
   dd869:	je     dd840 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b10>
   dd86b:	mov    %eax,%ecx
   dd86d:	and    $0x7f,%ecx
   dd870:	shr    $0x7,%rax
   dd874:	cmp    $0x1,%rcx
   dd878:	sbb    $0xffffffffffffffff,%rax
   dd87c:	mov    $0x8,%edi
   dd881:	cmp    $0x2,%rax
   dd885:	jae    dd8b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b80>
   dd887:	mov    0x38(%r10),%rax
   dd88b:	mov    %rax,%rsi
   dd88e:	cmp    $0x11,%rax
   dd892:	jb     dd8d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ba8>
   dd894:	mov    0x30(%r10),%rsi
   dd898:	mov    %rsi,%rdx
   dd89b:	sub    %rdi,%rdx
   dd89e:	jae    dd8e4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bb4>
   dd8a0:	jmp    e0419 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e9>
   dd8a5:	data16 cs nopw 0x0(%rax,%rax,1)
   dd8b0:	dec    %rax
   dd8b3:	bsr    %rax,%rcx
   dd8b7:	not    %ecx
   dd8b9:	mov    $0xffffffffffffffff,%rax
   dd8c0:	shr    %cl,%rax
   dd8c3:	lea    0x10(,%rax,8),%rdi
   dd8cb:	mov    0x38(%r10),%rax
   dd8cf:	mov    %rax,%rsi
   dd8d2:	cmp    $0x11,%rax
   dd8d6:	jae    dd894 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b64>
   dd8d8:	mov    %rsi,%rdx
   dd8db:	sub    %rdi,%rdx
   dd8de:	jb     e0419 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46e9>
   dd8e4:	cmp    $0x11,%rax
   dd8e8:	jb     dd8f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bc0>
   dd8ea:	mov    0x28(%r10),%rax
   dd8ee:	jmp    dd8f7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1bc7>
   dd8f0:	add    $0x28,%r10
   dd8f4:	mov    %r10,%rax
   dd8f7:	add    %rdi,%rax
   dd8fa:	lea    0x20(%r15),%rcx
   dd8fe:	mov    %r15,%r10
   dd901:	test   %rdx,%rdx
   dd904:	je     dd857 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1b27>
   dd90a:	xor    %r9d,%r9d
   dd90d:	mov    %r9,0xc8(%rsp)
   dd915:	mov    %r9,%rsi
   dd918:	xor    %r9d,%r9d
   dd91b:	mov    %rcx,%r15
   dd91e:	mov    %rax,0xb0(%rsp)
   dd926:	mov    %rdx,0xb8(%rsp)
   dd92e:	mov    %r15,0xf0(%rsp)
   dd936:	mov    %r9,0x100(%rsp)
   dd93e:	test   $0x7f,%sil
   dd942:	je     dd960 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c30>
   dd944:	movsbq (%rax),%rbp
   dd948:	test   %rbp,%rbp
   dd94b:	js     dd983 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c53>
   dd94d:	mov    $0x1,%edi
   dd952:	jmp    dd9e2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cb2>
   dd957:	nopw   0x0(%rax,%rax,1)
   dd960:	cmp    $0x7,%rdx
   dd964:	jbe    e0466 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4736>
   dd96a:	mov    (%rax),%rbp
   dd96d:	mov    %rbp,0xc0(%rsp)
   dd975:	add    $0xfffffffffffffff8,%rdx
   dd979:	mov    $0x8,%edi
   dd97e:	mov    %rdx,%rcx
   dd981:	jmp    dd9fe <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1cce>
   dd983:	cmp    $0x1,%rdx
   dd987:	je     e079a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a6a>
   dd98d:	and    $0x7f,%ebp
   dd990:	lea    -0x1(%rdx),%r8
   dd994:	mov    $0x7,%ecx
   dd999:	xor    %edi,%edi
   dd99b:	nopl   0x0(%rax,%rax,1)
   dd9a0:	movzbl 0x1(%rax,%rdi,1),%r9d
   dd9a6:	cmp    $0x3f,%ecx
   dd9a9:	jne    dd9b5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c85>
   dd9ab:	cmp    $0x1,%r9b
   dd9af:	ja     e079e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a6e>
   dd9b5:	movzbl %r9b,%r9d
   dd9b9:	test   %r9b,%r9b
   dd9bc:	jns    dd9d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1ca8>
   dd9be:	and    $0x7f,%r9d
   dd9c2:	shl    %cl,%r9
   dd9c5:	or     %r9,%rbp
   dd9c8:	add    $0x7,%ecx
   dd9cb:	inc    %rdi
   dd9ce:	cmp    %rdi,%r8
   dd9d1:	jne    dd9a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1c70>
   dd9d3:	jmp    e079a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a6a>
   dd9d8:	shl    %cl,%r9
   dd9db:	or     %r9,%rbp
   dd9de:	add    $0x2,%rdi
   dd9e2:	add    0xc0(%rsp),%rbp
   dd9ea:	mov    %rbp,0xc0(%rsp)
   dd9f2:	mov    %rdx,%rcx
   dd9f5:	sub    %rdi,%rcx
   dd9f8:	jb     e0445 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4715>
   dd9fe:	add    %rdi,%rax
   dda01:	mov    %rax,0xb0(%rsp)
   dda09:	mov    %rcx,0xb8(%rsp)
   dda11:	inc    %rsi
   dda14:	mov    %rsi,0xc8(%rsp)
   dda1c:	cmp    %r14,%rbp
   dda1f:	jb     dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   dda25:	jmp    dda8d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d5d>
   dda27:	xor    %r9d,%r9d
   dda2a:	mov    %r9,0xc8(%rsp)
   dda32:	xor    %r9d,%r9d
   dda35:	xor    %edx,%edx
   dda37:	movq   $0x0,0xb0(%rsp)
   dda43:	mov    %rdx,0xb8(%rsp)
   dda4b:	mov    %r15,0xf0(%rsp)
   dda53:	mov    %r9,0x100(%rsp)
   dda5b:	cmpq   $0x0,0xd0(%rsp)
   dda64:	je     ddf7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x224d>
   dda6a:	lea    0xd0(%rsp),%rdi
   dda72:	call   e2460 <<tk_train::trainers::bpe::positions::codec::Cursor as core::iter::traits::iterator::Iterator>::next>
   dda77:	cmp    $0x1,%rax
   dda7b:	jne    ddf7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x224d>
   dda81:	mov    %rdx,%rbp
   dda84:	cmp    %r14,%rbp
   dda87:	jb     dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   dda8d:	cmp    %r12,%rbp
   dda90:	jae    e0428 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46f8>
   dda96:	cmpb   $0x0,0x18(%rsp)
   dda9b:	je     ddab0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d80>
   dda9d:	mov    (%rbx,%rbp,4),%eax
   ddaa0:	cmpq   $0x0,0x8(%rsp)
   ddaa6:	jne    ddacf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1d9f>
   ddaa8:	jmp    dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddaad:	nopl   (%rax)
   ddab0:	movzwl (%rbx,%rbp,2),%eax
   ddab4:	cmp    $0xffff,%ax
   ddab8:	movzwl %ax,%eax
   ddabb:	mov    $0xffffffff,%ecx
   ddac0:	cmove  %ecx,%eax
   ddac3:	cmpq   $0x0,0x8(%rsp)
   ddac9:	je     dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddacf:	cmp    0x40(%rsp),%eax
   ddad3:	jne    dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddad9:	mov    0x130(%rsp),%rax
   ddae1:	lea    (%rax,%rbp,1),%r13
   ddae5:	cmp    %r12,%r13
   ddae8:	jae    dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddaee:	mov    0x8(%rsp),%rax
   ddaf3:	add    %rbp,%rax
   ddaf6:	cmpb   $0x0,0x18(%rsp)
   ddafb:	je     ddb15 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1de5>
   ddafd:	cmp    %r12,%rax
   ddb00:	jae    e08a7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b77>
   ddb06:	mov    (%rbx,%rax,4),%eax
   ddb09:	cmp    0x40(%rsp),%eax
   ddb0d:	jne    dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddb13:	jmp    ddb3b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e0b>
   ddb15:	cmp    %r12,%rax
   ddb18:	jae    e08b6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4b86>
   ddb1e:	movzwl (%rbx,%rax,2),%eax
   ddb22:	cmp    $0xffff,%ax
   ddb26:	movzwl %ax,%eax
   ddb29:	mov    $0xffffffff,%ecx
   ddb2e:	cmove  %ecx,%eax
   ddb31:	cmp    0x40(%rsp),%eax
   ddb35:	jne    dd5c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1890>
   ddb3b:	mov    0x38(%rsp),%rax
   ddb40:	cmp    0x60(%rsp),%rax
   ddb45:	jne    ddb5e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e2e>
   ddb47:	lea    0x60(%rsp),%rdi
   ddb4c:	call   79c10 <<alloc::raw_vec::RawVec<u64>>::grow_one>
   ddb51:	mov    0x68(%rsp),%rax
   ddb56:	mov    %rax,0x190(%rsp)
   ddb5e:	mov    0x190(%rsp),%rax
   ddb66:	mov    0x38(%rsp),%rcx
   ddb6b:	mov    %rbp,(%rax,%rcx,8)
   ddb6f:	inc    %rcx
   ddb72:	mov    %rcx,0x38(%rsp)
   ddb77:	mov    %rcx,0x70(%rsp)
   ddb7c:	jmp    dd5c3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1893>
   ddb81:	mov    0x28708(%rip),%rax        # 106290 <ahash::random_state::RAND_SOURCE>
   ddb88:	test   %rax,%rax
   ddb8b:	je     e0676 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4946>
   ddb91:	mov    (%rax),%r14
   ddb94:	mov    0x8(%rax),%r12
   ddb98:	mov    0x286f9(%rip),%rbx        # 106298 <ahash::random_state::get_fixed_seeds::SEEDS>
   ddb9f:	test   %rbx,%rbx
   ddba2:	je     e0680 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4950>
   ddba8:	mov    %r14,%rdi
   ddbab:	call   *0x18(%r12)
   ddbb0:	xor    0x8(%rbx),%rax
   ddbb4:	movabs $0x5851f42d4c957f2d,%r12
   ddbbe:	mul    %r12
   ddbc1:	mov    %rax,%rdi
   ddbc4:	xor    %rdx,%rdi
   ddbc7:	mov    (%rbx),%r10
   ddbca:	mov    0x20(%rbx),%r11
   ddbce:	mov    0x30(%rbx),%r9
   ddbd2:	mov    %r11,%rax
   ddbd5:	xor    %rdi,%rax
   ddbd8:	mul    %r12
   ddbdb:	xor    %r9,%rdx
   ddbde:	xor    %rdx,%rax
   ddbe1:	mul    %r12
   ddbe4:	mov    %rax,%rcx
   ddbe7:	xor    %rdx,%rcx
   ddbea:	mov    %rcx,%rax
   ddbed:	mul    %r10
   ddbf0:	mov    %rdx,%rsi
   ddbf3:	xor    %rax,%rsi
   ddbf6:	rol    %cl,%rsi
   ddbf9:	mov    0x28(%rbx),%r14
   ddbfd:	mov    0x38(%rbx),%rbx
   ddc01:	mov    %r14,%rax
   ddc04:	xor    %rdi,%rax
   ddc07:	mul    %r12
   ddc0a:	xor    %rbx,%rdx
   ddc0d:	xor    %rdx,%rax
   ddc10:	mul    %r12
   ddc13:	mov    %rax,%rcx
   ddc16:	xor    %rdx,%rcx
   ddc19:	mov    %rcx,%rax
   ddc1c:	mul    %r10
   ddc1f:	mov    %rdx,%r8
   ddc22:	xor    %rax,%r8
   ddc25:	rol    %cl,%r8
   ddc28:	xor    %rdi,%r9
   ddc2b:	mov    %r9,%rax
   ddc2e:	mul    %r12
   ddc31:	xor    %rdx,%r14
   ddc34:	xor    %r14,%rax
   ddc37:	mul    %r12
   ddc3a:	mov    %rax,%rcx
   ddc3d:	xor    %rdx,%rcx
   ddc40:	mov    %rcx,%rax
   ddc43:	mul    %r10
   ddc46:	mov    %rdx,%r9
   ddc49:	xor    %rax,%r9
   ddc4c:	rol    %cl,%r9
   ddc4f:	xor    %rbx,%rdi
   ddc52:	mov    %rdi,%rax
   ddc55:	mul    %r12
   ddc58:	xor    %rdx,%r11
   ddc5b:	xor    %r11,%rax
   ddc5e:	mul    %r12
   ddc61:	mov    %rax,%rcx
   ddc64:	xor    %rdx,%rcx
   ddc67:	mov    %rcx,%rax
   ddc6a:	mul    %r10
   ddc6d:	xor    %rax,%rdx
   ddc70:	rol    %cl,%rdx
   ddc73:	movaps 0x25216(%rip),%xmm0        # 102e90 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3398>
   ddc7a:	movaps %xmm0,0xc0(%rsp)
   ddc82:	movdqa 0x251f6(%rip),%xmm0        # 102e80 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3388>
   ddc8a:	movdqa %xmm0,0xb0(%rsp)
   ddc93:	lea    0xd0(%rsp),%r11
   ddc9b:	mov    %rsi,0xd0(%rsp)
   ddca3:	mov    %r8,0xd8(%rsp)
   ddcab:	mov    %r9,0xe0(%rsp)
   ddcb3:	mov    %rdx,0xe8(%rsp)
   ddcbb:	mov    0xc8(%rsp),%rax
   ddcc3:	mov    0x18(%rsp),%rsi
   ddcc8:	mov    %rsi,%rcx
   ddccb:	shr    $1,%rcx
   ddcce:	test   %rax,%rax
   ddcd1:	cmove  %rax,%rcx
   ddcd5:	sub    %rcx,%rsi
   ddcd8:	cmp    0xc0(%rsp),%rsi
   ddce0:	ja     e068d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x495d>
   ddce6:	xor    %ebx,%ebx
   ddce8:	jmp    dde06 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x20d6>
   ddced:	nopl   (%rax)
   ddcf0:	and    %rcx,%r13
   ddcf3:	movdqu 0x0(%rbp,%r13,1),%xmm1
   ddcfa:	movdqa %xmm1,%xmm2
   ddcfe:	pcmpeqb %xmm0,%xmm2
   ddd02:	pmovmskb %xmm2,%r8d
   ddd07:	test   %r8d,%r8d
   ddd0a:	je     ddd40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2010>
   ddd0c:	tzcnt  %r8d,%r9d
   ddd11:	add    %r13,%r9
   ddd14:	and    %rcx,%r9
   ddd17:	neg    %r9
   ddd1a:	lea    (%r9,%r9,2),%r10
   ddd1e:	cmp    -0xc(%rbp,%r10,4),%r14d
   ddd23:	jne    ddd30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2000>
   ddd25:	cmp    -0x8(%rbp,%r10,4),%r12d
   ddd2a:	je     dddec <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x20bc>
   ddd30:	lea    -0x1(%r8),%r9d
   ddd34:	and    %r8w,%r9w
   ddd38:	mov    %r9d,%r8d
   ddd3b:	jne    ddd0c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1fdc>
   ddd3d:	nopl   (%rax)
   ddd40:	cmp    $0x1,%rdi
   ddd44:	je     ddd58 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2028>
   ddd46:	pmovmskb %xmm1,%edx
   ddd4a:	test   %edx,%edx
   ddd4c:	je     ddd6f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x203f>
   ddd4e:	tzcnt  %edx,%edx
   ddd52:	add    %r13,%rdx
   ddd55:	and    %rcx,%rdx
   ddd58:	pcmpeqd %xmm2,%xmm2
   ddd5c:	pcmpeqb %xmm2,%xmm1
   ddd60:	pmovmskb %xmm1,%edi
   ddd64:	test   %edi,%edi
   ddd66:	jne    ddd90 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2060>
   ddd68:	mov    $0x1,%edi
   ddd6d:	jmp    ddd71 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2041>
   ddd6f:	xor    %edi,%edi
   ddd71:	add    %rsi,%r13
   ddd74:	add    $0x10,%r13
   ddd78:	add    $0x10,%rsi
   ddd7c:	jmp    ddcf0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1fc0>
   ddd81:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   ddd90:	movzbl 0x0(%rbp,%rdx,1),%esi
   ddd95:	test   %sil,%sil
   ddd98:	jns    ddeb2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2182>
   ddd9e:	and    $0x1,%sil
   ddda2:	lea    -0x10(%rdx),%rdi
   ddda6:	and    %rcx,%rdi
   ddda9:	mov    %al,0x0(%rbp,%rdx,1)
   dddad:	mov    %al,0x10(%rbp,%rdi,1)
   dddb1:	movdqa 0xc0(%rsp),%xmm0
   dddba:	movzbl %sil,%eax
   dddbe:	movd   %eax,%xmm1
   dddc2:	shufpd $0x2,-0xd265b(%rip),%xmm1        # b770 <GCC_except_table1280+0xe8>
   dddcb:	psubq  %xmm1,%xmm0
   dddcf:	movdqa %xmm0,0xc0(%rsp)
   dddd8:	neg    %rdx
   ddddb:	lea    (%rdx,%rdx,2),%rax
   ddddf:	mov    %r14d,-0xc(%rbp,%rax,4)
   ddde4:	mov    %r12d,-0x8(%rbp,%rax,4)
   ddde9:	mov    %rdx,%r9
   dddec:	lea    (%r9,%r9,2),%rax
   dddf0:	mov    0x8(%rsp),%ecx
   dddf4:	mov    %ecx,-0x4(%rbp,%rax,4)
   dddf8:	inc    %rbx
   dddfb:	cmp    0x18(%rsp),%rbx
   dde00:	je     ddeef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21bf>
   dde06:	imul   $0x38,%rbx,%rsi
   dde0a:	mov    0x28(%r15,%rsi,1),%r14d
   dde0f:	mov    0xd8(%rsp),%rax
   dde17:	xor    %r14,%rax
   dde1a:	movabs $0x5851f42d4c957f2d,%rcx
   dde24:	mul    %rcx
   dde27:	mov    0x2c(%r15,%rsi,1),%r12d
   dde2c:	xor    %r12,%rdx
   dde2f:	xor    %rdx,%rax
   dde32:	mul    %rcx
   dde35:	mov    %rax,%rcx
   dde38:	xor    %rdx,%rcx
   dde3b:	mov    %rcx,%rax
   dde3e:	mulq   0xd0(%rsp)
   dde46:	mov    %rdx,%r13
   dde49:	xor    %rax,%r13
   dde4c:	rol    %cl,%r13
   dde4f:	mov    0x30(%r15,%rsi,1),%eax
   dde54:	mov    %eax,0x8(%rsp)
   dde58:	cmpq   $0x0,0xc0(%rsp)
   dde61:	je     dde95 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2165>
   dde63:	mov    0xb0(%rsp),%rbp
   dde6b:	mov    0xb8(%rsp),%rcx
   dde73:	mov    %r13,%rax
   dde76:	shr    $0x39,%rax
   dde7a:	movd   %eax,%xmm0
   dde7e:	punpcklbw %xmm0,%xmm0
   dde82:	pshuflw $0x0,%xmm0,%xmm0
   dde87:	pshufd $0x44,%xmm0,%xmm0
   dde8c:	xor    %edi,%edi
   dde8e:	xor    %esi,%esi
   dde90:	jmp    ddcf0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1fc0>
   dde95:	mov    $0x1,%esi
   dde9a:	lea    0xb0(%rsp),%rdi
   ddea2:	mov    %r11,%rdx
   ddea5:	mov    %r11,%rbp
   ddea8:	call   c4760 <<hashbrown::raw::RawTable<((u32, u32), u32)>>::reserve_rehash::<hashbrown::map::make_hasher<(u32, u32), u32, ahash::random_state::RandomState>::{closure#0}>>
   ddead:	mov    %rbp,%r11
   ddeb0:	jmp    dde63 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2133>
   ddeb2:	movdqa 0x0(%rbp),%xmm0
   ddeb7:	pmovmskb %xmm0,%edx
   ddebb:	tzcnt  %edx,%edx
   ddebf:	movzbl 0x0(%rbp,%rdx,1),%esi
   ddec4:	jmp    ddd9e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x206e>
   ddec9:	mov    $0x1,%bpl
   ddecc:	mov    $0x8,%ebx
   dded1:	xor    %r14d,%r14d
   dded4:	mov    %fs:0xffffffffffffffd0,%rax
   ddedd:	test   %rax,%rax
   ddee0:	mov    %rbx,0x38(%rsp)
   ddee5:	je     ddf45 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2215>
   ddee7:	add    $0x110,%rax
   ddeed:	jmp    ddf4a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x221a>
   ddeef:	lea    0xb8(%rsp),%rcx
   ddef7:	mov    0x10(%rcx),%rax
   ddefb:	mov    %rax,0xa0(%rsp)
   ddf03:	movups (%rcx),%xmm0
   ddf06:	movaps %xmm0,0x90(%rsp)
   ddf0e:	movups (%r11),%xmm0
   ddf12:	movups 0x10(%r11),%xmm1
   ddf17:	movaps %xmm0,0x60(%rsp)
   ddf1c:	movaps %xmm1,0x70(%rsp)
   ddf21:	jmp    ddf9b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x226b>
   ddf23:	mov    0x68(%rsp),%rbx
   ddf28:	cmpq   $0x0,0x60(%rsp)
   ddf2e:	sete   %bpl
   ddf32:	mov    %fs:0xffffffffffffffd0,%rax
   ddf3b:	test   %rax,%rax
   ddf3e:	mov    %rbx,0x38(%rsp)
   ddf43:	jne    ddee7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x21b7>
   ddf45:	call   7f700 <rayon_core::registry::global_registry>
   ddf4a:	mov    (%rax),%rax
   ddf4d:	mov    0x208(%rax),%rcx
   ddf54:	test   %rcx,%rcx
   ddf57:	je     e08ef <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bbf>
   ddf5d:	mov    %r14,%rax
   ddf60:	or     %rcx,%rax
   ddf63:	shr    $0x20,%rax
   ddf67:	je     de05b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x232b>
   ddf6d:	mov    %r14,%rax
   ddf70:	xor    %edx,%edx
   ddf72:	div    %rcx
   ddf75:	mov    %rax,%rcx
   ddf78:	jmp    de064 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2334>
   ddf7d:	movdqu 0x60(%rsp),%xmm0
   ddf83:	movdqa %xmm0,0x90(%rsp)
   ddf8c:	mov    0x70(%rsp),%rax
   ddf91:	mov    %rax,0xa0(%rsp)
   ddf99:	xor    %ebp,%ebp
   ddf9b:	lea    0x270(%rsp),%rax
   ddfa3:	mov    %rax,0x1d0(%rsp)
   ddfab:	mov    0x20(%rsp),%rax
   ddfb0:	mov    %rax,0x1d8(%rsp)
   ddfb8:	mov    0x58(%rsp),%rax
   ddfbd:	mov    %rax,0x1e0(%rsp)
   ddfc5:	mov    %rbp,0x1e8(%rsp)
   ddfcd:	movaps 0x90(%rsp),%xmm0
   ddfd5:	movups %xmm0,0x1f0(%rsp)
   ddfdd:	mov    0xa0(%rsp),%rax
   ddfe5:	mov    %rax,0x200(%rsp)
   ddfed:	movdqa 0x60(%rsp),%xmm0
   ddff3:	movdqa 0x70(%rsp),%xmm1
   ddff9:	movdqu %xmm0,0x208(%rsp)
   de002:	movdqu %xmm1,0x218(%rsp)
   de00b:	movq   $0x0,0xb0(%rsp)
   de017:	movq   $0x8,0xb8(%rsp)
   de023:	movq   $0x0,0xc0(%rsp)
   de02f:	mov    0x280(%rsp),%rdx
   de037:	test   %rdx,%rdx
   de03a:	je     de3a5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2675>
   de040:	mov    0x278(%rsp),%rax
   de048:	cmp    $0x1,%rdx
   de04c:	jne    de651 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2921>
   de052:	xor    %ecx,%ecx
   de054:	xor    %ebx,%ebx
   de056:	jmp    de6a5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2975>
   de05b:	mov    %r14d,%eax
   de05e:	xor    %edx,%edx
   de060:	div    %ecx
   de062:	mov    %eax,%ecx
   de064:	xor    %eax,%eax
   de066:	test   %rdx,%rdx
   de069:	setne  %al
   de06c:	lea    (%rcx,%rax,1),%rdx
   de070:	cmp    $0x1,%rdx
   de074:	adc    %rax,%rcx
   de077:	movl   $0x0,0x2a0(%rsp)
   de082:	movb   $0x0,0x2a4(%rsp)
   de08a:	movq   $0x0,0x2a8(%rsp)
   de096:	movq   $0x0,0x160(%rsp)
   de0a2:	movq   $0x8,0x168(%rsp)
   de0ae:	movq   $0x0,0x170(%rsp)
   de0ba:	movb   $0x0,0x17(%rsp)
   de0bf:	mov    %rbx,0xb0(%rsp)
   de0c7:	mov    %r14,0xb8(%rsp)
   de0cf:	mov    %rcx,0xc0(%rsp)
   de0d7:	mov    %r15,0xc8(%rsp)
   de0df:	mov    0x20(%rsp),%r8
   de0e4:	mov    %r8,0xd0(%rsp)
   de0ec:	lea    0x300(%rsp),%rax
   de0f4:	mov    %rax,0xd8(%rsp)
   de0fc:	lea    0xe0(%rsp),%rsi
   de104:	lea    0x2a0(%rsp),%rdx
   de10c:	mov    %rdx,0xe0(%rsp)
   de114:	mov    %rbx,0x1d0(%rsp)
   de11c:	mov    %r14,0x1d8(%rsp)
   de124:	mov    %rcx,0x1e0(%rsp)
   de12c:	lea    0x1e8(%rsp),%rdi
   de134:	mov    %r15,0x1e8(%rsp)
   de13c:	mov    %r8,0x1f0(%rsp)
   de144:	mov    %rax,0x1f8(%rsp)
   de14c:	lea    0x200(%rsp),%r8
   de154:	mov    %r14,%rax
   de157:	or     %rcx,%rax
   de15a:	shr    $0x20,%rax
   de15e:	je     de16d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x243d>
   de160:	mov    %r14,%rax
   de163:	xor    %edx,%edx
   de165:	div    %rcx
   de168:	mov    %rax,%r13
   de16b:	jmp    de177 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2447>
   de16d:	mov    %r14d,%eax
   de170:	xor    %edx,%edx
   de172:	div    %ecx
   de174:	mov    %eax,%r13d
   de177:	mov    %bpl,0x40(%rsp)
   de17c:	cmp    $0x1,%rdx
   de180:	sbb    $0xffffffffffffffff,%r13
   de184:	mov    %r13,0x80(%rsp)
   de18c:	lea    0x17(%rsp),%rax
   de191:	mov    %rax,0x60(%rsp)
   de196:	mov    %rsi,0x68(%rsp)
   de19b:	mov    %r8,0x70(%rsp)
   de1a0:	mov    %rdi,0x78(%rsp)
   de1a5:	mov    %rcx,0x250(%rsp)
   de1ad:	mov    %rbx,0x240(%rsp)
   de1b5:	mov    %r14,0x248(%rsp)
   de1bd:	mov    %fs:0xffffffffffffffd0,%rax
   de1c6:	test   %rax,%rax
   de1c9:	je     de1d3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x24a3>
   de1cb:	add    $0x110,%rax
   de1d1:	jmp    de1d8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x24a8>
   de1d3:	call   7f700 <rayon_core::registry::global_registry>
   de1d8:	mov    (%rax),%rax
   de1db:	mov    0x208(%rax),%rcx
   de1e2:	sub    $0x8,%rsp
   de1e6:	lea    0x68(%rsp),%rax
   de1eb:	lea    0x150(%rsp),%rbp
   de1f3:	lea    0x248(%rsp),%r9
   de1fb:	mov    $0x1,%r8d
   de201:	mov    %rbp,%rdi
   de204:	mov    %r13,%rsi
   de207:	xor    %edx,%edx
   de209:	push   %rax
   de20a:	call   e2b90 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::chunks::ChunksProducer<usize>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::merge::Job, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::merge::Directories as core::default::Default>::default, <tk_train::trainers::bpe::merge::Batch>::prepare_cohort::{closure#1}>>>
   de20f:	add    $0x10,%rsp
   de213:	mov    0x158(%rsp),%rax
   de21b:	test   %rax,%rax
   de21e:	je     de24b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x251b>
   de220:	mov    0x148(%rsp),%rcx
   de228:	test   %rcx,%rcx
   de22b:	je     de24b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x251b>
   de22d:	xor    %edx,%edx
   de22f:	nop
   de230:	add    0x10(%rcx),%rdx
   de234:	dec    %rax
   de237:	je     de242 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2512>
   de239:	mov    0x18(%rcx),%rcx
   de23d:	test   %rcx,%rcx
   de240:	jne    de230 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2500>
   de242:	test   %rdx,%rdx
   de245:	jne    e04d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47a1>
   de24b:	movdqu 0x148(%rsp),%xmm0
   de254:	movdqa %xmm0,0x1d0(%rsp)
   de25d:	mov    0x158(%rsp),%r15
   de265:	mov    %r15,0x1e0(%rsp)
   de26d:	mov    0x1d0(%rsp),%rdi
   de275:	mov    0x1d8(%rsp),%rax
   de27d:	test   %rdi,%rdi
   de280:	je     de3a1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2671>
   de286:	mov    %rax,0x8(%rsp)
   de28b:	dec    %r15
   de28e:	mov    %r15,%rax
   de291:	jmp    de2b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2580>
   de293:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de2a0:	lea    -0x1(%r15),%rax
   de2a4:	mov    %rbx,%rdi
   de2a7:	test   %rbx,%rbx
   de2aa:	je     de3ac <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x267c>
   de2b0:	mov    %rax,%r15
   de2b3:	mov    0x18(%rdi),%rbx
   de2b7:	test   %rbx,%rbx
   de2ba:	je     de2d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x25a0>
   de2bc:	movq   $0x0,0x20(%rbx)
   de2c4:	jmp    de2d9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x25a9>
   de2c6:	cs nopw 0x0(%rax,%rax,1)
   de2d0:	movq   $0x0,0x8(%rsp)
   de2d9:	mov    (%rdi),%r12
   de2dc:	movdqu 0x8(%rdi),%xmm0
   de2e1:	movdqa %xmm0,0x60(%rsp)
   de2e7:	call   *0x25c03(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   de2ed:	cmp    $0xffffffffffffffff,%r12
   de2f1:	je     de3ae <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x267e>
   de2f7:	mov    %r12,0xb0(%rsp)
   de2ff:	movdqa 0x60(%rsp),%xmm0
   de305:	lea    0xb8(%rsp),%rax
   de30d:	movdqu %xmm0,(%rax)
   de311:	mov    0xb8(%rsp),%r13
   de319:	mov    0xc0(%rsp),%rbp
   de321:	mov    0x160(%rsp),%rax
   de329:	mov    0x170(%rsp),%r14
   de331:	sub    %r14,%rax
   de334:	cmp    %rax,%rbp
   de337:	ja     de379 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2649>
   de339:	test   %rbp,%rbp
   de33c:	je     de357 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2627>
   de33e:	imul   $0x38,%r14,%rdi
   de342:	add    0x168(%rsp),%rdi
   de34a:	imul   $0x38,%rbp,%rdx
   de34e:	mov    %r13,%rsi
   de351:	call   *0x25ba1(%rip)        # 103ef8 <memcpy@GLIBC_2.14>
   de357:	add    %rbp,%r14
   de35a:	mov    %r14,0x170(%rsp)
   de362:	test   %r12,%r12
   de365:	je     de2a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2570>
   de36b:	mov    %r13,%rdi
   de36e:	call   *0x25b7c(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   de374:	jmp    de2a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2570>
   de379:	mov    $0x8,%ecx
   de37e:	mov    $0x38,%r8d
   de384:	lea    0x160(%rsp),%rdi
   de38c:	mov    %r14,%rsi
   de38f:	mov    %rbp,%rdx
   de392:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   de397:	mov    0x170(%rsp),%r14
   de39f:	jmp    de33e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x260e>
   de3a1:	xor    %ebx,%ebx
   de3a3:	jmp    de3b3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2683>
   de3a5:	xor    %ebx,%ebx
   de3a7:	jmp    de6bb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x298b>
   de3ac:	xor    %ebx,%ebx
   de3ae:	mov    0x8(%rsp),%rax
   de3b3:	mov    %rbx,0x1d0(%rsp)
   de3bb:	mov    %rax,0x1d8(%rsp)
   de3c3:	mov    %r15,0x1e0(%rsp)
   de3cb:	lea    0x1d0(%rsp),%rdi
   de3d3:	call   e4cb0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>>
   de3d8:	movdqu 0x160(%rsp),%xmm0
   de3e1:	movdqa %xmm0,0x1a0(%rsp)
   de3ea:	mov    0x170(%rsp),%rax
   de3f2:	mov    %rax,0x1b0(%rsp)
   de3fa:	mov    0x2a8(%rsp),%r13
   de402:	mov    0x2b0(%rsp),%rbp
   de40a:	cmpb   $0x0,0x2a4(%rsp)
   de412:	jne    e050f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47df>
   de418:	test   %r13,%r13
   de41b:	je     de479 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2749>
   de41d:	mov    0x1a8(%rsp),%r14
   de425:	mov    0x1b0(%rsp),%rbx
   de42d:	test   %rbx,%rbx
   de430:	je     de463 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2733>
   de432:	mov    0x25ab7(%rip),%r15        # 103ef0 <free@GLIBC_2.2.5>
   de439:	mov    %r14,%r12
   de43c:	jmp    de451 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2721>
   de43e:	xchg   %ax,%ax
   de440:	mov    %r12,%rdi
   de443:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   de448:	add    $0x38,%r12
   de44c:	dec    %rbx
   de44f:	je     de463 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2733>
   de451:	cmpq   $0x3,0x28(%r12)
   de457:	jb     de440 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2710>
   de459:	mov    0x18(%r12),%rdi
   de45e:	call   *%r15
   de461:	jmp    de440 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2710>
   de463:	cmpq   $0x0,0x1a0(%rsp)
   de46c:	je     de4df <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27af>
   de46e:	mov    %r14,%rdi
   de471:	call   *0x25a79(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   de477:	jmp    de4df <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27af>
   de479:	mov    0x1a0(%rsp),%rax
   de481:	mov    0x1a8(%rsp),%r13
   de489:	mov    0x1b0(%rsp),%rbp
   de491:	cmp    $0xffffffffffffffff,%rax
   de495:	je     de4df <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27af>
   de497:	mov    0x98(%rsp),%r14
   de49f:	mov    0xa0(%rsp),%rcx
   de4a7:	cmpb   $0x0,0x40(%rsp)
   de4ac:	mov    %rbp,0x18(%rsp)
   de4b1:	mov    %r13,0x8(%rsp)
   de4b6:	mov    %rax,0x130(%rsp)
   de4be:	je     df01f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32ef>
   de4c4:	test   %rcx,%rcx
   de4c7:	je     df0e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33b9>
   de4cd:	mov    %rcx,%rbp
   de4d0:	xor    %r12d,%r12d
   de4d3:	mov    0x25a16(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   de4da:	jmp    de5ae <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x287e>
   de4df:	cmpb   $0x0,0x40(%rsp)
   de4e4:	mov    %rbp,0x18(%rsp)
   de4e9:	mov    %r13,0x8(%rsp)
   de4ee:	je     de608 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28d8>
   de4f4:	mov    0x98(%rsp),%rbx
   de4fc:	mov    0xa0(%rsp),%r12
   de504:	test   %r12,%r12
   de507:	je     de63d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x290d>
   de50d:	xor    %ebp,%ebp
   de50f:	mov    0x259da(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   de516:	mov    %r12,0x30(%rsp)
   de51b:	jmp    de537 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2807>
   de51d:	nopl   (%rax)
   de520:	mov    0x30(%rsp),%r12
   de525:	mov    %r14,%rdi
   de528:	call   *%r13
   de52b:	inc    %rbp
   de52e:	cmp    %r12,%rbp
   de531:	je     de63d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x290d>
   de537:	imul   $0x38,%rbp,%rcx
   de53b:	lea    (%rbx,%rcx,1),%rax
   de53f:	cmpq   $0x0,(%rbx,%rcx,1)
   de544:	je     de560 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2830>
   de546:	cmpq   $0x11,0x18(%rax)
   de54b:	jb     de52b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27fb>
   de54d:	mov    0x8(%rax),%r14
   de551:	jmp    de525 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27f5>
   de553:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de560:	mov    0x10(%rax),%r15
   de564:	test   %r15,%r15
   de567:	je     de52b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27fb>
   de569:	mov    0x8(%rax),%r14
   de56d:	lea    0x18(%r14),%r12
   de571:	jmp    de589 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2859>
   de573:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de580:	add    $0x20,%r12
   de584:	dec    %r15
   de587:	je     de520 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x27f0>
   de589:	cmpq   $0x11,(%r12)
   de58e:	jb     de580 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2850>
   de590:	mov    -0x10(%r12),%rdi
   de595:	call   *%r13
   de598:	jmp    de580 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2850>
   de59a:	mov    0x40(%rsp),%rdi
   de59f:	call   *%r13
   de5a2:	inc    %r12
   de5a5:	cmp    %rbp,%r12
   de5a8:	je     df0e9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33b9>
   de5ae:	imul   $0x38,%r12,%rcx
   de5b2:	lea    (%r14,%rcx,1),%rax
   de5b6:	cmpq   $0x0,(%r14,%rcx,1)
   de5bb:	je     de5d0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28a0>
   de5bd:	cmpq   $0x11,0x18(%rax)
   de5c2:	jb     de5a2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2872>
   de5c4:	mov    0x8(%rax),%rdi
   de5c8:	jmp    de59f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x286f>
   de5ca:	nopw   0x0(%rax,%rax,1)
   de5d0:	mov    0x10(%rax),%r15
   de5d4:	test   %r15,%r15
   de5d7:	je     de5a2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2872>
   de5d9:	mov    0x8(%rax),%rax
   de5dd:	mov    %rax,0x40(%rsp)
   de5e2:	lea    0x18(%rax),%rbx
   de5e6:	jmp    de5f9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28c9>
   de5e8:	nopl   0x0(%rax,%rax,1)
   de5f0:	add    $0x20,%rbx
   de5f4:	dec    %r15
   de5f7:	je     de59a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x286a>
   de5f9:	cmpq   $0x11,(%rbx)
   de5fd:	jb     de5f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28c0>
   de5ff:	mov    -0x10(%rbx),%rdi
   de603:	call   *%r13
   de606:	jmp    de5f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x28c0>
   de608:	mov    0x98(%rsp),%r14
   de610:	mov    0xa0(%rsp),%rbp
   de618:	movq   $0xffffffffffffffff,0x130(%rsp)
   de624:	mov    0x38(%rsp),%rdi
   de629:	call   *0x258c1(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   de62f:	test   %rbp,%rbp
   de632:	jne    df036 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3306>
   de638:	jmp    df0b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3389>
   de63d:	cmpq   $0x0,0x90(%rsp)
   de646:	je     defde <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32ae>
   de64c:	jmp    defd5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32a5>
   de651:	mov    %rdx,%rsi
   de654:	and    $0xfffffffffffffffe,%rsi
   de658:	xor    %ecx,%ecx
   de65a:	mov    %rax,%rdi
   de65d:	xor    %ebx,%ebx
   de65f:	jmp    de683 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2953>
   de661:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   de670:	add    %r8,%rbx
   de673:	add    %r9,%rbx
   de676:	add    $0x2,%rcx
   de67a:	add    $0x70,%rdi
   de67e:	cmp    %rcx,%rsi
   de681:	je     de6a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2970>
   de683:	mov    (%rdi),%r8
   de686:	sub    $0x1,%r8
   de68a:	jae    de690 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2960>
   de68c:	mov    0x18(%rdi),%r8
   de690:	mov    0x38(%rdi),%r9
   de694:	sub    $0x1,%r9
   de698:	jae    de670 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2940>
   de69a:	mov    0x50(%rdi),%r9
   de69e:	jmp    de670 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2940>
   de6a0:	test   $0x1,%dl
   de6a3:	je     de6bb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x298b>
   de6a5:	imul   $0x38,%rcx,%rdx
   de6a9:	mov    (%rax,%rdx,1),%rcx
   de6ad:	sub    $0x1,%rcx
   de6b1:	jae    de6b8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2988>
   de6b3:	mov    0x18(%rax,%rdx,1),%rcx
   de6b8:	add    %rcx,%rbx
   de6bb:	mov    %fs:0xffffffffffffffd0,%rax
   de6c4:	test   %rax,%rax
   de6c7:	je     de6d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x29a1>
   de6c9:	add    $0x110,%rax
   de6cf:	jmp    de6d6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x29a6>
   de6d1:	call   7f700 <rayon_core::registry::global_registry>
   de6d6:	mov    (%rax),%rax
   de6d9:	mov    0x208(%rax),%rcx
   de6e0:	test   %rcx,%rcx
   de6e3:	je     e0900 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bd0>
   de6e9:	shl    $0x2,%rcx
   de6ed:	mov    %rbx,%rax
   de6f0:	or     %rcx,%rax
   de6f3:	shr    $0x20,%rax
   de6f7:	je     de703 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x29d3>
   de6f9:	mov    %rbx,%rax
   de6fc:	xor    %edx,%edx
   de6fe:	div    %rcx
   de701:	jmp    de709 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x29d9>
   de703:	mov    %ebx,%eax
   de705:	xor    %edx,%edx
   de707:	div    %ecx
   de709:	cmp    $0x1,%rdx
   de70d:	sbb    $0xffffffffffffffff,%rax
   de711:	cmp    $0x1001,%rax
   de717:	mov    $0x1000,%ecx
   de71c:	cmovae %rax,%rcx
   de720:	mov    %rcx,0x18(%rsp)
   de725:	mov    %fs:0xffffffffffffffd0,%rax
   de72e:	test   %rax,%rax
   de731:	je     de73b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a0b>
   de733:	add    $0x110,%rax
   de739:	jmp    de740 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a10>
   de73b:	call   7f700 <rayon_core::registry::global_registry>
   de740:	mov    (%rax),%rax
   de743:	mov    0x208(%rax),%rcx
   de74a:	test   %rcx,%rcx
   de74d:	je     e0909 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4bd9>
   de753:	mov    %rbx,%rax
   de756:	or     %rcx,%rax
   de759:	shr    $0x20,%rax
   de75d:	je     de769 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a39>
   de75f:	mov    %rbx,%rax
   de762:	xor    %edx,%edx
   de764:	div    %rcx
   de767:	jmp    de76f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2a3f>
   de769:	mov    %ebx,%eax
   de76b:	xor    %edx,%edx
   de76d:	div    %ecx
   de76f:	cmp    $0x1,%rdx
   de773:	sbb    $0xffffffffffffffff,%rax
   de777:	cmp    $0x4000000,%rax
   de77d:	mov    $0x4000000,%ecx
   de782:	cmovb  %rax,%rcx
   de786:	test   %rax,%rax
   de789:	mov    $0x1,%r12d
   de78f:	cmovne %rcx,%r12
   de793:	mov    0x280(%rsp),%rax
   de79b:	test   %rax,%rax
   de79e:	je     dea4d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d1d>
   de7a4:	mov    0x278(%rsp),%rdx
   de7ac:	imul   $0x38,%rax,%rax
   de7b0:	add    %rdx,%rax
   de7b3:	mov    %rax,0x58(%rsp)
   de7b8:	test   %rbp,%rbp
   de7bb:	je     de951 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c21>
   de7c1:	mov    $0x8,%eax
   de7c6:	movq   $0x0,0x8(%rsp)
   de7cf:	movq   $0x0,0x130(%rsp)
   de7db:	jmp    de7fc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2acc>
   de7dd:	nopl   (%rax)
   de7e0:	mov    0x38(%rsp),%rdx
   de7e5:	add    $0x38,%rdx
   de7e9:	incq   0x130(%rsp)
   de7f1:	cmp    0x58(%rsp),%rdx
   de7f6:	je     dea56 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d26>
   de7fc:	mov    (%rdx),%rcx
   de7ff:	mov    $0x1,%r14d
   de805:	test   %rcx,%rcx
   de808:	jne    de80e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ade>
   de80a:	mov    0x10(%rdx),%r14
   de80e:	test   %r14,%r14
   de811:	mov    %rdx,0x38(%rsp)
   de816:	je     de7e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ab0>
   de818:	mov    %rdx,%r15
   de81b:	test   %rcx,%rcx
   de81e:	jne    de824 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2af4>
   de820:	mov    0x8(%rdx),%r15
   de824:	xor    %ebp,%ebp
   de826:	jmp    de89e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2b6e>
   de828:	nopl   0x0(%rax,%rax,1)
   de830:	shl    $0x5,%r13
   de834:	mov    0x40(%rsp),%rdi
   de839:	add    %rdi,%r13
   de83c:	mov    0x8(%rsp),%rdx
   de841:	imul   $0x70,%rdx,%rcx
   de845:	mov    0x130(%rsp),%rsi
   de84d:	mov    %rsi,(%rax,%rcx,1)
   de851:	movq   $0x1,0x8(%rax,%rcx,1)
   de85a:	movq   $0x0,0x10(%rax,%rcx,1)
   de863:	movq   $0x0,0x30(%rax,%rcx,1)
   de86c:	mov    %rdi,0x50(%rax,%rcx,1)
   de871:	mov    %r13,0x58(%rax,%rcx,1)
   de876:	mov    0x18(%rsp),%rsi
   de87b:	mov    %rsi,0x60(%rax,%rcx,1)
   de880:	mov    %rbx,0x68(%rax,%rcx,1)
   de885:	inc    %rdx
   de888:	mov    %rdx,0x8(%rsp)
   de88d:	mov    %rdx,0xc0(%rsp)
   de895:	test   %r14,%r14
   de898:	je     de7e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ab0>
   de89e:	mov    %r15,0x40(%rsp)
   de8a3:	mov    %r14,%rdx
   de8a6:	mov    %rbp,0x18(%rsp)
   de8ab:	xor    %ebx,%ebx
   de8ad:	nopl   (%rax)
   de8b0:	mov    %r12,%rcx
   de8b3:	sub    %rbx,%rcx
   de8b6:	mov    $0x0,%esi
   de8bb:	cmovae %rcx,%rsi
   de8bf:	cmp    $0x1,%rsi
   de8c3:	adc    $0x0,%rsi
   de8c7:	add    $0x7f,%esi
   de8ca:	and    $0xfffff80,%esi
   de8d0:	mov    (%r15),%rdi
   de8d3:	mov    %rbp,%rcx
   de8d6:	not    %rcx
   de8d9:	add    %rdi,%rcx
   de8dc:	cmp    %rsi,%rcx
   de8df:	cmovae %rsi,%rcx
   de8e3:	add    %rcx,%rbp
   de8e6:	dec    %rdi
   de8e9:	cmp    %rdi,%rbp
   de8ec:	jne    de8f7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2bc7>
   de8ee:	dec    %r14
   de8f1:	add    $0x20,%r15
   de8f5:	xor    %ebp,%ebp
   de8f7:	add    %rcx,%rbx
   de8fa:	cmp    %r12,%rbx
   de8fd:	jae    de904 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2bd4>
   de8ff:	test   %r14,%r14
   de902:	jne    de8b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2b80>
   de904:	mov    %rdx,%r13
   de907:	sub    %r14,%r13
   de90a:	cmp    $0x1,%rbp
   de90e:	sbb    $0xffffffffffffffff,%r13
   de912:	test   %rbx,%rbx
   de915:	je     de7e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ab0>
   de91b:	cmp    %rdx,%r13
   de91e:	ja     df100 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33d0>
   de924:	mov    0x8(%rsp),%rcx
   de929:	cmp    0xb0(%rsp),%rcx
   de931:	jne    de830 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2b00>
   de937:	lea    0xb0(%rsp),%rdi
   de93f:	call   e4f80 <<alloc::raw_vec::RawVec<(usize, either::Either<core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>)>>::grow_one>
   de944:	mov    0xb8(%rsp),%rax
   de94c:	jmp    de830 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2b00>
   de951:	mov    0x200(%rsp),%rax
   de959:	mov    %rax,0x130(%rsp)
   de961:	test   %rax,%rax
   de964:	je     dea4d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d1d>
   de96a:	mov    0x1f8(%rsp),%rax
   de972:	mov    %rax,0x190(%rsp)
   de97a:	mov    $0x8,%eax
   de97f:	movq   $0x0,0x8(%rsp)
   de988:	xor    %r12d,%r12d
   de98b:	jmp    de9aa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c7a>
   de98d:	nopl   (%rax)
   de990:	mov    %rcx,0x8(%rsp)
   de995:	mov    0x40(%rsp),%r12
   de99a:	mov    0x38(%rsp),%rdx
   de99f:	cmp    0x58(%rsp),%rdx
   de9a4:	je     dea56 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2d26>
   de9aa:	add    $0x38,%rdx
   de9ae:	mov    %rdx,0x38(%rsp)
   de9b3:	lea    0x1(%r12),%rcx
   de9b8:	mov    %rcx,0x40(%rsp)
   de9bd:	mov    0x8(%rsp),%rcx
   de9c2:	imul   $0x70,%rcx,%r15
   de9c6:	add    $0x18,%r15
   de9ca:	mov    0x190(%rsp),%rbp
   de9d2:	mov    0x130(%rsp),%rbx
   de9da:	jmp    dea22 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cf2>
   de9dc:	nopl   0x0(%rax)
   de9e0:	mov    0xb8(%rsp),%rax
   de9e8:	mov    %r12,-0x18(%rax,%r15,1)
   de9ed:	movq   $0x0,-0x10(%rax,%r15,1)
   de9f6:	mov    %rbp,-0x8(%rax,%r15,1)
   de9fb:	lea    0x0(%rbp,%r14,8),%rbp
   dea00:	sub    %r14,%rbx
   dea03:	mov    %rbp,(%rax,%r15,1)
   dea07:	mov    %r13,%rcx
   dea0a:	inc    %rcx
   dea0d:	mov    %rcx,0xc0(%rsp)
   dea15:	add    $0x70,%r15
   dea19:	test   %rbx,%rbx
   dea1c:	je     de990 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2c60>
   dea22:	mov    0x18(%rsp),%rdx
   dea27:	cmp    %rbx,%rdx
   dea2a:	mov    %rbx,%r14
   dea2d:	cmovb  %rdx,%r14
   dea31:	mov    %rcx,%r13
   dea34:	cmp    0xb0(%rsp),%rcx
   dea3c:	jne    de9e8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cb8>
   dea3e:	lea    0xb0(%rsp),%rdi
   dea46:	call   e4f80 <<alloc::raw_vec::RawVec<(usize, either::Either<core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>)>>::grow_one>
   dea4b:	jmp    de9e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2cb0>
   dea4d:	movq   $0x0,0x8(%rsp)
   dea56:	mov    0xb0(%rsp),%rax
   dea5e:	mov    0xb8(%rsp),%r14
   dea66:	movl   $0x0,0x300(%rsp)
   dea71:	movb   $0x0,0x304(%rsp)
   dea79:	movq   $0x0,0x308(%rsp)
   dea85:	mov    0x8(%rsp),%rbx
   dea8a:	mov    %rbx,0x250(%rsp)
   dea92:	lea    0x258(%rsp),%rcx
   dea9a:	lea    0x1d0(%rsp),%rdx
   deaa2:	mov    %rdx,0x258(%rsp)
   deaaa:	movq   $0x0,0x1a0(%rsp)
   deab6:	movq   $0x8,0x1a8(%rsp)
   deac2:	movq   $0x0,0x1b0(%rsp)
   deace:	movb   $0x0,0x17(%rsp)
   dead3:	mov    %rax,0x60(%rsp)
   dead8:	mov    %r14,0x68(%rsp)
   deadd:	mov    %rbx,0x70(%rsp)
   deae2:	mov    %rdx,0x78(%rsp)
   deae7:	lea    0x80(%rsp),%rdx
   deaef:	lea    0x300(%rsp),%rsi
   deaf7:	mov    %rsi,0x80(%rsp)
   deaff:	mov    %rax,0x240(%rsp)
   deb07:	mov    %r14,0x248(%rsp)
   deb0f:	lea    0x260(%rsp),%rsi
   deb17:	mov    %rax,0x148(%rsp)
   deb1f:	mov    %r14,0x150(%rsp)
   deb27:	mov    %rbx,0xd0(%rsp)
   deb2f:	lea    0x17(%rsp),%rdi
   deb34:	mov    %rdi,0xb0(%rsp)
   deb3c:	mov    %rdx,0xb8(%rsp)
   deb44:	mov    %rsi,0xc0(%rsp)
   deb4c:	mov    %rcx,0xc8(%rsp)
   deb54:	lea    0x148(%rsp),%rcx
   deb5c:	mov    %rcx,0x90(%rsp)
   deb64:	movq   $0x0,0x98(%rsp)
   deb70:	mov    %rbx,0xa0(%rsp)
   deb78:	mov    %rbx,0xa8(%rsp)
   deb80:	movq   $0x0,0x158(%rsp)
   deb8c:	cmp    %rbx,%rax
   deb8f:	jb     e054b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x481b>
   deb95:	mov    %fs:0xffffffffffffffd0,%rax
   deb9e:	test   %rax,%rax
   deba1:	je     debab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e7b>
   deba3:	add    $0x110,%rax
   deba9:	jmp    debb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2e80>
   debab:	call   7f700 <rayon_core::registry::global_registry>
   debb0:	mov    (%rax),%rax
   debb3:	mov    0x208(%rax),%rcx
   debba:	lea    0xb0(%rsp),%rax
   debc2:	lea    0x160(%rsp),%rdi
   debca:	mov    $0x1,%r8d
   debd0:	mov    %rbx,%rsi
   debd3:	xor    %edx,%edx
   debd5:	mov    %r14,%r9
   debd8:	push   %rax
   debd9:	push   %rbx
   debda:	call   e5090 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<(usize, either::Either<core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>)>, rayon::iter::map_with::MapInitConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<tk_train::trainers::bpe::merge::Job, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::merge::Directories as core::default::Default>::default, <tk_train::trainers::bpe::merge::Batch>::prepare::{closure#0}>>>
   debdf:	add    $0x10,%rsp
   debe3:	test   %rbx,%rbx
   debe6:	je     debf2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ec2>
   debe8:	cmp    %rbx,0x158(%rsp)
   debf0:	jne    debfe <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ece>
   debf2:	movq   $0x0,0x158(%rsp)
   debfe:	cmpq   $0x0,0x148(%rsp)
   dec07:	je     dec17 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2ee7>
   dec09:	mov    0x150(%rsp),%rdi
   dec11:	call   *0x252d9(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dec17:	mov    0x170(%rsp),%rax
   dec1f:	test   %rax,%rax
   dec22:	je     dec5b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f2b>
   dec24:	mov    0x160(%rsp),%rcx
   dec2c:	test   %rcx,%rcx
   dec2f:	je     dec5b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f2b>
   dec31:	xor    %edx,%edx
   dec33:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dec40:	add    0x10(%rcx),%rdx
   dec44:	dec    %rax
   dec47:	je     dec52 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f22>
   dec49:	mov    0x18(%rcx),%rcx
   dec4d:	test   %rcx,%rcx
   dec50:	jne    dec40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f10>
   dec52:	test   %rdx,%rdx
   dec55:	jne    e04f0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x47c0>
   dec5b:	movdqu 0x160(%rsp),%xmm0
   dec64:	movdqa %xmm0,0x60(%rsp)
   dec6a:	mov    0x170(%rsp),%r15
   dec72:	mov    %r15,0x70(%rsp)
   dec77:	mov    0x60(%rsp),%rdi
   dec7c:	mov    0x68(%rsp),%rax
   dec81:	test   %rdi,%rdi
   dec84:	je     deda5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3075>
   dec8a:	mov    %rax,0x8(%rsp)
   dec8f:	dec    %r15
   dec92:	mov    %r15,%rax
   dec95:	jmp    decb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f80>
   dec97:	nopw   0x0(%rax,%rax,1)
   deca0:	lea    -0x1(%r15),%rax
   deca4:	mov    %rbx,%rdi
   deca7:	test   %rbx,%rbx
   decaa:	je     deda9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3079>
   decb0:	mov    %rax,%r15
   decb3:	mov    0x18(%rdi),%rbx
   decb7:	test   %rbx,%rbx
   decba:	je     decd0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2fa0>
   decbc:	movq   $0x0,0x20(%rbx)
   decc4:	jmp    decd9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2fa9>
   decc6:	cs nopw 0x0(%rax,%rax,1)
   decd0:	movq   $0x0,0x8(%rsp)
   decd9:	mov    (%rdi),%r12
   decdc:	movups 0x8(%rdi),%xmm0
   dece0:	movaps %xmm0,0x90(%rsp)
   dece8:	call   *0x25202(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   decee:	cmp    $0xffffffffffffffff,%r12
   decf2:	je     dedab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x307b>
   decf8:	mov    %r12,0xb0(%rsp)
   ded00:	movdqa 0x90(%rsp),%xmm0
   ded09:	lea    0xb8(%rsp),%rax
   ded11:	movdqu %xmm0,(%rax)
   ded15:	mov    0xb8(%rsp),%r14
   ded1d:	mov    0xc0(%rsp),%r13
   ded25:	mov    0x1a0(%rsp),%rax
   ded2d:	mov    0x1b0(%rsp),%rbp
   ded35:	sub    %rbp,%rax
   ded38:	cmp    %rax,%r13
   ded3b:	ja     ded7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x304d>
   ded3d:	test   %r13,%r13
   ded40:	je     ded5b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x302b>
   ded42:	imul   $0x38,%rbp,%rdi
   ded46:	add    0x1a8(%rsp),%rdi
   ded4e:	imul   $0x38,%r13,%rdx
   ded52:	mov    %r14,%rsi
   ded55:	call   *0x2519d(%rip)        # 103ef8 <memcpy@GLIBC_2.14>
   ded5b:	add    %r13,%rbp
   ded5e:	mov    %rbp,0x1b0(%rsp)
   ded66:	test   %r12,%r12
   ded69:	je     deca0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f70>
   ded6f:	mov    %r14,%rdi
   ded72:	call   *0x25178(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   ded78:	jmp    deca0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f70>
   ded7d:	mov    $0x8,%ecx
   ded82:	mov    $0x38,%r8d
   ded88:	lea    0x1a0(%rsp),%rdi
   ded90:	mov    %rbp,%rsi
   ded93:	mov    %r13,%rdx
   ded96:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   ded9b:	mov    0x1b0(%rsp),%rbp
   deda3:	jmp    ded42 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3012>
   deda5:	xor    %ebx,%ebx
   deda7:	jmp    dedb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3080>
   deda9:	xor    %ebx,%ebx
   dedab:	mov    0x8(%rsp),%rax
   dedb0:	mov    %rbx,0x60(%rsp)
   dedb5:	mov    %rax,0x68(%rsp)
   dedba:	mov    %r15,0x70(%rsp)
   dedbf:	lea    0x60(%rsp),%rdi
   dedc4:	call   e4cb0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>>
   dedc9:	movdqu 0x1a0(%rsp),%xmm0
   dedd2:	movdqa %xmm0,0x2a0(%rsp)
   deddb:	mov    0x1b0(%rsp),%rax
   dede3:	mov    %rax,0x2b0(%rsp)
   dedeb:	mov    0x308(%rsp),%rax
   dedf3:	mov    0x310(%rsp),%r13
   dedfb:	cmpb   $0x0,0x304(%rsp)
   dee03:	jne    e0568 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4838>
   dee09:	test   %rax,%rax
   dee0c:	je     dee77 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3147>
   dee0e:	mov    %rax,0x8(%rsp)
   dee13:	mov    0x2a8(%rsp),%rbx
   dee1b:	mov    0x2b0(%rsp),%r12
   dee23:	test   %r12,%r12
   dee26:	je     dee61 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3131>
   dee28:	mov    0x250c1(%rip),%r15        # 103ef0 <free@GLIBC_2.2.5>
   dee2f:	mov    %rbx,%r14
   dee32:	jmp    dee51 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3121>
   dee34:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   dee40:	mov    %r14,%rdi
   dee43:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   dee48:	add    $0x38,%r14
   dee4c:	dec    %r12
   dee4f:	je     dee61 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3131>
   dee51:	cmpq   $0x3,0x28(%r14)
   dee56:	jb     dee40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3110>
   dee58:	mov    0x18(%r14),%rdi
   dee5c:	call   *%r15
   dee5f:	jmp    dee40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3110>
   dee61:	cmpq   $0x0,0x2a0(%rsp)
   dee6a:	je     deee1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31b1>
   dee6c:	mov    %rbx,%rdi
   dee6f:	call   *0x2507b(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dee75:	jmp    deee1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31b1>
   dee77:	mov    0x2a0(%rsp),%rbx
   dee7f:	mov    0x2a8(%rsp),%rax
   dee87:	mov    %rax,0x8(%rsp)
   dee8c:	mov    0x2b0(%rsp),%r13
   dee94:	cmp    $0xffffffffffffffff,%rbx
   dee98:	je     deee1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31b1>
   dee9a:	mov    0x1e8(%rsp),%rdi
   deea2:	mov    0x1f0(%rsp),%rax
   deeaa:	test   %rdi,%rdi
   deead:	je     df116 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33e6>
   deeb3:	test   %rax,%rax
   deeb6:	je     df129 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33f9>
   deebc:	lea    (%rax,%rax,2),%rcx
   deec0:	lea    0x1b(,%rcx,4),%rcx
   deec8:	and    $0xfffffffffffffff0,%rcx
   deecc:	add    %rcx,%rax
   deecf:	cmp    $0xffffffffffffffef,%rax
   deed3:	je     df129 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33f9>
   deed9:	sub    %rcx,%rdi
   deedc:	jmp    df123 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33f3>
   deee1:	mov    0x1e8(%rsp),%rdi
   deee9:	mov    0x1f0(%rsp),%rax
   deef1:	test   %rdi,%rdi
   deef4:	je     def19 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31e9>
   deef6:	test   %rax,%rax
   deef9:	je     def2c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31fc>
   deefb:	lea    (%rax,%rax,2),%rcx
   deeff:	lea    0x1b(,%rcx,4),%rcx
   def07:	and    $0xfffffffffffffff0,%rcx
   def0b:	add    %rcx,%rax
   def0e:	cmp    $0xffffffffffffffef,%rax
   def12:	je     def2c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31fc>
   def14:	sub    %rcx,%rdi
   def17:	jmp    def26 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31f6>
   def19:	test   %rax,%rax
   def1c:	je     def2c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x31fc>
   def1e:	mov    0x1f8(%rsp),%rdi
   def26:	call   *0x24fc4(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   def2c:	mov    %r13,0x18(%rsp)
   def31:	mov    0x278(%rsp),%rbx
   def39:	mov    0x280(%rsp),%r12
   def41:	test   %r12,%r12
   def44:	je     defca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x329a>
   def4a:	xor    %ebp,%ebp
   def4c:	mov    0x24f9d(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   def53:	mov    %r12,0x30(%rsp)
   def58:	jmp    def73 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3243>
   def5a:	nopw   0x0(%rax,%rax,1)
   def60:	mov    0x30(%rsp),%r12
   def65:	mov    %r14,%rdi
   def68:	call   *%r13
   def6b:	inc    %rbp
   def6e:	cmp    %r12,%rbp
   def71:	je     defca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x329a>
   def73:	imul   $0x38,%rbp,%rcx
   def77:	lea    (%rbx,%rcx,1),%rax
   def7b:	cmpq   $0x0,(%rbx,%rcx,1)
   def80:	je     def90 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3260>
   def82:	cmpq   $0x11,0x18(%rax)
   def87:	jb     def6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x323b>
   def89:	mov    0x8(%rax),%r14
   def8d:	jmp    def65 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3235>
   def8f:	nop
   def90:	mov    0x10(%rax),%r15
   def94:	test   %r15,%r15
   def97:	je     def6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x323b>
   def99:	mov    0x8(%rax),%r14
   def9d:	lea    0x18(%r14),%r12
   defa1:	jmp    defb9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3289>
   defa3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   defb0:	add    $0x20,%r12
   defb4:	dec    %r15
   defb7:	je     def60 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3230>
   defb9:	cmpq   $0x11,(%r12)
   defbe:	jb     defb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3280>
   defc0:	mov    -0x10(%r12),%rdi
   defc5:	call   *%r13
   defc8:	jmp    defb0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3280>
   defca:	cmpq   $0x0,0x270(%rsp)
   defd3:	je     defde <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32ae>
   defd5:	mov    %rbx,%rdi
   defd8:	call   *0x24f12(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   defde:	mov    0x18(%rsp),%rbp
   defe3:	mov    0x2c0(%rsp),%rax
   defeb:	mov    0x8(%rsp),%rcx
   deff0:	mov    %rcx,0x8(%rax)
   deff4:	mov    %rbp,0x10(%rax)
   deff8:	movq   $0x1,(%rax)
   defff:	mov    0x198(%rsp),%rdi
   df007:	call   *0x24ee3(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df00d:	add    $0x3e8,%rsp
   df014:	pop    %rbx
   df015:	pop    %r12
   df017:	pop    %r13
   df019:	pop    %r14
   df01b:	pop    %r15
   df01d:	pop    %rbp
   df01e:	ret
   df01f:	mov    %rcx,%rbp
   df022:	mov    0x38(%rsp),%rdi
   df027:	call   *0x24ec3(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df02d:	test   %rbp,%rbp
   df030:	je     df0b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3389>
   df036:	xor    %r12d,%r12d
   df039:	mov    0x24eb0(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   df040:	mov    %rbp,0x40(%rsp)
   df045:	jmp    df063 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3333>
   df047:	nopw   0x0(%rax,%rax,1)
   df050:	mov    0x40(%rsp),%rbp
   df055:	mov    %rbx,%rdi
   df058:	call   *%r13
   df05b:	inc    %r12
   df05e:	cmp    %rbp,%r12
   df061:	je     df0b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3389>
   df063:	imul   $0x38,%r12,%rcx
   df067:	lea    (%r14,%rcx,1),%rax
   df06b:	cmpq   $0x0,(%r14,%rcx,1)
   df070:	je     df080 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3350>
   df072:	cmpq   $0x11,0x18(%rax)
   df077:	jb     df05b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x332b>
   df079:	mov    0x8(%rax),%rbx
   df07d:	jmp    df055 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3325>
   df07f:	nop
   df080:	mov    0x10(%rax),%r15
   df084:	test   %r15,%r15
   df087:	je     df05b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x332b>
   df089:	mov    0x8(%rax),%rbx
   df08d:	lea    0x18(%rbx),%rbp
   df091:	jmp    df0a9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3379>
   df093:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   df0a0:	add    $0x20,%rbp
   df0a4:	dec    %r15
   df0a7:	je     df050 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3320>
   df0a9:	cmpq   $0x11,0x0(%rbp)
   df0ae:	jb     df0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3370>
   df0b0:	mov    -0x10(%rbp),%rdi
   df0b4:	call   *%r13
   df0b7:	jmp    df0a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3370>
   df0b9:	cmpq   $0x0,0x90(%rsp)
   df0c2:	je     df0cd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x339d>
   df0c4:	mov    %r14,%rdi
   df0c7:	call   *0x24e23(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df0cd:	mov    0x130(%rsp),%rax
   df0d5:	cmp    $0xffffffffffffffff,%rax
   df0d9:	mov    0x18(%rsp),%rbp
   df0de:	jne    df1ea <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34ba>
   df0e4:	jmp    defe3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32b3>
   df0e9:	cmpq   $0x0,0x90(%rsp)
   df0f2:	je     df1dd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34ad>
   df0f8:	mov    %r14,%rdi
   df0fb:	jmp    df1d7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34a7>
   df100:	lea    0x23fa9(%rip),%rcx        # 1030b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x35b8>
   df107:	xor    %edi,%edi
   df109:	mov    %r13,%rsi
   df10c:	call   70c70 <core::slice::index::slice_index_fail>
   df111:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   df116:	test   %rax,%rax
   df119:	je     df129 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x33f9>
   df11b:	mov    0x1f8(%rsp),%rdi
   df123:	call   *0x24dc7(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df129:	mov    %rbx,0x130(%rsp)
   df131:	mov    %r13,0x18(%rsp)
   df136:	mov    0x278(%rsp),%rbx
   df13e:	mov    0x280(%rsp),%rbp
   df146:	test   %rbp,%rbp
   df149:	je     df1c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3499>
   df14b:	xor    %r12d,%r12d
   df14e:	mov    0x24d9b(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   df155:	mov    %rbp,0x40(%rsp)
   df15a:	jmp    df173 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3443>
   df15c:	nopl   0x0(%rax)
   df160:	mov    0x40(%rsp),%rbp
   df165:	mov    %r14,%rdi
   df168:	call   *%r13
   df16b:	inc    %r12
   df16e:	cmp    %rbp,%r12
   df171:	je     df1c9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3499>
   df173:	imul   $0x38,%r12,%rcx
   df177:	lea    (%rbx,%rcx,1),%rax
   df17b:	cmpq   $0x0,(%rbx,%rcx,1)
   df180:	je     df190 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3460>
   df182:	cmpq   $0x11,0x18(%rax)
   df187:	jb     df16b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x343b>
   df189:	mov    0x8(%rax),%r14
   df18d:	jmp    df165 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3435>
   df18f:	nop
   df190:	mov    0x10(%rax),%r15
   df194:	test   %r15,%r15
   df197:	je     df16b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x343b>
   df199:	mov    0x8(%rax),%r14
   df19d:	lea    0x18(%r14),%rbp
   df1a1:	jmp    df1b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3489>
   df1a3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   df1b0:	add    $0x20,%rbp
   df1b4:	dec    %r15
   df1b7:	je     df160 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3430>
   df1b9:	cmpq   $0x11,0x0(%rbp)
   df1be:	jb     df1b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3480>
   df1c0:	mov    -0x10(%rbp),%rdi
   df1c4:	call   *%r13
   df1c7:	jmp    df1b0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3480>
   df1c9:	cmpq   $0x0,0x270(%rsp)
   df1d2:	je     df1dd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x34ad>
   df1d4:	mov    %rbx,%rdi
   df1d7:	call   *0x24d13(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df1dd:	mov    0x18(%rsp),%rbp
   df1e2:	mov    0x130(%rsp),%rax
   df1ea:	mov    %rax,0x90(%rsp)
   df1f2:	mov    0x8(%rsp),%rax
   df1f7:	mov    %rax,0x98(%rsp)
   df1ff:	mov    %rbp,0xa0(%rsp)
   df207:	mov    0x20(%rsp),%rax
   df20c:	mov    %rax,0xa8(%rsp)
   df214:	movq   $0x0,0x148(%rsp)
   df220:	movq   $0x8,0x150(%rsp)
   df22c:	movq   $0x0,0x158(%rsp)
   df238:	mov    %rbp,0x1a0(%rsp)
   df240:	test   %rbp,%rbp
   df243:	jne    e05a4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4874>
   df249:	mov    $0x8,%edx
   df24e:	movq   $0x0,0x1c0(%rsp)
   df25a:	xor    %r12d,%r12d
   df25d:	lea    (%r12,%r12,2),%rax
   df261:	mov    %rdx,0x128(%rsp)
   df269:	lea    (%rdx,%rax,8),%rax
   df26d:	movdqu 0x90(%rsp),%xmm0
   df276:	movdqu 0xa0(%rsp),%xmm1
   df27f:	movdqa %xmm1,0x70(%rsp)
   df285:	movdqa %xmm0,0x60(%rsp)
   df28b:	lea    0x78(%rsp),%rcx
   df290:	movdqa %xmm0,0x240(%rsp)
   df299:	mov    0x70(%rsp),%rbx
   df29e:	mov    %rbx,0x1e8(%rsp)
   df2a6:	mov    %rcx,0x1d0(%rsp)
   df2ae:	mov    %rax,0x1d8(%rsp)
   df2b6:	mov    %rbp,0x1e0(%rsp)
   df2be:	lea    0x240(%rsp),%rax
   df2c6:	mov    %rax,0xb0(%rsp)
   df2ce:	movq   $0x0,0xb8(%rsp)
   df2da:	mov    %rbx,0xc0(%rsp)
   df2e2:	mov    %rbx,0xc8(%rsp)
   df2ea:	movq   $0x0,0x250(%rsp)
   df2f6:	cmp    %rbx,0x240(%rsp)
   df2fe:	jb     e0602 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48d2>
   df304:	mov    0x248(%rsp),%r14
   df30c:	mov    %r14,0x160(%rsp)
   df314:	mov    %rbx,0x168(%rsp)
   df31c:	mov    %fs:0xffffffffffffffd0,%rax
   df325:	test   %rax,%rax
   df328:	je     df332 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3602>
   df32a:	add    $0x110,%rax
   df330:	jmp    df337 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3607>
   df332:	call   7f700 <rayon_core::registry::global_registry>
   df337:	mov    (%rax),%rax
   df33a:	mov    0x208(%rax),%rcx
   df341:	lea    0x1d0(%rsp),%rax
   df349:	lea    0x270(%rsp),%rdi
   df351:	mov    $0x1,%r8d
   df357:	mov    %rbx,%rsi
   df35a:	xor    %edx,%edx
   df35c:	mov    %r14,%r9
   df35f:	mov    %rax,0x338(%rsp)
   df367:	push   %rax
   df368:	push   %rbx
   df369:	call   e72e0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::vec::DrainProducer<tk_train::trainers::bpe::merge::Job>, rayon::iter::map::MapConsumer<rayon::iter::collect::consumer::CollectConsumer<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>, <tk_train::trainers::bpe::BpeTrainer>::merge_next::{closure#0}>>>
   df36e:	add    $0x10,%rsp
   df372:	lea    0xb0(%rsp),%rdi
   df37a:	call   e7150 <core::ptr::drop_glue::<rayon::vec::Drain<tk_train::trainers::bpe::merge::Job>>>
   df37f:	mov    0x248(%rsp),%rbx
   df387:	mov    0x250(%rsp),%r15
   df38f:	test   %r15,%r15
   df392:	je     df3c1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3691>
   df394:	mov    0x24b55(%rip),%r13        # 103ef0 <free@GLIBC_2.2.5>
   df39b:	mov    %rbx,%r14
   df39e:	jmp    df3b1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3681>
   df3a0:	mov    %r14,%rdi
   df3a3:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   df3a8:	add    $0x38,%r14
   df3ac:	dec    %r15
   df3af:	je     df3c1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3691>
   df3b1:	cmpq   $0x3,0x28(%r14)
   df3b6:	jb     df3a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3670>
   df3b8:	mov    0x18(%r14),%rdi
   df3bc:	call   *%r13
   df3bf:	jmp    df3a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3670>
   df3c1:	cmpq   $0x0,0x240(%rsp)
   df3ca:	je     df3d5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x36a5>
   df3cc:	mov    %rbx,%rdi
   df3cf:	call   *0x24b1b(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df3d5:	mov    0x280(%rsp),%r15
   df3dd:	mov    %r15,0x1d0(%rsp)
   df3e5:	cmp    %rbp,%r15
   df3e8:	jne    e061f <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x48ef>
   df3ee:	mov    %r12,%r15
   df3f1:	mov    0x30(%rsp),%rax
   df3f6:	mov    0x18(%rax),%rcx
   df3fa:	mov    %rcx,0x2d0(%rsp)
   df402:	mov    0x20(%rax),%rax
   df406:	mov    %rax,0x120(%rsp)
   df40e:	mov    $0x8,%eax
   df413:	mov    %rax,0x2d8(%rsp)
   df41b:	add    %rbp,%r15
   df41e:	mov    0x128(%rsp),%rbx
   df426:	je     df508 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37d8>
   df42c:	lea    (%r12,%rbp,1),%rcx
   df430:	dec    %rcx
   df433:	mov    %r15d,%eax
   df436:	and    $0x3,%eax
   df439:	cmp    $0x3,%rcx
   df43d:	jae    df446 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3716>
   df43f:	xor    %ecx,%ecx
   df441:	xor    %r14d,%r14d
   df444:	jmp    df481 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3751>
   df446:	mov    %r15,%rdx
   df449:	and    $0xfffffffffffffffc,%rdx
   df44d:	lea    0x58(%rbx),%rsi
   df451:	xor    %ecx,%ecx
   df453:	xor    %r14d,%r14d
   df456:	cs nopw 0x0(%rax,%rax,1)
   df460:	add    -0x48(%rsi),%r14
   df464:	add    -0x30(%rsi),%r14
   df468:	add    -0x18(%rsi),%r14
   df46c:	add    (%rsi),%r14
   df46f:	add    $0x4,%rcx
   df473:	add    $0x60,%rsi
   df477:	cmp    %rcx,%rdx
   df47a:	jne    df460 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3730>
   df47c:	test   %rax,%rax
   df47f:	je     df4ad <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x377d>
   df481:	lea    (%rcx,%rcx,2),%rcx
   df485:	lea    (%rbx,%rcx,8),%rcx
   df489:	add    $0x10,%rcx
   df48d:	shl    $0x3,%eax
   df490:	lea    (%rax,%rax,2),%rax
   df494:	xor    %edx,%edx
   df496:	cs nopw 0x0(%rax,%rax,1)
   df4a0:	add    (%rcx,%rdx,1),%r14
   df4a4:	add    $0x18,%rdx
   df4a8:	cmp    %rdx,%rax
   df4ab:	jne    df4a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3770>
   df4ad:	lea    0x0(,%r14,8),%rax
   df4b5:	lea    (%rax,%rax,4),%rsi
   df4b9:	movabs $0x333333333333333,%rax
   df4c3:	cmp    %rax,%r14
   df4c6:	jbe    df4d4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37a4>
   df4c8:	xor    %edi,%edi
   df4ca:	call   692b8 <alloc::raw_vec::handle_error>
   df4cf:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   df4d4:	test   %rsi,%rsi
   df4d7:	je     df508 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37d8>
   df4d9:	mov    %rsi,%rbx
   df4dc:	mov    $0x8,%esi
   df4e1:	mov    %rbx,%rdi
   df4e4:	call   68f50 <__rustc::__rust_alloc>
   df4e9:	mov    $0x8,%edi
   df4ee:	mov    %rbx,%rsi
   df4f1:	mov    0x128(%rsp),%rbx
   df4f9:	mov    %rax,0x2d8(%rsp)
   df501:	test   %rax,%rax
   df504:	jne    df50b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x37db>
   df506:	jmp    df4ca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x379a>
   df508:	xor    %r14d,%r14d
   df50b:	shl    $0x3,%r15
   df50f:	lea    (%r15,%r15,2),%rax
   df513:	lea    (%rbx,%rax,1),%rdx
   df517:	mov    %r14,0x90(%rsp)
   df51f:	mov    0x2d8(%rsp),%rcx
   df527:	mov    %rcx,0x98(%rsp)
   df52f:	movq   $0x0,0xa0(%rsp)
   df53b:	mov    %rbx,0xb0(%rsp)
   df543:	mov    %rbx,0xb8(%rsp)
   df54b:	mov    0x1c0(%rsp),%rcx
   df553:	mov    %rcx,0xc0(%rsp)
   df55b:	mov    %rdx,0x190(%rsp)
   df563:	mov    %rdx,0xc8(%rsp)
   df56b:	movq   $0x0,0xd0(%rsp)
   df577:	movq   $0x0,0xf0(%rsp)
   df583:	mov    %rbx,%r12
   df586:	add    %rax,%rbx
   df589:	add    $0xffffffffffffffe8,%rbx
   df58d:	xor    %r14d,%r14d
   df590:	lea    0x24281(%rip),%rax        # 103818 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d20>
   df597:	mov    %rax,0x328(%rsp)
   df59f:	xor    %ecx,%ecx
   df5a1:	mov    %rbx,0x180(%rsp)
   df5a9:	mov    %r14,0x2b8(%rsp)
   df5b1:	test   %rcx,%rcx
   df5b4:	sete   %al
   df5b7:	mov    %rbx,%rdx
   df5ba:	sub    %r12,%rdx
   df5bd:	mov    %rdx,0x188(%rsp)
   df5c5:	lea    0x18(%r12),%rdx
   df5ca:	mov    %rdx,0x18(%rsp)
   df5cf:	mov    %rcx,%r14
   df5d2:	mov    0x38(%rsp),%rcx
   df5d7:	jmp    df62e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38fe>
   df5d9:	nopl   0x0(%rax)
   df5e0:	mov    0x20(%rsp),%rax
   df5e5:	lea    (%rax,%rax,8),%rax
   df5e9:	lea    (%r14,%rax,8),%r15
   df5ed:	mov    %r14,0xd0(%rsp)
   df5f5:	mov    %r14,0xd8(%rsp)
   df5fd:	mov    0x130(%rsp),%rcx
   df605:	mov    %rcx,0xe0(%rsp)
   df60d:	mov    %r15,0xe8(%rsp)
   df615:	addq   $0xffffffffffffffe8,0x188(%rsp)
   df61e:	addq   $0x18,0x18(%rsp)
   df624:	xor    %eax,%eax
   df626:	mov    %r14,%rbp
   df629:	mov    0x40(%rsp),%r12
   df62e:	test   $0x1,%al
   df630:	je     df650 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3920>
   df632:	mov    %r14,0x8(%rsp)
   df637:	cmp    0x190(%rsp),%r12
   df63f:	jne    df695 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3965>
   df641:	jmp    dfc4b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f1b>
   df646:	cs nopw 0x0(%rax,%rax,1)
   df650:	cmp    %r15,%rbp
   df653:	jne    df7b9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a89>
   df659:	test   %r14,%r14
   df65c:	je     df672 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3942>
   df65e:	test   %rcx,%rcx
   df661:	je     df672 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3942>
   df663:	mov    %r14,%rdi
   df666:	mov    %rcx,%r14
   df669:	call   *0x24881(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df66f:	mov    %r14,%rcx
   df672:	movq   $0x0,0xd0(%rsp)
   df67e:	movq   $0x0,0x8(%rsp)
   df687:	cmp    0x190(%rsp),%r12
   df68f:	je     dfc4b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f1b>
   df695:	lea    0x18(%r12),%rax
   df69a:	mov    %rax,0xb8(%rsp)
   df6a2:	mov    (%r12),%rdx
   df6a6:	cmp    $0xffffffffffffffff,%rdx
   df6aa:	je     dfc08 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ed8>
   df6b0:	mov    %rdx,0x130(%rsp)
   df6b8:	mov    %rax,0x40(%rsp)
   df6bd:	mov    0x8(%r12),%r14
   df6c2:	mov    0x10(%r12),%rax
   df6c7:	mov    %rax,0x20(%rsp)
   df6cc:	cmpq   $0x0,0x8(%rsp)
   df6d2:	je     df5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38b0>
   df6d8:	mov    %rcx,0x38(%rsp)
   df6dd:	mov    %r14,0x58(%rsp)
   df6e2:	sub    %rbp,%r15
   df6e5:	je     df790 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a60>
   df6eb:	shr    $0x3,%r15
   df6ef:	movabs $0x8e38e38e38e38e39,%rax
   df6f9:	imul   %rax,%r15
   df6fd:	xor    %ebx,%ebx
   df6ff:	jmp    df721 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39f1>
   df701:	data16 data16 data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   df710:	mov    %r12,%rdi
   df713:	call   *0x247d7(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df719:	inc    %rbx
   df71c:	cmp    %r15,%rbx
   df71f:	je     df790 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a60>
   df721:	lea    (%rbx,%rbx,8),%rcx
   df725:	lea    0x0(,%rcx,8),%rax
   df72d:	add    %rbp,%rax
   df730:	cmpq   $0x0,0x0(%rbp,%rcx,8)
   df736:	je     df750 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a20>
   df738:	cmpq   $0x11,0x18(%rax)
   df73d:	jb     df719 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39e9>
   df73f:	mov    0x8(%rax),%r12
   df743:	jmp    df710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39e0>
   df745:	data16 cs nopw 0x0(%rax,%rax,1)
   df750:	mov    0x10(%rax),%r14
   df754:	test   %r14,%r14
   df757:	je     df719 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39e9>
   df759:	mov    0x8(%rax),%r12
   df75d:	lea    0x18(%r12),%r13
   df762:	jmp    df779 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a49>
   df764:	data16 data16 cs nopw 0x0(%rax,%rax,1)
   df770:	add    $0x20,%r13
   df774:	dec    %r14
   df777:	je     df710 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x39e0>
   df779:	cmpq   $0x11,0x0(%r13)
   df77e:	jb     df770 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a40>
   df780:	mov    -0x10(%r13),%rdi
   df784:	call   *0x24766(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df78a:	jmp    df770 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3a40>
   df78c:	nopl   0x0(%rax)
   df790:	cmpq   $0x0,0x38(%rsp)
   df796:	mov    0x180(%rsp),%rbx
   df79e:	mov    0x58(%rsp),%r14
   df7a3:	je     df5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38b0>
   df7a9:	mov    0x8(%rsp),%rdi
   df7ae:	call   *0x2473c(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   df7b4:	jmp    df5e0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x38b0>
   df7b9:	mov    %rcx,0x38(%rsp)
   df7be:	mov    %r15,0x188(%rsp)
   df7c6:	lea    0x48(%rbp),%r8
   df7ca:	mov    %r8,0xd8(%rsp)
   df7d2:	mov    0x0(%rbp),%rsi
   df7d6:	mov    0x8(%rbp),%rdx
   df7da:	mov    0x10(%rbp),%r13
   df7de:	mov    0x18(%rbp),%r9
   df7e2:	movdqu 0x20(%rbp),%xmm0
   df7e7:	movdqu 0x30(%rbp),%xmm1
   df7ec:	mov    0x40(%rbp),%rax
   df7f0:	mov    %rax,0x80(%rsp)
   df7f8:	movdqa %xmm1,0x70(%rsp)
   df7fe:	movdqa %xmm0,0x60(%rsp)
   df804:	lea    0x1f0(%rsp),%rcx
   df80c:	mov    %rax,0x20(%rcx)
   df810:	movdqu %xmm1,0x10(%rcx)
   df815:	movdqu %xmm0,(%rcx)
   df819:	mov    %rsi,0x1d0(%rsp)
   df821:	mov    %rdx,0x318(%rsp)
   df829:	mov    %rdx,0x1d8(%rsp)
   df831:	mov    %r13,0x1e0(%rsp)
   df839:	mov    %r9,0x1e8(%rsp)
   df841:	mov    0x200(%rsp),%r10
   df849:	test   %r10,%r10
   df84c:	mov    %r14,0x58(%rsp)
   df851:	mov    %r8,0x18(%rsp)
   df856:	mov    %r9,0x3c8(%rsp)
   df85e:	mov    %r10,0x8(%rsp)
   df863:	je     df8df <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3baf>
   df865:	cmpq   $0x0,0x120(%rsp)
   df86e:	je     e09ee <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cbe>
   df874:	mov    0x1f0(%rsp),%eax
   df87b:	mov    0x1f4(%rsp),%ecx
   df882:	movabs $0xbe5466cf34e90c6c,%rdx
   df88c:	xor    %rdx,%rax
   df88f:	movabs $0x5851f42d4c957f2d,%rdi
   df899:	mul    %rdi
   df89c:	xor    %rdx,%rcx
   df89f:	xor    %rcx,%rax
   df8a2:	mul    %rdi
   df8a5:	mov    %rax,%rcx
   df8a8:	xor    %rdx,%rcx
   df8ab:	mov    %rcx,%rax
   df8ae:	movabs $0x452821e638d01377,%rdx
   df8b8:	mul    %rdx
   df8bb:	xor    %rax,%rdx
   df8be:	rol    %cl,%rdx
   df8c1:	mov    %rdx,%rax
   df8c4:	mov    0x120(%rsp),%rcx
   df8cc:	or     %rcx,%rax
   df8cf:	shr    $0x20,%rax
   df8d3:	je     df903 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bd3>
   df8d5:	mov    %rdx,%rax
   df8d8:	xor    %edx,%edx
   df8da:	div    %rcx
   df8dd:	jmp    df909 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bd9>
   df8df:	sub    $0x1,%rsi
   df8e3:	setb   %cl
   df8e6:	cmovb  %r9,%rsi
   df8ea:	test   %rsi,%rsi
   df8ed:	je     df998 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c68>
   df8f3:	cmpq   $0x0,0x120(%rsp)
   df8fc:	jne    df921 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3bf1>
   df8fe:	jmp    e09ee <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cbe>
   df903:	mov    %edx,%eax
   df905:	xor    %edx,%edx
   df907:	div    %ecx
   df909:	mov    %rdx,0x2c8(%rsp)
   df911:	sub    $0x1,%rsi
   df915:	setb   %cl
   df918:	cmovb  %r9,%rsi
   df91c:	test   %rsi,%rsi
   df91f:	je     df998 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c68>
   df921:	mov    %cl,0x2f(%rsp)
   df925:	mov    0x1f8(%rsp),%eax
   df92c:	mov    0x1fc(%rsp),%ebp
   df933:	mov    %rax,0x20(%rsp)
   df938:	movabs $0xbe5466cf34e90c6c,%rcx
   df942:	xor    %rcx,%rax
   df945:	movabs $0x5851f42d4c957f2d,%rsi
   df94f:	mul    %rsi
   df952:	mov    %rbp,%rcx
   df955:	xor    %rdx,%rcx
   df958:	xor    %rcx,%rax
   df95b:	mul    %rsi
   df95e:	mov    %rax,%rcx
   df961:	xor    %rdx,%rcx
   df964:	mov    %rcx,%rax
   df967:	movabs $0x452821e638d01377,%rdx
   df971:	mul    %rdx
   df974:	xor    %rax,%rdx
   df977:	rol    %cl,%rdx
   df97a:	mov    %rdx,%rax
   df97d:	mov    0x120(%rsp),%rcx
   df985:	or     %rcx,%rax
   df988:	shr    $0x20,%rax
   df98c:	je     df9b4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c84>
   df98e:	mov    %rdx,%rax
   df991:	xor    %edx,%edx
   df993:	div    %rcx
   df996:	jmp    df9ba <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c8a>
   df998:	mov    %cl,0x2f(%rsp)
   df99c:	mov    0x1f8(%rsp),%eax
   df9a3:	mov    %rax,0x20(%rsp)
   df9a8:	mov    0x1fc(%rsp),%ebp
   df9af:	xor    %r15d,%r15d
   df9b2:	jmp    df9c5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3c95>
   df9b4:	mov    %edx,%eax
   df9b6:	xor    %edx,%edx
   df9b8:	div    %ecx
   df9ba:	mov    %rdx,0x320(%rsp)
   df9c2:	mov    $0x1,%r15b
   df9c5:	movsd  0x1f0(%rsp),%xmm0
   df9ce:	movaps %xmm0,0x130(%rsp)
   df9d6:	movdqu 0x208(%rsp),%xmm0
   df9df:	movdqa %xmm0,0x40(%rsp)
   df9e5:	mov    0x2b8(%rsp),%rax
   df9ed:	cmp    0x90(%rsp),%rax
   df9f5:	jne    dfa14 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ce4>
   df9f7:	lea    0x90(%rsp),%rdi
   df9ff:	call   e80e0 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::merge::Change<()>>>::grow_one>
   dfa04:	mov    0x98(%rsp),%rax
   dfa0c:	mov    %rax,0x2d8(%rsp)
   dfa14:	mov    0x2b8(%rsp),%rax
   dfa1c:	lea    0x1(%rax),%r14
   dfa20:	lea    (%rax,%rax,4),%rax
   dfa24:	mov    0x2d8(%rsp),%rcx
   dfa2c:	movaps 0x130(%rsp),%xmm0
   dfa34:	movlps %xmm0,(%rcx,%rax,8)
   dfa38:	mov    0x20(%rsp),%rdx
   dfa3d:	mov    %edx,0x8(%rcx,%rax,8)
   dfa41:	mov    %ebp,0xc(%rcx,%rax,8)
   dfa45:	mov    0x8(%rsp),%rdx
   dfa4a:	mov    %rdx,0x10(%rcx,%rax,8)
   dfa4f:	movdqa 0x40(%rsp),%xmm0
   dfa55:	movdqu %xmm0,0x18(%rcx,%rax,8)
   dfa5b:	mov    %r14,0xa0(%rsp)
   dfa63:	test   %rdx,%rdx
   dfa66:	je     dfb22 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3df2>
   dfa6c:	mov    0x2c8(%rsp),%rcx
   dfa74:	cmp    0x320(%rsp),%rcx
   dfa7c:	sete   %al
   dfa7f:	test   %al,%r15b
   dfa82:	je     dfae8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3db8>
   dfa84:	cmp    0x120(%rsp),%rcx
   dfa8c:	jae    e0a0d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cdd>
   dfa92:	imul   $0x70,%rcx,%rdi
   dfa96:	add    0x2d0(%rsp),%rdi
   dfa9e:	movdqu 0x1d0(%rsp),%xmm0
   dfaa7:	movdqu 0x1e0(%rsp),%xmm1
   dfab0:	lea    0x68(%rsp),%rax
   dfab5:	movdqu %xmm1,0x10(%rax)
   dfaba:	movdqu %xmm0,(%rax)
   dfabe:	movq   $0x1,0x60(%rsp)
   dfac7:	mov    0x2b8(%rsp),%rsi
   dfacf:	mov    $0x1,%edx
   dfad4:	lea    0x60(%rsp),%rcx
   dfad9:	call   e8150 <<tk_train::trainers::bpe::index::Shard>::push>
   dfade:	mov    0x18(%rsp),%rbp
   dfae3:	jmp    dfbf6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ec6>
   dfae8:	cmp    0x120(%rsp),%rcx
   dfaf0:	jae    e0a1c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4cec>
   dfaf6:	imul   $0x70,%rcx,%rdi
   dfafa:	add    0x2d0(%rsp),%rdi
   dfb02:	movq   $0x0,0x60(%rsp)
   dfb0b:	mov    0x2b8(%rsp),%rsi
   dfb13:	mov    $0x1,%edx
   dfb18:	lea    0x60(%rsp),%rcx
   dfb1d:	call   e8150 <<tk_train::trainers::bpe::index::Shard>::push>
   dfb22:	test   %r15b,%r15b
   dfb25:	je     dfb8d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e5d>
   dfb27:	mov    0x320(%rsp),%rax
   dfb2f:	cmp    0x120(%rsp),%rax
   dfb37:	mov    0x18(%rsp),%rbp
   dfb3c:	jae    e09fc <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ccc>
   dfb42:	imul   $0x70,%rax,%rdi
   dfb46:	add    0x2d0(%rsp),%rdi
   dfb4e:	movdqu 0x1d0(%rsp),%xmm0
   dfb57:	movdqu 0x1e0(%rsp),%xmm1
   dfb60:	lea    0x68(%rsp),%rax
   dfb65:	movdqu %xmm1,0x10(%rax)
   dfb6a:	movdqu %xmm0,(%rax)
   dfb6e:	movq   $0x1,0x60(%rsp)
   dfb77:	mov    0x2b8(%rsp),%rsi
   dfb7f:	xor    %edx,%edx
   dfb81:	lea    0x60(%rsp),%rcx
   dfb86:	call   e8150 <<tk_train::trainers::bpe::index::Shard>::push>
   dfb8b:	jmp    dfbf6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ec6>
   dfb8d:	cmpb   $0x0,0x2f(%rsp)
   dfb92:	mov    0x18(%rsp),%rbp
   dfb97:	je     dfbca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e9a>
   dfb99:	test   %r13,%r13
   dfb9c:	je     dfbf6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ec6>
   dfb9e:	mov    %rbx,%r15
   dfba1:	mov    0x318(%rsp),%rax
   dfba9:	lea    0x18(%rax),%rbx
   dfbad:	jmp    dfbb8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e88>
   dfbaf:	add    $0x20,%rbx
   dfbb3:	dec    %r13
   dfbb6:	je     dfbe5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3eb5>
   dfbb8:	cmpq   $0x11,(%rbx)
   dfbbc:	jb     dfbaf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e7f>
   dfbbe:	mov    -0x10(%rbx),%rdi
   dfbc2:	call   *0x24328(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dfbc8:	jmp    dfbaf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3e7f>
   dfbca:	cmpq   $0x11,0x3c8(%rsp)
   dfbd3:	jb     dfbf6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ec6>
   dfbd5:	mov    0x318(%rsp),%rdi
   dfbdd:	call   *0x2430d(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dfbe3:	jmp    dfbf6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3ec6>
   dfbe5:	mov    0x318(%rsp),%rdi
   dfbed:	call   *0x242fd(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dfbf3:	mov    %r15,%rbx
   dfbf6:	mov    0x188(%rsp),%r15
   dfbfe:	mov    0x58(%rsp),%rcx
   dfc03:	jmp    df5a9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3879>
   dfc08:	cmp    %rax,0x190(%rsp)
   dfc10:	je     dfc4b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f1b>
   dfc12:	movabs $0xaaaaaaaaaaaaaaab,%rcx
   dfc1c:	mov    0x188(%rsp),%rax
   dfc24:	mul    %rcx
   dfc27:	mov    %rdx,%rbx
   dfc2a:	shr    $0x4,%rbx
   dfc2e:	xchg   %ax,%ax
   dfc30:	mov    0x18(%rsp),%r14
   dfc35:	mov    %r14,%rdi
   dfc38:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   dfc3d:	add    $0x18,%r14
   dfc41:	mov    %r14,0x18(%rsp)
   dfc46:	dec    %rbx
   dfc49:	jne    dfc30 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f00>
   dfc4b:	cmpq   $0x0,0x1c0(%rsp)
   dfc54:	je     dfc64 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x3f34>
   dfc56:	mov    0x128(%rsp),%rdi
   dfc5e:	call   *0x2428c(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dfc64:	lea    0xd0(%rsp),%rdi
   dfc6c:	call   e82e0 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>>
   dfc71:	lea    0xf0(%rsp),%rdi
   dfc79:	call   e82e0 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>>
   dfc7e:	movdqu 0x90(%rsp),%xmm0
   dfc87:	movdqa %xmm0,0x160(%rsp)
   dfc90:	mov    0xa0(%rsp),%rax
   dfc98:	mov    %rax,0x170(%rsp)
   dfca0:	mov    0x30(%rsp),%rcx
   dfca5:	movzbl 0x30(%rcx),%eax
   dfca9:	mov    %al,0x17(%rsp)
   dfcad:	mov    0x28(%rcx),%rax
   dfcb1:	mov    %rax,0x2a0(%rsp)
   dfcb9:	movl   $0x0,0x148(%rsp)
   dfcc4:	movb   $0x0,0x14c(%rsp)
   dfccc:	movq   $0x0,0x150(%rsp)
   dfcd8:	movq   $0x0,0x240(%rsp)
   dfce4:	movq   $0x8,0x248(%rsp)
   dfcf0:	movq   $0x0,0x250(%rsp)
   dfcfc:	movb   $0x0,0x300(%rsp)
   dfd04:	mov    0x2d0(%rsp),%r15
   dfd0c:	mov    %r15,0xb0(%rsp)
   dfd14:	mov    0x120(%rsp),%r14
   dfd1c:	mov    %r14,0xb8(%rsp)
   dfd24:	lea    0x160(%rsp),%rax
   dfd2c:	mov    %rax,0xc0(%rsp)
   dfd34:	lea    0x17(%rsp),%rcx
   dfd39:	mov    %rcx,0xc8(%rsp)
   dfd41:	lea    0x2a0(%rsp),%rdx
   dfd49:	mov    %rdx,0xd0(%rsp)
   dfd51:	lea    0xd8(%rsp),%rsi
   dfd59:	lea    0x148(%rsp),%rdi
   dfd61:	mov    %rdi,0xd8(%rsp)
   dfd69:	mov    %r15,0x60(%rsp)
   dfd6e:	mov    %r14,0x68(%rsp)
   dfd73:	lea    0x70(%rsp),%rdi
   dfd78:	mov    %rax,0x70(%rsp)
   dfd7d:	mov    %rcx,0x78(%rsp)
   dfd82:	mov    %rdx,0x80(%rsp)
   dfd8a:	mov    %r15,0x1a0(%rsp)
   dfd92:	mov    %r14,0x1a8(%rsp)
   dfd9a:	lea    0x1b0(%rsp),%rax
   dfda2:	mov    %r14,0x1f0(%rsp)
   dfdaa:	lea    0x300(%rsp),%rcx
   dfdb2:	mov    %rcx,0x1d0(%rsp)
   dfdba:	mov    %rsi,0x1d8(%rsp)
   dfdc2:	mov    %rdi,0x1e0(%rsp)
   dfdca:	mov    %rax,0x1e8(%rsp)
   dfdd2:	mov    %fs:0xffffffffffffffd0,%rax
   dfddb:	test   %rax,%rax
   dfdde:	je     dfde8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40b8>
   dfde0:	add    $0x110,%rax
   dfde6:	jmp    dfded <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x40bd>
   dfde8:	call   7f700 <rayon_core::registry::global_registry>
   dfded:	mov    (%rax),%rax
   dfdf0:	mov    0x208(%rax),%rcx
   dfdf7:	xor    %eax,%eax
   dfdf9:	cmp    $0xffffffffffffffff,%r14
   dfdfd:	sete   %al
   dfe00:	cmp    %rcx,%rax
   dfe03:	cmova  %rax,%rcx
   dfe07:	lea    0x90(%rsp),%rbx
   dfe0f:	mov    $0x1,%r8d
   dfe15:	mov    %rbx,%rdi
   dfe18:	mov    %r14,%rsi
   dfe1b:	xor    %edx,%edx
   dfe1d:	mov    %r15,%r9
   dfe20:	lea    0x1d0(%rsp),%rax
   dfe28:	push   %rax
   dfe29:	push   %r14
   dfe2b:	call   e83f0 <rayon::iter::plumbing::bridge_producer_consumer::helper::<rayon::slice::IterMutProducer<tk_train::trainers::bpe::index::Shard>, rayon::iter::filter::FilterConsumer<rayon::iter::map::MapConsumer<rayon::iter::map::MapConsumer<rayon::iter::while_some::WhileSomeConsumer<rayon::iter::extend::ListVecConsumer>, <core::result::Result<_, _> as rayon::iter::FromParallelIterator<core::result::Result<_, _>>>::from_par_iter::ok<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>, alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>::{closure#0}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#1}>, <tk_train::trainers::bpe::index::PairIndex>::commit::{closure#0}>>>
   dfe30:	add    $0x10,%rsp
   dfe34:	mov    0xa0(%rsp),%rax
   dfe3c:	test   %rax,%rax
   dfe3f:	je     dfe6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x413b>
   dfe41:	mov    0x90(%rsp),%rcx
   dfe49:	test   %rcx,%rcx
   dfe4c:	je     dfe6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x413b>
   dfe4e:	xor    %edx,%edx
   dfe50:	add    0x10(%rcx),%rdx
   dfe54:	dec    %rax
   dfe57:	je     dfe62 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4132>
   dfe59:	mov    0x18(%rcx),%rcx
   dfe5d:	test   %rcx,%rcx
   dfe60:	jne    dfe50 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4120>
   dfe62:	test   %rdx,%rdx
   dfe65:	jne    e0706 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x49d6>
   dfe6b:	movdqu 0x90(%rsp),%xmm0
   dfe74:	movdqa %xmm0,0x1d0(%rsp)
   dfe7d:	mov    0xa0(%rsp),%r12
   dfe85:	mov    %r12,0x1e0(%rsp)
   dfe8d:	mov    0x1d0(%rsp),%rdi
   dfe95:	mov    0x1d8(%rsp),%rax
   dfe9d:	test   %rdi,%rdi
   dfea0:	je     dffd4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42a4>
   dfea6:	mov    %rax,0x8(%rsp)
   dfeab:	dec    %r12
   dfeae:	mov    %r12,%rax
   dfeb1:	jmp    dfed1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x41a1>
   dfeb3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   dfec0:	lea    -0x1(%r12),%rax
   dfec5:	mov    %rbx,%rdi
   dfec8:	test   %rbx,%rbx
   dfecb:	je     dffd8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42a8>
   dfed1:	mov    %rax,%r12
   dfed4:	mov    0x18(%rdi),%rbx
   dfed8:	test   %rbx,%rbx
   dfedb:	je     dfef0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x41c0>
   dfedd:	movq   $0x0,0x20(%rbx)
   dfee5:	jmp    dfef9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x41c9>
   dfee7:	nopw   0x0(%rax,%rax,1)
   dfef0:	movq   $0x0,0x8(%rsp)
   dfef9:	mov    (%rdi),%r15
   dfefc:	movdqu 0x8(%rdi),%xmm0
   dff01:	movdqa %xmm0,0x60(%rsp)
   dff07:	call   *0x23fe3(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dff0d:	cmp    $0xffffffffffffffff,%r15
   dff11:	je     dffda <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42aa>
   dff17:	mov    %r15,0xb0(%rsp)
   dff1f:	movdqa 0x60(%rsp),%xmm0
   dff25:	lea    0xb8(%rsp),%rax
   dff2d:	movdqu %xmm0,(%rax)
   dff31:	mov    0xb8(%rsp),%r14
   dff39:	mov    0xc0(%rsp),%r13
   dff41:	mov    0x240(%rsp),%rax
   dff49:	mov    0x250(%rsp),%rbp
   dff51:	sub    %rbp,%rax
   dff54:	cmp    %rax,%r13
   dff57:	ja     dffac <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x427c>
   dff59:	test   %r13,%r13
   dff5c:	je     dff8a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x425a>
   dff5e:	lea    0x0(,%rbp,2),%rdi
   dff66:	add    %rbp,%rdi
   dff69:	shl    $0x3,%rdi
   dff6d:	add    0x248(%rsp),%rdi
   dff75:	lea    0x0(,%r13,8),%rax
   dff7d:	lea    (%rax,%rax,2),%rdx
   dff81:	mov    %r14,%rsi
   dff84:	call   *0x23f6e(%rip)        # 103ef8 <memcpy@GLIBC_2.14>
   dff8a:	add    %r13,%rbp
   dff8d:	mov    %rbp,0x250(%rsp)
   dff95:	test   %r15,%r15
   dff98:	je     dfec0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4190>
   dff9e:	mov    %r14,%rdi
   dffa1:	call   *0x23f49(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   dffa7:	jmp    dfec0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4190>
   dffac:	mov    $0x8,%ecx
   dffb1:	mov    $0x18,%r8d
   dffb7:	lea    0x240(%rsp),%rdi
   dffbf:	mov    %rbp,%rsi
   dffc2:	mov    %r13,%rdx
   dffc5:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   dffca:	mov    0x250(%rsp),%rbp
   dffd2:	jmp    dff5e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x422e>
   dffd4:	xor    %ebx,%ebx
   dffd6:	jmp    dffdf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x42af>
   dffd8:	xor    %ebx,%ebx
   dffda:	mov    0x8(%rsp),%rax
   dffdf:	mov    %rbx,0x1d0(%rsp)
   dffe7:	mov    %rax,0x1d8(%rsp)
   dffef:	mov    %r12,0x1e0(%rsp)
   dfff7:	lea    0x1d0(%rsp),%rdi
   dffff:	call   eb970 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0004:	movdqu 0x240(%rsp),%xmm0
   e000d:	movdqa %xmm0,0x270(%rsp)
   e0016:	mov    0x250(%rsp),%rax
   e001e:	mov    %rax,0x280(%rsp)
   e0026:	mov    0x150(%rsp),%rax
   e002e:	mov    0x158(%rsp),%rbp
   e0036:	cmpb   $0x0,0x14c(%rsp)
   e003e:	jne    e074d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a1d>
   e0044:	test   %rax,%rax
   e0047:	je     e00b7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4387>
   e0049:	mov    %rax,0x8(%rsp)
   e004e:	mov    0x278(%rsp),%rbx
   e0056:	mov    0x280(%rsp),%r15
   e005e:	test   %r15,%r15
   e0061:	je     e0081 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4351>
   e0063:	mov    %rbx,%r14
   e0066:	cs nopw 0x0(%rax,%rax,1)
   e0070:	mov    %r14,%rdi
   e0073:	call   dae80 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   e0078:	add    $0x18,%r14
   e007c:	dec    %r15
   e007f:	jne    e0070 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4340>
   e0081:	cmpq   $0x0,0x270(%rsp)
   e008a:	je     e0095 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4365>
   e008c:	mov    %rbx,%rdi
   e008f:	call   *0x23e5b(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0095:	cmpq   $0x0,0x160(%rsp)
   e009e:	je     defe3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32b3>
   e00a4:	mov    0x168(%rsp),%rdi
   e00ac:	call   *0x23e3e(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e00b2:	jmp    defe3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32b3>
   e00b7:	mov    0x270(%rsp),%rax
   e00bf:	mov    0x278(%rsp),%rdx
   e00c7:	mov    0x280(%rsp),%rbp
   e00cf:	cmp    $0xffffffffffffffff,%rax
   e00d3:	je     e03ab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x467b>
   e00d9:	lea    0x0(,%rbp,2),%rcx
   e00e1:	add    %rbp,%rcx
   e00e4:	lea    (%rdx,%rcx,8),%rcx
   e00e8:	mov    %rdx,0xb0(%rsp)
   e00f0:	mov    %rdx,0xb8(%rsp)
   e00f8:	mov    %rax,0xc0(%rsp)
   e0100:	mov    %rcx,0xc8(%rsp)
   e0108:	lea    0xd0(%rsp),%r14
   e0110:	movq   $0x0,0xd0(%rsp)
   e011c:	movq   $0x0,0xf0(%rsp)
   e0128:	lea    0x1d8(%rsp),%r15
   e0130:	lea    0x1d0(%rsp),%r13
   e0138:	lea    0xb0(%rsp),%rbp
   e0140:	mov    0x30(%rsp),%rbx
   e0145:	jmp    e017e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x444e>
   e0147:	nopw   0x0(%rax,%rax,1)
   e0150:	xor    %r12d,%r12d
   e0153:	lea    (%r12,%r12,2),%rdi
   e0157:	shl    $0x4,%rdi
   e015b:	movdqu (%r15),%xmm0
   e0160:	movdqu 0x10(%r15),%xmm1
   e0166:	movdqu %xmm1,0x10(%rax,%rdi,1)
   e016c:	movdqu %xmm0,(%rax,%rdi,1)
   e0171:	mov    %rcx,0x20(%rax,%rdi,1)
   e0176:	mov    %edx,0x28(%rax,%rdi,1)
   e017a:	mov    %esi,0x2c(%rax,%rdi,1)
   e017e:	mov    %r13,%rdi
   e0181:	mov    %rbp,%rsi
   e0184:	call   dacb0 <<core::iter::adapters::flatten::FlattenCompat<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>, alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>> as core::iter::traits::iterator::Iterator>::next>
   e0189:	cmpl   $0x1,0x1d0(%rsp)
   e0191:	jne    e0277 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4547>
   e0197:	mov    0x10(%rbx),%r12
   e019b:	cmp    (%rbx),%r12
   e019e:	jne    e01a8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4478>
   e01a0:	mov    %rbx,%rdi
   e01a3:	call   e1480 <<alloc::raw_vec::RawVec<tk_train::trainers::bpe::index::Candidate>>::grow_one>
   e01a8:	mov    0x8(%rbx),%rax
   e01ac:	lea    (%r12,%r12,2),%rsi
   e01b0:	shl    $0x4,%rsi
   e01b4:	movdqu (%r15),%xmm0
   e01b9:	movdqu 0x10(%r15),%xmm1
   e01bf:	movdqu 0x20(%r15),%xmm2
   e01c5:	movdqu %xmm2,0x20(%rax,%rsi,1)
   e01cb:	movdqu %xmm1,0x10(%rax,%rsi,1)
   e01d1:	movdqu %xmm0,(%rax,%rsi,1)
   e01d6:	lea    0x1(%r12),%rcx
   e01db:	mov    %rcx,0x10(%rbx)
   e01df:	mov    0x20(%rax,%rsi,1),%rcx
   e01e4:	mov    0x28(%rax,%rsi,1),%edx
   e01e8:	mov    0x2c(%rax,%rsi,1),%esi
   e01ec:	test   %r12,%r12
   e01ef:	jne    e0242 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4512>
   e01f1:	jmp    e0150 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4420>
   e01f6:	cs nopw 0x0(%rax,%rax,1)
   e0200:	test   %r9b,%r9b
   e0203:	jne    e0153 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4423>
   e0209:	lea    (%r12,%r12,2),%r9
   e020d:	shl    $0x4,%r9
   e0211:	movdqu (%r8),%xmm0
   e0216:	movdqu 0x10(%r8),%xmm1
   e021c:	movdqu 0x20(%r8),%xmm2
   e0222:	movdqu %xmm2,0x20(%rax,%r9,1)
   e0229:	movdqu %xmm1,0x10(%rax,%r9,1)
   e0230:	movdqu %xmm0,(%rax,%r9,1)
   e0236:	mov    %rdi,%r12
   e0239:	test   %rdi,%rdi
   e023c:	je     e0150 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4420>
   e0242:	lea    -0x1(%r12),%rdi
   e0247:	shr    $0x3,%rdi
   e024b:	lea    (%rdi,%rdi,2),%r9
   e024f:	shl    $0x4,%r9
   e0253:	lea    (%rax,%r9,1),%r8
   e0257:	cmp    0x20(%rax,%r9,1),%rcx
   e025c:	setbe  %r9b
   e0260:	jne    e0200 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44d0>
   e0262:	cmp    %edx,0x28(%r8)
   e0266:	setbe  %r9b
   e026a:	jne    e0200 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44d0>
   e026c:	cmp    %esi,0x2c(%r8)
   e0270:	ja     e0209 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x44d9>
   e0272:	jmp    e0153 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4423>
   e0277:	mov    0xb0(%rsp),%r15
   e027f:	test   %r15,%r15
   e0282:	je     e02e5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45b5>
   e0284:	mov    0xb8(%rsp),%r12
   e028c:	mov    0xc8(%rsp),%rcx
   e0294:	mov    %rcx,%rax
   e0297:	sub    %r12,%rax
   e029a:	movabs $0xaaaaaaaaaaaaaaab,%rdx
   e02a4:	mul    %rdx
   e02a7:	cmp    %r12,%rcx
   e02aa:	je     e02d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45a1>
   e02ac:	mov    %rdx,%r13
   e02af:	shr    $0x4,%r13
   e02b3:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e02c0:	mov    %r12,%rdi
   e02c3:	call   dae80 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>
   e02c8:	add    $0x18,%r12
   e02cc:	dec    %r13
   e02cf:	jne    e02c0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4590>
   e02d1:	cmpq   $0x0,0xc0(%rsp)
   e02da:	je     e02e5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45b5>
   e02dc:	mov    %r15,%rdi
   e02df:	call   *0x23c0b(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e02e5:	mov    %r14,%rdi
   e02e8:	call   daf70 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>>>>
   e02ed:	lea    0xf0(%rsp),%rdi
   e02f5:	call   daf70 <core::ptr::drop_glue::<core::option::Option<alloc::vec::into_iter::IntoIter<tk_train::trainers::bpe::index::Candidate>>>>
   e02fa:	cmpq   $0x0,0x160(%rsp)
   e0303:	je     e0313 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x45e3>
   e0305:	mov    0x168(%rsp),%rdi
   e030d:	call   *0x23bdd(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0313:	mov    0x1c8(%rsp),%rcx
   e031b:	mov    0x58(%rcx),%rax
   e031f:	mov    0x68(%rcx),%rbx
   e0323:	sub    %rbx,%rax
   e0326:	mov    0x298(%rsp),%rdx
   e032e:	cmp    %rax,%rdx
   e0331:	ja     e0801 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ad1>
   e0337:	lea    0x0(,%rbx,8),%rdi
   e033f:	mov    0x1c8(%rsp),%r14
   e0347:	add    0x60(%r14),%rdi
   e034b:	mov    0x198(%rsp),%r15
   e0353:	mov    %r15,%rsi
   e0356:	mov    0x330(%rsp),%rdx
   e035e:	call   *0x23b94(%rip)        # 103ef8 <memcpy@GLIBC_2.14>
   e0364:	mov    0x298(%rsp),%r12
   e036c:	add    %r12,%rbx
   e036f:	mov    %rbx,0x68(%r14)
   e0373:	mov    %r15,%rdi
   e0376:	call   *0x23b74(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e037c:	mov    0x2c0(%rsp),%rax
   e0384:	movq   $0x2,0x8(%rax)
   e038c:	mov    %r12,0x10(%rax)
   e0390:	movq   $0x0,(%rax)
   e0397:	jmp    df00d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32dd>
   e039c:	xor    %edi,%edi
   e039e:	mov    %rdx,%rsi
   e03a1:	call   692b8 <alloc::raw_vec::handle_error>
   e03a6:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e03ab:	mov    %rdx,0x8(%rsp)
   e03b0:	cmpq   $0x0,0x160(%rsp)
   e03b9:	jne    e00a4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4374>
   e03bf:	jmp    defe3 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x32b3>
   e03c4:	mov    0x1d8(%rsp),%rcx
   e03cc:	mov    0x2c0(%rsp),%rdx
   e03d4:	mov    %rax,0x8(%rdx)
   e03d8:	mov    %rcx,0x10(%rdx)
   e03dc:	mov    $0x1,%eax
   e03e1:	mov    %rax,(%rdx)
   e03e4:	mov    0x388(%rsp),%rcx
   e03ec:	test   %rcx,%rcx
   e03ef:	jne    dd0fd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13cd>
   e03f5:	jmp    dd123 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13f3>
   e03fa:	mov    $0x19,%esi
   e03ff:	mov    0x180(%rsp),%rdi
   e0407:	mov    0x188(%rsp),%rdx
   e040f:	call   71e50 <core::option::expect_failed>
   e0414:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0419:	mov    %rsi,%rdx
   e041c:	lea    0x2316d(%rip),%rcx        # 103590 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a98>
   e0423:	jmp    e04bf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x478f>
   e0428:	lea    -0xc271c(%rip),%rdi        # 1dd13 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x3f6>
   e042f:	lea    0x230e2(%rip),%rdx        # 103518 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a20>
   e0436:	mov    $0x53,%esi
   e043b:	call   6eef0 <core::panicking::panic_fmt>
   e0440:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0445:	mov    %rdx,%rsi
   e0448:	lea    0x23771(%rip),%rcx        # 103bc0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x40c8>
   e044f:	jmp    e04bf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x478f>
   e0451:	mov    0x2c0(%rsp),%rdx
   e0459:	movq   $0x1,0x8(%rdx)
   e0461:	jmp    dd0eb <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x13bb>
   e0466:	mov    $0x8,%esi
   e046b:	xor    %edi,%edi
   e046d:	lea    0x235b4(%rip),%rcx        # 103a28 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f30>
   e0474:	jmp    e04bf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x478f>
   e0476:	call   c0750 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e047b:	jmp    dbddd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xad>
   e0480:	call   c05f0 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e0485:	mov    %rax,%rbx
   e0488:	jmp    dbdf4 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0xc4>
   e048d:	call   c0750 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e0492:	jmp    dbf0e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1de>
   e0497:	call   c05f0 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e049c:	mov    %rax,%rbx
   e049f:	jmp    dbf25 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1f5>
   e04a4:	mov    %r10,%rdi
   e04a7:	mov    %rdx,%rsi
   e04aa:	lea    0x23127(%rip),%rcx        # 1035d8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ae0>
   e04b1:	jmp    e04bf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x478f>
   e04b3:	mov    $0x8,%edi
   e04b8:	lea    0x23131(%rip),%rcx        # 1035f0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3af8>
   e04bf:	call   70c70 <core::slice::index::slice_index_fail>
   e04c4:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e04c9:	mov    %rdx,%rsi
   e04cc:	jmp    e041c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46ec>
   e04d1:	lea    0x160(%rsp),%rdi
   e04d9:	mov    $0x8,%ecx
   e04de:	mov    $0x38,%r8d
   e04e4:	xor    %esi,%esi
   e04e6:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e04eb:	jmp    de24b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x251b>
   e04f0:	lea    0x1a0(%rsp),%rdi
   e04f8:	mov    $0x8,%ecx
   e04fd:	mov    $0x38,%r8d
   e0503:	xor    %esi,%esi
   e0505:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e050a:	jmp    dec5b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x2f2b>
   e050f:	mov    %r13,0xb0(%rsp)
   e0517:	mov    %rbp,0xb8(%rsp)
   e051f:	lea    -0xc2910(%rip),%rdi        # 1dc16 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e0526:	lea    0x22cbb(%rip),%rcx        # 1031e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36f0>
   e052d:	lea    0x22abc(%rip),%r8        # 102ff0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e0534:	lea    0xb0(%rsp),%rdx
   e053c:	mov    $0x2b,%esi
   e0541:	call   71e90 <core::result::unwrap_failed>
   e0546:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e054b:	lea    -0xc2911(%rip),%rdi        # 1dc41 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x324>
   e0552:	lea    0x233c7(%rip),%rdx        # 103920 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3e28>
   e0559:	mov    $0x2f,%esi
   e055e:	call   70d40 <core::panicking::panic>
   e0563:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0568:	mov    %rax,0xb0(%rsp)
   e0570:	mov    %r13,0xb8(%rsp)
   e0578:	lea    -0xc2969(%rip),%rdi        # 1dc16 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e057f:	lea    0x22c62(%rip),%rcx        # 1031e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36f0>
   e0586:	lea    0x22a63(%rip),%r8        # 102ff0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e058d:	lea    0xb0(%rsp),%rdx
   e0595:	mov    $0x2b,%esi
   e059a:	call   71e90 <core::result::unwrap_failed>
   e059f:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e05a4:	lea    0x148(%rsp),%rdi
   e05ac:	mov    $0x8,%ecx
   e05b1:	mov    $0x18,%r8d
   e05b7:	xor    %esi,%esi
   e05b9:	mov    %rbp,%rdx
   e05bc:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e05c1:	mov    0x148(%rsp),%rax
   e05c9:	mov    0x158(%rsp),%r12
   e05d1:	mov    %rax,0x1c0(%rsp)
   e05d9:	sub    %r12,%rax
   e05dc:	cmp    %rbp,%rax
   e05df:	jae    e0a3b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d0b>
   e05e5:	lea    -0xc29ab(%rip),%rdi        # 1dc41 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x324>
   e05ec:	lea    0x22cb5(%rip),%rdx        # 1032a8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x37b0>
   e05f3:	mov    $0x2f,%esi
   e05f8:	call   70d40 <core::panicking::panic>
   e05fd:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0602:	lea    -0xc29c8(%rip),%rdi        # 1dc41 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x324>
   e0609:	lea    0x23310(%rip),%rdx        # 103920 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3e28>
   e0610:	mov    $0x2f,%esi
   e0615:	call   70d40 <core::panicking::panic>
   e061a:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e061f:	lea    0x1a0(%rsp),%rax
   e0627:	mov    %rax,0xb0(%rsp)
   e062f:	lea    -0x6c916(%rip),%rax        # 73d20 <<u64 as core::fmt::Display>::fmt>
   e0636:	mov    %rax,0xb8(%rsp)
   e063e:	lea    0x1d0(%rsp),%rcx
   e0646:	mov    %rcx,0xc0(%rsp)
   e064e:	mov    %rax,0xc8(%rsp)
   e0656:	lea    -0xcecd2(%rip),%rdi        # 1198b <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x5c0b>
   e065d:	lea    0x22884(%rip),%rdx        # 102ee8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x33f0>
   e0664:	lea    0xb0(%rsp),%rsi
   e066c:	call   6eef0 <core::panicking::panic_fmt>
   e0671:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0676:	call   c0750 <<once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<alloc::boxed::Box<dyn ahash::random_state::RandomSource + core::marker::Send + core::marker::Sync>>>::get_or_init<ahash::random_state::get_src::{closure#0}>::{closure#0}>>
   e067b:	jmp    ddb91 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e61>
   e0680:	call   c05f0 <<once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::init::<<once_cell::race::once_box::OnceBox<_>>::get_or_init::Void, <once_cell::race::once_box::OnceBox<[[u64; 4]; 2]>>::get_or_init<ahash::random_state::get_fixed_seeds::{closure#0}>::{closure#0}>>
   e0685:	mov    %rax,%rbx
   e0688:	jmp    ddba8 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1e78>
   e068d:	lea    0xb0(%rsp),%rdi
   e0695:	mov    %r11,%rdx
   e0698:	mov    %r11,%rbx
   e069b:	call   c4760 <<hashbrown::raw::RawTable<((u32, u32), u32)>>::reserve_rehash::<hashbrown::map::make_hasher<(u32, u32), u32, ahash::random_state::RandomState>::{closure#0}>>
   e06a0:	mov    %rbx,%r11
   e06a3:	jmp    ddce6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x1fb6>
   e06a8:	lea    0x60(%rsp),%rax
   e06ad:	mov    %rax,0xb0(%rsp)
   e06b5:	lea    0x22cf4(%rip),%rdx        # 1033b0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x38b8>
   e06bc:	jmp    e06d2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x49a2>
   e06be:	lea    0x60(%rsp),%rax
   e06c3:	mov    %rax,0xb0(%rsp)
   e06cb:	lea    0x22cf6(%rip),%rdx        # 1033c8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x38d0>
   e06d2:	lea    0x90(%rsp),%rax
   e06da:	lea    -0x6c9c1(%rip),%rcx        # 73d20 <<u64 as core::fmt::Display>::fmt>
   e06e1:	mov    0x330(%rsp),%rsi
   e06e9:	mov    %rcx,0x8(%rsi)
   e06ed:	mov    %rax,0x10(%rsi)
   e06f1:	mov    %rcx,0x18(%rsi)
   e06f5:	lea    -0xd066a(%rip),%rdi        # 10092 <serde_json::ser::Formatter::write_char_escape::HEX_DIGITS+0x4312>
   e06fc:	call   6eef0 <core::panicking::panic_fmt>
   e0701:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0706:	lea    0x240(%rsp),%rdi
   e070e:	mov    $0x8,%ecx
   e0713:	mov    $0x18,%r8d
   e0719:	xor    %esi,%esi
   e071b:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e0720:	jmp    dfe6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x413b>
   e0725:	mov    $0x1a,%esi
   e072a:	lea    -0xc292f(%rip),%rax        # 1de02 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x4e5>
   e0731:	mov    %rax,0x180(%rsp)
   e0739:	lea    0x23120(%rip),%rax        # 103860 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d68>
   e0740:	mov    %rax,0x188(%rsp)
   e0748:	jmp    e03ff <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x46cf>
   e074d:	mov    %rax,0xb0(%rsp)
   e0755:	mov    %rbp,0xb8(%rsp)
   e075d:	lea    -0xc2b4e(%rip),%rdi        # 1dc16 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x2f9>
   e0764:	lea    0x22a7d(%rip),%rcx        # 1031e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x36f0>
   e076b:	lea    0x2287e(%rip),%r8        # 102ff0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x34f8>
   e0772:	lea    0xb0(%rsp),%rdx
   e077a:	mov    $0x2b,%esi
   e077f:	call   71e90 <core::result::unwrap_failed>
   e0784:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0789:	lea    0x22c50(%rip),%rdi        # 1033e0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x38e8>
   e0790:	call   6eaa0 <compact_str::unwrap_with_msg_fail::<compact_str::ReserveError>>
   e0795:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e079a:	xor    %eax,%eax
   e079c:	jmp    e07a0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4a70>
   e079e:	mov    $0x1,%al
   e07a0:	mov    %al,0x90(%rsp)
   e07a7:	lea    -0xc265e(%rip),%rdi        # 1e150 <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x833>
   e07ae:	lea    0x22ad3(%rip),%rcx        # 103288 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3790>
   e07b5:	lea    0x233ec(%rip),%r8        # 103ba8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x40b0>
   e07bc:	lea    0x90(%rsp),%rdx
   e07c4:	mov    $0x13,%esi
   e07c9:	call   71e90 <core::result::unwrap_failed>
   e07ce:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e07d3:	lea    -0xc29be(%rip),%rdi        # 1de1c <core::unicode::unicode_data::white_space::WHITESPACE_MAP+0x4ff>
   e07da:	lea    0x23097(%rip),%rdx        # 103878 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d80>
   e07e1:	mov    $0x15,%esi
   e07e6:	call   71e50 <core::option::expect_failed>
   e07eb:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e07f0:	lea    0x23201(%rip),%rdi        # 1039f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f00>
   e07f7:	call   721c0 <core::panicking::panic_const::panic_const_rem_by_zero>
   e07fc:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0801:	mov    0x1c8(%rsp),%rax
   e0809:	lea    0x58(%rax),%rdi
   e080d:	mov    $0x4,%ecx
   e0812:	mov    $0x8,%r8d
   e0818:	mov    %rbx,%rsi
   e081b:	call   c9140 <<alloc::raw_vec::RawVecInner<_>>::reserve::do_reserve_and_handle::<alloc::alloc::Global>>
   e0820:	mov    0x1c8(%rsp),%rax
   e0828:	mov    0x68(%rax),%rbx
   e082c:	jmp    e0337 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4607>
   e0831:	mov    %r14,%r12
   e0834:	lea    0x22cc5(%rip),%rcx        # 103500 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3a08>
   e083b:	mov    %rcx,0x128(%rsp)
   e0843:	mov    %rax,%rdi
   e0846:	mov    %r12,%rsi
   e0849:	mov    0x128(%rsp),%rdx
   e0851:	call   6fd3b <core::panicking::panic_bounds_check>
   e0856:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e085b:	lea    0x22ede(%rip),%rax        # 103740 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c48>
   e0862:	mov    %rax,0x1c0(%rsp)
   e086a:	mov    %r14,%rsi
   e086d:	mov    0x1c0(%rsp),%rdx
   e0875:	call   6fd3b <core::panicking::panic_bounds_check>
   e087a:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e087f:	mov    $0x4,%edi
   e0884:	mov    %r15,%rsi
   e0887:	call   692b8 <alloc::raw_vec::handle_error>
   e088c:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0891:	shr    $0x7,%r9
   e0895:	xor    %eax,%eax
   e0897:	mov    %r9,%rdi
   e089a:	jmp    e093d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c0d>
   e089f:	mov    %rdx,%rdi
   e08a2:	jmp    e093d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c0d>
   e08a7:	mov    %rax,%rdi
   e08aa:	mov    %r12,%rax
   e08ad:	lea    0x22e8c(%rip),%rcx        # 103740 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c48>
   e08b4:	jmp    e0935 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c05>
   e08b6:	mov    %rax,%rdi
   e08b9:	mov    %r12,%rax
   e08bc:	lea    0x22e65(%rip),%rcx        # 103728 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c30>
   e08c3:	jmp    e0935 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4c05>
   e08c5:	lea    0x22e2c(%rip),%rdx        # 1036f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c00>
   e08cc:	xor    %edi,%edi
   e08ce:	xor    %esi,%esi
   e08d0:	call   6fd3b <core::panicking::panic_bounds_check>
   e08d5:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e08da:	lea    0x22ebf(%rip),%rdx        # 1037a0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3ca8>
   e08e1:	xor    %edi,%edi
   e08e3:	xor    %esi,%esi
   e08e5:	call   6fd3b <core::panicking::panic_bounds_check>
   e08ea:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e08ef:	lea    0x22e1a(%rip),%rdi        # 103710 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3c18>
   e08f6:	call   721a0 <core::panicking::panic_const::panic_const_div_by_zero>
   e08fb:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0900:	lea    0x22eb1(%rip),%rdi        # 1037b8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3cc0>
   e0907:	jmp    e0910 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4be0>
   e0909:	lea    0x22ec0(%rip),%rdi        # 1037d0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3cd8>
   e0910:	call   721a0 <core::panicking::panic_const::panic_const_div_by_zero>
   e0915:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e091a:	mov    $0x8,%edi
   e091f:	mov    $0x20,%esi
   e0924:	call   692b8 <alloc::raw_vec::handle_error>
   e0929:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e092e:	lea    0x227c3(%rip),%rcx        # 1030f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3600>
   e0935:	mov    %rcx,0x180(%rsp)
   e093d:	mov    %rax,%rsi
   e0940:	mov    0x180(%rsp),%rdx
   e0948:	call   6fd3b <core::panicking::panic_bounds_check>
   e094d:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0952:	mov    %rbx,%r13
   e0955:	lea    0x22a6c(%rip),%rax        # 1033c8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x38d0>
   e095c:	mov    %rax,0x120(%rsp)
   e0964:	mov    %r13,%rdi
   e0967:	mov    0x120(%rsp),%rdx
   e096f:	call   6fd3b <core::panicking::panic_bounds_check>
   e0974:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e0979:	lea    -0xc6a74(%rip),%rdi        # 19f0c <bitcannon::classify::atom_tables::BMP_RLE+0x2690>
   e0980:	lea    0x20111(%rip),%rcx        # 100a98 <bitcannon::classify::atom_tables::ATOM_TABLES+0xfa0>
   e0987:	lea    0x20142(%rip),%r8        # 100ad0 <bitcannon::classify::atom_tables::ATOM_TABLES+0xfd8>
   e098e:	lea    0x26f(%rsp),%rdx
   e0996:	mov    $0xe,%esi
   e099b:	call   71e90 <core::result::unwrap_failed>
   e09a0:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e09a5:	lea    0x2304c(%rip),%rdi        # 1039f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f00>
   e09ac:	call   721c0 <core::panicking::panic_const::panic_const_rem_by_zero>
   e09b1:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e09b6:	mov    %r15,%r13
   e09b9:	lea    0x22b10(%rip),%rax        # 1034d0 <bitcannon::classify::atom_tables::ATOM_TABLES+0x39d8>
   e09c0:	jmp    e09d1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ca1>
   e09c2:	mov    %rbx,%r13
   e09c5:	jmp    e09d9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ca9>
   e09c7:	mov    %r15,%r13
   e09ca:	lea    0x22b17(%rip),%rax        # 1034e8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x39f0>
   e09d1:	mov    %rax,0x128(%rsp)
   e09d9:	mov    %r13,%rdi
   e09dc:	mov    %r12,%rsi
   e09df:	mov    0x128(%rsp),%rdx
   e09e7:	call   6fd3b <core::panicking::panic_bounds_check>
   e09ec:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e09ee:	lea    0x23003(%rip),%rdi        # 1039f8 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3f00>
   e09f5:	call   721c0 <core::panicking::panic_const::panic_const_rem_by_zero>
   e09fa:	jmp    e0a39 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d09>
   e09fc:	mov    %rax,0x2c8(%rsp)
   e0a04:	lea    0x22e25(%rip),%rax        # 103830 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d38>
   e0a0b:	jmp    e0a14 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ce4>
   e0a0d:	lea    0x22dec(%rip),%rax        # 103800 <bitcannon::classify::atom_tables::ATOM_TABLES+0x3d08>
   e0a14:	mov    %rax,0x328(%rsp)
   e0a1c:	mov    0x2c8(%rsp),%rdi
   e0a24:	mov    0x120(%rsp),%rsi
   e0a2c:	mov    0x328(%rsp),%rdx
   e0a34:	call   6fd3b <core::panicking::panic_bounds_check>
   e0a39:	ud2
   e0a3b:	mov    0x150(%rsp),%rdx
   e0a43:	jmp    df25d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x352d>
   e0a48:	mov    %rax,%r14
   e0a4b:	mov    0x198(%rsp),%rdi
   e0a53:	call   *0x23497(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0a59:	mov    %r14,%rdi
   e0a5c:	call   fdf70 <_Unwind_Resume@plt>
   e0a61:	mov    %rax,%r14
   e0a64:	mov    %rbx,0x338(%rsp)
   e0a6c:	jmp    e0abf <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4d8f>
   e0a6e:	jmp    e0b40 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e10>
   e0a73:	jmp    e0d7e <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x504e>
   e0a78:	mov    %rax,%r14
   e0a7b:	lea    0x160(%rsp),%rdi
   e0a83:	jmp    e0afa <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4dca>
   e0a85:	mov    %rax,%r14
   e0a88:	mov    %rbp,0x18(%rsp)
   e0a8d:	jmp    e0b31 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e01>
   e0a92:	mov    %rax,%r14
   e0a95:	mov    %rbx,0x1d0(%rsp)
   e0a9d:	mov    0x8(%rsp),%rax
   e0aa2:	mov    %rax,0x1d8(%rsp)
   e0aaa:	mov    %r12,0x1e0(%rsp)
   e0ab2:	lea    0xb0(%rsp),%rdi
   e0aba:	call   dac50 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0abf:	mov    0x338(%rsp),%rdi
   e0ac7:	call   eb970 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0acc:	jmp    e0bd9 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4ea9>
   e0ad1:	mov    %rax,%r14
   e0ad4:	mov    %rbx,0x60(%rsp)
   e0ad9:	mov    0x8(%rsp),%rax
   e0ade:	mov    %rax,0x68(%rsp)
   e0ae3:	mov    %r15,0x70(%rsp)
   e0ae8:	lea    0xb0(%rsp),%rdi
   e0af0:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0af5:	lea    0x60(%rsp),%rdi
   e0afa:	call   e4cb0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>>
   e0aff:	jmp    e0e25 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x50f5>
   e0b04:	mov    %rax,%r14
   e0b07:	mov    %rbx,0x1d0(%rsp)
   e0b0f:	mov    0x8(%rsp),%rax
   e0b14:	mov    %rax,0x1d8(%rsp)
   e0b1c:	mov    %r15,0x1e0(%rsp)
   e0b24:	lea    0xb0(%rsp),%rdi
   e0b2c:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0b31:	mov    0x18(%rsp),%rdi
   e0b36:	call   e4cb0 <core::ptr::drop_glue::<alloc::collections::linked_list::LinkedList<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>>
   e0b3b:	jmp    e0c63 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4f33>
   e0b40:	mov    %rax,%r14
   e0b43:	mov    0xb0(%rsp),%rdi
   e0b4b:	mov    0xb8(%rsp),%rsi
   e0b53:	call   e2430 <core::ptr::drop_glue::<std::collections::hash::map::HashMap<(u32, u32), u32, ahash::random_state::RandomState>>>
   e0b58:	jmp    e0eb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5187>
   e0b5d:	mov    %rax,%r14
   e0b60:	jmp    e0b79 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e49>
   e0b62:	jmp    e0b69 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4e39>
   e0b64:	jmp    e0eca <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x519a>
   e0b69:	mov    %rax,%r14
   e0b6c:	lea    0x1d0(%rsp),%rdi
   e0b74:	call   e8260 <core::ptr::drop_glue::<tk_train::trainers::bpe::positions::Positions>>
   e0b79:	lea    0xb0(%rsp),%rdi
   e0b81:	call   e8050 <core::ptr::drop_glue::<core::iter::adapters::flatten::Flatten<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>>>
   e0b86:	cmpq   $0x0,0x90(%rsp)
   e0b8f:	je     e107a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x534a>
   e0b95:	mov    0x98(%rsp),%rdi
   e0b9d:	jmp    e1074 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5344>
   e0ba2:	mov    %rax,%r14
   e0ba5:	lea    0xb0(%rsp),%rdi
   e0bad:	call   e2230 <core::ptr::drop_glue::<tk_train::trainers::bpe::merge::Rule>>
   e0bb2:	jmp    e0ed7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51a7>
   e0bb7:	jmp    e0e9c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x516c>
   e0bbc:	mov    %rax,%r14
   e0bbf:	lea    0x160(%rsp),%rdi
   e0bc7:	call   e7280 <core::ptr::drop_glue::<rayon::vec::DrainProducer<tk_train::trainers::bpe::merge::Job>>>
   e0bcc:	jmp    e0d93 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5063>
   e0bd1:	jmp    e0e9c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x516c>
   e0bd6:	mov    %rax,%r14
   e0bd9:	lea    0x240(%rsp),%rdi
   e0be1:	call   dac50 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0be6:	mov    0x150(%rsp),%rdi
   e0bee:	mov    0x158(%rsp),%rsi
   e0bf6:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0bfb:	jmp    e0d3a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x500a>
   e0c00:	call   71f53 <core::panicking::panic_in_cleanup>
   e0c05:	mov    %rax,%r14
   e0c08:	jmp    e0da0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5070>
   e0c0d:	jmp    e0e9c <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x516c>
   e0c12:	mov    %rax,%r14
   e0c15:	cmpb   $0xd8,0x3e7(%rsp)
   e0c1d:	jne    e0ed7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51a7>
   e0c23:	mov    0x3d0(%rsp),%rdi
   e0c2b:	call   *0x232bf(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0c31:	jmp    e0ed7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51a7>
   e0c36:	mov    %rax,%r14
   e0c39:	lea    0x1d8(%rsp),%rdi
   e0c41:	call   db080 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e0c46:	lea    0xb0(%rsp),%rdi
   e0c4e:	call   db100 <core::ptr::drop_glue::<core::iter::adapters::flatten::Flatten<alloc::vec::into_iter::IntoIter<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>>
   e0c53:	jmp    e0d3a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x500a>
   e0c58:	mov    %rax,%r14
   e0c5b:	jmp    e0e78 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5148>
   e0c60:	mov    %rax,%r14
   e0c63:	lea    0x160(%rsp),%rdi
   e0c6b:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0c70:	mov    0x2a8(%rsp),%rdi
   e0c78:	mov    0x2b0(%rsp),%rsi
   e0c80:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0c85:	movzbl 0x40(%rsp),%ebp
   e0c8a:	jmp    e0e78 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5148>
   e0c8f:	call   71f53 <core::panicking::panic_in_cleanup>
   e0c94:	mov    %rax,%r14
   e0c97:	jmp    e0f7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x524d>
   e0c9c:	jmp    e0cae <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4f7e>
   e0c9e:	mov    %rax,%r14
   e0ca1:	jmp    e0e88 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5158>
   e0ca6:	mov    %rax,%r14
   e0ca9:	jmp    e0eec <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51bc>
   e0cae:	mov    %rax,%r14
   e0cb1:	lea    0x2e0(%rsp),%rdi
   e0cb9:	call   e2590 <core::ptr::drop_glue::<tk_train::trainers::bpe::merge::Batch>>
   e0cbe:	mov    %r14,%rdi
   e0cc1:	call   fdf70 <_Unwind_Resume@plt>
   e0cc6:	jmp    e0ea1 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5171>
   e0ccb:	jmp    e0d0d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x4fdd>
   e0ccd:	mov    %rax,%r14
   e0cd0:	cmpq   $0x0,0x60(%rsp)
   e0cd6:	je     e0e88 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5158>
   e0cdc:	mov    0x68(%rsp),%rdi
   e0ce1:	jmp    e0e82 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5152>
   e0ce6:	jmp    e0f16 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51e6>
   e0ceb:	jmp    e0f16 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51e6>
   e0cf0:	mov    %rax,%r14
   e0cf3:	cmpq   $0x0,0x40(%rsp)
   e0cf9:	je     e0ecd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x519d>
   e0cff:	mov    %rbp,%rdi
   e0d02:	call   *0x231e8(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0d08:	jmp    e0ecd <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x519d>
   e0d0d:	mov    %rax,%r14
   e0d10:	lea    0x1d0(%rsp),%rdi
   e0d18:	jmp    e0ed2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51a2>
   e0d1d:	mov    %rax,%r14
   e0d20:	lea    0xb0(%rsp),%rdi
   e0d28:	call   d81d0 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0d2d:	lea    0x270(%rsp),%rdi
   e0d35:	call   dac50 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::index::Candidate>>>>
   e0d3a:	cmpq   $0x0,0x160(%rsp)
   e0d43:	je     e107a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x534a>
   e0d49:	mov    0x168(%rsp),%rdi
   e0d51:	jmp    e1074 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5344>
   e0d56:	call   71f53 <core::panicking::panic_in_cleanup>
   e0d5b:	mov    %rax,%r14
   e0d5e:	test   %r15,%r15
   e0d61:	je     e0dad <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x507d>
   e0d63:	mov    0x270(%rsp),%rbx
   e0d6b:	mov    %rbx,%rdi
   e0d6e:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   e0d73:	add    $0x18,%rbx
   e0d77:	dec    %r15
   e0d7a:	jne    e0d6b <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x503b>
   e0d7c:	jmp    e0dad <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x507d>
   e0d7e:	mov    %rax,%r14
   e0d81:	lea    0x90(%rsp),%rdi
   e0d89:	call   e7f70 <core::ptr::drop_glue::<rayon::iter::collect::special_extend<rayon::iter::map::Map<rayon::vec::IntoIter<tk_train::trainers::bpe::merge::Job>, <tk_train::trainers::bpe::BpeTrainer>::merge_next::{closure#0}>, alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>::{closure#0}>>
   e0d8e:	jmp    e0dad <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x507d>
   e0d90:	mov    %rax,%r14
   e0d93:	lea    0xb0(%rsp),%rdi
   e0d9b:	call   e7150 <core::ptr::drop_glue::<rayon::vec::Drain<tk_train::trainers::bpe::merge::Job>>>
   e0da0:	lea    0x240(%rsp),%rdi
   e0da8:	call   e70d0 <core::ptr::drop_glue::<rayon::vec::IntoIter<tk_train::trainers::bpe::merge::Job>>>
   e0dad:	lea    0x148(%rsp),%rdi
   e0db5:	call   e7ff0 <core::ptr::drop_glue::<alloc::vec::Vec<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>>
   e0dba:	mov    0x198(%rsp),%rdi
   e0dc2:	call   *0x23128(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0dc8:	mov    %r14,%rdi
   e0dcb:	call   fdf70 <_Unwind_Resume@plt>
   e0dd0:	call   71f53 <core::panicking::panic_in_cleanup>
   e0dd5:	mov    %rax,%r14
   e0dd8:	lea    0xb0(%rsp),%rdi
   e0de0:	call   d81d0 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0de5:	lea    0x2a0(%rsp),%rdi
   e0ded:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0df2:	jmp    e0f32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5202>
   e0df7:	call   71f53 <core::panicking::panic_in_cleanup>
   e0dfc:	mov    %rax,%r14
   e0dff:	lea    0x90(%rsp),%rdi
   e0e07:	call   e4ff0 <core::ptr::drop_glue::<rayon::vec::Drain<(usize, either::Either<core::iter::adapters::copied::Copied<core::slice::iter::Iter<u64>>, core::iter::adapters::take::Take<core::iter::adapters::flatten::FlatMap<core::slice::iter::Iter<tk_train::trainers::bpe::positions::Leaf>, tk_train::trainers::bpe::positions::codec::Cursor, tk_train::trainers::bpe::positions::read::{closure#0}>>>)>>>
   e0e0c:	cmpq   $0x0,0x148(%rsp)
   e0e15:	je     e0e25 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x50f5>
   e0e17:	mov    0x150(%rsp),%rdi
   e0e1f:	call   *0x230cb(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0e25:	lea    0x1a0(%rsp),%rdi
   e0e2d:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0e32:	mov    0x308(%rsp),%rdi
   e0e3a:	mov    0x310(%rsp),%rsi
   e0e42:	call   d8170 <core::ptr::drop_glue::<std::sync::poison::mutex::Mutex<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0e47:	jmp    e0f32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5202>
   e0e4c:	call   71f53 <core::panicking::panic_in_cleanup>
   e0e51:	call   71f53 <core::panicking::panic_in_cleanup>
   e0e56:	mov    %rax,%r14
   e0e59:	lea    0xb0(%rsp),%rdi
   e0e61:	call   d81d0 <core::ptr::drop_glue::<std::sync::poison::PoisonError<core::option::Option<alloc::boxed::Box<dyn core::error::Error + core::marker::Send + core::marker::Sync>>>>>
   e0e66:	movzbl 0x40(%rsp),%ebp
   e0e6b:	lea    0x1a0(%rsp),%rdi
   e0e73:	call   e4c30 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Job>>>
   e0e78:	test   %bpl,%bpl
   e0e7b:	jne    e0e88 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5158>
   e0e7d:	mov    0x38(%rsp),%rdi
   e0e82:	call   *0x23068(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0e88:	lea    0x90(%rsp),%rdi
   e0e90:	call   e2590 <core::ptr::drop_glue::<tk_train::trainers::bpe::merge::Batch>>
   e0e95:	jmp    e0eb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5187>
   e0e97:	call   71f53 <core::panicking::panic_in_cleanup>
   e0e9c:	mov    %rax,%r14
   e0e9f:	jmp    e0ed7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x51a7>
   e0ea1:	mov    %rax,%r14
   e0ea4:	cmpq   $0x0,0x60(%rsp)
   e0eaa:	je     e0eb7 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5187>
   e0eac:	mov    0x68(%rsp),%rdi
   e0eb1:	call   *0x23039(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0eb7:	cmpb   $0x0,0x188(%rsp)
   e0ebf:	je     e0f7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x524d>
   e0ec5:	jmp    e107a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x534a>
   e0eca:	mov    %rax,%r14
   e0ecd:	lea    0x60(%rsp),%rdi
   e0ed2:	call   db080 <core::ptr::drop_glue::<tk_train::trainers::bpe::index::Candidate>>
   e0ed7:	mov    0x380(%rsp),%rdi
   e0edf:	mov    0x388(%rsp),%rsi
   e0ee7:	call   c8e30 <core::ptr::drop_glue::<ahash::hash_set::AHashSet<char>>>
   e0eec:	mov    0x340(%rsp),%rdi
   e0ef4:	mov    0x348(%rsp),%rsi
   e0efc:	call   c8e30 <core::ptr::drop_glue::<ahash::hash_set::AHashSet<char>>>
   e0f01:	lea    0x2e0(%rsp),%rdi
   e0f09:	call   e2590 <core::ptr::drop_glue::<tk_train::trainers::bpe::merge::Batch>>
   e0f0e:	mov    %r14,%rdi
   e0f11:	call   fdf70 <_Unwind_Resume@plt>
   e0f16:	mov    %rax,%r14
   e0f19:	cmpq   $0x0,0xb0(%rsp)
   e0f22:	je     e0f32 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5202>
   e0f24:	mov    0xb8(%rsp),%rdi
   e0f2c:	call   *0x22fbe(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0f32:	mov    0x1e8(%rsp),%rdi
   e0f3a:	mov    0x1f0(%rsp),%rax
   e0f42:	test   %rdi,%rdi
   e0f45:	jne    e0f56 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5226>
   e0f47:	test   %rax,%rax
   e0f4a:	je     e0f7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x524d>
   e0f4c:	mov    0x1f8(%rsp),%rdi
   e0f54:	jmp    e0f77 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5247>
   e0f56:	test   %rax,%rax
   e0f59:	je     e0f7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x524d>
   e0f5b:	lea    (%rax,%rax,2),%rcx
   e0f5f:	lea    0x1b(,%rcx,4),%rcx
   e0f67:	and    $0xfffffffffffffff0,%rcx
   e0f6b:	add    %rcx,%rax
   e0f6e:	cmp    $0xffffffffffffffef,%rax
   e0f72:	je     e0f7d <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x524d>
   e0f74:	sub    %rcx,%rdi
   e0f77:	call   *0x22f73(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0f7d:	mov    %r14,0x30(%rsp)
   e0f82:	mov    0x278(%rsp),%rbx
   e0f8a:	mov    0x280(%rsp),%r12
   e0f92:	test   %r12,%r12
   e0f95:	jne    e0fc6 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5296>
   e0f97:	cmpq   $0x0,0x270(%rsp)
   e0fa0:	je     e0fab <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x527b>
   e0fa2:	mov    %rbx,%rdi
   e0fa5:	call   *0x22f45(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0fab:	mov    0x30(%rsp),%r14
   e0fb0:	mov    0x198(%rsp),%rdi
   e0fb8:	call   *0x22f32(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e0fbe:	mov    %r14,%rdi
   e0fc1:	call   fdf70 <_Unwind_Resume@plt>
   e0fc6:	xor    %r13d,%r13d
   e0fc9:	mov    0x22f20(%rip),%rbp        # 103ef0 <free@GLIBC_2.2.5>
   e0fd0:	mov    %r12,0x8(%rsp)
   e0fd5:	jmp    e0ff2 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52c2>
   e0fd7:	nopw   0x0(%rax,%rax,1)
   e0fe0:	mov    0x8(%rsp),%r12
   e0fe5:	mov    %r15,%rdi
   e0fe8:	call   *%rbp
   e0fea:	inc    %r13
   e0fed:	cmp    %r12,%r13
   e0ff0:	je     e0f97 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5267>
   e0ff2:	imul   $0x38,%r13,%rcx
   e0ff6:	lea    (%rbx,%rcx,1),%rax
   e0ffa:	cmpq   $0x0,(%rbx,%rcx,1)
   e0fff:	je     e1010 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52e0>
   e1001:	cmpq   $0x11,0x18(%rax)
   e1006:	jb     e0fea <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52ba>
   e1008:	mov    0x8(%rax),%r15
   e100c:	jmp    e0fe5 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52b5>
   e100e:	xchg   %ax,%ax
   e1010:	mov    0x10(%rax),%r14
   e1014:	test   %r14,%r14
   e1017:	je     e0fea <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52ba>
   e1019:	mov    0x8(%rax),%r15
   e101d:	lea    0x18(%r15),%r12
   e1021:	jmp    e1039 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5309>
   e1023:	data16 data16 data16 cs nopw 0x0(%rax,%rax,1)
   e1030:	add    $0x20,%r12
   e1034:	dec    %r14
   e1037:	je     e0fe0 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x52b0>
   e1039:	cmpq   $0x11,(%r12)
   e103e:	jb     e1030 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5300>
   e1040:	mov    -0x10(%r12),%rdi
   e1045:	call   *%rbp
   e1047:	jmp    e1030 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5300>
   e1049:	mov    %rax,%r14
   e104c:	nopl   0x0(%rax)
   e1050:	mov    %rbx,%rdi
   e1053:	call   e4e90 <core::ptr::drop_glue::<alloc::vec::Vec<tk_train::trainers::bpe::merge::Change<tk_train::trainers::bpe::positions::Positions>>>>
   e1058:	add    $0x18,%rbx
   e105c:	dec    %r15
   e105f:	jne    e1050 <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x5320>
   e1061:	cmpq   $0x0,0x1c0(%rsp)
   e106a:	je     e107a <<tk_train::trainers::bpe::BpeTrainer>::merge_next+0x534a>
   e106c:	mov    0x128(%rsp),%rdi
   e1074:	call   *0x22e76(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e107a:	mov    0x198(%rsp),%rdi
   e1082:	call   *0x22e68(%rip)        # 103ef0 <free@GLIBC_2.2.5>
   e1088:	mov    %r14,%rdi
   e108b:	call   fdf70 <_Unwind_Resume@plt>
