
#define _GNU_SOURCE
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <string.h>
static uint64_t now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return (uint64_t)t.tv_sec*1000000000+t.tv_nsec; }
static inline void escape(void *p) { __asm__ __volatile__("" : : "r"(p) : "memory"); }
static double pair(int heap,size_t size,size_t n,unsigned char *arena) {
 uint64_t t=now();
 for(size_t i=0;i<n;i++) { void *p=heap?malloc(size):arena; if(!p)abort(); escape(p); if(heap)free(p); }
 return (double)(now()-t)/n;
}
static void batch(int heap,size_t size,size_t n,void **slots,unsigned char *arena,double *allocation,double *release) {
 uint64_t t=now();
 for(size_t i=0;i<n;i++) { void *p=heap?malloc(size):arena+i*size; if(!p)abort(); slots[i]=p; escape(p); }
 *allocation=(double)(now()-t)/n;
 t=now();
 for(size_t i=0;i<n;i++) { void *p=slots[i]; escape(p); if(heap)free(p); }
 *release=(double)(now()-t)/n;
}
int main(void) {
 const size_t n=1000000; void **slots=malloc(n*sizeof(void*)); unsigned char *arena=malloc(n*26); if(!slots||!arena)abort(); memset(arena,0,n*26);
 for(int repeat=0;repeat<7;repeat++)for(size_t size=24;size<=26;size+=2) {
  int first=repeat%2;for(int order=0;order<2;order++){int heap=order^first;double alloc,drop;batch(heap,size,n,slots,arena,&alloc,&drop);printf("{\"kind\":\"batch\",\"repeat\":%d,\"heap\":%d,\"bytes\":%zu,\"count\":%zu,\"allocation_ns\":%.6f,\"release_ns\":%.6f}\n",repeat,heap,size,n,alloc,drop);}
  for(int order=0;order<2;order++){int heap=order^first;double ns=pair(heap,size,n*10,arena);printf("{\"kind\":\"pair\",\"repeat\":%d,\"heap\":%d,\"bytes\":%zu,\"count\":%zu,\"allocation_release_ns\":%.6f}\n",repeat,heap,size,n*10,ns);}
 }
 free(arena);free(slots);return 0;
}
