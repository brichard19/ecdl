#ifndef _FP_X86_H
#define _FP_X86_H

#ifdef __cplusplus
extern "C" {
#endif

void x86_sub160(const unsigned long *a, const unsigned long *b, unsigned long *diff);
void x86_add160(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_add320(const unsigned long *a, const unsigned long *b, unsigned long *sum);
void x86_mul160(const unsigned long *a, const unsigned long *b, unsigned long *product);
void x86_mul_low160(const unsigned long *a, const unsigned long *b, unsigned long *product);

#ifdef __cplusplus
}
#endif

#endif