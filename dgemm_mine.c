#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE_L1 96
#define BLOCK_SIZE_L2 192
#define BLOCK_SIZE_L3 192

// 16x6 microkernel
void micro_kernel(double* A, double* B, double* C, int i, int j, int K, int Md, int Nd) {
    __m512d a0, a1, a2, a3, b0, b1;
    
    __m512d c00, c01, c02, c03, c04, c05;
    __m512d c10, c11, c12, c13, c14, c15;
    __m512d c20, c21, c22, c23, c24, c25;
    __m512d c30, c31, c32, c33, c34, c35;

    c00 = _mm512_load_pd(C + j * Md + i);
    c01 = _mm512_load_pd(C + (j+1) * Md + i);
    c02 = _mm512_load_pd(C + (j+2) * Md + i);
    c03 = _mm512_load_pd(C + (j+3) * Md + i);
    c04 = _mm512_load_pd(C + (j+4) * Md + i);
    c05 = _mm512_load_pd(C + (j+5) * Md + i);

    c10 = _mm512_load_pd(C + j * Md + (i+8));
    c11 = _mm512_load_pd(C + (j+1) * Md + (i+8));
    c12 = _mm512_load_pd(C + (j+2) * Md + (i+8));
    c13 = _mm512_load_pd(C + (j+3) * Md + (i+8));
    c14 = _mm512_load_pd(C + (j+4) * Md + (i+8));
    c15 = _mm512_load_pd(C + (j+5) * Md + (i+8));

    c20 = _mm512_load_pd(C + j * Md + (i+16));
    c21 = _mm512_load_pd(C + (j+1) * Md + (i+16));
    c22 = _mm512_load_pd(C + (j+2) * Md + (i+16));
    c23 = _mm512_load_pd(C + (j+3) * Md + (i+16));
    c24 = _mm512_load_pd(C + (j+4) * Md + (i+16));
    c25 = _mm512_load_pd(C + (j+5) * Md + (i+16));

    c30 = _mm512_load_pd(C + j * Md + (i+24));
    c31 = _mm512_load_pd(C + (j+1) * Md + (i+24));
    c32 = _mm512_load_pd(C + (j+2) * Md + (i+24));
    c33 = _mm512_load_pd(C + (j+3) * Md + (i+24));
    c34 = _mm512_load_pd(C + (j+4) * Md + (i+24));
    c35 = _mm512_load_pd(C + (j+5) * Md + (i+24));

    for (int k = 0; k < K; ++k) {
        a0 = _mm512_load_pd(A + i + k * Md);
        a1 = _mm512_load_pd(A + i + 8 + k * Md);
        a2 = _mm512_load_pd(A + i + 16 + k * Md);
        a3 = _mm512_load_pd(A + i + 24 + k * Md);

        b0 = _mm512_set1_pd(B[j * Nd + k]);
        b1 = _mm512_set1_pd(B[(j+1) * Nd + k]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        c20 = _mm512_fmadd_pd(a2, b0, c20);
        c30 = _mm512_fmadd_pd(a3, b0, c30);
        
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c11 = _mm512_fmadd_pd(a1, b1, c11);
        c21 = _mm512_fmadd_pd(a2, b1, c21);
        c31 = _mm512_fmadd_pd(a3, b1, c31);

        b0 = _mm512_set1_pd(B[(j+2) * Nd + k]);
        b1 = _mm512_set1_pd(B[(j+3) * Nd + k]);

        c02 = _mm512_fmadd_pd(a0, b0, c02);
        c12 = _mm512_fmadd_pd(a1, b0, c12);
        c22 = _mm512_fmadd_pd(a2, b0, c22);
        c32 = _mm512_fmadd_pd(a3, b0, c32);
        
        c03 = _mm512_fmadd_pd(a0, b1, c03);
        c13 = _mm512_fmadd_pd(a1, b1, c13);
        c23 = _mm512_fmadd_pd(a2, b1, c23);
        c33 = _mm512_fmadd_pd(a3, b1, c33);

        b0 = _mm512_set1_pd(B[(j+4) * Nd + k]);
        b1 = _mm512_set1_pd(B[(j+5) * Nd + k]);

        c04 = _mm512_fmadd_pd(a0, b0, c04);
        c14 = _mm512_fmadd_pd(a1, b0, c14);
        c24 = _mm512_fmadd_pd(a2, b0, c24);
        c34 = _mm512_fmadd_pd(a3, b0, c34);
        
        c05 = _mm512_fmadd_pd(a0, b1, c05);
        c15 = _mm512_fmadd_pd(a1, b1, c15);
        c25 = _mm512_fmadd_pd(a2, b1, c25);
        c35 = _mm512_fmadd_pd(a3, b1, c35);
    }

    _mm512_store_pd(C + j * Md + (i), c00); 
    _mm512_store_pd(C + j * Md + (i+8), c10); 
    _mm512_store_pd(C + j * Md + (i+16), c20); 
    _mm512_store_pd(C + j * Md + (i+24), c30); 

    _mm512_store_pd(C + (j+1) * Md + (i), c01); 
    _mm512_store_pd(C + (j+1) * Md + (i+8), c11);
    _mm512_store_pd(C + (j+1) * Md + (i+16), c21); 
    _mm512_store_pd(C + (j+1) * Md + (i+24), c31); 

    _mm512_store_pd(C + (j+2) * Md + (i), c02); 
    _mm512_store_pd(C + (j+2) * Md + (i+8), c12); 
    _mm512_store_pd(C + (j+2) * Md + (i+16), c22); 
    _mm512_store_pd(C + (j+2) * Md + (i+24), c32); 

    _mm512_store_pd(C + (j+3) * Md + (i), c03); 
    _mm512_store_pd(C + (j+3) * Md + (i+8), c13);
    _mm512_store_pd(C + (j+3) * Md + (i+16), c23); 
    _mm512_store_pd(C + (j+3) * Md + (i+24), c33); 

    _mm512_store_pd(C + (j+4) * Md + (i), c04); 
    _mm512_store_pd(C + (j+4) * Md + (i+8), c14);
    _mm512_store_pd(C + (j+4) * Md + (i+16), c24); 
    _mm512_store_pd(C + (j+4) * Md + (i+24), c34); 

    _mm512_store_pd(C + (j+5) * Md + (i), c05); 
    _mm512_store_pd(C + (j+5) * Md + (i+8), c15);
    _mm512_store_pd(C + (j+5) * Md + (i+16), c25); 
    _mm512_store_pd(C + (j+5) * Md + (i+24), c35); 
}


void square_dgemm(const int M, const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C) {

    int Md = (M + 31) / 32 * 32;
    int Nd = (M + 5) / 6 * 6;

    double *Ad = _mm_malloc(sizeof(double) * Md * M, 64);
    double *Bd = _mm_malloc(sizeof(double) * M * Nd, 64);
    double *Cd = _mm_malloc(sizeof(double) * Md * Nd, 64);

    for (int i = 0; i < M; ++i) {
        memcpy(&Ad[i * Md], &A[i * M], sizeof(double) * M);
        memcpy(&Bd[i * Nd], &B[i * M], sizeof(double) * M);
    }

    for (int j = 0; j < M; j += BLOCK_SIZE_L1) {
        int bN = min(BLOCK_SIZE_L1, M - j);
        for (int i = 0; i < M; i += BLOCK_SIZE_L2){
            int bM = min(BLOCK_SIZE_L2, M - i);
            for (int k = 0; k < M; k += BLOCK_SIZE_L3) {
                int bK = min(BLOCK_SIZE_L3, M - k);

                for (int x = 0; x < bM; x += 32) {
                    for (int y = 0; y < bN; y += 6) {
                        micro_kernel(Ad + i + k * Md, Bd + k + j * Nd, Cd + i + j * Md, x, y, bK, Md, Nd);
                    }
                }
            }
        }
    }    

    for (int i = 0; i < M; i++) {
        memcpy(C + i * M, Cd + i * Md, sizeof(double) * M);
    }

    free(Ad);
    free(Bd);
    free(Cd);
}
