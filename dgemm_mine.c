#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE 96

// 16x6 microkernel
void micro_kernel(double* A, double* B, double* C, int i, int j, int K, int Md, int Nd) {
    __m512d a0, a1, b0, b1;
    
    __m512d c00, c01, c02, c03, c04, c05;
    __m512d c10, c11, c12, c13, c14, c15;

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

    for (int k = 0; k < K; ++k) {
        a0 = _mm512_load_pd(A + i + k * Md);
        a1 = _mm512_load_pd(A + i + 8 + k * Md);

        b0 = _mm512_set1_pd(B[j + k * Nd]);
        b1 = _mm512_set1_pd(B[(j+1) + k * Nd]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c11 = _mm512_fmadd_pd(a1, b1, c11);

        b0 = _mm512_set1_pd(B[(j+2) + k * Nd]);
        b1 = _mm512_set1_pd(B[(j+3) + k * Nd]);

        c02 = _mm512_fmadd_pd(a0, b0, c02);
        c12 = _mm512_fmadd_pd(a1, b0, c12);
        
        c03 = _mm512_fmadd_pd(a0, b1, c03);
        c13 = _mm512_fmadd_pd(a1, b1, c13);

        b0 = _mm512_set1_pd(B[(j+4) + k * Nd]);
        b1 = _mm512_set1_pd(B[(j+5) + k * Nd]);

        c04 = _mm512_fmadd_pd(a0, b0, c04);
        c14 = _mm512_fmadd_pd(a1, b0, c14);
        
        c05 = _mm512_fmadd_pd(a0, b1, c05);
        c15 = _mm512_fmadd_pd(a1, b1, c15);
    }

    _mm512_store_pd(C + j * Md + (i), c00); 
    _mm512_store_pd(C + j * Md + (i+8), c10); 

    _mm512_store_pd(C + (j+1) * Md + (i), c01); 
    _mm512_store_pd(C + (j+1) * Md + (i+8), c11);

    _mm512_store_pd(C + (j+2) * Md + (i), c02); 
    _mm512_store_pd(C + (j+2) * Md + (i+8), c12); 

    _mm512_store_pd(C + (j+3) * Md + (i), c03); 
    _mm512_store_pd(C + (j+3) * Md + (i+8), c13);

    _mm512_store_pd(C + (j+4) * Md + (i), c04); 
    _mm512_store_pd(C + (j+4) * Md + (i+8), c14);

    _mm512_store_pd(C + (j+5) * Md + (i), c05); 
    _mm512_store_pd(C + (j+5) * Md + (i+8), c15);
}


void square_dgemm(const int M, const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C) {

    int Md = (M + 15) / 16 * 16;
    int Nd = (M + 5) / 6 * 6;

    // transpose B directly into Bd
    
    double *Bd = _mm_malloc(sizeof(double) * M * Nd, 64);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            Bd[j * Nd + i] = B[i * M + j];
        }
    }

    double *Ad = _mm_malloc(sizeof(double) * Md * M, 64);
    double *Cd = _mm_malloc(sizeof(double) * Md * Nd, 64);

    for (int i = 0; i < M; ++i) {
        memcpy(&Ad[i * Md], &A[i * M], sizeof(double) * M);
    }

    for (int j = 0; j < M; j += BLOCK_SIZE) {
        int bN = min(BLOCK_SIZE, M - j);
        for (int i = 0; i < M; i += BLOCK_SIZE){
            int bM = min(BLOCK_SIZE, M - i);
            for (int k = 0; k < M; k += BLOCK_SIZE) {
                int bK = min(BLOCK_SIZE, M - k);

                for (int x = 0; x < bM; x += 16) {
                    for (int y = 0; y < bN; y += 6) {
                        micro_kernel(Ad + i + k * Md, Bd + j + k * Nd, Cd + i + j * Md, x, y, bK, Md, Nd);
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
