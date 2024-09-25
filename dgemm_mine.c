#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE 96

// 16x6 microkernel
void micro_kernel(double* A, double* B, double* C, int i, int j, int K, int n) {
    __m512d a0, a1, b0, b1;
    
    __m512d c00, c01, c02, c03, c04, c05;
    __m512d c10, c11, c12, c13, c14, c15;

    c00 = _mm512_load_pd(C + j * n + i);
    c01 = _mm512_load_pd(C + (j+1) * n + i);
    c02 = _mm512_load_pd(C + (j+2) * n + i);
    c03 = _mm512_load_pd(C + (j+3) * n + i);
    c04 = _mm512_load_pd(C + (j+4) * n + i);
    c05 = _mm512_load_pd(C + (j+5) * n + i);

    c10 = _mm512_load_pd(C + j * n + (i+8));
    c11 = _mm512_load_pd(C + (j+1) * n + (i+8));
    c12 = _mm512_load_pd(C + (j+2) * n + (i+8));
    c13 = _mm512_load_pd(C + (j+3) * n + (i+8));
    c14 = _mm512_load_pd(C + (j+4) * n + (i+8));
    c15 = _mm512_load_pd(C + (j+5) * n + (i+8));

    for (int k = 0; k < K; ++k) {
        a0 = _mm512_load_pd(A + i + k * n);
        a1 = _mm512_load_pd(A + i + 8 + k * n);

        b0 = _mm512_set1_pd(B[j + k * n]);
        b1 = _mm512_set1_pd(B[(j+1) + k * n]);

        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c11 = _mm512_fmadd_pd(a1, b1, c11);

        b0 = _mm512_set1_pd(B[(j+2) + k * n]);
        b1 = _mm512_set1_pd(B[(j+3) + k * n]);

        c02 = _mm512_fmadd_pd(a0, b0, c02);
        c12 = _mm512_fmadd_pd(a1, b0, c12);
        
        c03 = _mm512_fmadd_pd(a0, b1, c03);
        c13 = _mm512_fmadd_pd(a1, b1, c13);

        b0 = _mm512_set1_pd(B[(j+4) + k * n]);
        b1 = _mm512_set1_pd(B[(j+5) + k * n]);

        c04 = _mm512_fmadd_pd(a0, b0, c04);
        c14 = _mm512_fmadd_pd(a1, b0, c14);
        
        c05 = _mm512_fmadd_pd(a0, b1, c05);
        c15 = _mm512_fmadd_pd(a1, b1, c15);
    }

    _mm512_store_pd(C + j * n + (i), c00); 
    _mm512_store_pd(C + j * n + (i+8), c10); 

    _mm512_store_pd(C + (j+1) * n + (i), c01); 
    _mm512_store_pd(C + (j+1) * n + (i+8), c11);

    _mm512_store_pd(C + (j+2) * n + (i), c02); 
    _mm512_store_pd(C + (j+2) * n + (i+8), c12); 

    _mm512_store_pd(C + (j+3) * n + (i), c03); 
    _mm512_store_pd(C + (j+3) * n + (i+8), c13);

    _mm512_store_pd(C + (j+4) * n + (i), c04); 
    _mm512_store_pd(C + (j+4) * n + (i+8), c14);

    _mm512_store_pd(C + (j+5) * n + (i), c05); 
    _mm512_store_pd(C + (j+5) * n + (i+8), c15);
}

double* alloc(int n) {
    double* ptr = (double*) aligned_alloc(64, sizeof(double) * n);
    memset(ptr, 0, sizeof(double) * n);
    return ptr;
}

void do_block(int Md, double *A, double *B, double *C, const int bM, const int bN, const int bK) {

    for (int i = 0; i < bM; i += 16) {
        for (int j = 0; j < bN; j += 6) {
            micro_kernel(A, B, C, i, j, bK, Md);
        }
    }
    
}


void square_dgemm(const int M, const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C) {

    int Md = (M + 47) / 48 * 48;

    // transpose B
    double *Bt = alloc(M * M);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < M; ++j) {
            Bt[i * M + j] = B[j * M + i];
        }
    }

    double *Ad = alloc(Md * Md);
    double *Bd = alloc(Md * Md);
    double *Cd = alloc(Md * Md);

    for (int i = 0; i < M; ++i) {
        memcpy(&Ad[i * Md], &A[i * M], sizeof(double) * M);
        memcpy(&Bd[i * Md], &Bt[i * M], sizeof(double) * M);
    }

    for (int j = 0; j < Md; j += BLOCK_SIZE) {
        int bN = min(BLOCK_SIZE, Md - j);
        for (int i = 0; i < Md; i += BLOCK_SIZE){
            int bM = min(BLOCK_SIZE, Md - i);
            for (int k = 0; k < Md; k += BLOCK_SIZE) {
                int bK = min(BLOCK_SIZE, Md - k);
                do_block(Md, Ad + i + k * Md, Bd + j + k * Md, Cd + i + j * Md, bM, bN, bK);
            }
        }
    }    

    for (int i = 0; i < M; i++) {
        memcpy(&C[i * M], &Cd[i * Md], sizeof(double) * M);
    }

    free(Ad);
    free(Bd);
    free(Cd);
    free(Bt);
}
