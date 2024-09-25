#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE 280

// 16x6 microkernel
void micro_kernel(double* A, double* B, double* C, int i, int j, int K, int Md) {
    __m512d a0, b0, b1;
    
    __m512d c0, c1, c2, c3, c4, c5, c6, c7;

    c0 = _mm512_loadu_pd(C + j * Md + i);
    c1 = _mm512_loadu_pd(C + (j+1) * Md + i);
    c2 = _mm512_loadu_pd(C + (j+2) * Md + i);
    c3 = _mm512_loadu_pd(C + (j+3) * Md + i);
    c4 = _mm512_loadu_pd(C + (j+4) * Md + i);
    c5 = _mm512_loadu_pd(C + (j+5) * Md + i);
    c6 = _mm512_loadu_pd(C + (j+6) * Md + i);
    c7 = _mm512_loadu_pd(C + (j+7) * Md + i);


    for (int k = 0; k < K; ++k) {
        a0 = _mm512_load_pd(A + i + k * Md);

        b0 = _mm512_set1_pd(B[j + k * Md]);
        b1 = _mm512_set1_pd(B[(j+1) + k * Md]);

        c0 = _mm512_fmadd_pd(a0, b0, c0);
        c1 = _mm512_fmadd_pd(a0, b1, c1);

        b0 = _mm512_set1_pd(B[(j+2) + k * Md]);
        b1 = _mm512_set1_pd(B[(j+3) + k * Md]);

        c2 = _mm512_fmadd_pd(a0, b0, c2);
        c3 = _mm512_fmadd_pd(a0, b1, c3);

        b0 = _mm512_set1_pd(B[(j+4) + k * Md]);
        b1 = _mm512_set1_pd(B[(j+5) + k * Md]);

        c4 = _mm512_fmadd_pd(a0, b0, c4);
        c5 = _mm512_fmadd_pd(a0, b1, c5);

        b0 = _mm512_set1_pd(B[(j+6) + k * Md]);
        b1 = _mm512_set1_pd(B[(j+7) + k * Md]);

        c6 = _mm512_fmadd_pd(a0, b0, c6);
        c7 = _mm512_fmadd_pd(a0, b1, c7);
    }

    _mm512_storeu_pd(C + j * Md + i, c0); 
    _mm512_storeu_pd(C + (j+1) * Md + i, c1); 
    _mm512_storeu_pd(C + (j+2) * Md + i, c2); 
    _mm512_storeu_pd(C + (j+3) * Md + i, c3); 
    _mm512_storeu_pd(C + (j+4) * Md + i, c4); 
    _mm512_storeu_pd(C + (j+5) * Md + i, c5); 
    _mm512_storeu_pd(C + (j+6) * Md + i, c6); 
    _mm512_storeu_pd(C + (j+7) * Md + i, c7);  
}

double* alloc(int n) {
    double* ptr = (double*) aligned_alloc(64, sizeof(double) * n);
    memset(ptr, 0, sizeof(double) * n);
    return ptr;
}


/*
    A is bM-by-bK
    B is bK-by-bN
    C is bM-by-bN
*/
void do_block(double *A, double *B, double *C, const int bM, const int bN, const int bK, int Md) {

    for (int i = 0; i < bM; i += 8) {
        for (int j = 0; j < bN; j += 8) {
            micro_kernel(A, B, C, i, j, bK, Md);
        }
    }
    
}


void square_dgemm(const int M, const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C) {

    int Md = (M + 7) / 8 * 8;

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

   for (int j = 0; j < M; j += BLOCK_SIZE) {
        int bN = min(BLOCK_SIZE, M - j);
        for (int i = 0; i < M; i += BLOCK_SIZE){
            int bM = min(BLOCK_SIZE, M - i);
            for (int k = 0; k < M; k += BLOCK_SIZE) {
                int bK = min(BLOCK_SIZE, M - k);
                do_block(Ad + i + k * Md, Bd + j + k * Md, Cd + i + j * Md, bM, bN, bK, Md);
            }
        }
    }    

    for (int i = 0; i < M; i++) {
        memcpy(C + i * M, Cd + i * Md, sizeof(double) * M);
    }

    free(Ad);
    free(Bd);
    free(Cd);
    free(Bt);
}
