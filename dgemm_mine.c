#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

// try 8x4 kernel first
void kernel(double* A, double* B, double* C, int i, int j, int r, int n) {
    __m256d a0, a1, b0, b1, b2, b3;
    
    __m256d c00 = _mm256_setzero_pd(); __m256d c10 = _mm256_setzero_pd(); 
    __m256d c01 = _mm256_setzero_pd(); __m256d c11 = _mm256_setzero_pd();

    __m256d c02 = _mm256_setzero_pd(); __m256d c12 = _mm256_setzero_pd(); 
    __m256d c03 = _mm256_setzero_pd(); __m256d c13 = _mm256_setzero_pd();

    for (int k = 0; k < r; k++) {
        a0 = _mm256_load_pd(&A[i + k * n]);
        a1 = _mm256_load_pd(&A[i + 4 + k * n]);

        b0 = _mm256_set1_pd(B[j + k * n]);
        b1 = _mm256_set1_pd(B[(j+1) + k * n]);
        b2 = _mm256_set1_pd(B[(j+2) + k * n]);
        b3 = _mm256_set1_pd(B[(j+3) + k * n]);

        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c11 = _mm256_fmadd_pd(a1, b1, c11);

        c02 = _mm256_fmadd_pd(a0, b2, c02);
        c12 = _mm256_fmadd_pd(a1, b2, c12);
        
        c03 = _mm256_fmadd_pd(a0, b3, c03);
        c13 = _mm256_fmadd_pd(a1, b3, c13);
    }

    _mm256_store_pd(&C[j * n + (i)], c00); 
    _mm256_store_pd(&C[j * n + (i+4)], c10); 

    _mm256_store_pd(&C[(j+1) * n + (i)], c01); 
    _mm256_store_pd(&C[(j+1) * n + (i+4)], c11);

    _mm256_store_pd(&C[(j+2) * n + (i)], c02); 
    _mm256_store_pd(&C[(j+2) * n + (i+4)], c12); 

    _mm256_store_pd(&C[(j+3) * n + (i)], c03); 
    _mm256_store_pd(&C[(j+3) * n + (i+4)], c13);
}

double* alloc(int n) {
    double* ptr = (double*) aligned_alloc(64, sizeof(double) * n);
    memset(ptr, 0, sizeof(double) * n);
    return ptr;
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    // we can use kernel size 8x4 to start
    // column major order
    int Mx = (M + 7) / 8 * 8;
    int My = (M + 3) / 4 * 4;

    double *Bt = alloc(M * M);
    for (int i = 0; i < M; i++) { 
        for (int j = 0; j < M; j++) {
            Bt[i * M + j] = B[j * M + i];
        }
    }

    double *a = alloc(Mx * My);
    double *b = alloc(Mx * My);
    double *c = alloc(Mx * My);

    for (int i = 0; i < M; i++) {
        memcpy(&a[i * My], &A[i * M], sizeof(double) * M);
        memcpy(&b[i * My], &Bt[i * M], sizeof(double) * M);
    }

    for (int i = 0; i < Mx; i += 8) {
        for (int j = 0; j < My; j += 4) {
            kernel(a, b, c, i, j, M, My);
        }
    }

    for (int i = 0; i < M; i++) {
        memcpy(&C[i * M], &c[i * My], sizeof(double) * M);
    }

    free(a);
    free(b);
    free(c);
    free(Bt);
}
