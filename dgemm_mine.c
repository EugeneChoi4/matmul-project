#include <immintrin.h>
#include <string.h>
#include <stdio.h>

const char* dgemm_desc = "My awesome dgemm.";

// try 8x8 kernel first
void kernel(double* __restrict__ A, double* __restrict__ B, double* C, int i, int j, int l, int r, int n) {
    __m256d a0, a1, a2, a3, b0, b1, b2, b3;
    
    __m256d c00 = _mm256_setzero_pd(); 
    __m256d c01 = _mm256_setzero_pd(); 
    __m256d c02 = _mm256_setzero_pd();
    __m256d c03 = _mm256_setzero_pd();

    __m256d c10 = _mm256_setzero_pd(); 
    __m256d c11 = _mm256_setzero_pd(); 
    __m256d c12 = _mm256_setzero_pd();
    __m256d c13 = _mm256_setzero_pd();

    for (int k = l; k < r; k++) {
        // a0 = _mm256_load_pd(A + i + k * n);
        // a1 = _mm256_load_pd(A + i + (k + 1) * n);
        // // printf("%f,%f\n",a0,a1);
        // // fflush(stdout);

        // b0 = _mm256_set1_pd(B[k * n + j]);
        // b1 = _mm256_set1_pd(B[(k+1) * n + j]);
        // b2 = _mm256_set1_pd(B[(k+2) * n + j]);
        // b3 = _mm256_set1_pd(B[(k+3) * n + j]);

        b0 = _mm256_load_pd(&B[k * n + j]);
        b1 = _mm256_load_pd(&B[k * n + j + 4]);
        b2 = _mm256_load_pd(&B[k * n + j + 8]);
        b3 = _mm256_load_pd(&B[k * n + j + 12]);

        a0 = _mm256_broadcast_pd(&A[i * n + k]);
        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c01 = _mm256_fmadd_pd(a0, b1, c01);
        c02 = _mm256_fmadd_pd(a0, b2, c02);
        c03 = _mm256_fmadd_pd(a0, b3, c03);

        a1 = _mm256_broadcast_pd(&A[(i+1) * n + k]);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c11 = _mm256_fmadd_pd(a1, b1, c11);
        c12 = _mm256_fmadd_pd(a1, b2, c12);
        c13 = _mm256_fmadd_pd(a1, b3, c13);
    }

    _mm256_store_pd(C, _mm256_add_pd(c00, _mm256_load_pd(C))); _mm256_store_pd(C+4, _mm256_add_pd(c01, _mm256_load_pd(C+4))); _mm256_store_pd(C+8, _mm256_add_pd(c02, _mm256_load_pd(C+8))); _mm256_store_pd(C+12, _mm256_add_pd(c03, _mm256_load_pd(C+12)));
    _mm256_store_pd(C+16, _mm256_add_pd(c10, _mm256_load_pd(C+16))); _mm256_store_pd(C+20, _mm256_add_pd(c11, _mm256_load_pd(C+20))); _mm256_store_pd(C+24, _mm256_add_pd(c12, _mm256_load_pd(C+24))); _mm256_store_pd(C+28, _mm256_add_pd(c13, _mm256_load_pd(C+28)));
}

void square_dgemm(const int M, const double *A, const double *B, double *C) {
    // we can use kernel size 8x8 to start
    // column major order
    int Mx = (M + 7) / 8 * 8;
    int My = (M + 7) / 8 * 8;

    double *a = malloc(Mx * My * sizeof(double));
    double *b = malloc(Mx * My * sizeof(double));
    double *c = malloc(Mx * My * sizeof(double));


    for (int i = 0; i < M; i++) {
        memcpy(&a[i * My], &A[i * M], sizeof(double) * M);
        memcpy(&b[i * My], &B[i * M], sizeof(double) * M);
    }

    for (int i = 0; i < Mx; i += 8) {
        for (int j = 0; j < My; j += 8) {
            kernel(a, b, c, i, j, 0, M, My);
        }
    }

    for (int i = 0; i < M; i++) {
        memcpy(&C[i * M], &c[i * My], sizeof(double) * M);
    }

    free(a);
    free(b);
    free(c);
}
