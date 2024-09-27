#include <immintrin.h>
#include <string.h>

const char* dgemm_desc = "My awesome dgemm.";

#define min(a, b) ((a) < (b) ? (a) : (b))
#define BLOCK_SIZE_L1 96
#define BLOCK_SIZE_L2 192
#define BLOCK_SIZE_L3 192

__m512d zero_vec;

// 16x6 microkernel to use registers for mask
void micro_kernel(double* A, double* B, double* C, int i, int j, int K, int M, int rem_cols, __mmask8 high_mask, __mmask8 low_mask) {
    __m512d a0, a1, b0, b1;
    
    __m512d c00, c01, c02, c03, c04, c05;
    __m512d c10, c11, c12, c13, c14, c15;

    c00 = _mm512_mask_load_pd(zero_vec, high_mask, C + j * M + i);
    c01 = _mm512_mask_load_pd(zero_vec, high_mask, C + (j+1) * M + i);
    c02 = _mm512_mask_load_pd(zero_vec, high_mask, C + (j+2) * M + i);
    c03 = _mm512_mask_load_pd(zero_vec, high_mask, C + (j+3) * M + i);
    c04 = _mm512_mask_load_pd(zero_vec, high_mask, C + (j+4) * M + i);
    c05 = _mm512_mask_load_pd(zero_vec, high_mask, C + (j+5) * M + i);

    c10 = _mm512_mask_load_pd(zero_vec, low_mask, C + j * M + (i+8));
    c11 = _mm512_mask_load_pd(zero_vec, low_mask, C + (j+1) * M + (i+8));
    c12 = _mm512_mask_load_pd(zero_vec, low_mask, C + (j+2) * M + (i+8));
    c13 = _mm512_mask_load_pd(zero_vec, low_mask, C + (j+3) * M + (i+8));
    c14 = _mm512_mask_load_pd(zero_vec, low_mask, C + (j+4) * M + (i+8));
    c15 = _mm512_mask_load_pd(zero_vec, low_mask, C + (j+5) * M + (i+8));

    for (int k = 0; k < K; ++k) {
        a0 = _mm512_mask_load_pd(zero_vec, high_mask, A + i + k * M);
        a1 = _mm512_mask_load_pd(zero_vec, low_mask, A + i + 8 + k * M);

        b0 = _mm512_set1_pd(B[j * M + k]);
        c00 = _mm512_fmadd_pd(a0, b0, c00);
        c10 = _mm512_fmadd_pd(a1, b0, c10);
        
        if (rem_cols == 1) continue;
        b1 = _mm512_set1_pd(B[(j+1) * M + k]);
        c01 = _mm512_fmadd_pd(a0, b1, c01);
        c11 = _mm512_fmadd_pd(a1, b1, c11);

        if (rem_cols == 2) continue;
        b0 = _mm512_set1_pd(B[(j+2) * M + k]);
        c02 = _mm512_fmadd_pd(a0, b0, c02);
        c12 = _mm512_fmadd_pd(a1, b0, c12);
        
        if (rem_cols == 3) continue;
        b1 = _mm512_set1_pd(B[(j+3) * M + k]);
        c03 = _mm512_fmadd_pd(a0, b1, c03);
        c13 = _mm512_fmadd_pd(a1, b1, c13);

        if (rem_cols == 4) continue;
        b0 = _mm512_set1_pd(B[(j+4) * M + k]);
        c04 = _mm512_fmadd_pd(a0, b0, c04);
        c14 = _mm512_fmadd_pd(a1, b0, c14);
        
        if (rem_cols == 5) continue;
        b1 = _mm512_set1_pd(B[(j+5) * M + k]);
        c05 = _mm512_fmadd_pd(a0, b1, c05);
        c15 = _mm512_fmadd_pd(a1, b1, c15);
    }

    _mm512_mask_store_pd(C + j * M + (i), high_mask, c00); 
    _mm512_mask_store_pd(C + j * M + (i+8), low_mask, c10); 

    _mm512_mask_store_pd(C + (j+1) * M + (i), high_mask, c01); 
    _mm512_mask_store_pd(C + (j+1) * M + (i+8), low_mask, c11);

    _mm512_mask_store_pd(C + (j+2) * M + (i), high_mask, c02); 
    _mm512_mask_store_pd(C + (j+2) * M + (i+8), low_mask, c12); 

    _mm512_mask_store_pd(C + (j+3) * M + (i), high_mask, c03); 
    _mm512_mask_store_pd(C + (j+3) * M + (i+8), low_mask, c13);

    _mm512_mask_store_pd(C + (j+4) * M + (i), high_mask, c04); 
    _mm512_mask_store_pd(C + (j+4) * M + (i+8), low_mask, c14);

    _mm512_mask_store_pd(C + (j+5) * M + (i), high_mask, c05); 
    _mm512_mask_store_pd(C + (j+5) * M + (i+8), low_mask, c15);
}

// want - A MxM, B MxM, C MxM
void square_dgemm(const int M, const double * restrict A, 
		  const double * restrict B, 
		  double * restrict C) {

    zero_vec = _mm512_setzero_pd();

    for (int j = 0; j < M; j += BLOCK_SIZE_L1) {
        int bN = min(BLOCK_SIZE_L1, M - j);
        for (int i = 0; i < M; i += BLOCK_SIZE_L2){
            int bM = min(BLOCK_SIZE_L2, M - i);
            for (int k = 0; k < M; k += BLOCK_SIZE_L3) {
                int bK = min(BLOCK_SIZE_L3, M - k);

                for (int x = 0; x < bM; x += 16) {
                    for (int y = 0; y < bN; y += 6) {
                        int rem_cols = M - (j + y);
                        int rem_rows = M - (i + x);

                        unsigned short full_mask = (rem_rows >= 16) ? 0xFFFF : (1U << rem_rows) - 1;
                        __mmask8 high_mask, low_mask;
                        
                        if (rem_rows >= 16) {
                            high_mask = 0xFF; // All bits set for high part
                            low_mask = 0xFF;  // All bits set for low part
                        } else if (rem_rows > 8) {
                            high_mask = 0xFF;
                            low_mask = (1U << (rem_rows - 8)) - 1;
                        } else {
                            high_mask = (1U << rem_rows) - 1;
                            low_mask = 0x00;  // No low part if rem_rows less than 8
                        }

                        micro_kernel(A + i + k * M, B + k + j * M, C + i + j * M, x, y, bK, M, rem_cols, high_mask, low_mask);
                    }
                }
            }
        }
    }    
}
