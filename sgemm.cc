#include <iostream>
#include <iomanip>
#include <immintrin.h>

typedef union {
  __m256 v;
  float f[8];
} v2f_t;

typedef union {
  __m256 v;
  float f[8];
} m256_t;

typedef union {
  __m128 v;
  float f[4];
} m128_t;

inline void pack_a(int k, const float* a, int lda, float* to) {
  for(int j=0; j<k; j++) {
    const float *a_ij_ptr = &a[(j*lda)+0]; 
    *to = *a_ij_ptr;
    *(to+1) = *(a_ij_ptr+1);
    *(to+2) = *(a_ij_ptr+2);
    *(to+3) = *(a_ij_ptr+3);
    *(to+4) = *(a_ij_ptr+4);
    *(to+5) = *(a_ij_ptr+5);
    *(to+6) = *(a_ij_ptr+6);
    *(to+7) = *(a_ij_ptr+7);
    to += 8;
  }
}

inline void pack_b(int k, const float* b, int lb, float* to) {
  int i;
  const float *b_i0_ptr = &b[0], *b_i1_ptr = &b[(1*lb)],
              *b_i2_ptr = &b[(2*lb)], *b_i3_ptr = &b[(3*lb)],
              *b_i4_ptr = &b[(4*lb)], *b_i5_ptr = &b[(5*lb)],
              *b_i6_ptr = &b[(6*lb)], *b_i7_ptr = &b[(7*lb)];
  for(i=0; i<k; i++) {
    *to     = *b_i0_ptr;
    *(to+1) = *(b_i1_ptr);
    *(to+2) = *(b_i2_ptr);
    *(to+3) = *(b_i3_ptr);
    *(to+4) = *(b_i4_ptr);
    *(to+5) = *(b_i5_ptr);
    *(to+6) = *(b_i6_ptr);
    *(to+7) = *(b_i7_ptr);
    to += 8;
    b_i0_ptr++; b_i1_ptr++; b_i2_ptr++;
    b_i3_ptr++; b_i4_ptr++; b_i5_ptr++;
    b_i6_ptr++; b_i7_ptr++;
  }
}


template <int mb, int kb, int th, int m, int n, int k>
inline void sgemm(float* a, float* b, float* c) {

  int i, ii, iii, iiii, iiiii;
  int ib, iib;

  for(i=0; i<k; i+=kb) {
    ib = std::min(k-i, kb);
    for(ii=0; ii<m; ii+=mb) {
      iib = std::min(m-ii, mb);

      // inner kernel
      float* pa = new alignas(32) float[ib*m];
      static float* pb = new alignas(32) float[iib*n];

      // stand in for the indexes given to inner kernel
      float* wa = &a[i*k+ii];
      float* wb = &b[i];
      float* wtc = &c[ii];

      for(iii=0; iii<n; iii+=8) { // loop over all columns of C unrolled by 8
        if(ii==0) pack_b(ib, &wb[(iii*n)], n, &pb[iii*ib]);

        // loop over rows of c until block, unrolled by 8
        for(iiii=0; iiii<ib; iiii+=8) { 
          if(iii==0) pack_a(ib, &wa[iiii], k, &pa[iiii*ib]);

          // kernel

          float* wpa = &pa[iiii*ib];
          float* wpb = &pb[iii*ib];
          float* wc = &wtc[iii*n+iiii];

          v2f_t c_0007_vreg, c_1017_vreg, c_2027_vreg, c_3037_vreg,
                c_4047_vreg, c_5057_vreg, c_6067_vreg, c_7077_vreg,
                a_1_vreg,
                b_p0_vreg;

          c_0007_vreg.v = _mm256_setzero_ps();
          c_1017_vreg.v = _mm256_setzero_ps();
          c_2027_vreg.v = _mm256_setzero_ps();
          c_3037_vreg.v = _mm256_setzero_ps();
          c_4047_vreg.v = _mm256_setzero_ps();
          c_5057_vreg.v = _mm256_setzero_ps();
          c_6067_vreg.v = _mm256_setzero_ps();
          c_7077_vreg.v = _mm256_setzero_ps();

          // loop over columns of a
          for(iiiii=0; iiiii<ib; iiiii++) {
            __builtin_prefetch(wpa+8);
            __builtin_prefetch(wpb+8);

            a_1_vreg.v = _mm256_load_ps( (float*) wpa );
            wpa += 8;

            b_p0_vreg.v = _mm256_load_ps( (float*) wpb);
            wpb += 8;

            c_0007_vreg.v += a_1_vreg.v * b_p0_vreg.f[0];
            c_1017_vreg.v += a_1_vreg.v * b_p0_vreg.f[1];
            c_2027_vreg.v += a_1_vreg.v * b_p0_vreg.f[2];
            c_3037_vreg.v += a_1_vreg.v * b_p0_vreg.f[3];
            c_4047_vreg.v += a_1_vreg.v * b_p0_vreg.f[4];
            c_5057_vreg.v += a_1_vreg.v * b_p0_vreg.f[5];
            c_6067_vreg.v += a_1_vreg.v * b_p0_vreg.f[6];
            c_7077_vreg.v += a_1_vreg.v * b_p0_vreg.f[7];

          }

          wc[(0*n)+0] += c_0007_vreg.f[0]; wc[(1*n)+0] += c_1017_vreg.f[0]; 
          wc[(2*n)+0] += c_2027_vreg.f[0]; wc[(3*n)+0] += c_3037_vreg.f[0]; 
          wc[(4*n)+0] += c_4047_vreg.f[0]; wc[(5*n)+0] += c_5057_vreg.f[0]; 
          wc[(6*n)+0] += c_6067_vreg.f[0]; wc[(7*n)+0] += c_7077_vreg.f[0]; 

          wc[(0*n)+1] += c_0007_vreg.f[1]; wc[(1*n)+1] += c_1017_vreg.f[1]; 
          wc[(2*n)+1] += c_2027_vreg.f[1]; wc[(3*n)+1] += c_3037_vreg.f[1]; 
          wc[(4*n)+1] += c_4047_vreg.f[1]; wc[(5*n)+1] += c_5057_vreg.f[1]; 
          wc[(6*n)+1] += c_6067_vreg.f[1]; wc[(7*n)+1] += c_7077_vreg.f[1]; 

          wc[(0*n)+2] += c_0007_vreg.f[2]; wc[(1*n)+2] += c_1017_vreg.f[2]; 
          wc[(2*n)+2] += c_2027_vreg.f[2]; wc[(3*n)+2] += c_3037_vreg.f[2]; 
          wc[(4*n)+2] += c_4047_vreg.f[2]; wc[(5*n)+2] += c_5057_vreg.f[2]; 
          wc[(6*n)+2] += c_6067_vreg.f[2]; wc[(7*n)+2] += c_7077_vreg.f[2]; 

          wc[(0*n)+3] += c_0007_vreg.f[3]; wc[(1*n)+3] += c_1017_vreg.f[3]; 
          wc[(2*n)+3] += c_2027_vreg.f[3]; wc[(3*n)+3] += c_3037_vreg.f[3]; 
          wc[(4*n)+3] += c_4047_vreg.f[3]; wc[(5*n)+3] += c_5057_vreg.f[3]; 
          wc[(6*n)+3] += c_6067_vreg.f[3]; wc[(7*n)+3] += c_7077_vreg.f[3]; 

          wc[(0*n)+4] += c_0007_vreg.f[4]; wc[(1*n)+4] += c_1017_vreg.f[4]; 
          wc[(2*n)+4] += c_2027_vreg.f[4]; wc[(3*n)+4] += c_3037_vreg.f[4]; 
          wc[(4*n)+4] += c_4047_vreg.f[4]; wc[(5*n)+4] += c_5057_vreg.f[4]; 
          wc[(6*n)+4] += c_6067_vreg.f[4]; wc[(7*n)+4] += c_7077_vreg.f[4]; 

          wc[(0*n)+5] += c_0007_vreg.f[5]; wc[(1*n)+5] += c_1017_vreg.f[5]; 
          wc[(2*n)+5] += c_2027_vreg.f[5]; wc[(3*n)+5] += c_3037_vreg.f[5]; 
          wc[(4*n)+5] += c_4047_vreg.f[5]; wc[(5*n)+5] += c_5057_vreg.f[5]; 
          wc[(6*n)+5] += c_6067_vreg.f[5]; wc[(7*n)+5] += c_7077_vreg.f[5]; 

          wc[(0*n)+6] += c_0007_vreg.f[6]; wc[(1*n)+6] += c_1017_vreg.f[6]; 
          wc[(2*n)+6] += c_2027_vreg.f[6]; wc[(3*n)+6] += c_3037_vreg.f[6]; 
          wc[(4*n)+6] += c_4047_vreg.f[6]; wc[(5*n)+6] += c_5057_vreg.f[6]; 
          wc[(6*n)+6] += c_6067_vreg.f[6]; wc[(7*n)+6] += c_7077_vreg.f[6]; 

          wc[(0*n)+7] += c_0007_vreg.f[7]; wc[(1*n)+7] += c_1017_vreg.f[7]; 
          wc[(2*n)+7] += c_2027_vreg.f[7]; wc[(3*n)+7] += c_3037_vreg.f[7]; 
          wc[(4*n)+7] += c_4047_vreg.f[7]; wc[(5*n)+7] += c_5057_vreg.f[7]; 
          wc[(6*n)+7] += c_6067_vreg.f[7]; wc[(7*n)+7] += c_7077_vreg.f[7]; 
        }
      }
    }
  }
}


#define N 16

int main() { 
  float* a = new alignas(32) float[N*N];
  float* b = new alignas(32) float[N*N];
  float* c = new alignas(32) float[N*N];

  for(int i=0; i<N*N; i++) {
    a[i] = i;
    b[i] = i;
  }

  sgemm<8, 8, 4, N, N, N>(a, b, c);

  std::cout << "\n\n";
  for(int i=0; i<N*N; i++) {
    if(i%N==0) std::cout << "\n";
    std::cout << std::setw(6) << c[i] << ", ";
  }
  std::cout << "\n\n";
}
