/**
 *  \file mat.c
 *  \desc Implements a sequential matrix multiply algorithm.
 */

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include "mat.h"

/* ------------------------------------------------------------ */

#if defined(USE_MKL)

extern void dgemm_ (const char* transa, const char* transb,
		    int* p_m, int* p_n, int* p_k,
		    const double* p_alpha,
		    const double* A, int* p_lda,
		    const double* B, int* p_ldb,
		    const double* p_beta,
		    double* C, int* p_ldc);

void
mat_multiply (int m, int n, int k,
	      const double* A, int lda, const double* B, int ldb,
	      double* C, int ldc)
{
  assert (A || m <= 0 || k <= 0); assert (lda >= m);
  assert (B || k <= 0 || n <= 0); assert (ldb >= k);
  assert (C || m <= 0 || n <= 0); assert (ldc >= m);
  const double ONE = 1.0;
  dgemm_ ("N", "N", &m, &n, &k, &ONE, A, &lda, B, &ldb, &ONE, C, &ldc);
}
#else
void
mat_multiply (int m, int n, int k,
	      const double* A, int lda, const double* B, int ldb,
	      double* C, int ldc)
{
  assert (A || m <= 0 || k <= 0); assert (lda >= m);
  assert (B || k <= 0 || n <= 0); assert (ldb >= k);
  assert (C || m <= 0 || n <= 0); assert (ldc >= m);
  for (int ii = 0; ii < m; ++ii) {
    for (int jj = 0; jj < n; ++jj) {
      double cij = C[ii + jj*ldc];
      for (int kk = 0; kk < k; ++kk) {
	double tij = A[ii + kk*lda] * B[kk + jj*ldb];
	cij += tij;
      }
      C[ii + jj*ldc] = cij;
    }
  }
}
#endif

/* ------------------------------------------------------------ */

void
mat_multiplyErrorbound (int m, int n, int k,
			const double* A, int lda, const double* B, int ldb,
			double* C, int ldc, double* C_bound, int ldc_bound)
{
  assert (A || m <= 0 || k <= 0); assert (lda >= m);
  assert (B || k <= 0 || n <= 0); assert (ldb >= k);
  assert (C || m <= 0 || n <= 0); assert (ldc >= m);
  for (int ii = 0; ii < m; ++ii) {
    for (int jj = 0; jj < n; ++jj) {
      double cij = 0.0;
      double cij_bound = 0.0;
      for (int kk = 0; kk < k; ++kk) {
	double tij = A[ii + kk*lda] * B[kk + jj*ldb];
	cij += tij;
	cij_bound += fabs (tij);
      }
      C[ii + jj*ldc] = cij;
      C_bound[ii + jj*ldc_bound] = cij_bound;
    }
  }
}

/* ------------------------------------------------------------ */

double *
mat_create (int m, int n)
{
  double* A = (double *)malloc (m * n * sizeof (double));
  assert (A);
  return A;
}

void
mat_free (double* A)
{
  if (A) free (A);
}

/* ------------------------------------------------------------ */

void
mat_randomize (int m, int n, double* A)
{
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      A[i + j*m] = drand48 ();
}

void
mat_setZero (int m, int n, double* A)
{
  if (A)
    bzero (A, m * n * sizeof (double));
}

/* ------------------------------------------------------------ */

void
mat_copyBlock (int m, int n,
	       const double* Src, int ld_src,
	       double* Dest, int ld_dest)
{
  const size_t strip_bytes = m * sizeof (double);
  for (int j = 0; j < n; ++j)
    memcpy (Dest + j*ld_dest, Src + j*ld_src, strip_bytes);
}

/* ------------------------------------------------------------ */

void
mat_dump (const char* tag, int m, int n, const double* A)
{
  fprintf (stderr, ">> Matrix: %s -- %d x %d\n", tag, m, n);
  for (int i = 0; i < m; ++i) {
    fprintf (stderr, "   Row %d:", i);
    for (int j = 0; j < n; ++j)
      fprintf (stderr, " %g", A[i + j*m]);
    fprintf (stderr, "\n");
  }
  fflush (stderr);
}

/* ------------------------------------------------------------ */

/* eof */
