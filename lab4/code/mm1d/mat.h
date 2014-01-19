/**
 *  \file mat.h
 *  \desc Implements a sequential matrix multiply algorithm.
 */

#if !defined (INC_MAT_H)
#define INC_MAT_H /*!< mat.h included */

/** \brief Returns an uninitialized buffer for an m x n matrix. */
double* mat_create (int m, int n);

/** \brief Free matrix buffer. */
void mat_free (double* A);

/** \brief Set matrix entries to random values in [0, 1]. */
void mat_randomize (int m, int n, double* A);

/** \brief Set matrix entries to zero. */
void mat_setZero (int m, int n, double* A);

/**
 *  \brief Copies an m x n submatrix from Src to Dest.
 *
 *  The matrices Src and Dest are stored in column-major order, with
 *  leading dimensions ld_src and ld_dest, respectively.
 */
void mat_copyBlock (int m, int n, const double* Src, int ld_src,
		    double* Dest, int ld_dest);

/**
 *  \brief Performs the sequential matrix multiply operation, C <- C +
 *  A*B.
 *
 *  \note Regarding matrix dimensions, A is m x k, B is k x n, and C
 *  is m x n. Moreover, A, B, and C are assumed to be stored in
 *  column-major order, with leading dimensions given by lda, ldb, and
 *  ldc, respectively.
 */
void mat_multiply (int m, int n, int k,
		   const double* A, int lda, const double* B, int ldb,
		   double* C, int ldc);


/** \brief Same as mat_multiply, but with a computed error bound. */
void mat_multiplyErrorbound (int m, int n, int k,
			     const double* A, int lda,
			     const double* B, int ldb,
			     double* C, int ldc,
			     double* C_bound, int ldc_bound);

/** \brief (Debug) Dump matrix entries. */
void mat_dump (const char* tag, int m, int n, const double* A);

#endif

/* eof */
