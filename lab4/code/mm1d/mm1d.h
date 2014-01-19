/**
 *  \file mm1d.h
 *  \desc Implements a 1D column-blocked matrix multiply algorithm.
 */

#if !defined (INC_MM1D_H)
#define INC_MM1D_H

#include <mpi.h>
#include "util.h"

/**
 *  \brief Returns the index of the first item on proc rank for n
 *  items distributed in consecutive chunks among P procs.
 */
inline int mm1d_getBlockStart (int n, int P, int rank)
{
  return (rank * (n / P)) + min_int (n % P, rank);
}

/**
 *  \brief Returns the number of items assigned to proc r if there are
 *  n items distributed in consecutive chunks among P procs.
 */
inline int mm1d_getBlockLength (int n, int P, int rank)
{
  return (n / P) + (rank < (n % P));
}

/**
 *  \brief Given an m x n matrix A stored on process 0, this
 *  collective routine distributes block columns of A among all
 *  processors in comm, returning a pointer to the local block.
 */
double* mm1d_distribute (int m, int n, const double* A, MPI_Comm comm);

/**
 *  \brief Performs a distributed 1D block row matrix multiply.
 *
 *  A is m x n, B is k x n, and C is m x n. All three matrices are
 *  distributed across the processors in comm by block columns. Caller
 *  may optionally provide non-NULL values for p_t_comp and p_t_comm
 *  to get the computation and communication time breakdown,
 *  respectively.
 */
void mm1d_mult (int m, int n, int k,
		const double* A_local, const double* B_local,
		double* C_local, MPI_Comm comm,
		double* p_t_comp, double* p_t_comm);

/**
 * \brief Allocates a M x N matrix across all processes in comm using
 * a 1D block column partitioning, returning a pointer to the local
 * block.
 */
double* mm1d_alloc (int m, int n, MPI_Comm comm);

/** \brief Sets matrix entries to random values in [0, 1]. */
void mm1d_randomize (int m, int n, double* A_local, MPI_Comm comm);

/** \brief Sets matrix entries to 0. */
void mm1d_setZero (int m, int n, double* A_local, MPI_Comm comm);

/** \brief Deallocates A. */
void mm1d_free (double* A_local, MPI_Comm comm);

/** \brief (Debug) Print the matrix. */
void mm1d_dump (const char* tag, int m, int n, const double* A_local,
		MPI_Comm comm);
#endif

/* eof */
