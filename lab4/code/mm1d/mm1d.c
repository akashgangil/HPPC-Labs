/**
 *  \file mm1d.c
 *  \desc Implements a 1D column-blocked matrix multiply algorithm.
 */

#include <stdlib.h>
#include <string.h>
#include "mat.h"
#include "mm1d.h"
#include "mpi_helper.h"

double *
mm1d_distribute (int m, int n, const double* A, MPI_Comm comm)
{
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  int n_local = mm1d_getBlockLength (n, P, rank);

  double* A_local = (double *)malloc (m * n_local * sizeof (double));
  mpih_assert (A_local != NULL);

  int* sendcounts = NULL;
  int* offsets = NULL;
  if (rank == 0) {
    sendcounts = (int *)malloc (P * sizeof (int));
    offsets = (int *)malloc (P * sizeof (int));
    mpih_assert (sendcounts && offsets);
    for (int r_dest = 0; r_dest < P; ++r_dest) {
      int j = mm1d_getBlockStart (n, P, r_dest);
      int n_j = mm1d_getBlockLength (n, P, r_dest);
      sendcounts[r_dest] = m * n_j;
      offsets[r_dest] = m * j;
    }
  }

  int retcode = MPI_Scatterv ((void *)A, sendcounts, offsets, MPI_DOUBLE,
			      A_local, m * n_local, MPI_DOUBLE,
			      0, comm);
  mpih_assert (retcode == MPI_SUCCESS);
  return A_local;
}

/* ------------------------------------------------------------ */

void
mm1d_mult (int m, int n, int k,
	   const double* A_local, const double* B_local,
	   double* C_local, MPI_Comm comm,
	   double* p_t_comp, double* p_t_comm)
{
  int P = mpih_getSize (comm); /* No. of processes */
  int r = mpih_getRank (comm); /* Rank (logical ID) of current process */
  int r_left = (r + P - 1) % P; /* Rank of left neighbor */
  int r_right = (r + 1) % P; /* Rank of right neighbor */

  int k_local = mm1d_getBlockLength (k, P, r);
  const int n_local = mm1d_getBlockLength (n, P, r);
  const int k_local_max = mm1d_getBlockLength (k, P, 0);

  double* A_local_working = (double *)malloc (m * k_local_max * sizeof (double));
  double* A_local_recv = (double *)malloc (m * k_local_max * sizeof (double));
  mpih_assert (A_local_working && A_local_recv);
  memcpy (A_local_working, A_local, m * k_local * sizeof (double));

  /* Internal timers */
  double t_comp = 0;
  double t_comm = 0;

  for (int iter = 0; iter < P; ++iter) {
    int r_effective = (r + P - iter) % P;
    int k0 = mm1d_getBlockStart (k, P, r_effective);
    k_local = mm1d_getBlockLength (k, P, r_effective);

    double t_start = MPI_Wtime ();
    mat_multiply (m, n_local, k_local,
		  A_local_working, m, &(B_local[k0]), k, C_local, m);
    t_comp += MPI_Wtime () - t_start;

    int r_effective_next = (r + P - iter - 1) % P;
    int k_local_next = mm1d_getBlockLength (k, P, r_effective_next);
    MPI_Status stat;
    t_start = MPI_Wtime ();
    MPI_Sendrecv (A_local_working, m * k_local, MPI_DOUBLE, r_right, r,
		  A_local_recv, m * k_local_next, MPI_DOUBLE, r_left, r_left,
		  comm, &stat);
    t_comm += MPI_Wtime () - t_start;
    swapPointers_double (&A_local_working, &A_local_recv);
  }

  free (A_local_working);
  free (A_local_recv);

  if (p_t_comp) *p_t_comp += t_comp;
  if (p_t_comm) *p_t_comm += t_comm;
}

/* ------------------------------------------------------------ */

double *
mm1d_alloc (int m, int n, MPI_Comm comm)
{
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  int n_local = mm1d_getBlockLength (n, P, rank);
  double* A_local = (double *)malloc (m * n_local * sizeof (double));
  mpih_assert (A_local != NULL);
  return A_local;
}

void
mm1d_randomize (int m, int n, double* A_local, MPI_Comm comm)
{
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  int n_local = mm1d_getBlockLength (n, P, rank);
  mpih_assert (A_local || !m || !n_local);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n_local; ++j)
      A_local[i + j*m] = drand48 ();
}

void
mm1d_setZero (int m, int n, double* A_local, MPI_Comm comm)
{
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  int n_local = mm1d_getBlockLength (n, P, rank);
  mpih_assert (A_local || !m || !n_local);
  if (A_local)
    bzero (A_local, m * n_local * sizeof (double));
}

void
mm1d_free (double* A_local, MPI_Comm comm)
{
  if (A_local) free (A_local);
}

/* ------------------------------------------------------------ */

void
mm1d_dump (const char* tag, int m, int n, const double* A_local, MPI_Comm comm)
{
  int rank = mpih_getRank (comm);
  int P = mpih_getSize (comm);
  for (int r = 0; r < P; ++r) {
    if (rank == r) {
      fflush (stderr);
      const int n_local = mm1d_getBlockLength (n, P, rank);
      mpih_debugmsg (comm, ">> Matrix: %s -- %d x %d (n=%d)\n", tag, m, n_local, n);
      for (int i = 0; i < m; ++i) {
	fprintf (stderr, "   Row %d:", i);
	for (int j = 0; j < n_local; ++j) {
	  fprintf (stderr, " %g", A_local[i + j*m]);
	}
	fprintf (stderr, "\n");
      }
    }
    MPI_Barrier (comm);
  }
}

/* eof */
