/*
 *  \file driver1d.c
 *
 *  \brief Driver program for a distributed 1D matrix multiply
 *  timing/testing program.
 *
 *  Adapted from code by Jason Riedy, David Bindel, David Garmire,
 *  Kent Czechowski, Aparna Chandramowlishwaran, and Richard Vuduc.
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>

#include <float.h>
#include <math.h>

#include "util.h"

#include <mpi.h>
#include "mpi_helper.h"

#include "mat.h" // sequential algorithm
#include "mm1d.h" // 1D block column algorithm

/* ------------------------------------------------------------ */

/** Prints help message */
static void usage__ (const char* progname);

/** \brief Checks the distributed matrix multiply routine */
static void verify__ (int m, int n, int k);

/**
 *  \brief Print aggregate execution time statistics for each of the
 *  given measurements 't[0..n_t-1]' on the local processor.
 *
 *  \note Set 'debug' to a non-zero value to print to stderr instead
 *  of stdout.
 */
static void summarize__ (int m, int n, int k,
			 const double* t, int n_t,
			 MPI_Comm comm, int debug);

/** \brief Checks the distributed matrix multiply routine */
static void benchmark__ (int m, int n, int k);

/* ------------------------------------------------------------ */

/** Program starts here */
int
main (int argc, char** argv)
{
  int retcode = MPI_Init (&argc, &argv);
  mpih_assert (retcode == MPI_SUCCESS);

  int rank = mpih_getRank (MPI_COMM_WORLD);
  int P = mpih_getSize (MPI_COMM_WORLD);

  srand48 ((long)rank);

  int M, N, K; /* matrix dimensions */
  if (rank == 0) { /* p0 parses the command-line arguments */
    if (argc != 4) {
      usage__ (argv[0]);
      MPI_Abort (MPI_COMM_WORLD, 1);
    }
    M = atoi (argv[1]);  mpih_assert (M > 0);
    N = atoi (argv[2]);  mpih_assert (N > 0);
    K = atoi (argv[3]);  mpih_assert (K > 0);
  }

  /* p0 then distributes the program arguments */
  MPI_Bcast (&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast (&K, 1, MPI_INT, 0, MPI_COMM_WORLD);
  mpih_debugmsg (MPI_COMM_WORLD, "Matrix dimensions: M=%d, N=%d, K=%d\n", M, N, K);

  verify__ (M, N, K);
  benchmark__ (M, N, K);

  MPI_Finalize ();
  return 0;
}

static
void
usage__ (const char* progname)
{
  fprintf (stderr, "\n");
  fprintf (stderr, "usage: %s <m> <n> <k>\n", progname);
  fprintf (stderr, "\n");
  fprintf (stderr,
	   "Performs C <- C + A*B using a 1D block row algorithm.\n");
  fprintf (stderr, "\n");
}

static
int
isEnvEnabled_mpi__ (MPI_Comm comm, const char* var, int def_val)
{
  int rank = mpih_getRank (comm);
  int val;
  if (rank == 0)
    val = env_isEnabled (var, def_val) || env_getInt (var, def_val);
  MPI_Bcast (&val, 1, MPI_INT, 0, comm);
  if (rank == 0)
    mpih_debugmsg (comm, "'%s' is%s enabled.\n", var, val ? "" : " not");
  return val;
}

/* ------------------------------------------------------------ */

/**
 *  \brief Creates a sequential baseline matrix multiply problem
 *  instance.
 */
static void setupSeqProblem__ (int m, int n, int k,
			       double** p_A, double** p_B,
			       double** p_C, double** p_C_bound);

static
void
verify__ (int m, int n, int k)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  if (!isEnvEnabled_mpi__ (comm, "VERIFY", 1)) return;

  double* A = NULL;
  double* B = NULL;
  double* C_soln = NULL;
  double* C_bound = NULL;

  /* First, run the trusted sequential version */
  int rank = mpih_getRank (comm);
  if (rank == 0) {
    setupSeqProblem__ (m, n, k, &A, &B, &C_soln, &C_bound);

    /* Measure time for the sequential problem. */
    mat_setZero (m, n, C_soln);
    double t_start = MPI_Wtime ();
    mat_multiply (m, n, k, A, m, B, k, C_soln, m);
    double dt_seq = MPI_Wtime () - t_start;
    mpih_debugmsg (comm, "t_seq = %g s\n", dt_seq);

    /* Recompute, to get the error bound this time */
    mpih_debugmsg (MPI_COMM_WORLD, "Estimating error bound...\n");
    mat_multiplyErrorbound (m, n, k, A, m, B, k, C_soln, m, C_bound, m);
  }

  /* Next, run the untrusted 1D algorithm */
  if (rank == 0) mpih_debugmsg (comm, "Distributing A, B, and C...\n");
  double* A_local = mm1d_distribute (m, k, A, comm);
  double* B_local = mm1d_distribute (k, n, B, comm);
  double* C_local = mm1d_alloc (m, n, comm);
  mm1d_setZero (m, n, C_local, comm);

  /* Do multiply */
  if (rank == 0) mpih_debugmsg (comm, "Computing C <- C + A*B...\n");
  mm1d_mult (m, n, k, A_local, B_local, C_local, comm, NULL, NULL);

  /* Compare the two answers (in parallel) */
  if (rank == 0) mpih_debugmsg (comm, "Verifying...\n");
  int P = mpih_getSize (comm);
  double* C_soln_local = mm1d_distribute (m, n, C_soln, comm);
  double* C_bound_local = mm1d_distribute (m, n, C_bound, comm);
  for (int i = 0; i < m; ++i) {
    int n_local = mm1d_getBlockLength (n, P, rank);
    for (int j = 0; j < n_local; ++j) {
      const double errbound = C_bound_local[i + j*m] * 3.0 * k * DBL_EPSILON;
      const double c_trusted = C_soln_local[i + j*m]; 
      const double c_untrusted = C_local[i + j*m];
      double delta = fabs (c_untrusted - c_trusted);
      if (delta > errbound)
	mpih_debugmsg (comm,
		       "*** Entry (%d, %d) --- Error bound violated ***\n    ==> |%g - %g| == %g > %g\n",
		       c_untrusted, c_trusted, delta, errbound, i, j);
      mpih_assert (delta <= errbound);
    }
  }
  if (rank == 0) mpih_debugmsg (comm, "Passed!\n");

  /* Cleanup */
  if (rank == 0) {
    free (A);
    free (B);
    free (C_soln);
    free (C_bound);
  }
  mm1d_free (A_local, comm);
  mm1d_free (B_local, comm);
  mm1d_free (C_local, comm);
}

/* ------------------------------------------------------------ */

static
void
setupSeqProblem__ (int m, int n, int k,
		   double** p_A, double** p_B,
		   double** p_C, double** p_C_bound)
{
  if (p_A) {
    *p_A = mat_create (m, k);
    mat_randomize (m, k, *p_A);
  }
  if (p_B) {
    *p_B = mat_create (k, n);
    mat_randomize (k, n, *p_B);
  }
  if (p_C) {
    *p_C = mat_create (m, n);
    mat_setZero (m, n, *p_C);
  }
  if (p_C_bound) {
    *p_C_bound = mat_create (m, n);
    mat_setZero (m, n, *p_C_bound);
  }
}

/* ------------------------------------------------------------ */

static
void
summarize__ (int m, int n, int k, const double* t, int n_t,
	     MPI_Comm comm, int debug)
{
  MPI_Barrier (comm);
  FILE* fp = debug ? stderr : stdout;
  int P = mpih_getSize (comm);
  int rank = mpih_getRank (comm);
  if (rank == 0)
    fprintf (fp, "%s%d %d %d %d", debug ? "DEBUG: " : "", m, n, k, P);
  for (int i = 0; i < n_t; ++i) {
    double* tt = (double *)t; /* remove cast */
    double ti_min;
    MPI_Reduce (&tt[i], &ti_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
    double ti_max;
    MPI_Reduce (&tt[i], &ti_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
    double ti_sum;
    MPI_Reduce (&tt[i], &ti_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    if (rank == 0)
      fprintf (fp, " %g %g %g", ti_min, ti_max, ti_sum / P);
  }
  if (rank == 0)
    fprintf (fp, "\n");
  MPI_Barrier (comm);
}

/* ------------------------------------------------------------ */

void
benchmark__ (int m, int n, int k)
{
  MPI_Comm comm = MPI_COMM_WORLD;
  if (!isEnvEnabled_mpi__ (comm, "BENCHMARK", 1)) return;

  int P = mpih_getSize (comm);

  /* Create a synthetic problem to benchmark. */
  double* A_local = mm1d_alloc (m, k, comm);
  double* B_local = mm1d_alloc (k, n, comm);
  double* C_local = mm1d_alloc (m, n, comm);

  mm1d_randomize (m, k, A_local, comm);
  mm1d_randomize (k, n, B_local, comm);

  const int TOTAL = 0;
  const int COMP = 1;
  const int COMM = 2;
  double t[3];  bzero (t, sizeof (t));

  const int MAX_TRIALS = 10;
  for (int trial = 0; trial < MAX_TRIALS; ++trial) {
    mm1d_setZero (m, n, C_local, comm);
    double t_start = MPI_Wtime ();
    mm1d_mult (m, n, k, A_local, B_local, C_local, comm, &t[COMP], &t[COMM]);
    t[TOTAL] += MPI_Wtime () - t_start;
  }
  t[TOTAL] /= MAX_TRIALS;
  t[COMP] /= MAX_TRIALS;
  t[COMM] /= MAX_TRIALS;
  summarize__ (m, n, k, t, 3, comm, 0);

  mm1d_free (A_local, comm);
  mm1d_free (B_local, comm);
  mm1d_free (C_local, comm);
}

/* eof */
