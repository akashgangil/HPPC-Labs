/**
 *  \file async.c
 *
 *  \brief Driver file for CSE 6230, Fall 2013, Lab 4:
 *         Asynchronous communication in MPI
 *
 *  \author Rich Vuduc <richie@gatech...>, adapted from Hager and
 *          Wellein "Intro to HPC for CSE" text (2010).
 */

#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#include <mpi.h>

#if defined (_OPENMP)
#  include <omp.h>
#endif

/** Pauses for approximately the specified number of seconds. */
double busywait (const double t_delay);

/* ============================================================ */

/**
 *  Asynchronous communication test between two ranks (Ranks 0 and 1).
 *  In principle, Rank 0 simultaneously (a) sends a message to rank 1,
 *  and (b) "busy waits" for a specified amount of time (in seconds).
 */
double
async_comm_test (const double t_delay, const int rank, int* msgbuf, const int len)
{
  const int MSG_TAG = 1000; /* Arbitrary message tag number */
  double t_start = MPI_Wtime ();
  if (rank == 0) {
    MPI_Request req;
    MPI_Status stat;
    #pragma omp task 
    MPI_Send (msgbuf, len, MPI_INT, 1, MSG_TAG, MPI_COMM_WORLD);
    busywait (t_delay);
    #pragma omp taskwait
 //   MPI_Wait (&req, &stat);
  } else { /* rank == 1 */
    MPI_Status stat;
    MPI_Recv (msgbuf, len, MPI_INT, 0, MSG_TAG, MPI_COMM_WORLD, &stat);
  }
  return MPI_Wtime () - t_start;
}

/* ============================================================ */

/**
 *  Initialize the message as it will be used in the
 *  microbenchmark. Upon return, msgbuf[i] == i on rank 0 and
 *  msgbuf[i] == 0 on rank 1.
 */
static
void
init_message (const int rank, int* msgbuf, const int len)
{
  if (rank == 0) {
    int i;
    for (i = 0; i < len; ++i)
      msgbuf[i] = i;
  } else {
    assert (rank == 1);
    bzero (msgbuf, len * sizeof (int));
  }
}

/**
 *  Checks (asserts) that every element of the given buffer has its
 *  "expected" value. The expected value is defined as the *initial*
 *  value on the given rank. That is, if rank == 0, then the expected
 *  value of msgbuf[i] is i; where rank == 1, msgbuf[i] == 0.
 */
static
void
test_message (const int rank, const int* msgbuf, const int len)
{
  if (rank == 0) {
    int i;
    for (i = 0; i < len; ++i)
      assert (msgbuf[i] == i);
  } else {
    assert (rank == 1);
    int i;
    for (i = 0; i < len; ++i)
      assert (msgbuf[i] == 0);
  }
}

/* ============================================================ */

/** Program start */
int
main (int argc, char *argv[])
{
  /* MPI stuff */
  int rank = 0;
  int np = 0;
  char hostname[MPI_MAX_PROCESSOR_NAME+1];
  int namelen = 0;

  /* Output file */
  const char* outfile = "async.dat";
  FILE *fp = NULL; /* output file, only valid on rank 0 */

  /* Message buffer */
  int* msgbuf = NULL;
  const int msglen = (1 << 21); /* 2^21 words */

  double t_delay = 0; /* Current delay (seconds) */
  const double delay_step = 1e-4; /* Delay step */
  const double max_delay = 1000*delay_step; /* Maximum delay */

  /* Start MPI */
#if defined(_OPENMP)
  int mpi_omp_support_level;
  MPI_Init_thread (&argc, &argv, MPI_THREAD_MULTIPLE, &mpi_omp_support_level);
#else
  MPI_Init (&argc, &argv);
#endif

  MPI_Comm_rank (MPI_COMM_WORLD, &rank);	/* Get process id */
  MPI_Comm_size (MPI_COMM_WORLD, &np);	/* Get number of processes */
  MPI_Get_processor_name (hostname, &namelen); /* Get hostname of node */
  fprintf (stderr, "[%s:rank %d of %d] Hello, world!\n", hostname, rank, np);

  /* This benchmark must be run between only two processes */
  assert (np == 2);

  if (rank == 0) {
    fprintf (stderr, "\n");
    fprintf (stderr, "Experimental parameters:\n");
    fprintf (stderr, "  Delay step: %g seconds\n", delay_step);
    fprintf (stderr, "  Maximum delay: %g seconds\n", max_delay);
#if defined (_OPENMP)
    fprintf (stderr, "  OpenMP enabled; support level: ");
    switch (mpi_omp_support_level) {
    case MPI_THREAD_SINGLE: fprintf (stderr, "MPI_THREAD_SINGLE"); break;
    case MPI_THREAD_FUNNELED: fprintf (stderr, "MPI_THREAD_FUNNELED"); break;
    case MPI_THREAD_SERIALIZED: fprintf (stderr, "MPI_THREAD_SERIALIZED"); break;
    case MPI_THREAD_MULTIPLE: fprintf (stderr, "MPI_THREAD_MULTIPLE"); break;
    default: fprintf (stderr, "(unknown)");
    }
    fprintf (stderr, "\n");
#else
    fprintf (stderr, "  OpenMP disabled.\n");
#endif
    fprintf (stderr, "  Output file: %s\n", outfile);
    fprintf (stderr, "\n");
  }

  /* Open a file for writing results */
  fprintf (stderr, "[%s:rank %d of %d] Opening output file, %s...\n",
	   hostname, rank, np, outfile);
  if (rank == 0) {
    fp = fopen (outfile, "w");
    assert (fp != NULL);
  }

  /* Create a message buffer */
  fprintf (stderr, "[%s:rank %d of %d] Creating message buffer of size %d ints (%d bytes)...\n",
	   hostname, rank, np, msglen, msglen * sizeof (int));
  msgbuf = (int *)malloc (msglen * sizeof (int));
  assert (msgbuf);

  /* Runs the asynchronous test-delay protocol */
  while (t_delay < max_delay) {
    double t_elapsed = -1;
    init_message (rank, msgbuf, msglen); /* reset the message */
    test_message (rank, msgbuf, msglen); /* redundant check */
    MPI_Barrier (MPI_COMM_WORLD);

    #pragma omp parallel
    {
	#pragma omp single nowait
 	t_elapsed = async_comm_test (t_delay, rank, msgbuf, msglen);
    }

    /* Check that the msgbufs match initial values on rank 0 */
    test_message (0, msgbuf, msglen);

    /* Write out the timing result */
    if (rank == 0) {
      printf ("%g\t%g\n", t_delay, t_elapsed); fflush (stdout);
      fprintf (fp, "%g\t%g\n", t_delay, t_elapsed); fflush (fp);
    }

    t_delay += delay_step;
  } while (t_delay <= max_delay);

  fprintf (stderr, "[%s:rank %d of %d] Done! Cleaning up...\n", hostname, rank, np);
  free (msgbuf);

  if (rank == 0) {
    fclose (fp); /* Close output file */
  }
  fprintf (stderr, "[%s:rank %d of %d] Shutting down MPI...\n", hostname, rank, np);
  MPI_Finalize ();
  fprintf (stderr, "[%s:rank %d of %d] Bye bye.\n", hostname, rank, np);
  return 0;
}

double
busywait (const double t_delay)
{
  double t_start = MPI_Wtime ();
  double t_elapsed = 0;
  do {
    t_elapsed = MPI_Wtime () - t_start;
  } while (t_elapsed < t_delay);
  return t_elapsed;
}

/* eof */
