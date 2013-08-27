/**
 *  \file sort.cc
 *  \brief Lab 1: Multithreaded sorting
 */

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <stdio.h>
#include "timer.c"

typedef unsigned long keytype;

static
int
compare (const void* a, const void* b)
{
  keytype ka = *(const keytype *)a;
  keytype kb = *(const keytype *)b;
  if (ka < kb)
    return -1;
  else if (ka == kb)
    return 0;
  else
    return 1;
}

/**
 *  Sorts an input array, In, writing the sorted result to Out, using
 *  a sequential algorithm.
 */
void
sortSequential (int N, keytype* A)
{
  qsort (A, N, sizeof (keytype), compare);
}

/* ============================================================
 * EXERCISE 4: Implement a Cilk Plus copy function
 */

/** Returns a new uninitialized array of length N */
keytype *
newKeys (int N)
{
  keytype* A = (keytype *)malloc (N * sizeof (keytype));
  assert (A);
  return A;
}

/** Returns a new copy of A[0:N-1] */
keytype *
newCopy (int N, const keytype* A)
{
  keytype* A_copy = newKeys (N);

  /* REPLACE THIS SEQUENTIAL CODE: */
  memcpy (A_copy, A, N * sizeof (keytype));

  return A_copy;
}

/* ============================================================
 * EXERCISE 5: Implement a Cilk Plus Quicksort
 */

/**
 *  Pivots the keys of A[0:N-1] around a given pivot value. The number
 *  of keys less than the pivot is returned in *p_n_lt; the number
 *  equal in *p_n_eq; and the number greater in *p_n_gt. The
 *  rearranged keys are stored back in A as follows:
 *
 * - The first *p_n_lt elements of A are all the keys less than the
 *    pivot. That is, they appear in A[0:(*p_n_lt)-1].
 *
 * - The next *p_n_eq elements of A are all keys equal to the
 *   pivot. That is, they appear in A[(*p_n_lt):(*p_n_lt)+(*p_n_eq)-1].
 *
 * - The last *p_n_gt elements of A are all keys greater than the
 *   pivot. That is, they appear in
 *   A[(*p_n_lt)+(*p_n_eq):(*p_n_lt)+(*p_n_eq)+(*p_n_gt)-1].
 */
void
partition (keytype pivot, int N, keytype* A,
	   int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
  /* Count how many elements of A are less than (lt), equal to (eq),
     or greater than (gt) the pivot value. */
  int n_lt = 0, n_eq = 0, n_gt = 0;
  for (int i = 0; i < N; ++i) {
    if (A[i] < pivot) ++n_lt;
    else if (A[i] == pivot) ++n_eq;
    else ++n_gt;
  }

  keytype* A_orig = newCopy (N, A);

  /* Next, rearrange A so that:
   *   A_lt == A[0:n_lt-1] == subset of A < pivot
   *   A_eq == A[n_lt:(n_lt+n_eq-1)] == subset of A == pivot
   *   A_gt == A[(n_lt+n_eq):(N-1)] == subset of A > pivot
   */
  int i_lt = 0; /* next open slot in A_lt */
  int i_eq = n_lt; /* next open slot in A_eq */
  int i_gt = n_lt + n_eq; /* next open slot in A_gt */
  for (int i = 0; i < N; ++i) {
    keytype ai = A_orig[i];
    if (ai < pivot)
      A[i_lt++] = ai;
    else if (ai > pivot)
      A[i_gt++] = ai;
    else
      A[i_eq++] = ai;
  }
  assert (i_lt == n_lt);
  assert (i_eq == (n_lt+n_eq));
  assert (i_gt == N);

  free (A_orig);

  if (p_n_lt) *p_n_lt = n_lt;
  if (p_n_eq) *p_n_eq = n_eq;
  if (p_n_gt) *p_n_gt = n_gt;
}

void
quickSort (int N, keytype* A)
{
  const int G = 100; /* base case size, a tuning parameter */
  if (N < G)
    sortSequential (N, A);
  else {
    keytype pivot = A[rand () % N];
    int n_less = -1, n_equal = -1, n_greater = -1;
    partition (pivot, N, A, &n_less, &n_equal, &n_greater);
    assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
    quickSort (n_less, A);
    quickSort (n_greater, A + n_less + n_equal);
  }
}

/* ============================================================
 * EXERCISE 6 (BONUS): Implement a Cilk Plus mergesort
 */

/** Merge two sorted arrays, A[0:na-1] and B[0:nb-1], into a sorted
 *  array C[0:(na+nb-1)].
 */
void
merge (keytype* C, int na, const keytype* A, int nb, const keytype* B)
{
  int ia = 0, ib = 0, ic = 0;
  while (ia < na && ib < nb) {
    keytype a = A[ia];
    keytype b = B[ib];
    if (a <= b) {
      C[ic] = a;
      ++ia;
    } else {
      C[ic] = b;
      ++ib;
    }
    ++ic;
  }
  if (ia < na)
    memcpy (C+ic, A+ia, (na-ia) * sizeof (keytype));
  else if (ib < nb) /* ib < nb */
    memcpy (C+ic, B+ib, (nb-ib) * sizeof (keytype));
}

void
mergeSort (int N, keytype* A)
{
  const int G = 100; /* base case size, a tuning parameter */
  if (N < G)
    sortSequential (N, A);
  else {
    keytype* A_tmp = newCopy (N, A);
    int N_half = N / 2;
    mergeSort (N_half, A_tmp);
    mergeSort (N - N_half, A_tmp+N_half);
    merge (A, N_half, A_tmp, N-N_half, A_tmp+N_half);
    free (A_tmp);
  }
}

/* ============================================================
 * Code for checking the sorted results
 */

void
assertIsSorted (int N, const keytype* A)
{
  for (int i = 1; i < N; ++i) {
    if (A[i-1] > A[i]) {
      fprintf (stderr, "*** ERROR ***\n");
      fprintf (stderr, "  A[i=%d] == %lu > A[%d] == %lu\n", i-1, A[i-1], i, A[i]);
      assert (A[i-1] <= A[i]);
    }
  } /* i */
  fprintf (stderr, "\t(Array is sorted.)\n");
}

void
assertEqual (int N, const keytype* A, const keytype* B)
{
  for (int i = 0; i < N; ++i) {
    if (A[i] != B[i]) {
      fprintf (stderr, "*** ERROR ***\n");
      fprintf (stderr, "  A[i=%d] == %lu, but B[%d] == %lu\n", i, A[i], i, B[i]);
      assert (A[i] == B[i]);
    }
  } /* i */
  fprintf (stderr, "\t(Computed answer seems correct.)\n");
}

int
main (int argc, char* argv[])
{
  int N = -1;

  if (argc == 2) {
    N = atoi (argv[1]);
    assert (N > 0);
  } else {
    fprintf (stderr, "usage: %s <n>\n", argv[0]);
    fprintf (stderr, "where <n> is the length of the list to sort.\n");
    return -1;
  }

  stopwatch_init ();
  struct stopwatch_t* timer = stopwatch_create (); assert (timer);

  /* Create an input array of length N, initialized to random values */
  keytype* A_in = newKeys (N);
  for (int i = 0; i < N; ++i)
    A_in[i] = lrand48 ();

  printf ("\nN == %d\n\n", N);

  /* Sort sequentially */
  keytype* A_seq = newCopy (N, A_in);
  stopwatch_start (timer);
  sortSequential (N, A_seq);
  long double t_seq = stopwatch_stop (timer);
  printf ("Sequential: %Lg seconds ==> %Lg million keys per second\n",
	  t_seq, 1e-6 * N / t_seq);
  assertIsSorted (N, A_seq);

  /* Quicksort */
  keytype* A_qs = newCopy (N, A_in);
  stopwatch_start (timer);
  quickSort (N, A_qs);
  long double t_qs = stopwatch_stop (timer);
  printf ("Quicksort: %Lg seconds ==> %Lg million keys per second\n",
	  t_qs, 1e-6 * N / t_qs);
  assertIsSorted (N, A_qs);
  assertEqual (N, A_qs, A_seq);

  /* Mergesort */
  keytype* A_ms = newCopy (N, A_in);
  stopwatch_start (timer);
  mergeSort (N, A_ms);
  long double t_ms = stopwatch_stop (timer);
  printf ("Mergesort: %Lg seconds ==> %Lg million keys per second\n",
	  t_ms, 1e-6 * N / t_ms);
  assertIsSorted (N, A_ms);
  assertEqual (N, A_ms, A_seq);

  printf ("\n");

  /* Cleanup */
  free (A_ms);
  free (A_qs);
  free (A_seq);
  free (A_in);
  stopwatch_destroy (timer);
  return 0;
}

/* eof */
