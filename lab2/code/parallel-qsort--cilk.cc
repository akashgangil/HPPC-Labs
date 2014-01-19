/**
 *  \file parallel-qsort--cilk.cc
 *
 *  \brief Sample solution to the Quicksort problem using Cilk Plus
 *  and a fully-parallel 3-way partitioning scheme.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <algorithm> /* For 'std::swap' template routine */

#include "sort.hh"

/**
 *  Pivots the keys of A[0:N-1] around a given pivot value. The number
 *  of keys less than the pivot is returned in *p_n_lt; the number
 *  equal in *p_n_eq; and the number greater in *p_n_gt. The
 *  rearranged keys are stored back in A as follows:
 *
 * - The first *p_n_lt elements of A are all the keys less than the
 *   pivot. That is, they appear in A[0:(*p_n_lt)-1].
 *
 * - The next *p_n_eq elements of A are all keys equal to the
 *   pivot. That is, they appear in A[(*p_n_lt):(*p_n_lt)+(*p_n_eq)-1].
 *
 * - The last *p_n_gt elements of A are all keys greater than the
 *   pivot. That is, they appear in
 *   A[(*p_n_lt)+(*p_n_eq):(*p_n_lt)+(*p_n_eq)+(*p_n_gt)-1].
 */
void partition__seq (keytype pivot, int N, keytype* A,
		     int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
  /* The following implementation is based on the Dutch National Flag
   * solution suggested by someone on Piazza. See also:
   * http://en.wikipedia.org/wiki/Dutch_national_flag_problem
   */
  int p = -1, q = N;
  int i = 0;
  while (i < q) {
    if (A[i] == pivot) {
      std::swap (A[i++], A[++p]);
    } else if (A[i] >= pivot) {
      std::swap (A[i], A[--q]);
    } else {
      ++i;
    }
  }

  /* After the above loop completes:
   * - A[0:p] == equal to pivot
   * - A[p+1:q-1] == less than pivot
   * - A[q:n-1] == greater than pivot
   *
   * Therefore, need to move equal elements into the middle.
   */
  for (int k = 0; k <= p; ++k)
    std::swap (A[k], A[q-1-k]);

  if (p_n_lt) *p_n_lt = q-1-p;
  if (p_n_eq) *p_n_eq = p+1;
  if (p_n_gt) *p_n_gt = N-q;
}

/* ===== A parallel partitioning algorithm =====

The algorithm implemented here is in-place. It does this by

- splitting the input array into two halves, and recursively
  partitioning each half in-place;

- merging the two subpartitions using "reverse" operations, which can
  be implemented using parallel-for loops that repeatedly swap
  elements.

 */

/**
 *  Partially reverses an array. In particular, given an input array
 *  A[0:n-1], this routine swaps A[i] with A[n-1-i] for all 0 <= i <
 *  k.
 */
void reversePartial (int n, keytype* A, int k)
{
  assert (k <= (n >> 1)); /* k < (n/2) */
  _Cilk_for (int i = 0; i < k; ++i)
    std::swap (A[i], A[n-1-i]);
}

/**
 *  Reverses an array.
 */
void
reverse (int n, keytype* A)
{
  reversePartial (n, A, n >> 1);
}

// A | B | C  ==>  {C} | {B} | {A}
void
regroup3 (int na, int nb, int nc, keytype* X)
{
  // A | B | C
  if (na <= nc) {
    // ==>  {C_hi} | B | C_lo | {A}
    reversePartial (na + nb + nc, X, na);
    if (nb <= (nc - na)) {
      // ==>  {C_hi} | {C_lo_hi} | C_lo_lo | {B} | {A}
      reversePartial (nb + nc - na, X + na, nb);
    } else { // nc - na < nb
      // ==>  {C_hi} | {C_lo} | B_hi | {B_lo} | {A}
      reversePartial (nb + nc - na, X + na, nc - na);
    }
  } else { // nc < na
    // ==>  {C} | A_hi | B | {A_lo}
    reversePartial (na + nb + nc, X, nc);
    if (nb <= (na - nc)) {
      // ==>  {C} | {B} | A_hi_hi | {A_hi_lo} | {A_lo}
      reversePartial (na - nc + nb, X + nc, nb);
    } else { // na - nc < nb
      // ==>  {C} | {B_hi} | B_lo | {A_hi} | {A_lo}
      reversePartial (na - nc + nb, X + nc, na - nc);
    }
  }
}

void
mergePartitions (keytype* X
		 , int n1a, int n1b, int n1c
		 , int n2a, int n2b, int n2c)
{
#if 1
  // A1 | B1 | C1 | A2 | B2 | C2
  //   ==>  A1 | {A2} | {C1} | {B1} | B2 | C2
  //   ==>  A1 | {A2} | {B2} | {{B1}} | {{C1}} | C2
  regroup3 (n1b, n1c, n2a, X + n1a);
  regroup3 (n1c, n1b, n2b, X + n1a + n2a);
#else
  // A1 | B1 | C1 | A2 | B2 | C2
  //   ==>  A1 | A2' | C1' | B1' | B2 | C2
  //   ==>  A1 | A2' | B2' | B1 | C1 | C2
  reverse (n1b + n1c + n2a, X + n1a);
  reverse (n1c + n1b + n2b, X + n1a + n2a);
#endif
}

// In-place partition
void
partition (keytype pivot, int N, keytype* A,
	   int* p_n_lt, int* p_n_eq, int* p_n_gt)
{
  assert (p_n_lt != NULL);
  assert (p_n_eq != NULL);
  assert (p_n_gt != NULL);

  const int G = 1024*1024;
  if (N <= G) {
    partition__seq (pivot, N, A, p_n_lt, p_n_eq, p_n_gt);
    return;
  }
  // N > G
  int N_mid = N >> 1; // i.e., floor (N / 2)
  int n1_lt = -1, n1_eq = -1, n1_gt = -1;
  _Cilk_spawn partition (pivot, N_mid, A, &n1_lt, &n1_eq, &n1_gt);
  int n2_lt = -1, n2_eq = -1, n2_gt = -1;
  partition (pivot, N-N_mid, A+N_mid, &n2_lt, &n2_eq, &n2_gt);
  _Cilk_sync;
  mergePartitions (A, n1_lt, n1_eq, n1_gt, n2_lt, n2_eq, n2_gt);
  *p_n_lt = n1_lt + n2_lt;
  *p_n_eq = n1_eq + n2_eq;
  *p_n_gt = n1_gt + n2_gt;
}

/* ===== Quicksort with parallelized recursive calls ===== */

void
quickSort (int N, keytype* A)
{
  const int G = 1024; /* base case size, a tuning parameter */
  if (N < G)
    sequentialSort (N, A);
  else {
    keytype pivot = A[rand () % N];
    int n_less = -1, n_equal = -1, n_greater = -1;
    partition (pivot, N, A, &n_less, &n_equal, &n_greater);
    assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
    _Cilk_spawn quickSort (n_less, A);
    quickSort (n_greater, A + n_less + n_equal);
  }
}

void
parallelSort (int N, keytype* A)
{
  quickSort (N, A);
}

/* eof */
