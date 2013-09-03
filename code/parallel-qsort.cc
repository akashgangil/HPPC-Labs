/**
 *  \file parallel-qsort.cc
 *
 *  \brief Implement your parallel quicksort using Cilk Plus in this
 *  file, given an initial sequential implementation.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "sort.hh"
#include <cilk/cilk.h>

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
void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{

  int n_lt = 0, n_eq = 0, n_gt = 0;

  int* x = (int *) malloc(N * sizeof(int));
  x[0] = 0;
  for(int i=1; i < N; i++){
    x[i] = !compare(x[i], pivot);
  }

  int* b = (int *) malloc( N * sizeof(int));
  memcpy(b, x, N *sizeof(int));

  for(int j = 1; j <= ceil(log(N)/log(2)); j++){
    int offset = (int)pow(2, j-1);
    cilk_for (int k = 0; k < N ; k++){
	if( k >= k - offset) x[k] += x[k-offset];   
    }
  } 

  for(int i = 0; i < N; i++){
    if(b[i] == 1) { 
      swap(&A[i], &A[b[i]]);
      n_lt++;
    }
  }

  free(x);
  free(b);

  if (p_n_lt) *p_n_lt = n_lt;
  if (p_n_eq) *p_n_eq = n_eq;
  if (p_n_gt) *p_n_gt = n_gt;
}

void
quickSort (int N, keytype* A)
{
  const int G = 100; /* base case size, a tuning parameter */
  if (N < G)
    sequentialSort (N, A);
  else {
    // Choose pivot at random
    keytype pivot = A[rand () % N];

    // Partition around the pivot. Upon completion, n_less, n_equal,
    // and n_greater should each be the number of keys less than,
    // equal to, or greater than the pivot, respectively. Moreover, the array
    int n_less = -1, n_equal = -1, n_greater = -1;
    partition (pivot, N, A, &n_less, &n_equal, &n_greater);
    assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
    cilk_spawn quickSort (n_less, A);
    quickSort (n_greater, A + n_less + n_equal);
  }
}

void
parallelSort (int N, keytype* A)
{
  quickSort (N, A);
}

/* eof */
