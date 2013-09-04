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

int compare(keytype a, keytype b){
  if(a <= b) return 1;
  else return 0;
}

void swap(keytype* a, keytype* b){
  keytype temp = *a;
  *a = *b;
  *b = temp;
}

void display(int *x, int N){
	for(int j=0; j < N; j++){
		printf("%d ", x[j]);
	}
	printf("\n");
}

void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{

  int n_lt = 0, n_eq = 0, n_gt = 0;

  int* x = (int *) malloc(N * sizeof(int));
  memset(x, 0, N*sizeof(int));
 
  for(int i=1; i < N; i++){
    x[i] = compare(A[i], pivot);
  }

  printf("\nCOMPARE x[i]\n");
  display(x, N);

  int* b = (int *) malloc( N * sizeof(int));
  memcpy(b, x, N *sizeof(int));

  for(int j = 1; j <= ceil(log(N)/log(2)); j++){
    int offset = (int)pow(2, j-1);
    for (int k = 0; k < N ; k++){
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
  const int G = 1; /* base case size, a tuning parameter */
  if (N < G)
    sequentialSort (N, A);
  else {
    // Choose pivot at random
    keytype pivot = A[rand () % N];

    printf("\n PIVOT = %ld\n", pivot);

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
