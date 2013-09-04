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
#include <cilk/reducer_opadd.h>
//#define DEBUG

//#define EX_DEBUG


/*Compares two keytypes*/
int compare(keytype a, keytype b){
  if(a < b) return 1;
  else return 0;
}

/*Swaps two keytypes*/
void swap(keytype* a, keytype* b){
  keytype temp = *a;
  *a = *b;
  *b = temp;
}

/*DEBUG: print an integer array*/
void display(int *x, int N){
	for(int j=0; j < N; j++){
		printf("%d ", x[j]);
	}
	printf("\n");
}

/*DEBUG: print a keytype array*/
void display_arr(keytype *x, int N){
	for(int j=0; j < N; j++){
		printf("%ld ", x[j]);
	}
	printf("\n");
}

/* Performs exclusive scan on input array x, 
 * we use exclusive scan because indexs start
 * from 0 */
void exclusive_scan(int* x, int *e, int N){

	if(N == 1) e[0] = 0;

#ifdef EX_DEBUG
  printf("\n X[i] Before\n");
  display(x, N);
#endif

	//intializing the exclusive scan array to the input array
	cilk_for(int i= 0; i < N; i++){
		e[i] = x[i];
	}

	//since we just need to go logN to base 2 steps 
	for(int step = 0; (1 << step) <= N; step++){		
		cilk_for(int i = 1<<step ; i < N; i += 1 ){
			e[i] = e[i] + x[i - (1 << step)];
		}
	
		#ifdef EX_DEBUG
		  printf("\nE[i] After Step %d \n", step);
		  display(e, N);
		#endif
				
		cilk_for(int i = 0; i < N; i++){
		   x[i] = e[i];
		}
	}
}

void partition (keytype pivot, int N, keytype* A,
		int* p_n_lt, int* p_n_eq, int* p_n_gt)
{

  int n_lt = 0, n_eq = 0, n_gt = 0;

  int *x = (int *) malloc(N * sizeof(int));
  memset(x, 0, N*sizeof(int));
 
  for(int i=0; i < N; i++){
    x[i] = compare(A[i], pivot);
  }

 int *b = (int *) malloc(N * sizeof(int));
  memcpy(b, x, N *sizeof(int));

  #ifdef DEBUG 
  printf("\nCOMPARE B[i]\n");
  display(b, N);
  #endif


 int *e = (int *) malloc(N * sizeof(int));
 exclusive_scan(x, e, N);

 //to get a exclusive scan 
 cilk_for(int j=0; j<N ; j++)
	x[j] = x[j] - b[j];
  

 #ifdef DEBUG
 printf("\nPREFIX SCAN X[i]\n");
 display(x, N);

 printf("\nA[i] BEFORE\n");
 display_arr(A, N);
 #endif


//for all the elements who were 1 in the compare array,
//we swap them with the corresponding
//indexes we get in the exclusive scan output.
//We also increment n_lt to keep a track ofthe partition point.
  for(int i = 0; i < N; i++){
    if(b[i] == 1) { 
      swap(&A[i], &A[x[i]]);
      n_lt++;
    }
  }

#ifdef DEBUG
 printf("\n A[i] AFTER\n");
 display_arr(A, N);


 printf("\nCounter Values\n");
 printf("\n n_lt = %d, n_gt = %d", n_lt, N - n_lt);
#endif

  free(x);
  free(b);
  free(e);


  if (p_n_lt) *p_n_lt = n_lt;
  if (p_n_eq) *p_n_eq = n_eq;
  if (p_n_gt) *p_n_gt = N - n_lt;
}

void
quickSort (int N, keytype* A)
{

#ifdef DEBUG
  printf("\nStarting the PARTITION subroutine with N = %d\n", N);
  display_arr(A, N);
#endif

  const int G = 1; /* base case size, a tuning parameter */
  if (N<=1)
    return;
    //sequentialSort (N, A);
  else {
    // Choose pivot at random
    keytype pivot = A[rand () % N];
#ifdef DEBUG
    printf("\n PIVOT = %ld\n", pivot);
#endif

    // Partition around the pivot. Upon completion, n_less, n_equal,
    // and n_greater should each be the number of keys less than,
    // equal to, or greater than the pivot, respectively. Moreover, the array
    int n_less = -1, n_equal = -1, n_greater = -1;
    partition (pivot, N, A, &n_less, &n_equal, &n_greater);
    assert (n_less >= 0 && n_equal >= 0 && n_greater >= 0);
    quickSort (n_less, A);
    quickSort (n_greater, A + n_less + n_equal);
  }
}

void
parallelSort (int N, keytype* A)
{
  quickSort (N, A);
}

/* eof */
