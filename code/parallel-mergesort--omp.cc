/**
 *  \file parallel-mergesort--omp.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sort.hh"
//#define DEBUG 

void display(keytype* a, int start, int N)
{
  for(int i=start; i<N; i++)
    printf("%lu ", a[i]);
  printf("\n");
}


keytype* merge(keytype *left, keytype *right, int start, int end, int mid)
{
  return newArr;
}


keytype* mergeSort(int start, int end, keytype* a)
{
  int N = end - start;
  if(N <= 100)
  {
    sequentialSort(N, a); 
    return a;
  }
  
  int mid = (end - start)/2;

  keytype* left = mergeSort(start, start+mid, a);
  keytype* right = mergeSort(start+mid, end, a);
  
  keytype *newArr = merge(left, right, start, end, mid);
  memcpy(a+start, newArr, N*sizeof(int));
  free(newArr);
  return a;
}

void
parallelSort (int N, keytype* A)
{
  /* Lucky you, you get to start from scratch */
  mergeSort(0, N, A);
}

