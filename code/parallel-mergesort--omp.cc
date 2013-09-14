/**
 *  \file parallel-mergesort--omp.cc
 *
 *  \brief Implement your parallel mergesort in this file.
 */

#include <assert.h>
#include<stdio.h>
#include<stdlib.h>
#include<string.h>

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
#ifdef DEBUG		
  printf("**********************");
  printf("\nINSIDE MERGE SORT\n");
#endif
  int N = end - start;
  if(N <= 100)
  {
    sequentialSort(N, a); 
    return a;
  }
  
  if(N == 1)
  {
#ifdef DEBUG
    printf("SINGLE ELEMENT\n");
    printf("ELEMENT: %d\n", a[start]);
#endif				
    return a;
  }
  
  int mid = (end - start)/2;

#ifdef DEBUG
  printf("Start: %3d  End: %3d\n", start, end);
  printf("N : %3d  mid: %3d\n", N, mid);
#endif
  keytype* left = mergeSort(start, start+mid, a);
  keytype* right = mergeSort(start+mid, end, a);
#ifdef DEBUG
  printf("LEFT ARRAY\n");
  display(a, 0, 8);
  printf("RIGHT ARRAY\n");
  display(a, 0, 8);

  printf("\n");
  printf("MERGING!!!!!!! with params start %3d end %3d mid %3d N %3d\n", start, end, mid, N);
#endif
  keytype *newArr = merge(left, right, start, end, mid);
  memcpy(a+start, newArr, N*sizeof(int));
  free(newArr);
  display(a, 0, 8);
  return a;
}

void
parallelSort (int N, keytype* A)
{
  /* Lucky you, you get to start from scratch */
  mergeSort(0, N, A);
}

