#include<stdio.h>
#include<stdlib.h>
#include<string.h>

#include "sort.hh"
//#define DEBUG 

void display(keytype* a, int start, int N)
{
  for(int i=start; i<N; i++)
    printf("%ld ", a[i]);
  printf("\n");
}

keytype* merge(keytype *left, keytype *right, int start, int end, int mid)
{
  int N = end - start;

  keytype *newArr = (keytype*)malloc(N*sizeof(keytype));
  int l_max = start+ mid -1;
  int r_max = end-1; 
  int l = start;
  int r = start+mid;
#ifdef DEBUG
  printf("PARAMETERS PASSED %3d %3d  %3d\n", N, start, end);
  printf("Inside the MERGE FUNCTION l = %3d  r = %3d  l_max = %3d r_max = %3d\n", l, r, l_max, r_max);
#endif
  int i=0;
  while( l <= l_max && r <= r_max)
  {
      //printf("Comparing %3d and %3d\n", left[l], right[r]);
      if(left[l] > right[r])
      {
	newArr[i] = right[r];
        r++;
      }
      else
      {
        newArr[i] = left[l];
	l++;
      }
      i++;
  }

  while(l <= l_max)
  {
    //printf("DEBUG: %3d %3d %3d\n", i, l, left[l]);
    newArr[i] = left[l];
    i++;
    l++;
  }

  while(r <= r_max)
  {
    newArr[i] = right[r];
    i++;
    r++;
  }
	
  //printf("NEW ARRAY is : \n");
  display(newArr, 0, N);
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

