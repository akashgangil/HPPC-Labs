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


/*Displays the Array*/
void display(keytype* a, int start, int N)
{
#ifdef DEBUG
  printf("INSIDE DISPLAY : VALUES start %3d N %3d\n", start, N);
#endif
  while(N--)  
  {
    printf("%lu ", a[start]);
    start++;
  }
  printf("\n");
}


/*Serial Merge Routine*/
keytype* smerge(keytype *arr, int l_start, int l_end, int r_start, int r_end)
{

  int l_len = l_end - l_start + 1;
  int r_len = r_end - r_start;

#ifdef DEBUG   
  printf("Full Array : \n"); display(arr, 0, r_len+l_len); 
  printf("Left Array : \n"); display(arr, 0, l_len);
  printf("Right Array : \n"); display(arr, r_start, r_len);
  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n", l_start, l_end, r_start, r_end, l_len, r_len);
#endif
  int i=0;
  
  keytype *newArr = (keytype*) malloc((r_len + l_len)*sizeof(keytype));
  for(;l_start <= l_end && r_start < r_end;)
  {
    if(arr[l_start] < arr[r_start])
    {
	newArr[i] = arr[l_start];
	i++;
	l_start++;	
    }
    else
    {
	newArr[i] = arr[r_start];
	i++;
    	r_start++;
    }
  }

#ifdef DEBUG  
  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n", l_start, l_end, r_start, r_end, l_len, r_len);
#endif

  for(;l_start <= l_end;)
  {
    newArr[i] = arr[l_start];
    i++;
    l_start++;
  }

  for(;r_start < r_end;)
  {
    newArr[i] = arr[r_start];
    i++;
    r_start++;
  }

  return newArr;
}

/*Binary Search Routine*/
int binarySearch(keytype* a, int mid, int r_start, int r_end)
{
  int low = r_start;
  int high = r_end;
  while(low < high)
  {
    int m = (low + high)/2;
    if(a[mid] < a[m]) high = m;
    else low = m + 1;
  }
  return high;
}


/*Parallel Merge Routine*/
keytype* pmerge(keytype *arr, int l_start, int l_end, int r_start, int r_end)
{
  int l_len = l_end - l_start;
  int r_len = r_end - r_start;
  int N = l_len + r_len ;

  keytype *newArr = (keytype*)malloc(N*sizeof(keytype));
 

  //middle element in the first array
  int m = (l_start + l_end)/2;

  int r_partition = binarySearch(arr, m, r_start, r_end-1);
 
  int m_index = (m - l_start) + (r_partition - r_start) + 1;
  newArr[m_index] = arr[m];

  //copying the lesser than elements;first part of the array  
  memcpy(newArr, arr, m-l_start);
  //copying the lesser than, second part of the array
  memcpy(newArr+m-l_start, arr, r_partition-r_start+1);

  //copying the greater than, first part of the array
  memcpy(newArr+m_index+1, arr, l_end-m+1);
  //copying the greater than, second part of the array 
  memcpy(newArr+r_partition, arr, r_end-r_partition+1);

  pmerge(arr, l_start, m-1, r_start, r_partition);
  pmerge(arr, m+1, l_end, r_partition+1, r_end);

  return newArr;
}


void mergeSort(int start, int end, keytype* a)
{
  int N = end - start;

  const int G = 1000; 
  if(N <= G)
  {
    sequentialSort(N, a+start); 
    return;
  }
  
  int mid = (end - start)/2;

  /* [start, end) */
  #pragma omp task default(none) shared(start, mid, a)
  mergeSort(start, start+mid, a);

  mergeSort(start+mid, end, a);

  #pragma omp taskwait
    
  keytype *newArr = smerge(a, start, start+mid-1, start+mid, end);
  
  memcpy(a+start, newArr, N * sizeof(keytype));

  free(newArr);
}

void
parallelSort (int N, keytype* A)
{
  #pragma omp parallel
  /* Lucky you, you get to start from scratch */
  #pragma omp single nowait
  mergeSort(0, N, A);
}

