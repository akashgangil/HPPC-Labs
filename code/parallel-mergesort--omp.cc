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

//#define DEBUG1

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
void smerge(keytype *arr, int l_start, int l_end, int r_start, int r_end)
{

  int l_len = l_end - l_start + 1;
  int r_len = r_end - r_start;

#ifdef DEBUG   
  printf("Full Array : \n"); display(arr, 0, r_len+l_len); 
  printf("Left Array : \n"); display(arr, 0, l_len);
  printf("Right Array : \n"); display(arr, r_start, r_len);
  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n",
				 l_start, l_end, r_start, r_end, l_len, r_len);
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
  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n",
				 l_start, l_end, r_start, r_end, l_len, r_len);
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

  //copy back to the original array
  memcpy(arr+l_start, newArr,(r_len+l_len)*sizeof(keytype));
  free(newArr);
}

/*Binary Search Routine: Returns the index of the first element in the second array which 
 * is greater than the median of the first array*/
int binarySearch(keytype* a, int median, int r_start, int r_end)
{
  int low = r_start;
  int high = r_end;
#ifdef DEBUG1
  printf("BINARY SEARCH arguments : median %3d r_start %3d r_end %3d\n", median, r_start, r_end);
  printf("Inside Binary Routine\n");
  printf("median is %3lu\n", a[median]);
  printf("The target array is : \n"); display(a, r_start, r_end-r_start+1); 
#endif

  if(a[median] < a[r_start]) return r_start;
 
  while(low < high)
  {
    int mid = (low + high)/2;
    if(a[median] < a[mid]) high = mid;
    else low = mid + 1;
  }
#ifdef DEBUG1
  printf("high is %3d\n", high);
#endif
  return high;
}


/*Parallel Merge Routine*/
void pmerge(keytype *arr, int l_start, int l_end, int r_start, int r_end)
{

  /*fml*/
  int anomaly = 0;

  int l_len = l_end - l_start + 1;
  int r_len = r_end - r_start;
  int N = l_len + r_len ;

#ifdef DEBUG1
  printf("INSIDE P MERGE ROUTINE\n"); 
  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n",
				 l_start, l_end, r_start, r_end, l_len, r_len);
  printf(" N is %3d\n", N);
  const int G = 2;
#endif

  const int G = 10;
  if(N <= G)
  {
#ifdef DEBUG1
    printf("smerge!! was called \n");
#endif
    smerge(arr, l_start, l_end, r_start, r_end);
    return; 
  }

  //if the r_len is 0, we just have the left part sorted out
  if(!r_len)
  {
     int mid = l_start + (l_end - l_start)/2; 
     pmerge(arr, l_start, mid, mid+1, l_end);
     return;
  }
  keytype *newArr = (keytype*)malloc(N*sizeof(keytype));
 

  //middle element in the first array
  int m = l_start + (l_end - l_start)/2;
 

  int r_partition = binarySearch(arr, m, r_start, r_end-1);
 
  int m_index = (m - l_start) + (r_partition - r_start + anomaly) ;

  /* Accounting for the case when all the elements in the second
   * array are less than m
   */
  if(arr[r_partition] < arr[m]) anomaly = 1;
  
  if(r_partition-1 == r_start || r_partition == r_start) return; //the sub array is already sorted

#ifdef DEBUG1
  printf("Other Parameters: m %3d r_partition %3d m_index %3d\n", m , r_partition, m_index);
  printf("*******Arr******* : \n"); display(arr, 0, N);
#endif

  //copying the lesser than elements;first part of the array  
  memcpy(newArr, arr, (m-l_start)*sizeof(keytype));
  //copying the lesser than, second part of the array
  memcpy(newArr+m-l_start, arr+l_end+1, (r_partition-r_start+anomaly)*sizeof(keytype));

  //place the first array mid element
  newArr[m_index] = arr[m];
  
  //copying the greater than, first part of the array
  memcpy(newArr+m_index+1, arr+m+1, (l_end-m)*sizeof(keytype));
  //copying the greater than, second part of the array 
  memcpy(newArr+m_index + l_end - m +1, arr+r_partition, (r_end-r_partition)*sizeof(keytype));

  //copy back the entire array
  memcpy(arr+l_start, newArr, N*sizeof(keytype));
 
  free(newArr);
#ifdef DEBUG1
  printf("*********newArr******* : \n"); display(newArr, 0, N);

  printf("l_start  %3d l_end %3d  r_start %3d r_end %3d l_len %3d r_len %3d\n",
				 l_start, l_end, r_start, r_end, l_len, r_len);
  printf("N value is %3d\n", N);
  printf("Partition Values is %3d\n", r_partition);
  printf(" m is %3d anomaly is %3d\n", m, anomaly);
#endif
  pmerge(arr, l_start, l_start + m , r_start, r_partition+anomaly);
  pmerge(arr, m+1, l_end, r_partition, r_end);
}


void mergeSort(int start, int end, keytype* a)
{
  int N = end - start;

#ifdef DEBUG1
  printf("N  value is %3d\n", N);
  const int G = 3; 
#endif

  const int G = 100;
  if(N <= G)
  {
    sequentialSort(N, a+start); 
    return;
  }
  
  int mid = (end - start)/2;

  /* [start, end) */
 // #pragma omp task default(none) shared(start, mid, a)
  mergeSort(start, start+mid, a);

  mergeSort(start+mid, end, a);

 // #pragma omp taskwait
    
  pmerge(a, start, start+mid-1, start+mid, end);
}

void
parallelSort (int N, keytype* a)
{
//  #pragma omp parallel
  /* Lucky you, you get to start from scratch */
 // #pragma omp single nowait

#ifdef DEBUG1  
  keytype a[] = {7, 19,  4, 3, 11, 54, 7, 111};
  printf("Input Array\n : ");display(a, 0,  8);
  mergeSort(0, 8, a);
#endif

  mergeSort(0, N, a);
#ifdef DEBUG1
  printf("Output Array\n :");display(a, 0, 8);
#endif
}

