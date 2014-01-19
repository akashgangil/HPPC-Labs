#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;

void display(dtype *A, int N){
	for(int i = 0; i < N; i++){
		printf("%f ", A[i]);
	}
	printf("\n");
}

__global__ 
void matTrans(dtype* AT, size_t pitch_trans, dtype* A, size_t pitch, int N)  {
	/* Fill your code here */

//	FIRST METHOD
/*
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	int nThreads = gridDim.x * blockDim.x;
	int total_elements = N * N;
	int num_pass = total_elements/nThreads + 1;

	for(unsigned int i = 1; i <= num_pass; i++) {
		if(gid < total_elements)
		{
			unsigned int index = (gid / N) + ( (gid % N) * N );
		 	AT[gid] = A[index];
		}
		gid += nThreads;
	}
*/
//	SECOND METHOD
/*
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int index = x ;
	int transIndex = ( x / N) + ( (x % N) * pitch );
	if(blockIdx.x == 0)
		printf("N %3d pitch %3d BlockDim.x  %3d  BlockIdX:  %3d   x:  %3d    transIndex: %3d\n",
				blockDim.x, blockIdx.x,  x, transIndex);

	int normal_total_elems = N * N;
	int trans_total_elems = N * pitch;
	if (index < normal_total_elems && transIndex < trans_total_elems)
	{
		printf("ThreadIdx %3d\n", threadIdx.x);
		printf("Total Elements  %3d TransTotal Elements %3d BlockId %3d BlockDim %3d\n",
				 normal_total_elems, trans_total_elems, blockIdx.x, blockDim.x);
      		printf("N  %3d  Pitch  %3d  Index %3d transIndex %3d A  %3d\n", N, pitch, index, transIndex, A[index]);
		AT[transIndex] = A[index] ;
	}
*/
	//int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	//int yIndex = blockIdx.y * blockDim.y + threadIdx.y;

	int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
	int tot_elems = N * N;

	//if(xIndex < N && yIndex < N)  
	if(xIndex < tot_elems)
	{  
	     int xi = xIndex % N;
	     int yi = xIndex / N;	
//	     printf("THREAD X  %3d THREAD Y  %3d X INDEX %3d TOTAL ELEMS %3d PITCH %3d\n", xi, yi, xIndex, tot_elems, pitch);
	     // update the pointer to point to the beginning of the next row  
	     dtype* rowData = (dtype*)(((char*)A) + (yi * pitch));  
	     dtype* transRowData = (dtype*)(((char*)AT)+(xi * pitch_trans));
	     transRowData[yi] = rowData[xi];  
	}  

}

void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}


void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}

void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  struct stopwatch_t* timer = NULL;
  long double t_gpu, t_pcie, t_malloc;
  dtype *d_a, *d_at;
	
  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();
 
  size_t pitch;
   size_t pitch_trans;

  stopwatch_start (timer);
	/* warup */
  CUDA_CHECK_ERROR( cudaFree(0) );

	CUDA_CHECK_ERROR ( cudaMallocPitch((void**)&d_a, &pitch, N*sizeof(dtype), N) );
	CUDA_CHECK_ERROR ( cudaMallocPitch((void**)&d_at, &pitch_trans, N*sizeof(dtype), N) );
        //CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_at, N * N * sizeof (dtype)));
	t_malloc = stopwatch_stop (timer);
	fprintf (stderr, "cudaMalloc: %Lg seconds\n", t_malloc);

	/* run your kernel here */
//	printf("PITCH : %3d \n", pitch);

//	printf("Matrix before transpose\n");
//	display(A, N*N);
//	printf("PITCH_TRANS : %3d\n", pitch_trans);

  stopwatch_start (timer);

	// copy arrays to device via PCIe
	CUDA_CHECK_ERROR ( cudaMemcpy2D(d_a,pitch,A,sizeof(dtype)*N,sizeof(dtype)*N,N,cudaMemcpyHostToDevice) );
	t_pcie = stopwatch_stop (timer);
	fprintf (stderr, "cudaMemcpy: %Lg seconds\n", t_pcie);


	/* do not change this number */
	int nThreads = 1048576;
	int tbSize = 1024;
	int numTB = (nThreads + tbSize - 1) / tbSize;

//	dim3 threadsPerBlock(16, 16);
//	dim3 numBlocks(nThreads / threadsPerBlock.x, nThreads / threadsPerBlock.y);

	stopwatch_start (timer);

	// kernel invocation
//	matTrans <<<numBlocks, threadsPerBlock>>> (d_at, d_a, N, pitch);
	matTrans <<<numTB, tbSize>>> (d_at, pitch_trans, d_a, pitch, N);

	cudaThreadSynchronize ();
	t_gpu = stopwatch_stop (timer);
	fprintf (stderr, "GPU transpose %Lg seconds ==> %Lg billion elements per second\n", 
						t_gpu, (N / t_gpu) * 1e-9);

	// copy results back from device to host
	//CUDA_CHECK_ERROR (cudaMemcpy (AT, d_at, N * N * sizeof (dtype), 
	//									cudaMemcpyDeviceToHost));
	CUDA_CHECK_ERROR (cudaMemcpy2D (AT,sizeof(dtype)*N,d_at,pitch_trans,sizeof(dtype)*N,N,cudaMemcpyDeviceToHost));
	// copy results back from device to host
        //CUDA_CHECK_ERROR (cudaMemcpy (AT, d_at, N * N * sizeof (dtype), 
        //                                                                        cudaMemcpyDeviceToHost));
//	printf("Matrix after transpose\n");
//	display(AT, N*N);

	// free memory on device
	CUDA_CHECK_ERROR (cudaFree (d_a));
	CUDA_CHECK_ERROR (cudaFree (d_at));

}

int 
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);

  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stderr, "Transpose successful\n");
	}

	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
