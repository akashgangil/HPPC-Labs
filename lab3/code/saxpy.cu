#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "cuda_utils.h"
#include "timer.c"


typedef float dtype;


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
cpuSaxpy (dtype a, dtype* x, dtype* y, int N)
{
	int i;
	
	for(i = 0;i < N; i++) {
		y[i] = a * x[i] + y[i];
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


__global__ void
saxpy (dtype a, dtype* x, dtype* y, int N)
{
	/* fill in your code here */
	int gid = blockIdx.x * blockDim.x + threadIdx.x;

	int nThreads = gridDim.x * blockDim.x;
	int num_pass = N/nThreads + 1;

	for(int i=0; i< num_pass; i++){
		if(gid < N) y[gid] = a * x[gid] + y[gid];	
		gid +=nThreads;
	}
}


void
gpuSaxpy (dtype a, dtype* h_x, dtype* h_y, int N)
{
	dtype *d_x, *d_y;
	int nThreads, tbSize, numTB;

	struct stopwatch_t* timer;
	long double t_gpu, t_pcie, t_malloc;


	// create timers
	stopwatch_init ();
	timer = stopwatch_create ();
	assert (timer);

	stopwatch_start (timer);
	// allocate memory on device
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_x, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_y, N * sizeof (dtype)));
	t_malloc = stopwatch_stop (timer);
	fprintf (stderr, "cudaMalloc: %Lg seconds\n", t_malloc);


	stopwatch_start (timer);
	// copy arrays to device via PCIe
	CUDA_CHECK_ERROR (cudaMemcpy (d_x, h_x, N * sizeof (dtype), 
										cudaMemcpyHostToDevice));
	CUDA_CHECK_ERROR (cudaMemcpy (d_y, h_y, N * sizeof (dtype), 
										cudaMemcpyHostToDevice));
	t_pcie = stopwatch_stop (timer);
	fprintf (stderr, "cudaMemcpy: %Lg seconds\n", t_pcie);


	/* do not change this number */
	nThreads = 1048576;
	tbSize = 256;
	numTB = (nThreads + tbSize - 1) / 256;

	stopwatch_start (timer);
	// kernel invocation
	saxpy <<<numTB, tbSize>>> (a, d_x, d_y, N);
	cudaThreadSynchronize ();
	t_gpu = stopwatch_stop (timer);
	fprintf (stderr, "SAXPY: %Lg seconds ==> %Lg billiion elements per second\n", 
						t_gpu, (N / t_gpu) * 1e-9);

	// copy results back from device to host
	CUDA_CHECK_ERROR (cudaMemcpy (h_y, d_y, N * sizeof (dtype), 
										cudaMemcpyDeviceToHost));

	// free memory on device
	CUDA_CHECK_ERROR (cudaFree (d_x));
	CUDA_CHECK_ERROR (cudaFree (d_y));
}


void
initArr (dtype* in, int N)
{
	int i;
	
	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void initA (dtype* a)
{
	*a = (dtype) rand () / RAND_MAX;
}

void copyArr (dtype* dst, dtype* src, int N)
{
	int i;

	for(i = 0;i < N; i++) {
		dst[i] = src[i];
	}
}


int 
main (int argc, char** argv)
{	
	dtype *x, *y, *y_cpu;
	dtype a;

	int N;
	int err;

	struct stopwatch_t* timer = NULL;
	long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

	/* create host data structures */
	x = (dtype*) malloc (N * sizeof (dtype));
	y = (dtype*) malloc (N * sizeof (dtype));
	y_cpu = (dtype*) malloc (N * sizeof (dtype));

	/* initialize arrays */
	initArr (x, N);
	initArr (y, N);
	copyArr (y_cpu, y, N);
	initA (&a);

	/* create timers */
	stopwatch_init ();
	timer = stopwatch_create ();
	assert (timer);

	/* call function for GPU SAXPY */
	/* y = ax + y */
	gpuSaxpy (a, x, y, N);

	stopwatch_start (timer);
	/* verify results on CPU */
	cpuSaxpy (a, x, y_cpu, N);
	t_cpu = stopwatch_stop (timer);
	fprintf (stderr, "CPU: %Lg seconds\n", t_cpu);

	/* compare answers */
	err = cmpArr (y, y_cpu, N);
	if(!err) {
		fprintf (stderr, "Correct answer\n");
	} else {
		fprintf (stderr, "Wrong answers: %d out of %d\n", err, N);
	}	

	free (x);
	free (y);
	free (y_cpu);
	
	return 0;
}
