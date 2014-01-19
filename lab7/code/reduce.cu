#include "driver.h"
#include "reduce.h"
#include "cuda_utils.h"

dtype
reduceCpu (dtype* h_A, unsigned int N)
{
  int i;
  dtype ans;

  ans = (dtype) 0.0;
  for(i = 0; i < N; i++) {
    ans += h_A[i];
  }

  return ans;
}

__global__ void 
reduceNaiveKernel (dtype* In, dtype *Out, unsigned int N)
{
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	/* reduce in shared memory */
	for(stride = 1; stride < blockDim.x; stride *= 2) {
		if(threadIdx.x % (stride * 2) == 0) {
			buffer[threadIdx.x] += buffer[threadIdx.x + stride];
		}
		__syncthreads ();
	}

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}

dtype
reduceNaive (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;
	

	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceNaiveKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceNonDivergeKernel (dtype* In, dtype *Out, unsigned int N)
{
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int stride;
	

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	int num_threads = BS/2;
        /* reduce in shared memory */
        for(stride = 1; stride < blockDim.x; stride <<= 1) {
                if(threadIdx.x < num_threads)
                {
		        int pow = stride * 2 * threadIdx.x;
                        buffer[pow] += buffer[pow + stride];
                }
                __syncthreads();
	    num_threads >>= 1;
        }
	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}



dtype
reduceNonDiverge (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceNonDivergeKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceSeqAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	/* load data to buffer */
	if(tid < N) {
		buffer[threadIdx.x] = In[tid];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();
	int num_threads = BS/2;
        /* reduce in shared memory */
        for(; num_threads > 0; num_threads >>= 1) {
            if( threadIdx.x < num_threads)
	    {
	    	buffer[threadIdx.x] += buffer[threadIdx.x + num_threads];
	        __syncthreads();
	    }
        }
	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}



dtype
reduceSeqAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = N;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceSeqAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceFirstAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* As it can be seen from `reduceSeqAdd`, the total number of threads
		 have been halved */
	/* Thus, you need to load 2 elements from the global memory, add them, and
		 then store the sum in the shared memory before reduction over the shared
		 memory occurs */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	/* load data to buffer */
	if(tid < (N+1)/2) {
		buffer[threadIdx.x] = In[tid] + In[tid + N/2];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();
	int num_threads = BS/2;
        /* reduce in shared memory */
        for(; num_threads > 0; num_threads >>= 1) {
            if( threadIdx.x < num_threads)
	    {
	    	buffer[threadIdx.x] += buffer[threadIdx.x + num_threads];
	        __syncthreads();
	    }
        }
	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = buffer[0];
	}
}



dtype
reduceFirstAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceFirstAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceUnrollLastKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* unroll the loop when there are fewer than 32 threads working */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	/* load data to buffer */
	if(tid < (N+1)/2) {
		buffer[threadIdx.x] = In[tid] + In[tid + N/2];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();
	int num_threads = BS/2;
        /* reduce in shared memory */
        for(; num_threads > 32; num_threads >>= 1) {
            if( threadIdx.x < num_threads)
	    {
	    	buffer[threadIdx.x] += buffer[threadIdx.x + num_threads];
	        __syncthreads();
	    }
        }

	volatile dtype *sm = buffer;
	if(threadIdx.x < 32)
	{
		sm[threadIdx.x] += sm[threadIdx.x + num_threads];
		sm[threadIdx.x] += sm[threadIdx.x + num_threads/2];
		sm[threadIdx.x] += sm[threadIdx.x + num_threads/4];
		sm[threadIdx.x] += sm[threadIdx.x + num_threads/8];
		sm[threadIdx.x] += sm[threadIdx.x + num_threads/16];
		sm[threadIdx.x] += sm[threadIdx.x + num_threads/32];
	}	

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = sm[0];
	}
}



dtype
reduceUnrollLast (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceUnrollLastKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}

__global__ void 
reduceUnrollAllKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	/* load data to buffer */
	if(tid < (N+1)/2) {
		buffer[threadIdx.x] = In[tid] + In[tid + N/2];
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	if(BS >= 1024) {
		if(threadIdx.x < 512) {buffer[threadIdx.x] += buffer[threadIdx.x + 512]; __syncthreads();}
	}

	if(BS >= 512) {
		if(threadIdx.x < 256) {buffer[threadIdx.x] += buffer[threadIdx.x + 256]; __syncthreads();}
	}

	if(BS >= 256) {
		if(threadIdx.x < 128) {buffer[threadIdx.x] += buffer[threadIdx.x + 128]; __syncthreads ();}
	}

	if(BS >= 128) {
		if(threadIdx.x < 64) {buffer[threadIdx.x] += buffer[threadIdx.x + 64]; __syncthreads ();}
	}


	volatile dtype *sm = buffer;
	if(threadIdx.x < 32)
	{
		sm[threadIdx.x] += sm[threadIdx.x + 32];
		sm[threadIdx.x] += sm[threadIdx.x + 16];
		sm[threadIdx.x] += sm[threadIdx.x + 8];
		sm[threadIdx.x] += sm[threadIdx.x + 4];
		sm[threadIdx.x] += sm[threadIdx.x + 2];
		sm[threadIdx.x] += sm[threadIdx.x + 1];
	}	

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = sm[0];
	}
}



dtype
reduceUnrollAll (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 2;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceUnrollAllKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}



__global__ void 
reduceMultAddKernel (dtype* In, dtype *Out, unsigned int N)
{
	/* Fill in your code here */
	/* Instead of just adding 2 elements in the beginning, try adding more 
		 before reducing the partial sums over the shared memory */
	__shared__ dtype buffer[BS];
	unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;

	/* load data to buffer */
	if(tid < (N+1)/32) {
		buffer[threadIdx.x] = In[tid] + In[tid + 1*(N/32)]+ In[tid + 2*(N/32)] + In[tid + 3*(N/32)] 
				     + In[tid + 4*(N/32)] + In[tid + 5*(N/32)] + In[tid + 6*(N/32)] + In[tid + 7*(N/32)]
				     + In[tid + 8*(N/32)] + In[tid + 9*(N/32)] + In[tid + 10*(N/32)] + In[tid + 11*(N/32)] 
				     + In[tid + 12*(N/32)] + In[tid + 13*(N/32)] + In[tid + 14*(N/32)] + In[tid + 15*(N/32)] 
				     + In[tid + 16*(N/32)] + In[tid + 17*(N/32)] + In[tid + 18*(N/32)] + In[tid + 19*(N/32)] 
				     + In[tid + 20*(N/32)] + In[tid + 21*(N/32)] + In[tid + 22*(N/32)] + In[tid + 23*(N/32)] 
				     + In[tid + 24*(N/32)] + In[tid + 25*(N/32)] + In[tid + 26*(N/32)] + In[tid + 27*(N/32)] 
				     + In[tid + 28*(N/32)] + In[tid + 29*(N/32)] + In[tid + 30*(N/32)] + In[tid + 31*(N/32)];

	//	for(int i=0; i<32; ++i){
	//		buffer[threadIdx.x] += In[tid + i*((N+1)/32)];
	//	}
	} else {
		buffer[threadIdx.x] = (dtype) 0.0;
	}
	__syncthreads ();

	if(BS >= 1024) {
		if(threadIdx.x < 512) {buffer[threadIdx.x] += buffer[threadIdx.x + 512]; __syncthreads();}
	}

	if(BS >= 512) {
		if(threadIdx.x < 256) {buffer[threadIdx.x] += buffer[threadIdx.x + 256]; __syncthreads();}
	}

	if(BS >= 256) {
		if(threadIdx.x < 128) {buffer[threadIdx.x] += buffer[threadIdx.x + 128]; __syncthreads ();}
	}

	if(BS >= 128) {
		if(threadIdx.x < 64) {buffer[threadIdx.x] += buffer[threadIdx.x + 64]; __syncthreads ();}
	}


	volatile dtype *sm = buffer;
	if(threadIdx.x < 32)
	{
		sm[threadIdx.x] += sm[threadIdx.x + 32];
		sm[threadIdx.x] += sm[threadIdx.x + 16];
		sm[threadIdx.x] += sm[threadIdx.x + 8];
		sm[threadIdx.x] += sm[threadIdx.x + 4];
		sm[threadIdx.x] += sm[threadIdx.x + 2];
		sm[threadIdx.x] += sm[threadIdx.x + 1];
	}	

	/* store back the reduced result */
	if(threadIdx.x == 0) {
		Out[blockIdx.x] = sm[0];
	}

}



dtype
reduceMultAdd (dtype* d_In, dtype* d_Out, dtype* h_Out, unsigned int N)
{
	unsigned int i, nThreads, tbSize, nBlocks;
	dtype ans;


	nThreads = (N + 1) / 32;
	tbSize = BS;
	nBlocks = (nThreads + tbSize - 1) / tbSize;

	dim3 grid (nBlocks);
	dim3 block (tbSize);

	for(i = 0; i < NUM_ITER; i++) {
		reduceMultAddKernel <<<grid, block>>> (d_In, d_Out, N);
		cudaThreadSynchronize ();
	}

	CUDA_CHECK_ERROR (cudaMemcpy (h_Out, d_Out, nBlocks * sizeof (dtype),
																cudaMemcpyDeviceToHost));

	ans = reduceCpu (h_Out, nBlocks);

	return ans;

}





void
initCudaArray (dtype **d_A, dtype *h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_A, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_A, h_A, N * sizeof (dtype),
																cudaMemcpyHostToDevice));
}

void
cudaReduction (dtype *A, unsigned int N, unsigned int OPT, dtype *ret)
{
	dtype *h_Out, *d_Out;
	unsigned int nBlocks;

	cudaEvent_t start, stop;
	float elapsedTime;

	dtype ans;

	nBlocks = (N + BS - 1) / BS;
	h_Out = (dtype*) malloc (nBlocks * sizeof (dtype));
	CUDA_CHECK_ERROR (cudaMalloc ((void**) &d_Out, nBlocks * sizeof (dtype)));
	
	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));

	fprintf (stderr, "Executing test case [%d]\n", OPT);
	fprintf (stderr, "[1]: Naive | [2]: Non-divergent | [3]: Sequential Add. | [4]: First add | [5]: Unroll last warp | [6]: Complete unroll | [7] Multiple Adds\n");

	
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	/* execute kernel */
	switch (OPT) {
		case 1:
			ans = reduceNaive (A, d_Out, h_Out, N);	
			break;
		case 2:
			ans = reduceNonDiverge (A, d_Out, h_Out, N);	
			break;
		case 3:
			ans = reduceSeqAdd (A, d_Out, h_Out, N);	
			break;
		case 4:
			ans = reduceFirstAdd (A, d_Out, h_Out, N);	
			break;
		case 5:
			ans = reduceUnrollLast (A, d_Out, h_Out, N);	
			break;
		case 6:
			ans = reduceUnrollAll (A, d_Out, h_Out, N);	
			break;
		case 7:
			ans = reduceMultAdd (A, d_Out, h_Out, N);	
			break;
		default:
			ans = reduceNaive (A, d_Out, h_Out, N);	
	} 
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime = elapsedTime / NUM_ITER;


	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Equivalent performance: %f GB/s\n", 
						(N * sizeof (dtype) / elapsedTime) * 1e-6);

	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

	free (h_Out);
	CUDA_CHECK_ERROR (cudaFree (d_Out));

	*ret = ans;	
}


