#include "driver.h"
#include "mm.h"
#include "cuda_utils.h"

void
initCudaArray (dtype **d_A, dtype *h_A, unsigned int N)
{
	CUDA_CHECK_ERROR (cudaMalloc ((void**) d_A, N * sizeof (dtype)));
	CUDA_CHECK_ERROR (cudaMemcpy (*d_A, h_A, N * sizeof (dtype),
																cudaMemcpyHostToDevice));
}


__global__
void
mmSharedKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{

	

	/* block indices */
	int bidx = blockIdx.x;
	int bidy = blockIdx.y;

	/* thread indices */
	int tidx = threadIdx.x;
	int tidy = threadIdx.y;

	/* row  index of first sub-block of matrix A processed by this thread block */
	int aStart = N * (BLOCK_SIZE * bidy);
	/* row  index of last sub-block of matrix A processed by this thread block */
	int aEnd   = aStart + N - 1;
	/* increment size for sub-block of matrix A */
	int aInc = BLOCK_SIZE;

	/* col index of first sub-blcok of matrx B processed by this thread block */
	int bStart = BLOCK_SIZE * bidx;
	/* last sub block is not needed since it'll have 1-on-1 match to A */
	/* increment size for sub-block of matrix B */
	int bInc = BLOCK_SIZE * N;

	/* temporary variable for accummulating the partial results */
	float cSub = 0;

	/* Loop over the sub-matrices of A and B */
	for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
		/* declaration of shared memory for storing sub-block of A */
		__shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

		/* declaration of shared memory for storing sub-block of B */
		__shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

		/* load the matrices from memory to shared memory */
		As[tidy][tidx] = A[a + N * tidy + tidx];
		Bs[tidy][tidx] = B[b + N * tidy + tidx];
		__syncthreads();

		/* multiply the two matrices together */
		/* one thread per element of C */
#pragma unroll
		for (int k = 0; k < BLOCK_SIZE; ++k)
			cSub += As[tidy][k] * Bs[k][tidx];

		/* synchornize before loading next sub-blocks */
		__syncthreads();
	}

	/* write back the results */
	int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub;

}
void
mmShared (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE);	

	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmSharedKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}



__global__
void
mmNaiveKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	int i;
	dtype sum;
	int gidx = threadIdx.x + blockIdx.x * blockDim.x; /* column (j) */
	int gidy = threadIdx.y + blockIdx.y * blockDim.y; /* row (i) */
	int gid = gidx + gidy * N;

	sum = 0.0;
	for(i = 0; i < N; i++) {
		sum += A[gidy * N + i] * B[i * N + gidx];
	}
	C[gid] = sum;
}
void
mmNaive (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE);	


	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();


	mmNaiveKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}


__global__
void
mmShared2Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
        /* block indices */
        int bidx = blockIdx.x;
        int bidy = blockIdx.y;

        /* thread indices */
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        /* row  index of first sub-block of matrix A processed by this thread block */
        int aStart = N * (BLOCK_SIZE * bidy);
        /* row  index of last sub-block of matrix A processed by this thread block */
        int aEnd   = aStart + N - 1;
        /* increment size for sub-block of matrix A */
        int aInc = BLOCK_SIZE;

        /* col index of first sub-blcok of matrx B processed by this thread block */
        int bStart = BLOCK_SIZE * bidx;
        /* last sub block is not needed since it'll have 1-on-1 match to A */
        /* increment size for sub-block of matrix B */
        int bInc = BLOCK_SIZE * N;


	/* temporary variable for accummulating the partial results */
        float cSub = 0;
        float cSub_1 = 0;

        int incr = BLOCK_SIZE >> 1;
        /* Loop over the sub-matrices of A and B */
        for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
                /* declaration of shared memory for storing sub-block of A */
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

                /* declaration of shared memory for storing sub-block of B */
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                /* load the matrices from memory to shared memory */
                As[tidy][tidx] = A[a + N * tidy + tidx];
                As[tidy + incr][tidx] = A[a + N * (tidy + incr) + tidx];

                Bs[tidy][tidx] = B[b + N * tidy + tidx];
                Bs[tidy + incr][tidx]  = B[b + N * (tidy + incr) + tidx];

                __syncthreads();

                /* multiply the two matrices together */
                /* one thread per element of C */
#pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k){
                        cSub += As[tidy][k] * Bs[k][tidx];
                        cSub_1 += As[tidy + incr][k] * Bs[k][tidx];
                }
                /* synchornize before loading next sub-blocks */
                __syncthreads();
        }

        /* write back the results */
        int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
        C[c + N * tidy + tidx] = cSub;
        C[c + N * (tidy + incr) + tidx] = cSub_1;

}
void
mmShared2 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 2);	

	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared2Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}


__global__
void
mmShared4Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	
	/* block indices */
        int bidx = blockIdx.x;
        int bidy = blockIdx.y;

        /* thread indices */
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        /* row  index of first sub-block of matrix A processed by this thread block */
        int aStart = N * (BLOCK_SIZE * bidy);
        /* row  index of last sub-block of matrix A processed by this thread block */
        int aEnd   = aStart + N - 1;
        /* increment size for sub-block of matrix A */
        int aInc = BLOCK_SIZE;

        /* col index of first sub-blcok of matrx B processed by this thread block */
        int bStart = BLOCK_SIZE * bidx;
        /* last sub block is not needed since it'll have 1-on-1 match to A */
        /* increment size for sub-block of matrix B */
        int bInc = BLOCK_SIZE * N;


	/* temporary variable for accummulating the partial results */
        float cSub = 0;
        float cSub_1 = 0;
	float cSub_2 = 0;
	float cSub_3 = 0;

        int incr = BLOCK_SIZE >> 2;
        /* Loop over the sub-matrices of A and B */
        for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
                /* declaration of shared memory for storing sub-block of A */
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

                /* declaration of shared memory for storing sub-block of B */
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                /* load the matrices from memory to shared memory */
                As[tidy][tidx] = A[a + N * tidy + tidx];
                As[tidy + incr][tidx] = A[a + N * (tidy + incr) + tidx];
                As[tidy + 2*incr][tidx] = A[a + N * (tidy + 2*incr) + tidx];
                As[tidy + 3*incr][tidx] = A[a + N * (tidy + 3*incr) + tidx];

                Bs[tidy][tidx] = B[b + N * tidy + tidx];
                Bs[tidy + incr][tidx]  = B[b + N * (tidy + incr) + tidx];
                Bs[tidy + 2*incr][tidx]  = B[b + N * (tidy + 2*incr) + tidx];
                Bs[tidy + 3*incr][tidx]  = B[b + N * (tidy + 3*incr) + tidx];

                __syncthreads();

                /* multiply the two matrices together */
                /* one thread per element of C */
#pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k){
                        cSub += As[tidy][k] * Bs[k][tidx];
                        cSub_1 += As[tidy + incr][k] * Bs[k][tidx];
			cSub_2 += As[tidy + 2*incr][k] * Bs[k][tidx];
			cSub_3 += As[tidy + 3*incr][k] * Bs[k][tidx];
                }
                /* synchornize before loading next sub-blocks */
                __syncthreads();
        }

        /* write back the results */
        int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
        C[c + N * tidy + tidx] = cSub;
        C[c + N * (tidy + incr) + tidx] = cSub_1;
        C[c + N * (tidy + 2*incr) + tidx] = cSub_2;
        C[c + N * (tidy + 3*incr) + tidx] = cSub_3;

}
void
mmShared4 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 4);	

	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared4Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}



__global__
void
mmShared8Kernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
        /* block indices */
        int bidx = blockIdx.x;
        int bidy = blockIdx.y;

        /* thread indices */
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        /* row  index of first sub-block of matrix A processed by this thread block */
        int aStart = N * (BLOCK_SIZE * bidy);
        /* row  index of last sub-block of matrix A processed by this thread block */
        int aEnd   = aStart + N - 1;
        /* increment size for sub-block of matrix A */
        int aInc = BLOCK_SIZE;

        /* col index of first sub-blcok of matrx B processed by this thread block */
        int bStart = BLOCK_SIZE * bidx;
        /* last sub block is not needed since it'll have 1-on-1 match to A */
        /* increment size for sub-block of matrix B */
        int bInc = BLOCK_SIZE * N;


        /* temporary variable for accummulating the partial results */
        float cSub = 0;
	float cSub_1 = 0;
	float cSub_2 = 0;
	float cSub_3 = 0;
	float cSub_4 = 0;
	float cSub_5 = 0;
	float cSub_6 = 0;
	float cSub_7 = 0;
	

        int incr = BLOCK_SIZE >> 3;
        /* Loop over the sub-matrices of A and. B */
        for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
                /* declaration of shared memory for storing sub-block of A */
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
	

                /* declaration of shared memory for storing sub-block of B */
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                /* load the matrices from memory to shared memory */
                As[tidy][tidx] = A[a + N * tidy + tidx];
                As[tidy + incr][tidx] = A[a + N * (tidy + incr) + tidx];
                As[tidy + 2*incr][tidx] = A[a + N * (tidy + 2*incr) + tidx];
                As[tidy + 3*incr][tidx] = A[a + N * (tidy + 3*incr) + tidx];
                As[tidy + 4*incr][tidx] = A[a + N * (tidy + 4*incr) + tidx];
                As[tidy + 5*incr][tidx] = A[a + N * (tidy + 5*incr) + tidx];
                As[tidy + 6*incr][tidx] = A[a + N * (tidy + 6*incr) + tidx];
                As[tidy + 7*incr][tidx] = A[a + N * (tidy + 7*incr) + tidx];


                Bs[tidy][tidx] = B[b + N * tidy + tidx];
                Bs[tidy + incr][tidx]  = B[b + N * (tidy + incr) + tidx];
                Bs[tidy + 2*incr][tidx]  = B[b + N * (tidy + 2*incr) + tidx];
                Bs[tidy + 3*incr][tidx]  = B[b + N * (tidy + 3*incr) + tidx];
                Bs[tidy + 4*incr][tidx]  = B[b + N * (tidy + 4*incr) + tidx];
                Bs[tidy + 5*incr][tidx]  = B[b + N * (tidy + 5*incr) + tidx];
                Bs[tidy + 6*incr][tidx]  = B[b + N * (tidy + 6*incr) + tidx];
                Bs[tidy + 7*incr][tidx]  = B[b + N * (tidy + 7*incr) + tidx];

                __syncthreads();

                /* multiply the two matrices together */
                /* one thread per element of C */
#pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k){
                        cSub += As[tidy][k] * Bs[k][tidx];
                        cSub_1 += As[tidy + incr][k] * Bs[k][tidx];
                        cSub_2 += As[tidy + 2*incr][k] * Bs[k][tidx];
                        cSub_3 += As[tidy + 3*incr][k] * Bs[k][tidx];
                        cSub_4 += As[tidy + 4*incr][k] * Bs[k][tidx];
                        cSub_5 += As[tidy + 5*incr][k] * Bs[k][tidx];
                        cSub_6 += As[tidy + 6*incr][k] * Bs[k][tidx];
                        cSub_7 += As[tidy + 7*incr][k] * Bs[k][tidx];
                }
                /* synchornize before loading next sub-blocks */
                __syncthreads();
        }

        /* write back the results */
        int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
        C[c + N * tidy + tidx] = cSub;
        C[c + N * (tidy + incr) + tidx] = cSub_1;
        C[c + N * (tidy + 2*incr) + tidx] = cSub_2;
        C[c + N * (tidy + 3*incr) + tidx] = cSub_3;
        C[c + N * (tidy + 4*incr) + tidx] = cSub_4;
        C[c + N * (tidy + 5*incr) + tidx] = cSub_5;
        C[c + N * (tidy + 6*incr) + tidx] = cSub_6;
        C[c + N * (tidy + 7*incr) + tidx] = cSub_7;

}
void
mmShared8 (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE, BLOCK_SIZE / 8);	

	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmShared8Kernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}

__global__
void
mmMyOwnKernel (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	/* block indices */
        int bidx = blockIdx.x;
        int bidy = blockIdx.y;

        /* thread indices */
        int tidx = threadIdx.x;
        int tidy = threadIdx.y;

        /* row  index of first sub-block of matrix A processed by this thread block */
        int aStart = N * (BLOCK_SIZE * bidy);
        /* row  index of last sub-block of matrix A processed by this thread block */
        int aEnd   = aStart + N - 1;
        /* increment size for sub-block of matrix A */
        int aInc = BLOCK_SIZE;

        /* col index of first sub-blcok of matrx B processed by this thread block */
        int bStart = BLOCK_SIZE * bidx;
        /* last sub block is not needed since it'll have 1-on-1 match to A */
        /* increment size for sub-block of matrix B */
        int bInc = BLOCK_SIZE * N;


        /* temporary variable for accummulating the partial results */
        float cSub = 0;
        float cSub_1 = 0;
        float cSub_2 = 0;
        float cSub_3 = 0;
        float cSub_4 = 0;
        float cSub_5 = 0;
        float cSub_6 = 0;
        float cSub_7 = 0;
        float cSub_8 = 0;
        float cSub_9 = 0;
        float cSub_10 = 0;
        float cSub_11 = 0;
        float cSub_12 = 0;
        float cSub_13 = 0;
        float cSub_14 = 0;
	float cSub_15 = 0;

        int y_incr = BLOCK_SIZE >> 3;
	int x_incr = BLOCK_SIZE >> 1;
        /* Loop over the sub-matrices of A and B */
        for (int a = aStart, b = bStart; a <= aEnd; a += aInc, b += bInc) {
                /* declaration of shared memory for storing sub-block of A */
                __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];


                /* declaration of shared memory for storing sub-block of B */
                __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

                /* load the matrices from memory to shared memory */

		for(int i=0; i<8; i++){
			As[tidy + i*y_incr][tidx] = A[a + N * (tidy + i * y_incr) + tidx];
			As[tidy + i*y_incr][tidx + x_incr] = A[a + N * (tidy + i * y_incr) + tidx + x_incr];
		}	
/*
                As[tidy][tidx] = A[a + N * tidy + tidx];
                As[tidy][tidx + x_incr] = A[a + N * tidy + tidx + x_incr];
                
		As[tidy + y_incr][tidx] = A[a + N * (tidy + y_incr) + tidx];
                As[tidy + y_incr][tidx + x_incr] = A[a + N * (tidy + y_incr) + tidx + x_incr];
                
		As[tidy + 2*y_incr][tidx] = A[a + N * (tidy + 2*y_incr) + tidx];
                As[tidy + 2*y_incr][tidx + x_incr] = A[a + N * (tidy + 2*y_incr) + tidx + x_incr];
                
		As[tidy + 3*y_incr][tidx] = A[a + N * (tidy + 3*y_incr) + tidx];
                As[tidy + 3*y_incr][tidx + x_incr] = A[a + N * (tidy + 3*y_incr) + tidx + x_incr];
                
		As[tidy + 4*y_incr][tidx] = A[a + N * (tidy + 4*y_incr) + tidx];
                As[tidy + 4*y_incr][tidx + x_incr] = A[a + N * (tidy + 4*y_incr) + tidx + x_incr];
                
		As[tidy + 5*y_incr][tidx] = A[a + N * (tidy + 5*y_incr) + tidx];
                As[tidy + 5*y_incr][tidx + x_incr] = A[a + N * (tidy + 5*y_incr) + tidx + x_incr];
                
		As[tidy + 6*y_incr][tidx] = A[a + N * (tidy + 6*y_incr) + tidx];
                As[tidy + 6*y_incr][tidx + x_incr] = A[a + N * (tidy + 6*y_incr) + tidx + x_incr];
                
		As[tidy + 7*y_incr][tidx] = A[a + N * (tidy + 7*y_incr) + tidx];
                As[tidy + 7*y_incr][tidx + x_incr] = A[a + N * (tidy + 7*y_incr) + tidx + x_incr];
              
*/		
		for(int j=0; j<8; j++){
			Bs[tidy + j * y_incr][tidx] = B[b + N * (tidy + j * y_incr) + tidx];
			Bs[tidy + j * y_incr][tidx + x_incr] = B[b + N * (tidy + j * y_incr) + tidx + x_incr];
		}
/*

                Bs[tidy][tidx] = B[b + N * tidy + tidx];
                Bs[tidy][tidx + x_incr] = B[b + N * tidy + tidx + x_incr];
                
		Bs[tidy + y_incr][tidx]  = B[b + N * (tidy + y_incr) + tidx];
                Bs[tidy + y_incr][tidx + x_incr]  = B[b + N * (tidy + y_incr) + tidx + x_incr];
                
		Bs[tidy + 2*y_incr][tidx]  = B[b + N * (tidy + 2*y_incr) + tidx];
                Bs[tidy + 2*y_incr][tidx + x_incr]  = B[b + N * (tidy + 2*y_incr) + tidx + x_incr];
                
		Bs[tidy + 3*y_incr][tidx]  = B[b + N * (tidy + 3*y_incr) + tidx];
                Bs[tidy + 3*y_incr][tidx + x_incr]  = B[b + N * (tidy + 3*y_incr) + tidx + x_incr];
                
		Bs[tidy + 4*y_incr][tidx]  = B[b + N * (tidy + 4*y_incr) + tidx];
                Bs[tidy + 4*y_incr][tidx + x_incr]  = B[b + N * (tidy + 4*y_incr) + tidx + x_incr];
                
		Bs[tidy + 5*y_incr][tidx]  = B[b + N * (tidy + 5*y_incr) + tidx];
                Bs[tidy + 5*y_incr][tidx + x_incr]  = B[b + N * (tidy + 5*y_incr) + tidx + x_incr];
                
		Bs[tidy + 6*y_incr][tidx]  = B[b + N * (tidy + 6*y_incr) + tidx];
                Bs[tidy + 6*y_incr][tidx + x_incr]  = B[b + N * (tidy + 6*y_incr) + tidx + x_incr];
                
		Bs[tidy + 7*y_incr][tidx]  = B[b + N * (tidy + 7*y_incr) + tidx];
                Bs[tidy + 7*y_incr][tidx + x_incr]  = B[b + N * (tidy + 7*y_incr) + tidx + x_incr];
  */              
                
		__syncthreads();

                /* multiply the two matrices together */
                /* one thread per element of C */
#pragma unroll
                for (int k = 0; k < BLOCK_SIZE; ++k){
                        cSub += As[tidy][k] * Bs[k][tidx];
			cSub_8 += As[tidy][k] * Bs[k][tidx + x_incr];

                        cSub_1 += As[tidy + y_incr][k] * Bs[k][tidx];
			cSub_9 += As[tidy + y_incr][k] * Bs[k][tidx + x_incr];

			cSub_2 += As[tidy + 2*y_incr][k] * Bs[k][tidx];
			cSub_10 += As[tidy + 2*y_incr][k] * Bs[k][tidx + x_incr];

                        cSub_3 += As[tidy + 3*y_incr][k] * Bs[k][tidx];
			cSub_11 += As[tidy + 3*y_incr][k] * Bs[k][tidx + x_incr];                       

			cSub_4 += As[tidy + 4*y_incr][k] * Bs[k][tidx];
                        cSub_12 += As[tidy + 4*y_incr][k] * Bs[k][tidx + x_incr];  

			cSub_5 += As[tidy + 5*y_incr][k] * Bs[k][tidx];
			cSub_13 += As[tidy + 5*y_incr][k] * Bs[k][tidx + x_incr];                        

			cSub_6 += As[tidy + 6*y_incr][k] * Bs[k][tidx];
			cSub_14 += As[tidy + 6*y_incr][k] * Bs[k][tidx + x_incr];
                        
			cSub_7 += As[tidy + 7*y_incr][k] * Bs[k][tidx];
			cSub_15 += As[tidy + 7*y_incr][k] * Bs[k][tidx + x_incr];
                }
                /* synchornize before loading next sub-blocks */
                __syncthreads();
        }

        /* write back the results */
        int c = N * BLOCK_SIZE * bidy + BLOCK_SIZE * bidx;
	C[c + N * tidy + tidx] = cSub;
        C[c + N * tidy + tidx + x_incr] = cSub_8;
        
	C[c + N * (tidy + y_incr) + tidx] = cSub_1;
	C[c + N * (tidy + y_incr) + tidx + x_incr] = cSub_9;
        
	C[c + N * (tidy + 2*y_incr) + tidx] = cSub_2;
	C[c + N * (tidy + 2*y_incr) + tidx + x_incr] = cSub_10;
        
	C[c + N * (tidy + 3*y_incr) + tidx] = cSub_3;
	C[c + N * (tidy + 3*y_incr) + tidx + x_incr] = cSub_11;
        
	C[c + N * (tidy + 4*y_incr) + tidx] = cSub_4;
	C[c + N * (tidy + 4*y_incr) + tidx + x_incr] = cSub_12;
        
	C[c + N * (tidy + 5*y_incr) + tidx] = cSub_5;
	C[c + N * (tidy + 5*y_incr) + tidx + x_incr] = cSub_13;
        
	C[c + N * (tidy + 6*y_incr) + tidx] = cSub_6;
	C[c + N * (tidy + 6*y_incr) + tidx + x_incr] = cSub_14;
        
	C[c + N * (tidy + 7*y_incr) + tidx] = cSub_7;
	C[c + N * (tidy + 7*y_incr) + tidx + x_incr] = cSub_15;

}

void
mmMyOwn (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int nBlocks;


	nBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

	dim3 grid (nBlocks, nBlocks);	
	dim3 block (BLOCK_SIZE / 2, BLOCK_SIZE / 8);	

	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
	mmMyOwnKernel <<<grid, block>>> (A, B, C, N);
	cudaThreadSynchronize ();
}




void
cudaMM (dtype *A, dtype* B, dtype* C, unsigned int N, unsigned int OPT, dtype* h_C)
{
	cudaEvent_t start, stop;
	float elapsedTime;

	CUDA_CHECK_ERROR (cudaEventCreate (&start));
	CUDA_CHECK_ERROR (cudaEventCreate (&stop));

	fprintf (stderr, "Executing test case [%d]\n", OPT);
	fprintf (stderr, "[1]: Naive | [2]: shared memory| [3]: SM 2 per thread | [4]: SM 4 per thread | [5]: SM 8 per thread | [6]: my own implementation \n");

	
	CUDA_CHECK_ERROR (cudaEventRecord (start, 0));
	/* execute kernel */
	switch (OPT) {
		case 1:
			mmNaive (A, B, C, N);	
			break;
		case 2:
			mmShared (A, B, C, N);	
			break;
		case 3:
			mmShared2 (A, B, C, N);	
			break;
		case 4:
			mmShared4 (A, B, C, N);	
			break;
		case 5:
			mmShared8 (A, B, C, N);	
			break;
		case 6:
			mmMyOwn (A, B, C, N);
			break;
		default:
			mmNaive (A, B, C, N);	
	} 
	CUDA_CHECK_ERROR (cudaEventRecord (stop, 0));
	CUDA_CHECK_ERROR (cudaEventSynchronize (stop));
	CUDA_CHECK_ERROR (cudaEventElapsedTime (&elapsedTime, start, stop));
	elapsedTime = elapsedTime / 5;

	CUDA_CHECK_ERROR (cudaMemcpy (h_C, C, N * N * sizeof (dtype), 
																cudaMemcpyDeviceToHost));

	fprintf (stderr, "Execution time: %f ms\n", elapsedTime);
	fprintf (stderr, "Equivalent performance: %f GFLOP/s\n", 
						1e-6 * 2 * N * N * N / elapsedTime );

	CUDA_CHECK_ERROR (cudaEventDestroy (start));
	CUDA_CHECK_ERROR (cudaEventDestroy (stop));

}


