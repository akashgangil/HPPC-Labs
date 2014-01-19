#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "driver.h"

int
compareL2fe(const float* reference, const float* data, const unsigned int len, 
						const float epsilon)
{
	unsigned int i;
	float error = 0;
	float ref = 0;

	for(i = 0; i < len; ++i) {
		float diff = reference[i] - data[i];
		error += diff * diff;
		ref += reference[i] * reference[i];
	}

	float normRef = sqrtf(ref);
	if (fabs(ref) < 1e-7) {
			return 0;
	}
	float normError = sqrtf(error);
	error = normError / normRef;

	int result;
	if(error < epsilon) result = 1; else result = 0;

	return result;
}

void
cpuMM (dtype* A, dtype* B, dtype* C, unsigned int N)
{
	unsigned int i, j, k;
	for (i = 0; i < N; ++i) {
		for (j = 0; j < N; ++j) {
			double sum = 0;
			for (k = 0; k < N; ++k) {
				double a = A[i * N + k];
				double b = B[k * N + j];
				sum += a * b;
			}
			C[i * N + j] = (float)sum;
		}
	}	

}

void
parseArgs (int argc, char** argv, unsigned int *N, unsigned int *OPT)
{
	if(argc < 3) {
		fprintf (stderr, "usage: %s <N> <test type>\n", argv[0]);
		exit (EXIT_FAILURE);
	} else {
		*N = atoi (argv[1]);
		*OPT = atoi (argv[2]);
	}
}

void
initArray (dtype* A, unsigned int N)
{
	unsigned int i;

	// srand48 (time (NULL));
	srand (2006);

	for(i = 0;i < N; i++) {
		// A[i] = drand48 ();
		// A[i] = (rand () & 0xFF) / ((dtype) RAND_MAX);
		A[i] = rand () / (float) RAND_MAX;
	}
}

int
cmpArr (dtype* a, dtype* b, int N, float tol)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(fabs(a[i] - b[i]) > tol) cnt++;
	}

	return cnt;
}


int main (int argc, char** argv)
{
	/* declare variables */
	dtype *h_A, *d_A, *h_B, *d_B, *h_C, *d_C, *h_Reference;
	unsigned int N, OPT;
	int cnt;


	/* read arguments */
	N = 0;
	OPT = 0;
	parseArgs (argc, argv, &N, &OPT);
	assert ((N > 0));
	assert ((OPT > 0));

	/* declare and initialize data */
	h_A = (dtype*) malloc (N * N * sizeof (dtype));
	h_B = (dtype*) malloc (N * N * sizeof (dtype));
	h_C = (dtype*) malloc (N * N * sizeof (dtype));
	h_Reference = (dtype*) malloc (N * N * sizeof (dtype));
	initArray (h_A, N * N);
	initArray (h_B, N * N);
	initCudaArray (&d_A, h_A, N * N);
	initCudaArray (&d_B, h_B, N * N);
	initCudaArray (&d_C, h_C, N * N);

	/* do matrix multiply */
	cudaMM (d_A, d_B, d_C, N, OPT, h_C);

	/* compare answers */	
	cpuMM (h_A, h_B, h_Reference, N);
	if(compareL2fe(h_Reference, h_C, N * N, 1.0e-6f)) {
		fprintf (stderr, "Correct answer\n");
	} else {
		fprintf (stderr, "Incorrect answer\n");
		cnt = cmpArr (h_Reference, h_C, N * N, 1.0e-5f);
	}


	free (h_A);
	free (h_B);
	
	return 0;
}
