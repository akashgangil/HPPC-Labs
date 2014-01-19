#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>

#include "driver.h"

dtype
reduceCpu (dtype* h_A, unsigned int N)
{
	int i;
	// dtype ans, ans2;
	double ans;

	ans = 0.0;
	for(i = 0; i < N; i++) {
		ans += h_A[i];
	}

	return ans;
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

	srand48 (time (NULL));

	for(i = 0;i < N; i++) {
		// A[i] = drand48 ();
		A[i] = (rand () & 0xFF) / ((dtype) RAND_MAX);
	}
}

int main (int argc, char** argv)
{
	/* declare variables */
	dtype *h_A, *d_A, ans;
	unsigned int N, OPT;


	/* read arguments */
	N = 0;
	OPT = 0;
	parseArgs (argc, argv, &N, &OPT);
	assert ((N > 0));
	assert ((OPT > 0));

	/* declare and initialize data */
	h_A = (dtype*) malloc (N * sizeof (dtype));
	initArray (h_A, N);
	initCudaArray (&d_A, h_A, N);

	/* do reduction */
	cudaReduction (d_A, N, OPT, &ans);

	if(fabs ((double) ans - (double) reduceCpu (h_A, N)) > (1e-8 * N)) {
		fprintf (stderr, "Answer is %f\n", ans);
		fprintf (stderr, "Reference answer is %f\n", reduceCpu (h_A, N));
	} else {
		fprintf (stderr, "Answer is correct\n");
		fprintf (stderr, "Reference answer is %f\n", reduceCpu (h_A, N));
		fprintf (stderr, "GPU answer is %f\n", ans);
	}

	free (h_A);
	
	return 0;
}
