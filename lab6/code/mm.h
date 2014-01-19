#define BLOCK_SIZE 32

#ifdef __cplusplus
extern "C" {
#endif

void initCudaArray (dtype **d_A, dtype *h_A, unsigned int N);
void cudaMM (dtype *A, dtype* B, dtype* C, unsigned int N, 
									unsigned int OPT, dtype* h_C);

#ifdef __cplusplus
}
#endif

