#ifdef __cplusplus
extern "C" {
#endif

void initCudaArray (dtype **d_A, dtype *h_A, unsigned int N);
void cudaReduction (dtype* A, unsigned int N, unsigned int OPT, dtype* ret);

#ifdef __cplusplus
}
#endif

