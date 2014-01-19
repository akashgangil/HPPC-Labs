CC = gcc 
NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart 
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g
LDFLAGS =


reduce_CUSRCS = reduce.cu
reduce_CSRCS = driver.c timer.c
reduce_CUOBJS = $(reduce_CUSRCS:.cu=.o__cu)
reduce_COBJS = $(reduce_CSRCS:.c=.o__c)

reduce: $(reduce_CUOBJS) $(reduce_COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 


%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< -DNUM_ITER=5 -DBS=512

clean:
	rm -f core *.o__cu *.o__c *~ reduce

# eof
