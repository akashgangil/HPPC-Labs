CC = gcc 
NVCC = nvcc
CUDA_PATH = /opt/cuda-4.2/cuda
CFLAGS = -L$(CUDA_PATH)/lib64 -lcudart 
NVCCFLAGS= -arch=compute_20 -code=sm_20 -I$(CUDA_SDK_PATH)/C/common/inc
COPTFLAGS = -O3 -g
LDFLAGS =


mm_CUSRCS = mm.cu
mm_CSRCS = driver.c timer.c
mm_CUOBJS = $(mm_CUSRCS:.cu=.o__cu)
mm_COBJS = $(mm_CSRCS:.c=.o__c)

mm: $(mm_CUOBJS) $(mm_COBJS)
	$(CC) $(CFLAGS) $^ -o $@ 


%.o__c: %.c
	$(CC) -o $@ -c $<

%.o__cu: %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $< -DBS=512 -Xptxas -v

clean:
	rm -f core *.o__cu *.o__c *~ mm

# eof
