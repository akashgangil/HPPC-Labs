CC = icpc
CFLAGS =
COPTFLAGS = -O3 -g
LDFLAGS =

# OpenMP flags
# To prevent mixing of Cilk Plus and OpenMP, the extra parameters cause Cilk keywords to be errors
OMPFLAGS = -openmp -D_Cilk_for=\#error -D_Cilk_spawn=\#error -D_Cilk_sync=\#error

default:
	@echo "=================================================="
	@echo "To build the original Cilk Plus code, use:"
	@echo "  make qsort-cilk"
	@echo ""
	@echo "To build your OpenMP code, use:"
	@echo "  make qsort-omp"
	@echo ""
	@echo "To clean this subdirectory (remove object files"
	@echo "and other junk), use:"
	@echo "  make clean"
	@echo "=================================================="

# Cilk driver
qsort-cilk: driver.o sort.o parallel-qsort--cilk.o
	$(CC) $(COPTFLAGS) -o $@ $^

# Default rules -- assume Cilk
%.o: %.cc
	$(CC) $(CFLAGS) $(COPTFLAGS) -o $@ -c $<

# Quicksort driver using OpenMP
qsort-omp: driver.o sort.o parallel-qsort--omp.o
	$(CC) $(COPTFLAGS) $(OMPFLAGS) -o $@ $^

parallel-qsort--omp.o: parallel-qsort--omp.cc
	$(CC) $(COPTFLAGS) $(OMPFLAGS) -o $@ -c $<

# Mergesort driver using OpenMP
mergesort-omp: driver.o sort.o parallel-mergesort--omp.o
	$(CC) $(COPTFLAGS) $(OMPFLAGS) -o $@ $^

parallel-mergesort--omp.o: parallel-mergesort--omp.cc
	$(CC) $(COPTFLAGS) $(OMPFLAGS) -o $@ -c $<

clean:
	rm -f core *.o *~ qsort-cilk qsort-omp mergesort-omp

# eof
