CC = icpc
CFLAGS =
COPTFLAGS = -O3 -g
LDFLAGS =

qsort: driver.o sort.o sequential-sort.o parallel-qsort.o
	$(CC) $(COPTFLAGS) -o $@ $^

%.o: %.cc
	$(CC) $(CFLAGS) $(COPTFLAGS) -o $@ -c $<

clean:
	rm -f core *.o *~

# eof
