ifeq ($(CXX),icpc)
    override CXXFLAGS += -O3 -xSSE4.2 -no-intel-extensions
    override LDFLAGS += -static-intel
else
    override CXXFLAGS += -O3 -march=corei7
    override LDFLAGS += -static-libgcc
endif

all: pagerank-test pr-turbo

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -std=gnu++0x -g -I. -c -o $@ $<

%.po : %.cpp
	$(CXX) -fPIC $(CXXFLAGS) -std=gnu++0x -I. -c -o $@ $<

pr-turbo: libprturbo.so

libprturbo.so: pagerank-turbo.po
	$(CXX) $(LDFLAGS) -shared -fPIC -o $@ $^

pagerank-test: pagerank-test.o pagerank-reference.o timer.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lrt -ldl -lpthread

queue: all
	qsub pagerank.pbs

run-local: pagerank-test pr-turbo
	@./pagerank-test

report: probabilities-naive.log probabilities-optimized.log
	@python show-urls.py

grade: libprturbo.so
	@curl -F "student=$(LOGNAME)" -F "lab=10" -F "submission=@libprturbo.so" http://mochalisa.cc.gatech.edu:8080/submit

clean:
	rm -f *.o
	rm -f *.po
	rm -f *.log
	rm -f pagerank-test
	rm -f libprturbo.so
