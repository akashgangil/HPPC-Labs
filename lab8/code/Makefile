ifeq ($(CXX),icpc)
    override CXXFLAGS += -O3 -xSSE4.2 -no-intel-extensions
    override LDFLAGS += -static-intel
else
    override CXXFLAGS += -O3 -march=corei7
    override LDFLAGS += -static-libgcc
endif

all: image-test simdimage

%.o : %.cpp
	$(CXX) $(CXXFLAGS) -I. -c -o $@ $<
%.po : %.cpp
	$(CXX) -fPIC $(CXXFLAGS) -I. -c -o $@ $<

simdimage: libsimdimage.so

libsimdimage.so: image-simd.po
	$(CXX) $(LDFLAGS) -shared -fPIC -Wl,-Bstatic -lc -Wl,-Bstatic -lm -o $@ $^

image-test: image-test.o image-reference.o image-io.o timer.o
	$(CXX) $(LDFLAGS) -o $@ $^ -lrt -ldl

grade: libsimdimage.so
	@curl -F "student=$(LOGNAME)" -F "submission=@libsimdimage.so" http://coffeelab.cc.gt.atl.ga.us:8080/submit

clean:
	rm -f *.o
	rm -f *.po
	rm -f *.bmp
	rm -f image-test
	rm -f libsimdimage.so
