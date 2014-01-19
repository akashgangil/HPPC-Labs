# Questions for Lab 5


## Part 0: Getting started

*What is the name of the processor which you used for this assignment?*

Answer: jinx15, Intel(R) Xeon(R) CPU X5650 @ 2.67GHz




Part 1: Profiling
---------------------

1. What is the IPC of the utility in our use case? Is it good?

Command:	perf stat -e instructions -e cycles bin/convert -blur 15x15 -colorspace gray input.jpg output.png
Answer: IPC is 1.29. A value of 2 is average expected value, so we can do better.

------------------------------------------------------------------------------------------------
 Performance counter stats for 'bin/convert -blur 15x15 -colorspace gray input.jpg output.png':

       21680445171 instructions              #    1.29  insns per cycle        
       16804138631 cycles                    #    0.000 GHz                    

       6.526487107 seconds time elapsed
-------------------------------------------------------------------------------------------------

2. What is the fraction of mispredicted branches? Is it acceptable?

Command:	 perf stat -e branch-misses bin/convert -blur 15x15 -colorspace gray input.jpg output.png
Answer: 15410977 branch misses. No, this is not acceptable.
---------------------------------------------------------------------------------------------
 Performance counter stats for 'bin/convert -blur 15x15 -colorspace gray input.jpg output.png':

          15410977 branch-misses                                               

       6.084576183 seconds time elapsed
-----------------------------------------------------------------------------------------------

3. What is the rate of cache misses? How do you feel about this number?

Command: 	perf stat -e cache-misses bin/convert -blur 15x15 -colorspace gray input.jpg output.png
Answer: 4503959 cache-misses. This would adversly affect the performance, as this then leads to a memory access. We can do better.

-----------------------------------------------------------------------------------------------
 Performance counter stats for 'bin/convert -blur 15x15 -colorspace gray input.jpg output.png':

           4503959 cache-misses                                                

       6.881817227 seconds time elapsed
------------------------------------------------------------------------------------------------

4. Which two functions take most of the execution time? What do they do?
Command: perf record -e cycles bin/convert -blur 15x15 -colorspace gray input.jpg output.png
	 perf report

Since more the cycles, more time would the function take to execute. Below are the two function that take max time.

 52.39%  convert  convert             [.] MorphologyApply   Blurs the image
 35.27%  convert  convert             [.] WriteOnePNGImage  Save the Image



Part 2: Compiler Optimizations
------------------------------

1. What is the "User Time" for program execution before you start optimizing?
Command: time bin/convert -blur 15x15 -colorspace gray input.jpg output.png
Answer: 6.686 seconds

2. What is the "User Time" for program execution after you completed **all** three steps and rune the program with `-fprofile-use`?

Using GCC -fprofile-generate / -O2 / -fprofile-use I get, 
-------------------------------------------------
2.16  insns per cycle 
-------------------------------------------------------------------------------------------------
Performance counter stats for 'bin/convert -blur 15x15 -colorspace gray input.jpg output.png':

       17607621373 instructions              #    2.15  insns per cycle        
        8170804390 cycles                    #    0.000 GHz                    

       3.872584594 seconds time elapsed
------------------------------------------------------------------------------------------------

USER TIME: 3.85 seconds
----------------------------------------------------------------
time bin/convert -blur 15x15 -colorspace gray input.jpg output.png

real	0m4.038s
user	0m3.858s
sys	0m0.039s
-----------------------------------------------------------------

Part 3: Out of Class. 
--------------------

I got a minimum time of 3.5 seconds on login node. 2.7seconds on compute node. Over an average of 5 runs.
---------------------------------------------------------------------------------------------------------

I noticed high variations when running convert on login node at different times. However, my executable consistently executes
under under 3-3.2 seconds on the compute node.

I tried both gcc and icc ompiler. From my experiments gcc performed better than icc. I used the following options with each.

The GCC compiled excutable gave 2.7 seconds of execution time on compute node on an average of 5 runs. But showed a high 
degree of variation in runtime on login node.

I also specified CXX flags because the C++ files were getting compiled using those flags.

*gcc compiler* with the following 5 options:
--------------------------------------------

1. *-fprofile-generate* /*-fprofile-use* for Profile guided optimization.

2. *-Ofast* for all O3 optimizations and -ffast-math math optimizations.

3. *-march=corei7* for machine specific optimizations

4. *-flto* for Interprocedural Optimizations

5. *-msse2* to take advantage of SSE2 instruction set.

Steps: 
------

1. ./configure ... CFLAGS="-fprofile-generate -Ofast -march=corei7 -flto -msse2" LDFLAGS="-fprofile-generate -Ofast -march=corei7 -flto -msse2" CXXFLAGS="-fprofile-generate -Ofast -march=corei7 -flto -msse2"

2. make -j8

3. make install

4. bin/convert -blur 15x15 -colorspace gray input.jpg output.png

5. make clean

6. ./configure ... CFLAGS="-fprofile-use -Ofast -march=corei7 -flto -msse2" LDFLAGS="-fprofile-use -Ofast -march=corei7 -flto -msse2" CXXFLAGS="-fprofile-use -Ofast -march=corei7 -flto -msse2"

*icc compiler* with the following 8 options:
------------------------------------------------
1. *-prof-gen* / *-prof-use* for profile guided optimizations

2. *-ipo* for Inter Procedural Optimizations

3. *-O3* for other general compiler optimizations

4. *-xhost* to optimiza the code for the running CPU by utilizing any of its special instrcution set.

5. *-axsse4.2* to optimize SIMD instructions execution.

6. *-no-prev-div* to speed up the division process like A/B would be breaken down to the multiplication of A * (1/B)

7. *-finline* to inline required functions to prevent the call.

8. *-funroll-all-loops* to optimize the loops.


------------------------------------------------------

a) Compiled using -prof-gen
./configure ....  CFLAGS="-prof-gen -O3 -ipo -no-prec-div -opt-prefetch -xhost -axsse4.2 -finline -funroll-all-loops" LDFLAGS="-prof-gen -O3 -ipo -no-prec-div -opt-prefetch -xhost -axsse4.2 -finline -funroll-all-loops" AR=xiar CC=icc

b) make -j8 && 
make install

c) ran the program to give the compiler profiling info. 
bin/convert -blur 15x15 -colorspace gray input.jpg output.png

d) make clean (to clean up previously generated object files)

e) compiled again using -prof-use to do profile guided optimization
./configure ....  CFLAGS="-prof-use -O3 -ipo -no-prec-div -opt-prefetch -xhost -axsse4.2 -finline -funroll-all-loops" LDFLAGS="-prof-use -O3 -ipo -no-prec-div -opt-prefetch -xhost -axsse4.2 -finline -funroll-all-loops" AR=xiar CC=icc

f) make -j8 &&
make install
---------------------------------------------------------------
