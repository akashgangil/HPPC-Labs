PROJID = lab2

# Try to find an MPICC compiler
ifeq ($(MPICC),)
  MPICC = $(shell which mpicc)
endif
ifeq ($(MPICC),)
  MPICC = $(shell which openmpicc)
endif

MPICFLAGS = -std=gnu99
MPICOPTFLAGS = -O2 -g
MPILDFLAGS =

HOST := $(shell hostname -f)
ifeq ($(HOST),daffy3)
  MPICFLAGS += -DUSE_MKL
  MKLDIR = /opt/intel/Compiler/11.1/080/Frameworks/mkl/lib/em64t
  MPILDFLAGS += -L$(MKLDIR) -lmkl_intel_lp64 -lmkl_sequential
else ifeq ($(HOST),jinx-login.cc.gatech.edu)
  MPICFLAGS += -DUSE_MKL
  MKLDIR = /opt/intel/Compiler/11.1/059/mkl/lib/em64t
  MPILDFLAGS += -L$(MKLDIR) -Wl,-rpath -Wl,$(MKLDIR) -lmkl_intel_lp64 -lmkl_sequential -lmkl_core
endif

EXEEXT =

.DEFAULT_GOAL := all

TARGETS = mm1d$(EXEEXT)
CLEANFILES =
DISTFILES = Makefile

HDRS_COMMON = util.h mat.h mpi_helper.h
SRCS_COMMON = $(HDRS_COMMON:.h=.c)
OBJS_COMMON = $(SRCS_COMMON:.c=.o)
DISTFILES += $(HDRS_COMMON) $(SRCS_COMMON)

#------------------------------------------------------------
all: check-mpicc $(TARGETS)

#------------------------------------------------------------
HDRS_1D = mm1d.h
SRCS_1D = $(HDRS_1D:.h=.c) driver1d.c
OBJS_1D = $(SRCS_1D:.c=.o)
DISTFILES += $(HDRS_1D) $(SRCS_1D)
DISTFILES += mm1d.pbs mm1d-test.pbs

mm1d$(EXEEXT): $(OBJS_1D) $(OBJS_COMMON)
	$(MPICC) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

#------------------------------------------------------------
HDRS_SUMMA = mm1d.h summa.h
SRCS_SUMMA = $(HDRS_SUMMA:.h=.c) driversumma.c
OBJS_SUMMA = $(SRCS_SUMMA:.c=.o)
DISTFILES += $(HDRS_SUMMA) $(SRCS_SUMMA)

summa$(EXEEXT): $(OBJS_SUMMA) $(OBJS_COMMON)
	$(MPICC) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

#------------------------------------------------------------
HDRS_SUMMA_SOLN = mm1d.h summa.h
SRCS_SUMMA_SOLN = mm1d.c summa--soln.c driversumma.c
OBJS_SUMMA_SOLN = $(SRCS_SUMMA_SOLN:.c=.o)

summa--soln$(EXEEXT): $(OBJS_SUMMA_SOLN) $(OBJS_COMMON)
	$(MPICC) $(MPICOPTFLAGS) -o $@ $^ $(MPILDFLAGS)

#------------------------------------------------------------
%.o: %.c
	$(MPICC) $(MPICFLAGS) $(MPICOPTFLAGS) -o $@ -c $^

#------------------------------------------------------------
check-mpicc:
	@if ! test -x "$(MPICC)" ; then \
	  echo "*** No MPI compiler specified (via MPICC) ***" ; \
	  exit 1 ; \
	fi
	@echo "MPICC = $(MPICC)"

#------------------------------------------------------------
clean:
	rm -rf core *~ *.o $(TARGETS) $(CLEANFILES)
	rm -rf $(PROJID)/ $(PROJID).tar.gz

runclean:
	test -d archive || mkdir -p archive
	@-mv mm1d.o[0-9][0-9]* mm1d.e[0-9][0-9]* archive
	if test -f strong_scaling.txt ; then mv strong_scaling.txt archive ; fi

#------------------------------------------------------------
dist: $(PROJID).tar.gz
# $(PROJID)b.tar.gz

$(PROJID).tar.gz: $(DISTFILES)
	rm -rf $(PROJID)/
	mkdir -p $(PROJID)
	if test -n "$(DISTFILES)" ; then \
	  cp $(DISTFILES) $(PROJID) ; \
	fi
	tar cvf - $(PROJID)/ | gzip -9c > $@

#------------------------------------------------------------
# eof
