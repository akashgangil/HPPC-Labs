/**
 *  \file mpi_helper.c
 *  \brief Miscellaneous MPI helper utilities
 */

#include <stdarg.h>
#include <mpi.h>
#include "mpi_helper.h"
#include "util.h"

void
mpih_assert__ (const char* file, size_t line, const char* cond_str, int cond)
{
  int rank, np;
  if (cond) return;
  mpih_fprintf (MPI_COMM_WORLD, stderr,
	       "[%s:%lu] *** Assertion, '%s', failed ***\n",
	       file, (unsigned long)line, cond_str ? cond_str : "(unknown condition)");
  MPI_Abort (MPI_COMM_WORLD, cond);
}

void
mpih_printf (MPI_Comm comm, const char* fmt, ...)
{
  va_list args;
  va_start (args, fmt);
  mpih_vfprintf (comm, stdout, fmt, args);
  va_end (args);
}

void
mpih_fprintf (MPI_Comm comm, FILE* fp, const char* fmt, ...)
{
  va_list args;
  va_start (args, fmt);
  mpih_vfprintf (comm, fp, fmt, args);
  va_end (args);
}

void
mpih_debugmsg__ (const char* file, size_t line, MPI_Comm comm, const char* fmt, ...)
{
  if (env_isEnabled ("VERBOSE", 1) || env_getInt ("VERBOSE", 1)) {
    va_list args;
    va_start (args, fmt);
    mpih_fprintf (comm, stderr, "<%s:%d> ", file, line);
    vfprintf (stderr, fmt, args);
    va_end (args);
  }
}

void
mpih_vfprintf (MPI_Comm comm, FILE* fp, const char* fmt, va_list args)
{
  int r = mpih_getRank (comm); /* rank in 'comm' */
  int P = mpih_getSize (comm); /* no. of processes in 'comm' */

  /* Get hostname of node */
  char hostname[MPI_MAX_PROCESSOR_NAME+1];
  int namelen = 0;
  MPI_Get_processor_name (hostname, &namelen);

  fprintf (fp, "[p%d/%d:%s] ", r, P, hostname);
  vfprintf (fp, fmt, args);
  fflush (fp);
}

/* eof */
