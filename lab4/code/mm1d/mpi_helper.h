/**
 *  \file mpi_helper.h
 *  \brief Miscellaneous MPI helper utilities
 */

#if !defined (INC_MPI_HELPER_H)
#define INC_MPI_HELPER_H

#include <stdio.h>
#include <stdarg.h>
#include <mpi.h>

/** Shortcut C-style wrapper around MPI_Comm_rank (). */
inline int mpih_getRank (MPI_Comm comm) {
  int rank;
  MPI_Comm_rank (comm, &rank);
  return rank;
}

/** Shortcut C-style wrapper around MPI_Comm_size (). */
inline int mpih_getSize (MPI_Comm comm) {
  int size;
  MPI_Comm_size (comm, &size);
  return size;
}

/** Aborts the program if the given condition is false. */
#define mpih_assert(cond)  mpih_assert__ (__FILE__, __LINE__, #cond, cond)

/** Use wrapper 'mpih_assert' instead. */
void mpih_assert__ (const char* file, size_t line, const char* cond_str, int cond);

/** \name printf-style wrappers that prepend MPI info */
/*@{*/
void mpih_printf (MPI_Comm comm, const char* fmt, ...);
void mpih_fprintf (MPI_Comm comm, FILE* fp, const char* fmt, ...);
void mpih_vfprintf (MPI_Comm comm, FILE* fp, const char* fmt, va_list args);
/*@}*/

/** \brief Prints only if verbose messaging is enabled.
 *
 *  Enable (or disable) verbose messaging by setting the environment
 *  variable, 'VERBOSE', to yes or some positive integer (or no or
 *  non-positive integer).
 */
#define mpih_debugmsg(comm, fmt, args...) mpih_debugmsg__ (__FILE__, __LINE__, (comm), (fmt), ##args)

/** \brief Prints only if verbose messaging is enabled.
 *
 *  \note Use wrapper, 'mpih_debugmsg', instead.
 */
void mpih_debugmsg__ (const char* file, size_t line, MPI_Comm comm, const char* fmt, ...);


#endif

/* eof */
