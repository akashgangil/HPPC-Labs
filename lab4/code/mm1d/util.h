/**
 *  \file util.h
 *  \brief Miscellaneous utilities
 */

#if !defined (INC_UTIL_H)
#define INC_UTIL_H

/** Returns the minimum of two integer values. */
inline int min_int (int a, int b) { return (a < b) ? a : b; }

/** Returns the maximum of two integer values. */
inline int max_int (int a, int b) { return (a < b) ? b : a; }

/** Swaps two generic pointers */
inline void swapPointers_double (double** pa, double** pb) {
  double* c = *pa;
  *pa = *pb;
  *pb = c;
}

/** Returns a non-zero value if the given environment variable is
 *  "yes" or "y"; the value 0 if the given environment variable is set to
 *  "no" or "n"; or the given default value otherwise.
 */
int env_isEnabled (const char* var, int def_val);

/** Returns the value of the integer environment variable, or def_val
 *  if the value is unavailable.
 */
int env_getInt (const char* var, int def_val);

#endif

/* eof */
