/**
 *  \file util.c
 *  \brief Miscellaneous utilities
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

int
env_getInt (const char* var, int def_val)
{
  if (var) {
    char* val = getenv (var);
    if (val && strlen (val) > 0) {
      return atoi (val);
    }
  }
  return def_val;
}

int
env_isEnabled (const char* var, int def_val)
{
  if (var) {
    char* val = getenv (var);
    if (val && strlen (val) > 0) {
      if (strcasecmp (val, "yes") == 0 || strcasecmp (val, "y") == 0)
	return 1;
      if (strcasecmp (val, "no") == 0 || strcasecmp (val, "n") == 0)
	return 0;
    }
  }
  return def_val;
}

/* eof */
