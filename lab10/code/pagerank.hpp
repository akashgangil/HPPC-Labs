#pragma once

#include <hpcdefs.hpp>

typedef void (*page_rank_iteration_function)(double *CSE6230_RESTRICT probabilities_new, const double *CSE6230_RESTRICT probabilities_old,
	const double *CSE6230_RESTRICT matrix, const int32_t*CSE6230_RESTRICT columns, const int32_t*CSE6230_RESTRICT rows,
	const int32_t *CSE6230_RESTRICT link_free_pages, int32_t pages_count, int32_t link_free_pages_count);

void page_rank_iteration_naive(double *CSE6230_RESTRICT probabilities_new, const double *CSE6230_RESTRICT probabilities_old,
	const double *CSE6230_RESTRICT matrix, const int32_t*CSE6230_RESTRICT columns, const int32_t*CSE6230_RESTRICT rows,
	const int32_t *CSE6230_RESTRICT link_free_pages, int32_t pages_count, int32_t link_free_pages_count);

extern "C" void page_rank_iteration_optimized(double *CSE6230_RESTRICT probabilities_new, const double *CSE6230_RESTRICT probabilities_old,
	const double *CSE6230_RESTRICT matrix, const int32_t*CSE6230_RESTRICT columns, const int32_t*CSE6230_RESTRICT rows,
	const int32_t *CSE6230_RESTRICT link_free_pages, int32_t pages_count, int32_t link_free_pages_count);

/* Utility vector operations */
void vector_set(double *CSE6230_RESTRICT vector, size_t length, double constant);
double vector_sum(const double *CSE6230_RESTRICT vector, size_t length);
double vector_max_abs_diff(const double *CSE6230_RESTRICT vector_x, const double *CSE6230_RESTRICT vector_y, size_t length);
bool check_vector(const double* vector, const double* vector_ref, double eps_error, size_t length);
