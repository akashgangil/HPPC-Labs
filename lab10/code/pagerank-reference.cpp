#include <hpcdefs.hpp>
#include <pagerank.hpp>
#include <math.h>
#include <assert.h>
#include <float.h>

void page_rank_iteration_naive(double *CSE6230_RESTRICT probabilities_new, const double *CSE6230_RESTRICT probabilities_old,
	const double *CSE6230_RESTRICT matrix, const int32_t*CSE6230_RESTRICT  columns, const int32_t*CSE6230_RESTRICT rows,
	const int32_t *CSE6230_RESTRICT link_free_pages, int32_t pages_count, int32_t link_free_pages_count)
{
	for (int32_t page = 0; page < pages_count; page++) {
		double transition_probability = 0.0;

		/* First process transitions from link-free pages */
		for (int32_t link_free_page_index = 0; link_free_page_index < link_free_pages_count; link_free_page_index++) {
			const int32_t column_index = link_free_pages[link_free_page_index];
			transition_probability += probabilities_old[column_index];
		}
		transition_probability /= double(pages_count);

		/* Not process transitions form pages with links */
		for (int32_t index = rows[page], row_end = rows[page + 1]; index != row_end; index++) {
			const int32_t column_index = columns[index];
			transition_probability += matrix[index] * probabilities_old[column_index];
		}
		probabilities_new[page] = transition_probability;
	}
}

void vector_set(double *CSE6230_RESTRICT vector, size_t length, double constant) {
	for (size_t i = 0; i < length; i++) {
		vector[i] = constant;
	}
}

double vector_sum(const double *CSE6230_RESTRICT vector, size_t length) {
	double sum = 0.0;
	for (size_t i = 0; i < length; i++) {
		sum += vector[i];
	}
	return sum;
}

double vector_max_abs_diff(const double *CSE6230_RESTRICT vector_x, const double *CSE6230_RESTRICT vector_y, size_t length) {
	double max_abs_diff = 0.0;
	for (size_t i = 0; i < length; i++) {
		max_abs_diff = fmax(max_abs_diff, fabs(vector_x[i] - vector_y[i]));
	}
	return max_abs_diff;
}

bool check_vector(const double *CSE6230_RESTRICT vector, const double *CSE6230_RESTRICT vector_ref, double eps_error, size_t length) {
	for (size_t i = 0; i < length; i++) {
		if (fabs(vector[i] - vector_ref[i]) > fabs(vector_ref[i]) * DBL_EPSILON * eps_error) {
			return false;
		}
	}
	return true;
}

