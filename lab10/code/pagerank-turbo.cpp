#include <hpcdefs.hpp>
#include <pagerank.hpp>

void page_rank_iteration_optimized(double *CSE6230_RESTRICT probabilities_new, const double *CSE6230_RESTRICT probabilities_old,
	const double *CSE6230_RESTRICT matrix, const int32_t*CSE6230_RESTRICT  columns, const int32_t*CSE6230_RESTRICT rows,
	const int32_t *CSE6230_RESTRICT link_free_pages, int32_t pages_count, int32_t link_free_pages_count)
{
		double transition_probability = 0.0;
		/* First process transitions from link-free pages */
		for (int32_t link_free_page_index = 0; link_free_page_index < link_free_pages_count; link_free_page_index++) {
			const int32_t column_index = link_free_pages[link_free_page_index];
			transition_probability += probabilities_old[column_index];
		}
		const double const_transition_probability = transition_probability / double(pages_count);

	for (int32_t page = 0; page < pages_count; page++) {
    transition_probability = const_transition_probability;
		/* Not process transitions form pages with links */
		for (int32_t index = rows[page], row_end = rows[page + 1]; index != row_end; index++) {
			const int32_t column_index = columns[index];
			transition_probability += matrix[index] * probabilities_old[column_index];
		}
		probabilities_new[page] = transition_probability;
	}
}
