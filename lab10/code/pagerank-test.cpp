#include <hpcdefs.hpp>
#include <pagerank.hpp>
#include <timer.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <assert.h>
#include <dlfcn.h>

template<class T>
void swap(T& a, T& b) {
	T tmp = a;
	a = b;
	b = tmp;
}

void test_page_rank(const char* method_name, page_rank_iteration_function page_rank_iteration,
	double* probabilities_new, double* probabilities_old, double* probabilities_ref,
	const double* matrix, const int32_t* columns, const int32_t* rows, const int32_t* link_free_pages,
	int32_t pages_count, int32_t link_free_pages_count,
	size_t experiments_count, bool is_naive)
{
	vector_set(probabilities_old, pages_count, 1.0 / double(pages_count));
	vector_set(probabilities_new, pages_count, 0.0);

	timer page_rank_timer;
	page_rank_iteration(probabilities_new, probabilities_old, matrix, columns, rows,
		link_free_pages, pages_count, link_free_pages_count);
	double min_page_rank_ms = page_rank_timer.get_ms();

	page_rank_iteration_naive(probabilities_ref, probabilities_old, matrix, columns, rows,
		link_free_pages, pages_count, link_free_pages_count);

	bool conversion_test_passed = check_vector(probabilities_new, probabilities_ref, sqrt(double(pages_count)), pages_count);
	swap(probabilities_old, probabilities_new);

	page_rank_iteration(probabilities_new, probabilities_old, matrix, columns, rows,
		link_free_pages, pages_count, link_free_pages_count);
	page_rank_iteration_naive(probabilities_ref, probabilities_old, matrix, columns, rows,
		link_free_pages, pages_count, link_free_pages_count);
	conversion_test_passed &= check_vector(probabilities_new, probabilities_ref, sqrt(double(pages_count)), pages_count);

	vector_set(probabilities_old, pages_count, 1.0 / double(pages_count));
	vector_set(probabilities_new, pages_count, 0.0);
	for (size_t experiment = 0; experiment < experiments_count; experiment++) {
		timer page_rank_timer;
		page_rank_iteration(probabilities_new, probabilities_old, matrix, columns, rows,
			link_free_pages, pages_count, link_free_pages_count);
		const double page_rank_ms = page_rank_timer.get_ms();
		if (page_rank_ms < min_page_rank_ms)
			min_page_rank_ms = page_rank_ms;
	}
	printf("\t%s\n", method_name);
	if (is_naive) {
		if (!conversion_test_passed) {
			printf("\t\tUnit test:         " CSE6230_ESCAPE_RED_COLOR "FAILED" CSE6230_ESCAPE_NORMAL_COLOR "\n");
		}
	} else {
		printf("\t\tUnit test:         %s\n", (conversion_test_passed ?
			CSE6230_ESCAPE_GREEN_COLOR "PASSED" CSE6230_ESCAPE_NORMAL_COLOR:
			CSE6230_ESCAPE_RED_COLOR "FAILED" CSE6230_ESCAPE_NORMAL_COLOR));
	}
	printf("\t\tPerformance test:  %.3lf ms (%.1lf IPS)\n", min_page_rank_ms, (1000.0 / min_page_rank_ms));
}

int main(int argc, char** argv) {
#if defined(DEBUG) || defined(_DEBUG)
	const size_t experiments_count = 3;
#else
	const size_t experiments_count = 50;
#endif
	
	void* libprturbo = dlopen("./libprturbo.so", RTLD_NOW | RTLD_LOCAL);
	if (libprturbo == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	page_rank_iteration_function page_rank_iteration_optimized =
		reinterpret_cast<page_rank_iteration_function>(dlsym(libprturbo, "page_rank_iteration_optimized"));
	if (page_rank_iteration_optimized == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	uint32_t links_count = 0, pages_count = 0;
	FILE* matrix_file = fopen("web-16384.mat", "r");
	assert(matrix_file != NULL);
	ssize_t elements_read;
	elements_read = fread(&links_count, sizeof(uint32_t), 1, matrix_file);
	assert(elements_read == 1);
	elements_read = fread(&pages_count, sizeof(uint32_t), 1, matrix_file);
	assert(elements_read == 1);

	int32_t* columns = (int32_t*)allocate_aligned_memory(links_count * sizeof(int32_t), 64);
	int32_t* rows = (int32_t*)allocate_aligned_memory((pages_count + 1) * sizeof(int32_t), 64);
	double* matrix = (double*)allocate_aligned_memory(links_count * sizeof(double), 64);
	vector_set(matrix, links_count, 1.0);

	double* probabilities_old = (double*)allocate_aligned_memory(pages_count * sizeof(double), 64);
	double* probabilities_new = (double*)allocate_aligned_memory(pages_count * sizeof(double), 64);
	double* probabilities_ref = (double*)allocate_aligned_memory(pages_count * sizeof(double), 64);
	vector_set(probabilities_old, pages_count, 1.0 / double(pages_count));

	int32_t* page_links_count = (int32_t*)allocate_aligned_memory(pages_count * sizeof(int32_t), 64);
	memset(page_links_count, 0, pages_count * sizeof(int32_t));

	elements_read = fread(columns, sizeof(int32_t), links_count, matrix_file);
	assert(elements_read == links_count);
	elements_read = fread(rows, sizeof(int32_t), (pages_count + 1), matrix_file);
	assert(elements_read == (pages_count + 1));

	fclose(matrix_file);

	/* Convert 1-based indices of CSR format to 0-base indices */
	for (size_t link = 0; link < links_count; link++) {
		columns[link] -= 1;
	}
	for (size_t page = 0; page <= pages_count; page++) {
		rows[page] -= 1;
	}

	for (size_t link = 0; link < links_count; link++) {
		const int32_t column_index = columns[link];
		page_links_count[column_index]++;
	}

	/* Count the number of pages which have no outgoing links */
	int32_t link_free_pages_count = 0;
	for (size_t page = 0; page < pages_count; page++) {
		if (page_links_count[page] == 0) {
			link_free_pages_count++;
		}
	}

	/* Indices of pages (matrix columns) which have no outgoing links */
	int32_t* link_free_pages = (int32_t*)allocate_aligned_memory(link_free_pages_count * sizeof(int32_t), 64);
	{
		size_t link_free_page_index = 0;
		for (size_t page = 0; page < pages_count; page++) {
			if (page_links_count[page] == 0) {
				link_free_pages[link_free_page_index++] = page;
			}
		}
	}

	/* Normalize transition probabilities in the matrix */
	for (size_t link = 0; link < links_count; link++) {
		int32_t column_index = columns[link];
		matrix[link] /= double(page_links_count[column_index]);
	}
	
	printf("Page rank iteration on %u webpages\n", pages_count);
	test_page_rank("Naive", page_rank_iteration_naive,
		probabilities_new, probabilities_old, probabilities_ref,
		matrix, columns, rows, link_free_pages,
		pages_count, link_free_pages_count,
		experiments_count, true);
	test_page_rank("Optimized", page_rank_iteration_optimized,
		probabilities_new, probabilities_old, probabilities_ref,
		matrix, columns, rows, link_free_pages,
		pages_count, link_free_pages_count,
		experiments_count, false);

	{
		printf("Computing page rank probabilities with naive implementation");
		fflush(stdout);
		vector_set(probabilities_old, pages_count, 1.0 / double(pages_count));
		for (size_t iteration = 0; iteration < 100; iteration++) {
			page_rank_iteration_naive(probabilities_new, probabilities_old, matrix, columns, rows, link_free_pages,
				pages_count, link_free_pages_count);
			if (iteration % 20 == 0) {
				printf(".");
				fflush(stdout);
			}
			swap(probabilities_old, probabilities_new);
			if (vector_max_abs_diff(probabilities_old, probabilities_new, pages_count) < 1.0e-9) {
				printf("Iteration stopped\n");
				break;
			}
		}
		printf("done\n");

		printf("Writing results to probabilities-naive.log...");
		fflush(stdout);
		FILE* naive_probabilities_file = fopen("probabilities-naive.log", "w");
		assert(naive_probabilities_file != NULL);
		for (int32_t page = 0; page < pages_count; page++) {
			fprintf(naive_probabilities_file, "%.18lg\n", probabilities_old[page]);
		}
		printf("done\n");
		fclose(naive_probabilities_file);
	}
	{
		printf("Computing page rank probabilities with optimized implementation");
		fflush(stdout);
		vector_set(probabilities_old, pages_count, 1.0 / double(pages_count));
		for (size_t iteration = 0; iteration < 100; iteration++) {
			page_rank_iteration_optimized(probabilities_new, probabilities_old, matrix, columns, rows, link_free_pages,
				pages_count, link_free_pages_count);
			if (iteration % 20 == 0) {
				printf(".");
				fflush(stdout);
			}
			swap(probabilities_old, probabilities_new);
			if (vector_max_abs_diff(probabilities_old, probabilities_new, pages_count) < 1.0e-9) {
				printf("Iteration stopped\n");
				break;
			}
		}
		printf("done\n");

		printf("Writing results to probabilities-optimized.log...");
		fflush(stdout);
		FILE* optimized_probabilities_file = fopen("probabilities-optimized.log", "w");
		assert(optimized_probabilities_file != NULL);
		for (int32_t page = 0; page < pages_count; page++) {
			fprintf(optimized_probabilities_file, "%.18lg\n", probabilities_old[page]);
		}
		printf("done\n");
		fclose(optimized_probabilities_file);
	}

	release_aligned_memory(link_free_pages);
	release_aligned_memory(page_links_count);
	release_aligned_memory(probabilities_old);
	release_aligned_memory(probabilities_new);
	release_aligned_memory(probabilities_ref);
	release_aligned_memory(matrix);
	release_aligned_memory(rows);
	release_aligned_memory(columns);

	dlclose(libprturbo);
}
