#include <hpcdefs.hpp>
#include <image.hpp>
#include <timer.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <dlfcn.h>

template <class T>
void swap(T& a, T& b) {
	T tmp = a;
	a = b;
	b = tmp;
}

void write_integral_error_image(const char* integral_error_image_path,
	const uint32_t* integral_image, const uint32_t* reference_integral_image,
	size_t image_width, size_t image_height)
{
	size_t integral_error_image_size = image_width * image_height;
	uint8_t* integral_error_image = static_cast<uint8_t*>(
		allocate_aligned_memory(integral_error_image_size, 64));
	for (size_t i = 0; i < image_height; i++) {
		for (size_t j = 0; j < image_width; j++) {
			const uint32_t integral_pixel = integral_image[i * image_width + j];
			const uint32_t reference_integral_pixel = reference_integral_image[i * image_width + j];
			integral_error_image[i * image_width + j] = (integral_pixel == reference_integral_pixel) ? 0xFF : 0x00;
		}
	}
	write_bmp_image(integral_error_image_path, integral_error_image, image_width, image_height);
	release_aligned_memory(integral_error_image);
}

void test_conversion(const char* method_name, convert_to_floating_point_function convert_to_floating_point,
	const uint8_t* fixed_point_images, double* floating_point_images, double* floating_point_images_upper, double* floating_point_images_lower,
	size_t image_width, size_t image_height, size_t image_count, size_t experiments_count, bool is_naive, double& fps)
{
	memset(floating_point_images, 0, image_width * image_height * image_count * sizeof(double));

	timer conversion_timer;
	convert_to_floating_point(fixed_point_images, floating_point_images, image_width, image_height, image_count);
	double min_conversion_ms = conversion_timer.get_ms();

	convert_to_floating_point_upper(fixed_point_images, floating_point_images_upper, image_width, image_height, image_count);
	convert_to_floating_point_lower(fixed_point_images, floating_point_images_lower, image_width, image_height, image_count);

	bool conversion_test_passed = check_images(floating_point_images, floating_point_images_lower, floating_point_images_upper, image_width, image_height, image_count);
	for (size_t experiment = 0; experiment < experiments_count; experiment++) {
		timer conversion_timer;
		convert_to_floating_point(fixed_point_images, floating_point_images, image_width, image_height, image_count);
		double conversion_ms = conversion_timer.get_ms();
		if (conversion_ms < min_conversion_ms)
			min_conversion_ms = conversion_ms;
	}
	fps = 1000.0 / min_conversion_ms;
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
	printf("\t\tPerformance test:  %.3lf ms (%.1lf FPS)\n", min_conversion_ms, (1000.0 / min_conversion_ms));
}

double* test_multiplication(const char* method_name, matrix_vector_multiplication_function matrix_vector_multiplication,
	double* vector_old, double* vector_new, double* vector_ref, double* vector_abs, const double* matrix, size_t length,
	size_t experiments_count, bool is_naive, double& fps)
{
	bool multiplication_test_passed = true;
	vector_set(vector_old, length, 1.0 / sqrt(double(length)));

	timer multiplication_timer;
	matrix_vector_multiplication(vector_new, matrix, vector_old, length, length);
	double min_multiplication_ms = multiplication_timer.get_ms();

	for (size_t experiment = 0; experiment < experiments_count; experiment++) {
		timer multiplication_timer;
		matrix_vector_multiplication(vector_new, matrix, vector_old, length, length);
		double multiplication_ms = multiplication_timer.get_ms();
		if (multiplication_ms < min_multiplication_ms)
			min_multiplication_ms = multiplication_ms;
	}

	if (!is_naive) {
		for (size_t iteration = 0; iteration < 10000; iteration++) {
			matrix_vector_multiplication(vector_new, matrix, vector_old, length, length);
			matrix_vector_multiplication_naive(vector_ref, matrix, vector_old, length, length);
			matrix_vector_multiplication_abs(vector_abs, matrix, vector_old, length, length);
			if (multiplication_test_passed) {
				multiplication_test_passed = check_vector(vector_new, vector_ref, vector_abs, sqrt(double(length)), length);
			}
			normalize_vector(vector_new, length);
			const double dp = dot_product(vector_new, vector_old, length);
			swap(vector_old, vector_new);
			if (iteration != 0)
				if (fabs(1.0 - dp) <= sqrt(DBL_EPSILON))
					break;
		}
	}

	fps = 1000.0 / min_multiplication_ms;
	printf("\t%s\n", method_name);
	if (!is_naive) {
		printf("\t\tUnit test:         %s\n", (multiplication_test_passed ?
			CSE6230_ESCAPE_GREEN_COLOR "PASSED" CSE6230_ESCAPE_NORMAL_COLOR:
			CSE6230_ESCAPE_RED_COLOR "FAILED" CSE6230_ESCAPE_NORMAL_COLOR));
	}
	printf("\t\tPerformance test:  %.3lf us (%.1lf FPS)\n", min_multiplication_ms * 1000.0, (1000.0 / min_multiplication_ms));
	return vector_old;
}

int main(int argc, char** argv) {
#if defined(DEBUG) || defined(_DEBUG)
	const size_t experiments_count = 3;
#else
	const size_t experiments_count = 50;
#endif
	
	void* libsimdimage = dlopen("./libsimdimage.so", RTLD_NOW | RTLD_LOCAL);
	if (libsimdimage == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	convert_to_floating_point_function convert_to_floating_point_optimized =
		reinterpret_cast<convert_to_floating_point_function>(dlsym(libsimdimage, "convert_to_floating_point_optimized"));
	if (convert_to_floating_point_optimized == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}
	matrix_vector_multiplication_function matrix_vector_multiplication_optimized =
		reinterpret_cast<matrix_vector_multiplication_function>(dlsym(libsimdimage, "matrix_vector_multiplication_optimized"));
	if (matrix_vector_multiplication_optimized == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	const size_t image_width = 120;
	const size_t image_height = 120;
	const size_t image_count = 199;

	const size_t image_pixels = image_width * image_height;
	const size_t image_collection_pixels = image_pixels * image_count;
	uint8_t* fixed_point_images = static_cast<uint8_t*>(allocate_aligned_memory(image_collection_pixels * sizeof(uint8_t), 64));

	double* floating_point_images = static_cast<double*>(allocate_aligned_memory(image_collection_pixels * sizeof(double), 64));
	double* floating_point_images_upper = static_cast<double*>(allocate_aligned_memory(image_collection_pixels * sizeof(double), 64));
	double* floating_point_images_lower = static_cast<double*>(allocate_aligned_memory(image_collection_pixels * sizeof(double), 64));
	double* squared_matrix = static_cast<double*>(allocate_aligned_memory(image_count * image_count * sizeof(double), 64));
	double* eigenvector_old = static_cast<double*>(allocate_aligned_memory(image_count * sizeof(double), 64));
	double* eigenvector_new = static_cast<double*>(allocate_aligned_memory(image_count * sizeof(double), 64));
	double* eigenvector_ref = static_cast<double*>(allocate_aligned_memory(image_count * sizeof(double), 64));
	double* eigenvector_abs = static_cast<double*>(allocate_aligned_memory(image_count * sizeof(double), 64));
	double* floating_point_eigencat = static_cast<double*>(allocate_aligned_memory(image_pixels * sizeof(double), 64));
	uint8_t* fixed_point_eigencat = static_cast<uint8_t*>(allocate_aligned_memory(image_pixels * sizeof(uint8_t), 64));

	for (size_t image_number = 1; image_number <= image_count; image_number++) {
		char image_name[64];
		sprintf(image_name, "cats/%04u.y", unsigned(image_number));
		read_raw_image(image_name, &fixed_point_images[image_pixels * (image_number - 1)], image_width, image_height);
	}

	printf("Conversion from fixed-point to floating-point\n");
	double naive_conversion_fps = 0.0, naive_multiplication_fps = 0.0, simd_conversion_fps = 0.0, simd_multiplication_fps = 0.0;
	test_conversion("Naive", convert_to_floating_point_naive, 
		fixed_point_images, floating_point_images, floating_point_images_upper, floating_point_images_lower,
		image_width, image_height, image_count, experiments_count, true, naive_conversion_fps);
	test_conversion("Optimized", convert_to_floating_point_optimized,
		fixed_point_images, floating_point_images, floating_point_images_upper, floating_point_images_lower,
		image_width, image_height, image_count, experiments_count, false, simd_conversion_fps);
	printf("\t\tPerformance boost: %.1lfx\n", simd_conversion_fps / naive_conversion_fps);

	printf("Matrix-vector multiplication:\n");
	{
		convert_to_floating_point_naive(fixed_point_images, floating_point_images, image_width, image_height, image_count);
		demean_images(floating_point_images, image_width, image_height, image_count);
		square_matrix(squared_matrix, floating_point_images, image_pixels, image_count);

		double* eigenvector = test_multiplication("Naive", matrix_vector_multiplication_naive,
			eigenvector_old, eigenvector_new, eigenvector_ref, eigenvector_abs, squared_matrix, image_count,
			experiments_count, true, naive_multiplication_fps);

		vector_matrix_multiplication(floating_point_eigencat, eigenvector, floating_point_images, image_pixels, image_count);
		normalize_vector(floating_point_eigencat, image_pixels);
		convert_to_fixed_point(floating_point_eigencat, fixed_point_eigencat, image_width, image_height);
		write_bmp_image("eigencat-naive.bmp", fixed_point_eigencat, image_width, image_height);
	}
	{
		convert_to_floating_point_optimized(fixed_point_images, floating_point_images, image_width, image_height, image_count);
		demean_images(floating_point_images, image_width, image_height, image_count);
		square_matrix(squared_matrix, floating_point_images, image_pixels, image_count);

		double* eigenvector = test_multiplication("Naive", matrix_vector_multiplication_optimized,
			eigenvector_old, eigenvector_new, eigenvector_ref, eigenvector_abs, squared_matrix, image_count,
			experiments_count, false, simd_multiplication_fps);

		vector_matrix_multiplication(floating_point_eigencat, eigenvector, floating_point_images, image_pixels, image_count);
		normalize_vector(floating_point_eigencat, image_pixels);
		convert_to_fixed_point(floating_point_eigencat, fixed_point_eigencat, image_width, image_height);
		write_bmp_image("eigencat-optimized.bmp", fixed_point_eigencat, image_width, image_height);
	}
	printf("\t\tPerformance boost: %.1lfx\n", simd_multiplication_fps / naive_multiplication_fps);

	printf("Total (geometric mean):\n");
	printf("\tNaive FPS:         %.1lf\n", sqrt(naive_conversion_fps * naive_multiplication_fps));
	printf("\tOptimized FPS:     %.1lf\n", sqrt(simd_conversion_fps * simd_multiplication_fps));
	printf("\tPerformance boost: %.1lfx\n", sqrt((simd_conversion_fps * simd_multiplication_fps) / (naive_conversion_fps * naive_multiplication_fps)));

	release_aligned_memory(fixed_point_images);

	release_aligned_memory(floating_point_images);
	release_aligned_memory(floating_point_images_upper);
	release_aligned_memory(floating_point_images_lower);
	release_aligned_memory(squared_matrix);
	release_aligned_memory(eigenvector_old);
	release_aligned_memory(eigenvector_new);
	release_aligned_memory(eigenvector_ref);
	release_aligned_memory(eigenvector_abs);
	release_aligned_memory(floating_point_eigencat);
	release_aligned_memory(fixed_point_eigencat);

	dlclose(libsimdimage);
}
