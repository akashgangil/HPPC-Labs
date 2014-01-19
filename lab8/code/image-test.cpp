#include <hpcdefs.hpp>
#include <image.hpp>
#include <timer.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <dlfcn.h>

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

void test_conversion(const char* method_name, const char* grayscale_image_path, const char* integral_error_image_path,
	convert_rgb_to_grayscale_function convert_rgb_to_grayscale, integrate_image_function integrate_image,
	void* rgb_image, void* grayscale_image, void* reference_grayscale_image, void* integral_image, void* reference_integral_image,
	size_t image_width, size_t image_height, size_t experiments_count, bool is_naive)
{
	memset(grayscale_image, 0, image_width * image_height * sizeof(uint8_t));
	memset(integral_image, 0, image_width * image_height * sizeof(uint32_t));

	timer conversion_timer;
	convert_rgb_to_grayscale(static_cast<const uint8_t*>(rgb_image), static_cast<uint8_t*>(grayscale_image), image_width, image_height);
	double min_conversion_ms = conversion_timer.get_ms();

		timer integration_timer;
	integrate_image(static_cast<const uint8_t*>(grayscale_image), static_cast<uint32_t*>(integral_image), image_width, image_height);
	double min_integration_ms = integration_timer.get_ms();

	bool conversion_test_passed = memcmp(grayscale_image, reference_grayscale_image, image_width * image_height * sizeof(uint8_t)) == 0;
	bool integration_test_passed = memcmp(integral_image, reference_integral_image, image_width * image_height * sizeof(uint32_t)) == 0;
	for (size_t experiment = 0; experiment < experiments_count; experiment++) {
		{
			timer conversion_timer;
			convert_rgb_to_grayscale(static_cast<const uint8_t*>(rgb_image), static_cast<uint8_t*>(grayscale_image), image_width, image_height);
			double conversion_ms = conversion_timer.get_ms();
			if (conversion_ms < min_conversion_ms)
				min_conversion_ms = conversion_ms;
		}
		{
			timer integration_timer;
			integrate_image(static_cast<const uint8_t*>(grayscale_image), static_cast<uint32_t*>(integral_image), image_width, image_height);
			double integration_ms = integration_timer.get_ms();
			if (integration_ms < min_integration_ms)
				min_integration_ms = integration_ms;
		}
	}
	printf("%s\n", method_name);
	printf("\tPerformance test:\n");
	printf("\t\tConversion:  %.2lf\n", min_conversion_ms);
	printf("\t\tIntegration: %.2lf\n", min_integration_ms);
	printf("\t\tTotal:       %.2lf\n", (min_conversion_ms + min_integration_ms));
	printf("\t\tFPS:         %.1lf\n", (1000.0 / (min_conversion_ms + min_integration_ms)));
	if (!is_naive) {
		printf("\tUnit test:\n");
		printf("\t\tConversion:  %s\n", (conversion_test_passed ?
			CSE6230_ESCAPE_GREEN_COLOR "PASSED" CSE6230_ESCAPE_NORMAL_COLOR:
			CSE6230_ESCAPE_RED_COLOR "FAILED" CSE6230_ESCAPE_NORMAL_COLOR));
		if (integration_test_passed) {
			printf("\t\tIntegration: %s\n", CSE6230_ESCAPE_GREEN_COLOR "PASSED" CSE6230_ESCAPE_NORMAL_COLOR);
		} else {
			printf("\t\tIntegration: %s (mask for erroneus pixels saved to %s)\n", CSE6230_ESCAPE_RED_COLOR "FAILED" CSE6230_ESCAPE_NORMAL_COLOR, integral_error_image_path);
		}
		if (!integration_test_passed) {
			write_integral_error_image(integral_error_image_path,
				static_cast<const uint32_t*>(integral_image),
				static_cast<const uint32_t*>(reference_integral_image),
				image_width, image_height);
		}
	}
	write_bmp_image(grayscale_image_path, grayscale_image, image_width, image_height);
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

	convert_rgb_to_grayscale_function convert_rgb_to_grayscale_optimized =
		reinterpret_cast<convert_rgb_to_grayscale_function>(dlsym(libsimdimage, "convert_rgb_to_grayscale_optimized"));
	if (convert_rgb_to_grayscale_optimized == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}
	integrate_image_function integrate_image_optimized =
		reinterpret_cast<integrate_image_function>(dlsym(libsimdimage, "integrate_image_optimized"));
	if (integrate_image_optimized == NULL) {
		fprintf(stderr, "Error: %s\n", dlerror());
		exit(EXIT_FAILURE);
	}

	const size_t image_width = 520;
	const size_t image_height = 390;
	const size_t rgb_image_size = image_width * image_height * 3; // 24 bits per pixel
	const size_t grayscale_image_size = image_width * image_height; // 8 bits per pixel
	const size_t integral_image_size = image_width * image_height * 4; // 32 bits per pixel

	void* rgb_image = allocate_aligned_memory(rgb_image_size, 64);
	void* grayscale_image = allocate_aligned_memory(grayscale_image_size, 64);
	void* reference_grayscale_image = allocate_aligned_memory(grayscale_image_size, 64);
	void* integral_image = allocate_aligned_memory(integral_image_size, 64);
	void* reference_integral_image = allocate_aligned_memory(integral_image_size, 64);

	read_raw_image("cat.rgb", rgb_image, image_width, image_height);
	convert_rgb_to_grayscale_naive(static_cast<const uint8_t*>(rgb_image), static_cast<uint8_t*>(reference_grayscale_image), image_width, image_height);
	integrate_image_naive(static_cast<const uint8_t*>(reference_grayscale_image), static_cast<uint32_t*>(reference_integral_image), image_width, image_height);
	write_bmp_image("cat.bmp", reference_grayscale_image, image_width, image_height);

	printf("%15s\t%10s\t%10s\n", "Version", "Time (ms)", "Unit test");

	test_conversion("Naive", "cat-grayscale-naive.bmp", NULL,
		&convert_rgb_to_grayscale_naive, &integrate_image_naive,
		rgb_image, grayscale_image, reference_grayscale_image,
		integral_image, reference_integral_image,
		image_width, image_height, experiments_count, true);

	test_conversion("Optimized", "cat-grayscale-optimized.bmp", "cat-integral-errors.bmp",
		convert_rgb_to_grayscale_optimized, integrate_image_optimized,
		rgb_image, grayscale_image, reference_grayscale_image,
		integral_image, reference_integral_image,
		image_width, image_height, experiments_count, false);

	release_aligned_memory(reference_integral_image);
	release_aligned_memory(integral_image);
	release_aligned_memory(reference_grayscale_image);
	release_aligned_memory(grayscale_image);
	release_aligned_memory(rgb_image);

	dlclose(libsimdimage);
}
