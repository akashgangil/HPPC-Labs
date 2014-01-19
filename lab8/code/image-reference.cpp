#include <hpcdefs.hpp>
#include <image.hpp>

void convert_rgb_to_grayscale_naive(const uint8_t *CSE6230_RESTRICT rgb_image, uint8_t *CSE6230_RESTRICT grayscale_image, size_t width, size_t height) {
	for (size_t i = 0; i < height; i++) {
		for (size_t j = 0; j < width; j++) {
			const uint16_t red   = rgb_image[(i * width + j) * 3 + 0];
			const uint16_t green = rgb_image[(i * width + j) * 3 + 1];
			const uint16_t blue  = rgb_image[(i * width + j) * 3 + 2];
			const uint8_t  luma  = (red * 54 + green * 183 + blue * 19) >> 8;
			grayscale_image[i * width + j] = luma;
		}
	}
}

void integrate_image_naive(const uint8_t *CSE6230_RESTRICT source_image, uint32_t *CSE6230_RESTRICT integral_image, size_t width, size_t height) {
	for (size_t i = 0; i < height; i++) {
		uint32_t integral = 0;
		for (size_t j = 0; j < width; j++) {
			integral += source_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}
	for (size_t j = 0; j < width; j++) {
		uint32_t integral = 0;
		for (size_t i = 0; i < height; i++) {
			integral += integral_image[i * width + j];
			integral_image[i * width + j] = integral;
		}
	}
}
