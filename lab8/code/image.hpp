#pragma once

#include <hpcdefs.hpp>

typedef void (*convert_rgb_to_grayscale_function)(const uint8_t*, uint8_t*, size_t, size_t);
typedef void (*integrate_image_function)(const uint8_t*, uint32_t*, size_t, size_t);

extern "C" void convert_rgb_to_grayscale_naive(const uint8_t* rgb_image, uint8_t* grayscale_image, size_t image_width, size_t image_height);
extern "C" void integrate_image_naive(const uint8_t* grayscale_image, uint32_t* integral_image, size_t image_width, size_t image_height);

extern "C" void convert_rgb_to_grayscale_optimized(const uint8_t* rgb_image, uint8_t* grayscale_image, size_t image_width, size_t image_height);
extern "C" void integrate_image_optimized(const uint8_t* grayscale_image, uint32_t* integral_image, size_t image_width, size_t image_height);

void read_raw_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height);
void write_bmp_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height);
