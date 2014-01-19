#pragma once

#include <hpcdefs.hpp>

typedef void (*convert_to_floating_point_function)(const uint8_t*, double*, size_t, size_t, size_t);
typedef void (*matrix_vector_multiplication_function)(double*, const double*, const double*, size_t, size_t);

void convert_to_floating_point_naive(const uint8_t* input_images, double* output_images, size_t image_width, size_t image_height, size_t image_count);
void matrix_vector_multiplication_naive(double* output_vector, const double* matrix, const double* input_vector, size_t matrix_width, size_t matrix_height);

extern "C" void convert_to_floating_point_optimized(const uint8_t* input_images, double* output_images, size_t image_width, size_t image_height, size_t image_count);
extern "C" void matrix_vector_multiplication_optimized(double* output_vector, const double* matrix, const double* input_vector, size_t matrix_width, size_t matrix_height);

void convert_to_floating_point_upper(const uint8_t* input_images, double* output_images, size_t image_width, size_t image_height, size_t image_count);
void convert_to_floating_point_lower(const uint8_t* input_images, double* output_images, size_t image_width, size_t image_height, size_t image_count);
void matrix_vector_multiplication_abs(double* output_vector, const double* matrix, const double* input_vector, size_t matrix_width, size_t matrix_height);

/* Utility matrix and image operations */
void square_matrix(double* output_matrix, const double* input_matrix, size_t matrix_width, size_t matrix_height);
void demean_images(double* images, size_t image_width, size_t image_height, size_t image_count);
void vector_matrix_multiplication(double* output_vector, const double* input_vector, const double* matrix, size_t matrix_width, size_t matrix_height);
void min_max_image(const double* image_data, size_t image_width, size_t image_height, double& image_min, double& image_max);
void convert_to_fixed_point(const double* input_image, uint8_t* output_image, size_t image_width, size_t image_height);
bool check_images(const double* images, const double* images_lower, const double* images_upper, size_t image_width, size_t image_height, size_t image_count);

/* Utility vector operations */
void vector_set(double *CSE6230_RESTRICT vector, size_t length, double constant);
void vector_scale(double *CSE6230_RESTRICT vector, size_t length, double factor);
double sum_squares(const double *CSE6230_RESTRICT vector, size_t length);
double dot_product(const double *CSE6230_RESTRICT vector_x, const double *CSE6230_RESTRICT vector_y, size_t length);
void normalize_vector(double *CSE6230_RESTRICT vector, size_t length);
bool check_vector(const double* vector, const double* vector_ref, const double* vector_abs, double eps_error, size_t length);

void read_raw_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height);
void write_bmp_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height);
