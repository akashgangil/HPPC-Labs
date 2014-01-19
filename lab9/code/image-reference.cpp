#include <hpcdefs.hpp>
#include <image.hpp>
#include <math.h>
#include <assert.h>
#include <fenv.h>
#include <float.h>

void convert_to_floating_point_naive(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {
	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] =
					double(fixed_point_images[(image_number * image_height + image_row) * image_width + image_column]) / 255.0;
			}
		}
	}
}

#pragma STDC FENV_ACCESS ON
void convert_to_floating_point_upper(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {
	const int original_rounding_mode = fegetround();
	fesetround(FE_UPWARD);
	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] =
					double(fixed_point_images[(image_number * image_height + image_row) * image_width + image_column]) / 255.0;
			}
		}
	}
	fesetround(original_rounding_mode);
}

void convert_to_floating_point_lower(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {
	const int original_rounding_mode = fegetround();
	fesetround(FE_DOWNWARD);
	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] =
					double(fixed_point_images[(image_number * image_height + image_row) * image_width + image_column]) / 255.0;
			}
		}
	}
	fesetround(original_rounding_mode);
}
#pragma STDC FENV_ACCESS OFF

void matrix_vector_multiplication_naive(double *CSE6230_RESTRICT output_vector, const double *CSE6230_RESTRICT matrix, const double *CSE6230_RESTRICT input_vector, size_t matrix_width, size_t matrix_height) {
	for (size_t i = 0; i < matrix_height; i++) {
		double accumulated_sum = 0.0;
		for (size_t j = 0; j < matrix_width; j++) {
			accumulated_sum += matrix[i * matrix_width + j] * input_vector[j];
		}
		output_vector[i] = accumulated_sum;
	}
}

void matrix_vector_multiplication_abs(double *CSE6230_RESTRICT output_vector, const double *CSE6230_RESTRICT matrix, const double *CSE6230_RESTRICT input_vector, size_t matrix_width, size_t matrix_height) {
	for (size_t i = 0; i < matrix_height; i++) {
		double accumulated_sum = 0.0;
		for (size_t j = 0; j < matrix_width; j++) {
			accumulated_sum += fabs(matrix[i * matrix_width + j]) * fabs(input_vector[j]);
		}
		output_vector[i] = accumulated_sum;
	}
}

bool check_images(const double *CSE6230_RESTRICT images, const double *CSE6230_RESTRICT images_lower, const double *CSE6230_RESTRICT images_upper, size_t image_width, size_t image_height, size_t image_count) {
	for (size_t image_number = 0; image_number < image_count; image_number++) {
		for (size_t image_row = 0; image_row < image_height; image_row++) {
			for (size_t image_column = 0; image_column < image_width; image_column++) {
				const double pixel = images[(image_number * image_height + image_row) * image_width + image_column];
				const double pixel_upper = images_upper[(image_number * image_height + image_row) * image_width + image_column];
				const double pixel_lower = images_lower[(image_number * image_height + image_row) * image_width + image_column];
				if ((pixel > pixel_upper) || (pixel < pixel_lower)) {
					return false;
				}
			}
		}
	}
	return true;
}

void square_matrix(double *CSE6230_RESTRICT output_matrix, const double *CSE6230_RESTRICT input_matrix, size_t matrix_width, size_t matrix_height) {
	for (size_t i = 0; i < matrix_height; i++) {
		for (size_t j = 0; j < matrix_height; j++) {
			double accumulated_sum = 0.0;
			for (size_t k = 0; k < matrix_width; k++) {
				accumulated_sum += input_matrix[i * matrix_width + k] * input_matrix[j * matrix_width + k];
			}
			output_matrix[i * matrix_height + j] = accumulated_sum;
		}
	}
}

void demean_images(double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {
	for (size_t image_row = 0; image_row < image_height; image_row++) {
		for (size_t image_column = 0; image_column < image_width; image_column++) {
			/* Sum pixel (image_row, image_column) of all images */
			double pixel_sum = 0.0;
			for (size_t image_number = 0; image_number < image_count; image_number++) {
				pixel_sum += floating_point_images[(image_number * image_height + image_row) * image_width + image_column];
			}
			const double pixel_mean = pixel_sum / double(image_count);
			/* Subtract the mean for pixel (image_row, image_column) on all images */
			for (size_t image_number = 0; image_number < image_count; image_number++) {
				floating_point_images[(image_number * image_height + image_row) * image_width + image_column] -= pixel_mean;
			}
		}
	}
}

void vector_matrix_multiplication(double *CSE6230_RESTRICT output_vector, const double *CSE6230_RESTRICT input_vector, const double *CSE6230_RESTRICT matrix, size_t matrix_width, size_t matrix_height) {
	for (size_t j = 0; j < matrix_width; j++) {
		double accumulated_sum = 0.0;
		for (size_t i = 0; i < matrix_height; i++) {
			accumulated_sum += matrix[i * matrix_width + j] * input_vector[i];
		}
		output_vector[j] = accumulated_sum;
	}
}

void min_max_image(const double *CSE6230_RESTRICT image_data, size_t image_width, size_t image_height, double& image_min, double& image_max) {
	double current_min = *image_data;
	double current_max = *image_data;
	for (size_t image_row = 0; image_row < image_height; image_row++) {
		for (size_t image_column = 0; image_column < image_width; image_column++) {
			current_min = fmin(current_min, image_data[image_row * image_width + image_column]);
			current_max = fmax(current_max, image_data[image_row * image_width + image_column]);
		}
	}
	image_min = current_min;
	image_max = current_max;
}

void convert_to_fixed_point(const double *CSE6230_RESTRICT input_image, uint8_t *CSE6230_RESTRICT output_image, size_t image_width, size_t image_height) {
	double image_min, image_max;
	min_max_image(input_image, image_width, image_height, image_min, image_max);
	const double image_range = image_max - image_min;
	const double image_scale = 255.0 / image_range;
	for (size_t image_row = 0; image_row < image_height; image_row++) {
		for (size_t image_column = 0; image_column < image_width; image_column++) {
			output_image[image_row * image_width + image_column] = static_cast<uint8_t>(
				(input_image[image_row * image_width + image_column] - image_min) * image_scale);
		}
	}
}

void vector_set(double *CSE6230_RESTRICT vector, size_t length, double constant) {
	for (size_t i = 0; i < length; i++) {
		vector[i] = constant;
	}
}

void vector_scale(double *CSE6230_RESTRICT vector, size_t length, double factor) {
	for (size_t i = 0; i < length; i++) {
		vector[i] *= factor;
	}
}

double sum_squares(const double *CSE6230_RESTRICT vector, size_t length) {
	double accumulated_sum_squares = 0.0;
	for (size_t i = 0; i < length; i++) {
		accumulated_sum_squares += vector[i] * vector[i];
	}
	return accumulated_sum_squares;
}

double dot_product(const double *CSE6230_RESTRICT vector_x, const double *CSE6230_RESTRICT vector_y, size_t length) {
	double accumulated_dot_product = 0.0;
	for (size_t i = 0; i < length; i++) {
		accumulated_dot_product += vector_x[i] * vector_y[i];
	}
	return accumulated_dot_product;
}

void normalize_vector(double *CSE6230_RESTRICT vector, size_t length) {
	const double scale_factor = 1.0 / sqrt(sum_squares(vector, length));
	vector_scale(vector, length, scale_factor);
}

bool check_vector(const double *CSE6230_RESTRICT vector, const double *CSE6230_RESTRICT vector_ref, const double *CSE6230_RESTRICT vector_abs, double eps_error, size_t length) {
	for (size_t i = 0; i < length; i++) {
		if (fabs(vector[i] - vector_ref[i]) > vector_abs[i] * DBL_EPSILON * eps_error) {
			return false;
		}
	}
	return true;
}

