#include <image.hpp>

#include <stdio.h>
#include <stdlib.h>

#pragma pack(push, 1)

struct bitmap_file_header {
	uint16_t magic;
	uint32_t file_size;
	uint16_t reserved[2];
	uint32_t data_offset;
};

struct bitmap_core_header {
	uint32_t structure_size;
	uint16_t image_width;
	uint16_t image_height;
	uint16_t image_planes;
	uint16_t image_bpp; // Bits per pixel
};

struct r8g8b8 {
	uint8_t red;
	uint8_t green;
	uint8_t blue;
};

#pragma pack(pop)

void read_raw_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height) {
	const size_t expected_file_size = image_width * image_height; // Grayscale, 8 bits per pixel

	FILE* image_file = fopen(image_file_path, "r");
	if (image_file != 0) {
		const size_t image_bytes_read = fread(image_buffer, 1, expected_file_size, image_file);
		if (image_bytes_read == expected_file_size) {
			fclose(image_file);
		} else {
			fprintf(stderr, "Could only read %u out of %u expected bytes from image %s\n",
				unsigned(image_bytes_read), unsigned(expected_file_size), image_file_path);
			exit(-1);
		}
	} else {
		fprintf(stderr, "Failed to open the input file %s\n", image_file_path);
		exit(-1);
	}
}

void write_bmp_image(const char* image_file_path, void* image_buffer, size_t image_width, size_t image_height) {
	const size_t image_size = image_width * image_height;
	
	FILE* image_file = fopen(image_file_path, "w");
	if (image_file != 0) {
		bitmap_file_header file_header;
		file_header.magic = 0x4D42; // 'BM'
		file_header.file_size = sizeof(bitmap_file_header) + sizeof(bitmap_core_header) + 256 * 3 + image_size;
		file_header.reserved[0] = 0;
		file_header.reserved[1] = 0;
		file_header.data_offset = sizeof(bitmap_file_header) + sizeof(bitmap_core_header) + 256 * 3;
		size_t image_bytes_written = fwrite(&file_header, 1, sizeof(bitmap_file_header), image_file);
		if (image_bytes_written == sizeof(bitmap_file_header)) {
			bitmap_core_header bitmap_header;
			bitmap_header.structure_size = sizeof(bitmap_core_header);
			bitmap_header.image_width = image_width;
			bitmap_header.image_height = image_height;
			bitmap_header.image_planes = 1;
			bitmap_header.image_bpp = 8;
			image_bytes_written = fwrite(&bitmap_header, 1, sizeof(bitmap_core_header), image_file);
			if (image_bytes_written == sizeof(bitmap_core_header)) {
				r8g8b8 palette[256];
				for (size_t index = 0; index < 256; index++) {
					palette[index].red = index;
					palette[index].green = index;
					palette[index].blue = index;
				}
				image_bytes_written = fwrite(palette, 1, sizeof(palette), image_file);
				if (image_bytes_written != image_size) {
					const uint8_t* image_data = static_cast<const uint8_t*>(image_buffer);
					for (size_t row = image_height; row != 0; row--) {
						image_bytes_written = fwrite(image_data + (row - 1) * image_width, 1, image_width, image_file);
						if (image_bytes_written != image_width) {
							fprintf(stderr, "Could only write %u out of %u expected bytes to image %s\n",
								unsigned(image_bytes_written), unsigned(image_size), image_file_path);
							break;
						}
					}
				} else {
					fprintf(stderr, "Could only write %u out of %u expected bytes to image %s\n",
						unsigned(image_bytes_written), unsigned(sizeof(palette)), image_file_path);
				}
			} else {
				fprintf(stderr, "Could only write %u out of %u expected bytes to image %s\n",
					unsigned(image_bytes_written), unsigned(sizeof(bitmap_core_header)), image_file_path);
			}						
		} else {
			fprintf(stderr, "Could only write %u out of %u expected bytes to image %s\n",
				unsigned(image_bytes_written), unsigned(sizeof(bitmap_file_header)), image_file_path);
		}
		
		fclose(image_file);
	} else {
		fprintf(stderr, "Failed to open the output file %s\n", image_file_path);
	}
}
