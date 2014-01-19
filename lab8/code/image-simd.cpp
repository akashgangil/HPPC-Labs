#include <hpcdefs.hpp>
#include <image.hpp>

void convert_rgb_to_grayscale_optimized(const uint8_t *CSE6230_RESTRICT rgb_image, uint8_t *CSE6230_RESTRICT grayscale_image, size_t width, size_t height) {

	const __m128i rconst = _mm_set1_epi16(54);
	const __m128i gconst = _mm_set1_epi16(183);
	const __m128i bconst = _mm_set1_epi16(19);

	__m128i gray_mask1 = _mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,14,12,10,8,6,4,2,0);
	__m128i gray_mask2 = _mm_set_epi8(14,12,10,8,6,4,2,0,-1,-1,-1,-1,-1,-1,-1,-1);
	__m128i mask1 = _mm_set_epi8(-1,7,-1,6,-1,5,-1,4,-1,3,-1,2,-1,1,-1,0);
	__m128i mask2 = _mm_set_epi8(-1,15,-1,14,-1,13,-1,12,-1,11,-1,10,-1,9,-1,8);	

	const __m128i zero = _mm_setzero_si128();
	size_t ptr=0;

	while(ptr < (width*height-32))
	{
		/*Loading 16 8-bit ints to perform deinterleaving*/
		__m128i base = _mm_loadu_si128((__m128i*)(rgb_image));
		__m128i base2 = _mm_loadu_si128((__m128i*)(rgb_image+16));
		__m128i base3 = _mm_loadu_si128((__m128i*)(rgb_image+32));
		__m128i base4 = _mm_loadu_si128((__m128i*)(rgb_image+48));
		__m128i base5 = _mm_loadu_si128((__m128i*)(rgb_image+64));
		__m128i base6 = _mm_loadu_si128((__m128i*)(rgb_image+80));

		/*Deinterleaving starts here*/
		__m128i lvl11 = _mm_unpacklo_epi8(base, base4);
		__m128i lvl12 = _mm_unpackhi_epi8(base, base4);
		__m128i lvl13 = _mm_unpacklo_epi8(base2, base5);
		__m128i lvl14 = _mm_unpackhi_epi8(base2, base5);
		__m128i lvl15 = _mm_unpacklo_epi8(base3, base6);
		__m128i lvl16 = _mm_unpackhi_epi8(base3, base6);

		__m128i lvl21 = _mm_unpacklo_epi8(lvl11, lvl14);
		__m128i lvl22 = _mm_unpackhi_epi8(lvl11, lvl14);
		__m128i lvl23 = _mm_unpacklo_epi8(lvl12, lvl15);
		__m128i lvl24 = _mm_unpackhi_epi8(lvl12, lvl15);
		__m128i lvl25 = _mm_unpacklo_epi8(lvl13, lvl16);
		__m128i lvl26 = _mm_unpackhi_epi8(lvl13, lvl16);

		__m128i lvl31 = _mm_unpacklo_epi8(lvl21, lvl24);
		__m128i lvl32 = _mm_unpackhi_epi8(lvl21, lvl24);
		__m128i lvl33 = _mm_unpacklo_epi8(lvl22, lvl25);
		__m128i lvl34 = _mm_unpackhi_epi8(lvl22, lvl25);
		__m128i lvl35 = _mm_unpacklo_epi8(lvl23, lvl26);
		__m128i lvl36 = _mm_unpackhi_epi8(lvl23, lvl26);

		__m128i lvl41 = _mm_unpacklo_epi8(lvl31, lvl34);
		__m128i lvl42 = _mm_unpackhi_epi8(lvl31, lvl34);
		__m128i lvl43 = _mm_unpacklo_epi8(lvl32, lvl35);
		__m128i lvl44 = _mm_unpackhi_epi8(lvl32, lvl35);
		__m128i lvl45 = _mm_unpacklo_epi8(lvl33, lvl36);
		__m128i lvl46 = _mm_unpackhi_epi8(lvl33, lvl36);

		__m128i lvl51 = _mm_unpacklo_epi8(lvl41, lvl44);
		__m128i lvl52 = _mm_unpackhi_epi8(lvl41, lvl44);
		__m128i lvl53 = _mm_unpacklo_epi8(lvl42, lvl45);
		__m128i lvl54 = _mm_unpackhi_epi8(lvl42, lvl45);
		__m128i lvl55 = _mm_unpacklo_epi8(lvl43, lvl46);
		__m128i lvl56 = _mm_unpackhi_epi8(lvl43, lvl46);

		/*Deinterleaving ends - lvl51 and lvl52 contain red, 3 and 4 contain green, 5 and 6 blue*/

		/*converting 8 bit unsigned ints to 16bit unsigned ints - using unpack for red and green; shuffle for blue*/
		__m128i red1 = _mm_unpacklo_epi8(lvl51,lvl51);
		__m128i red2 = _mm_unpackhi_epi8(lvl51,lvl51);
		__m128i red3 = _mm_unpacklo_epi8(lvl52,lvl52);
		__m128i red4 = _mm_unpackhi_epi8(lvl52,lvl52);
		__m128i green1 = _mm_unpacklo_epi8(lvl53,lvl53);
		__m128i green2 = _mm_unpackhi_epi8(lvl53,lvl53);
		__m128i green3 = _mm_unpacklo_epi8(lvl54,lvl54);
		__m128i green4 = _mm_unpackhi_epi8(lvl54,lvl54);
		red1 = _mm_srli_epi16(red1,8);
		red2 = _mm_srli_epi16(red2,8);
		red3 = _mm_srli_epi16(red3,8);
		red4 = _mm_srli_epi16(red4,8);
		green1 = _mm_srli_epi16(green1,8);
		green2 = _mm_srli_epi16(green2,8);
		green3 = _mm_srli_epi16(green3,8);
		green4 = _mm_srli_epi16(green4,8);

		__m128i blue1 = _mm_shuffle_epi8(lvl55,mask1);
		__m128i blue2 = _mm_shuffle_epi8(lvl55,mask2);
		__m128i blue3 = _mm_shuffle_epi8(lvl56,mask1);
		__m128i blue4 = _mm_shuffle_epi8(lvl56,mask2);	
		/*conversion ends*/
	
		/*Multiply each color with respective constant*/
		__m128i r1 = _mm_mullo_epi16(red1 , rconst);
		__m128i g1 = _mm_mullo_epi16(green1, gconst);
		__m128i b1 = _mm_mullo_epi16(blue1, bconst);
		__m128i r2 =_mm_mullo_epi16(red2 , rconst);
		__m128i g2 = _mm_mullo_epi16(green2, gconst);
		__m128i b2 = _mm_mullo_epi16(blue2, bconst);
		__m128i r3 = _mm_mullo_epi16(red3 , rconst);
		__m128i g3 =  _mm_mullo_epi16(green3, gconst);
		__m128i b3 =  _mm_mullo_epi16(blue3, bconst);
		__m128i r4 = _mm_mullo_epi16(red4 , rconst);
		__m128i g4 =  _mm_mullo_epi16(green4, gconst);
		__m128i b4 =  _mm_mullo_epi16(blue4, bconst);

		/*Add all each rgb set together */
		__m128i rgb1 = _mm_add_epi16(r1,g1);
		rgb1 = _mm_add_epi16(rgb1,b1);
		__m128i rgb2 = _mm_add_epi16(r2,g2);
		rgb2 = _mm_add_epi16(rgb2,b2);
		__m128i rgb3 = _mm_add_epi16(r3,g3);
		rgb3 = _mm_add_epi16(rgb3,b3);
		__m128i rgb4 = _mm_add_epi16(r4,g4);
		rgb4 = _mm_add_epi16(rgb4,b4);

		/*Right shift the sum*/
		__m128i rgb11 = _mm_srli_epi16(rgb1,8);
		__m128i rgb12 = _mm_srli_epi16(rgb2,8);
		__m128i rgb13 = _mm_srli_epi16(rgb3,8);
		__m128i rgb14 = _mm_srli_epi16(rgb4,8);

		/*Extract only lower bits using shuffle to convert back into 16 8bit uint*/
		__m128i gray1 = _mm_or_si128(_mm_shuffle_epi8(rgb12,gray_mask2),_mm_shuffle_epi8(rgb11,gray_mask1));
		__m128i gray2 = _mm_or_si128(_mm_shuffle_epi8(rgb14,gray_mask2),_mm_shuffle_epi8(rgb13,gray_mask1));

		/*Store*/
		_mm_storeu_si128((__m128i*)grayscale_image,gray1);
		_mm_storeu_si128((__m128i*)(grayscale_image+16),gray2);

		ptr +=32;
		grayscale_image += 32;
		rgb_image += 96;

	}

	/*Handles cases which exceed a multiple of 96*/
	while(ptr < (width*height))
	{
		const uint16_t red   = *(rgb_image);
		const uint16_t green = *(rgb_image+ 1);
		const uint16_t blue  = *(rgb_image + 2);
		const uint8_t  luma  = (red * 54 + green * 183 + blue * 19) >> 8;
		*(grayscale_image++) = luma;
		ptr++;
		rgb_image+=3;
	}
}

void integrate_image_optimized(const uint8_t *CSE6230_RESTRICT source_image, 
		uint32_t *CSE6230_RESTRICT integral_image, size_t width, size_t height) {

	__m128i c = _mm_set1_epi32(0);
	size_t j = 0;       

	int prefix_last_ele = 0;

	for(size_t i=0;i<height; ++i) {

		__m128i carry = _mm_set1_epi32(0);
	
		size_t start_addr = i * width; 	
		j = 0;

		for(; j + 16 < width; j += 16) {

			//load 128 bits or 16 8-bit integers
			__m128i elements = _mm_loadu_si128((const __m128i*)(source_image + start_addr + j));

			//take lower 8 8-bit integers and convert them to 16 bit
			__m128i al_8_16 = _mm_unpacklo_epi8(elements, c);
			//do the same for higher 8-bit integers
			__m128i ah_8_16 = _mm_unpackhi_epi8(elements, c);

			//prefix sum vector containing lower 8 integers
			al_8_16 = _mm_add_epi16(al_8_16, _mm_slli_si128(al_8_16, 2));
			al_8_16 = _mm_add_epi16(al_8_16, _mm_slli_si128(al_8_16, 4));
			al_8_16 = _mm_add_epi16(al_8_16, _mm_slli_si128(al_8_16, 8));

			//prefix sum vector containing higher 8 integers
			ah_8_16 = _mm_add_epi16(ah_8_16, _mm_slli_si128(ah_8_16, 2));
			ah_8_16 = _mm_add_epi16(ah_8_16, _mm_slli_si128(ah_8_16, 4));
			ah_8_16 = _mm_add_epi16(ah_8_16, _mm_slli_si128(ah_8_16, 8));

			//add the last element from the previous vector and store it in integral image
			//which defaults to zero for the first set
			__m128i al_8_16l_32 = _mm_add_epi32(_mm_unpacklo_epi16(al_8_16, c), carry);
			_mm_storeu_si128((__m128i *) (integral_image + start_addr + j + 0), al_8_16l_32);

			__m128i al_8_16h_32 = _mm_add_epi32(_mm_unpackhi_epi16(al_8_16, c), carry);
			_mm_storeu_si128((__m128i *) (integral_image + start_addr + j + 4), al_8_16h_32);

			//take the last element of the higher 32 bit vector, to be added to the next 8 32 bit elements
			prefix_last_ele = _mm_extract_epi32(al_8_16h_32, 3);                
			carry = _mm_set1_epi32(prefix_last_ele);

			//unpack to 32 bit and add the last element
			__m128i ah_8_16l_32 = _mm_add_epi32(_mm_unpacklo_epi16(ah_8_16, c), carry);
			_mm_storeu_si128((__m128i *) (integral_image + start_addr + j + 8), ah_8_16l_32);

			__m128i ah_8_16h_32 = _mm_add_epi32(_mm_unpackhi_epi16(ah_8_16, c), carry);
			_mm_storeu_si128((__m128i *) (integral_image + start_addr + j + 12), ah_8_16h_32);

			//prepare the carry to contain the last element for adding to following vectors
			prefix_last_ele = _mm_extract_epi32(ah_8_16h_32, 3);
			carry = _mm_set1_epi32(prefix_last_ele);
		}

		//now i have like 8 elements remaining
                for (; j < width ; ++j) {
                        prefix_last_ele += source_image[i * width + j];
                        integral_image[i * width + j] = prefix_last_ele;
                }

        }


	//now performing the columnwise prefix sum
	for(size_t i = 1; i < height ; ++i) {

		size_t j = 0;
		for(j = 0; j + 16 < width; j += 16) {
			__m128i row1 = _mm_loadu_si128((const __m128i*)(integral_image + ((i-1) * width) + j));
			__m128i row2 = _mm_loadu_si128((const __m128i*)(integral_image + (i * width) + j));
			row2 = _mm_add_epi32(row1, row2);
                        _mm_storeu_si128((__m128i *) (integral_image + (i * width) + j ), row2);

			row1 = _mm_loadu_si128((const __m128i*)(integral_image + ((i-1) * width) + j + 4));
			row2 = _mm_loadu_si128((const __m128i*)(integral_image + (i * width) + j + 4));
			row2 = _mm_add_epi32(row1, row2);
                        _mm_storeu_si128((__m128i *) (integral_image + (i * width) + j + 4), row2);

			row1 = _mm_loadu_si128((const __m128i*)(integral_image + ((i-1) * width) + j + 8));
			row2 = _mm_loadu_si128((const __m128i*)(integral_image + (i * width) + j + 8));
			row2 = _mm_add_epi32(row1, row2);
			_mm_storeu_si128((__m128i *) (integral_image + (i * width) + j + 8), row2);

			row1 = _mm_loadu_si128((const __m128i*)(integral_image + ((i-1) * width) + j + 12));
			row2 = _mm_loadu_si128((const __m128i*)(integral_image + (i * width) + j + 12));
			row2 = _mm_add_epi32(row1, row2);
			_mm_storeu_si128((__m128i *) (integral_image + (i * width) + j + 12), row2);
		}

		for (; j < width; ++j) {
			integral_image[ i * width + j] += integral_image[ (i-1) * width + j];
		}
	}
}

