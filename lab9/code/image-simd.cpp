#include <hpcdefs.hpp>
#include <image.hpp>

void convert_to_floating_point_optimized(const uint8_t *CSE6230_RESTRICT fixed_point_images, double *CSE6230_RESTRICT floating_point_images, size_t image_width, size_t image_height, size_t image_count) {

    __m128i mask1=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,1,-1,-1,-1,0);
    __m128i mask2=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,3,-1,-1,-1,2);
    __m128i mask3=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,5,-1,-1,-1,4);
    __m128i mask4=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,7,-1,-1,-1,6);
    __m128i mask5=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,9,-1,-1,-1,8);
    __m128i mask6=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,11,-1,-1,-1,10);
    __m128i mask7=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,13,-1,-1,-1,12);
    __m128i mask8=_mm_set_epi8(-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,15,-1,-1,-1,14);
    __m128d constant = _mm_set1_pd(1.0/255);
    size_t ptr=0;
    size_t im_size = image_width * image_height * image_count;
    while(ptr <= (im_size-16))
    {
        __m128i fixed=_mm_loadu_si128((const __m128i*)fixed_point_images);
        fixed_point_images+=16;
        __m128d fl1 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask1)),constant);
        __m128d fl2 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask2)),constant);
        __m128d fl3 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask3)),constant);
        __m128d fl4 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask4)),constant);
        __m128d fl5 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask5)),constant);
        __m128d fl6 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask6)),constant);
        __m128d fl7 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask7)),constant);
        __m128d fl8 = _mm_mul_pd(_mm_cvtepi32_pd(_mm_shuffle_epi8(fixed,mask8)),constant);  

        _mm_storeu_pd((floating_point_images),fl1);
        _mm_storeu_pd((floating_point_images+2),fl2);     
        _mm_storeu_pd((floating_point_images+4),fl3);
        _mm_storeu_pd((floating_point_images+6),fl4);
        _mm_storeu_pd((floating_point_images+8),fl5);
        _mm_storeu_pd((floating_point_images+10),fl6);
        _mm_storeu_pd((floating_point_images+12),fl7);
        _mm_storeu_pd((floating_point_images+14),fl8);

        floating_point_images+=16;
        ptr+=16;
    }


}

void matrix_vector_multiplication_optimized(double *CSE6230_RESTRICT output_vector, const double *CSE6230_RESTRICT matrix, const double *CSE6230_RESTRICT input_vector, size_t matrix_width, size_t matrix_height) {

    size_t ptr=0,i,j;
    __m128d result1,result2,result3,result2b,result3b,result4;
    while((ptr)<(matrix_height-4))
    {
         /*Processing 4 rows at once - need to find the right balance between handling more stuff outside the vectorization loop vs inside - the greater the number of rows processed inside, more rows tend to get left out of the above while loop - i<matrix_width-4. At the same time, we also need to exploit the vectorization component by using sufficient number of rows to be processed inside. In short, the degree of unrolling is a factor of the ratio of number of rows processes by vectors to the number of rows processes naively*/

        i=0;
        result3=_mm_setzero_pd();
        result4=_mm_setzero_pd();
        while(i<(matrix_width-4))
        {
            __m128d mat = _mm_loadu_pd(matrix+(ptr*matrix_width)+i);
            __m128d mat1a = _mm_loadu_pd(matrix+(ptr*matrix_width)+i+2);
            __m128d mat1 = _mm_loadu_pd(matrix+((ptr+1)*matrix_width+i));
            __m128d mat1b = _mm_loadu_pd(matrix+((ptr+1)*matrix_width+i+2));
            __m128d mat2 = _mm_loadu_pd(matrix+((ptr+2)*matrix_width+i));
            __m128d mat2b = _mm_loadu_pd(matrix+((ptr+2)*matrix_width+i+2));
            __m128d mat3 = _mm_loadu_pd(matrix+((ptr+3)*matrix_width+i));
            __m128d mat3b = _mm_loadu_pd(matrix+((ptr+3)*matrix_width+i+2));

            __m128d i_vec = _mm_loadu_pd(input_vector+i);
            __m128d i_veca = _mm_loadu_pd(input_vector+i+2);

            __m128d prod = _mm_mul_pd(mat,i_vec);
            __m128d prod1a = _mm_mul_pd(mat1a,i_veca);
            __m128d prod1 = _mm_mul_pd(mat1,i_vec);
            __m128d prod1b = _mm_mul_pd(mat1b,i_veca);
            __m128d prod2 = _mm_mul_pd(mat2,i_vec);
            __m128d prod2b = _mm_mul_pd(mat2b,i_veca);  
            __m128d prod3 = _mm_mul_pd(mat3,i_vec);
            __m128d prod3b = _mm_mul_pd(mat3b,i_veca);

            result1=_mm_hadd_pd(prod,prod1a);
            result2=_mm_hadd_pd(prod1,prod1b);
            result3=_mm_add_pd(result3,_mm_hadd_pd(result1,result2));
            result2b=_mm_hadd_pd(prod2,prod2b);
            result3b=_mm_hadd_pd(prod3,prod3b);
            result4=_mm_add_pd(result4,_mm_hadd_pd(result2b,result3b));

            i+=4;
        }
        _mm_storeu_pd(output_vector+ptr,result3);
        _mm_storeu_pd(output_vector+ptr+2,result4);
        double accumulated_sum=0.0, accumulated_sum1=0.0,accumulated_sum2=0.0,accumulated_sum3=0.0;
        for(j=i;j<matrix_width;++j)
        {
            accumulated_sum += matrix[ptr * matrix_width + j] * input_vector[j];
            accumulated_sum1 +=matrix[(ptr+1) *matrix_width +j] * input_vector[j];
            accumulated_sum2 +=matrix[(ptr+2) *matrix_width +j] * input_vector[j];
            accumulated_sum3 +=matrix[(ptr+3) *matrix_width +j] * input_vector[j];

        }
        output_vector[ptr]+= accumulated_sum;
        output_vector[ptr+1] += accumulated_sum1;
        output_vector[ptr+2] +=accumulated_sum2;
        output_vector[ptr+3] +=accumulated_sum3;

        ptr+=4;
    }

    for (size_t i = ptr; i < matrix_height; i++) {
         double accumulated_sum = 0.0;
        for (size_t j = 0; j < matrix_width; j++) {
            accumulated_sum += matrix[i * matrix_width + j] * input_vector[j];
        }
        output_vector[i] = accumulated_sum;

    }

}
