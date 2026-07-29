#ifndef  TEST_HELPER
#define TEST_HELPER

#include "livero_types.h"
#include "util.h"
#include <stdint.h>
#include <float.h>
#include <math.h>

#define F32_EPS FLT_EPSILON

static float rand_f32(void){
    const uint32_t random_value = xorshift();
    float output = (float)((double)random_value / 4294967296.0);
    output = random_value %2 == 0 ? -output:output;
    return output;
}

static int8_t rand_i8(void)
{
    return -128 + (int8_t)(xorshift() % 255);
}

static void fill_f32_vector(float* vector, const LVDim32_t dim){
    for(LVDim32_t i=0; i<dim; ++i){
        vector[i] = rand_f32();
    }
}

static void fill_i8_vector(int8_t* vector, const LVDim32_t dim){
    for(LVDim32_t i=0; i<dim; ++i){
        vector[i] = rand_i8();
    }
}

// Summing n floats accumulates rounding error that grows roughly with n
// (each addition can round by up to ~0.5 ULP of the running sum). So the
// tolerance must scale with BOTH the magnitude of the result AND the number
// of terms summed — a fixed ULP count only works for a fixed n.
static int f32_close(float got, float want, LVDim32_t dim) {
  const float abs_diff = fabsf(got - want);
  const float scale = fabsf(want) > 1.0f ? fabsf(want) : 1.0f;
  // magnitude-relative, and proportional to the number of accumulations.
  return abs_diff <= scale * F32_EPS * (float)dim;
}

#endif
