//Vectorization SSE for x86_64 architectures

#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>


//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

//
typedef struct particle_s {

    f32 *x, *y, *z;
    f32 *vx, *vy, *vz;
} particle_t;

//
static inline void init(particle_t p, u64 n)
{

  for (u64 i = 0; i < n; i++)
    {
      //
      u64 r1 = (u64)rand();
      u64 r2 = (u64)rand();
      f32 sign = (r1 > r2) ? 1 : -1;
      
      //
      p.x[i] = sign * (f32)rand() / (f32)RAND_MAX;
      p.y[i] = (f32)rand() / (f32)RAND_MAX;
      p.z[i] = sign * (f32)rand() / (f32)RAND_MAX;

      //
       p.vx[i] = (f32)rand() / (f32)RAND_MAX;
       p.vy[i] = sign * (f32)rand() / (f32)RAND_MAX;
       p.vz[i] = (f32)rand() / (f32)RAND_MAX;
    }
}

static inline void move_particles(particle_t p, const f32 dt,const u64 n)
{
  //
  const f32 softening = 1e-20;
 
  //set les elements en AVX
  __m128 softening_mm = _mm_set1_ps(softening);
  __m128 dt_mm = _mm_set1_ps(dt);

  //
#pragma unroll
  for (u64 i = 0; i < n; i++)
    {
      //
      // f32 fx = 0.0;
      // f32 fy = 0.0;
      // f32 fz = 0.0;

      __m128 fx = _mm_setzero_ps();
      __m128 fy = _mm_setzero_ps();
      __m128 fz = _mm_setzero_ps();
      __m128 dx_mm = _mm_set1_ps(p.x[i]);
      __m128 dy_mm = _mm_set1_ps(p.y[i]);
      __m128 dz_mm = _mm_set1_ps(p.z[i]);

    //23 floating-point operations
    #pragma unroll
      for (u64 j = 0; j < n; j= j+8)
	{   

    //Newton's law
    // const f32 dx = p.x[j] - p.x[i]; //1 (sub)
    // const f32 dy = p.y[j] - p.y[i]; //2 (sub)
    // const f32 dz = p.z[j] - p.z[i]; //3 (sub)

    __m128 dx = _mm_sub_ps(_mm_loadu_ps(p.x+j),dx_mm);
    __m128 dy = _mm_sub_ps(_mm_loadu_ps(p.y+j),dy_mm);
    __m128 dz = _mm_sub_ps(_mm_loadu_ps(p.z+j),dz_mm);

    //const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening; //9 (mul, add)
    //const f32 d_2 = 1.0 / sqrtf((dx * dx) + (dy * dy) + (dz * dz) + softening); //11 (mul,add,div,sqrtf)

    __m128 d_2 = _mm_fmadd_ps(dz, dz,_mm_fmadd_ps(dy, dy,_mm_fmadd_ps(dx, dx, softening_mm)));

    __m128 d_2_mm = _mm_rsqrt_ps (d_2);
    __m128 d_3 =  _mm_mul_ps(_mm_mul_ps(d_2_mm,d_2_mm),d_2_mm);

    //Net force
    //fx += dx / d_3_over_2; //13 (add, div)
    //fy += dy / d_3_over_2; //15 (add, div)
    //fz += dz / d_3_over_2; //17 (add, div)

    //fx += dx * d_3_over_2; //15 (add, mul)
    //fy += dy * d_3_over_2; //17 (add, mul)
    //fz += dz * d_3_over_2; //19 (add, mul)

      fx = _mm_fmadd_ps(dx,d_3, fx);
      fy = _mm_fmadd_ps(dy, d_3, fy);
      fz = _mm_fmadd_ps(dz, d_3, fz);


	}
  //
  //  p.vx[i] += dt * fx; //21 (mul, add)
  //  p.vy[i] += dt * fy; //23 (mul, add)
  //  p.vz[i] += dt * fz; //25 (mul, add)
    
      __m128 fx_sum = _mm_hadd_ps(fx, fx);
      fx_sum = _mm_hadd_ps(fx_sum, fx_sum);
      fx_sum = _mm_hadd_ps(fx_sum, fx_sum);
      float fx_result = _mm_cvtss_f32(fx_sum);

      p.vx[i] += dt * fx_result;

      __m128 fy_sum = _mm_hadd_ps(fy, fy);
      fy_sum = _mm_hadd_ps(fy_sum, fy_sum);
      fy_sum = _mm_hadd_ps(fy_sum, fy_sum);
      float fy_result = _mm_cvtss_f32(fy_sum);

      p.vy[i] += dt * fy_result;

      __m128 fz_sum = _mm_hadd_ps(fz, fz);
      fz_sum = _mm_hadd_ps(fz_sum, fz_sum);
      fz_sum = _mm_hadd_ps(fz_sum, fz_sum);
      float fz_result = _mm_cvtss_f32(fz_sum);

    }
  
  //3 floating-point operations
  #pragma unroll
  for (u64 i = 0; i < n; i=i+8)
    {
      // p.x[i] += dt *  p.vx[i];
      // p.y[i] += dt *  p.vy[i];
      // p.z[i] += dt *  p.vz[i];
     
    __m128 xx = _mm_fmadd_ps(dt_mm,_mm_loadu_ps(p.vx+i),_mm_loadu_ps(p.x+i));
    __m128 yy = _mm_fmadd_ps(dt_mm,_mm_loadu_ps(p.vy+i),_mm_loadu_ps(p.y+i));
    __m128 zz = _mm_fmadd_ps(dt_mm,_mm_loadu_ps(p.vz+i),_mm_loadu_ps(p.z+i));

    _mm_storeu_ps(p.x + i, xx);
    _mm_storeu_ps(p.y + i, yy);
    _mm_storeu_ps(p.z + i, zz); 


    }

}


//
int main(int argc, char **argv)
{
  //
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;
  const u64 steps= 10;
  const f32 dt = 0.01;

  //
  f64 rate = 0.0, drate = 0.0;

  //Steps to skip for warm up
  const u64 warmup = 3;
  
  //
  
    particle_t p;
    const u64 al = 64;
    p.x = aligned_alloc(al, n * sizeof(f32));
    p.y = aligned_alloc(al, n * sizeof(f32));
    p.z = aligned_alloc(al, n * sizeof(f32));
    p.vx = aligned_alloc(al, n * sizeof(f32));
    p.vy = aligned_alloc(al, n * sizeof(f32));
    p.vz = aligned_alloc(al, n * sizeof(f32));

  //
  init(p, n);

  const u64 s = sizeof(particle_t) * n;
  
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
  
  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  
  //
  for (u64 i = 0; i < steps; i++)
    {
      //Measure
      const f64 start = omp_get_wtime();
      move_particles(p, dt, n);

      const f64 end = omp_get_wtime();

      //Number of interactions/iterations
      const f32 h1 = (f32)(n) * (f32)(n - 1);

      //GFLOPS
      const f32 h2 = (23.0 * h1 + 3.0 * (f32)n) * 1e-9;
      
      if (i >= warmup)
	{
	  rate += h2 / (end - start);
	  drate += (h2 * h2) / ((end - start) * (end - start));
	}
      
      //
      printf("%5llu %10.3e %10.3e %8.1f %s\n",
	     i,
	     (end - start),
	     h1 / (end - start),
	     h2 / (end - start),
	     (i < warmup) ? "*" : "");
      
      fflush(stdout);
    }
  
  //
  rate /= (f64)(steps - warmup);
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));
  
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");
  
  //
  free(p.x);
  free(p.y);
  free(p.z);
  free(p.vx);
  free(p.vy);
  free(p.vz);


  //
  return 0;
}
