/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <omp.h>


double bench_t_start, bench_t_end;

static
double rtclock()
{
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, NULL);
    if (stat != 0)
      printf ("Error return from gettimeofday: %d", stat);
    return (Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}

void bench_timer_start()
{
  bench_t_start = rtclock ();
}

void bench_timer_stop()
{
  bench_t_end = rtclock ();
}

void bench_timer_print()
{
  printf ("Time in seconds = %0.6lf\n", bench_t_end - bench_t_start);
}


static
void init_array (int n,
   float A[n],
   float B[n])
{
  int i;

  for (i = 0; i < n; i++)
      {
 A[i] = ((float) i+ 2) / n;
 B[i] = ((float) i+ 3) / n;
      }
}

static
void print_array(int n,
   float A[n])

{
  int i;

  fprintf(stderr, "==BEGIN DUMP_ARRAYS==\n");
  fprintf(stderr, "begin dump: %s", "A");
  for (i = 0; i < n; i++)
    {
      if (i % 20 == 0) fprintf(stderr, "\n");
      fprintf(stderr, "%0.2f ", A[i]);
    }
  fprintf(stderr, "\nend   dump: %s\n", "A");
  fprintf(stderr, "==END   DUMP_ARRAYS==\n");
}

static
void kernel_jacobi_1d(int tsteps,
       int n,
       float A[n],
       float B[n])
{
  int t, i;


    {
        for (t = 0; t < tsteps; t++) {
            #pragma omp parallel shared(B)
            {
                #pragma omp for
                for (i = 1; i < n - 1; i++) {
                    B[i] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
                }
            }
            #pragma omp barrier
            #pragma omp parallel shared(A)
            {
                #pragma omp for
                for (i = 1; i < n - 1; i++) {
                    A[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
                }
            }
            #pragma omp barrier
        }
    }
}


int main(int argc, char** argv)
{
    omp_set_num_threads(omp_get_num_procs());
    printf("%d\n", omp_get_num_procs());
    int nes[] =      {30, 120, 400, 2000, 4000, 16000, 32000, 64000, 128000};
    int tstepses[] = {20, 40, 100, 500, 1000, 4000, 8000, 16000, 32000};
    int n, tsteps;
    
    for (int i = 0; i < 7; i++) {
        n = nes[i];
        tsteps = tstepses[i];
  
        float (*A)[n];
        A = (float(*)[n]) malloc((n) * sizeof(float));
  
        float (*B)[n]; 
        B = (float(*)[n]) malloc ((n) * sizeof(float));

        init_array(n, *A, *B);

        bench_timer_start();
  
        printf("%d, %d\n", n, tsteps);
        //print_array(n, A);
        kernel_jacobi_1d(tsteps, n, *A, *B);

        bench_timer_stop();
        bench_timer_print();

        //if (argc > 42 && ! strcmp(argv[0], "")) 
        print_array(n, *A);

        free((void*)A);
        free((void*)B);
    }
    return 0;
}
