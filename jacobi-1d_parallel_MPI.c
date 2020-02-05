/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <mpi.h>


double bench_t_start, bench_t_end;

int nProcs, id, block, left, right;

MPI_Status status;

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
   float B[n])
{
    int i;

    for (i = 0; i < n; i++) {
        B[i] = ((float) i+ 2) / n;
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
void calc(int n, float B[n]) 
{
    int i;
    for (i = left; i < right; i++) {
        B[i] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
    }
}

static
void kernel_jacobi_1d(int tsteps,
       int n,
       float B[n])
{
    int t;

    MPI_Status status;
    block = (n - 2) / nProcs;
    if (id == nProcs - 1) {
        left = id * block + 1;
        right = n - 1;
    } else {
        left = id * block + 1;
        right = (id + 1) * block + 1;
    }
    for (t = 0; t < tsteps * 2; t++) {
        calc(n, B);
        if (t < tsteps * 2 - 1) {
            if (id == 0) {
                MPI_Send(&(B[right- 1]),  1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&(B[right]),     1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD, &status);
            } else if (id == nProcs - 1) {
                MPI_Recv(&(B[left - 1]),  1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &status); 
                MPI_Send(&(B[left]),      1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD);
            } else {
                MPI_Recv(&(B[left - 1]),  1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&(B[right - 1]), 1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD);
                MPI_Recv(&(B[right]),     1, MPI_FLOAT, id + 1, 0, MPI_COMM_WORLD, &status);
                MPI_Send(&(B[left]),      1, MPI_FLOAT, id - 1, 0, MPI_COMM_WORLD);
            }
            MPI_Barrier(MPI_COMM_WORLD);
        } else {
            MPI_Gather(&(B[id * block + 1]), block, MPI_FLOAT, &(B[1]), block, MPI_FLOAT, nProcs - 1, MPI_COMM_WORLD);
        } 
    }
}


int main(int argc, char** argv)
{
    int nes[] =      {30, 120, 400, 2000, 4000, 16000, 32000, 64000, 128000};
    int tstepses[] = {20, 40, 100, 500, 1000, 4000, 8000, 16000, 32000};
    int n, tsteps;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int i;
    for (int i = 0; i < 1; i++) {
        n = nes[i];
        tsteps = tstepses[i];
  
        float (*B)[n]; 
        B = (float(*)[n]) malloc ((n) * sizeof(float));

        init_array(n, *B);
        if (id == nProcs - 1) {
            printf("%d, %d\n", n, tsteps);
            bench_timer_start();
        }
        kernel_jacobi_1d(tsteps, n, *B);
        if (id == nProcs - 1) {
            bench_timer_stop();
            bench_timer_print();
            print_array(n, *B);
        }
    
        free((void*)B);
    }
    MPI_Finalize();
    return 0;
}
