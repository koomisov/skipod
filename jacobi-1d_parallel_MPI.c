/* Include benchmark-specific header. */
#include "jacobi-1d.h"
#include <mpi.h>


double bench_t_start, bench_t_end;

int nProcs, id;

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
        int len = n / (nProcs - 1) + 1;
        float C[len];
        for (t = 0; t < tsteps; t++) {
            if (id != 0) {
                int j = 0;
                for (i = 1 + (id - 1); i < n - 1; i += (nProcs - 1)) {
                    C[j] = 0.33333 * (A[i-1] + A[i] + A[i + 1]);
                    j++;
                }
                MPI_Send(C, len, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
                MPI_Recv(B, n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                int k = 0;
                for (k = 1; k < nProcs; k++) {
                    MPI_Recv(C, len, MPI_FLOAT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int j = 0;
                    for (i = 1 + (k - 1); i < n - 1; i += (nProcs - 1)) {
                        B[i] = C[j];
                        j++;
                    }
                }
                for (i = 1; i < nProcs; i++) {
                    MPI_Send(B, n, MPI_FLOAT, i, 0, MPI_COMM_WORLD);
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
            if (id != 0) {
                int j = 0;
                for (i = 1 + (id - 1); i < n - 1; i += (nProcs - 1)) {
                    C[j] = 0.33333 * (B[i-1] + B[i] + B[i + 1]);
                    j++;
                }
                MPI_Send(C, len, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
            } else {
                int k;
                for (k = 1; k < nProcs; k++) {
                    MPI_Recv(C, len, MPI_FLOAT, k, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    int j = 0;
                    for (i = 1 + (k - 1); i < n - 1; i += (nProcs - 1)) {
                        A[i] = C[j];
                        j++;
                    }
                 }
            }
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
    
    for (int i = 0; i < 9; i++) {
        MPI_Barrier(MPI_COMM_WORLD);
        n = nes[i];
        tsteps = tstepses[i];
  
        float (*A)[n];
        A = (float(*)[n]) malloc((n) * sizeof(float));
  
        float (*B)[n]; 
        B = (float(*)[n]) malloc ((n) * sizeof(float));

        init_array(n, *A, *B);

        bench_timer_start();
        //printf("%d\n", id);
        if (id == 0) {
            printf("%d, %d\n", n, tsteps);
            //printf("%d\n", nProcs);
        }
        //print_array(n, A);
        kernel_jacobi_1d(tsteps, n, *A, *B);
        MPI_Barrier(MPI_COMM_WORLD);
        if (id == 0) {
            bench_timer_stop();
            bench_timer_print();
            print_array(n, *A);
        }

        //if (argc > 42 && ! strcmp(argv[0], ""))
        

        free((void*)A);
        free((void*)B);
    }
    MPI_Finalize();
    return 0;
}
