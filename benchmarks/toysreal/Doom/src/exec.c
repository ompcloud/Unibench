#include <stdio.h>
#include <stdlib.h>
#include <unistd.h> /* for fork */
#include <sys/types.h> /* for pid_t */
#include <sys/wait.h> /* for wait */
#include <sys/time.h>
double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

int main()
{
    /*Spawn a child to run the program.*/
    double t_start, t_end;
    int status;


    pid_t pid=fork();

    if (pid==0) { /* child process */
        printf("Running parallel version of Doom\n");
        //static char *argv[]={"echo","-iwad doom1.wad -timedemo demo3 -window",NULL};
        t_start = rtclock();
        status =system("../input/chocolate-doom-parallel -iwad ../input/doom1.wad -timedemo demo3 -window");
        //execv("./chocolate-doom",argv);
        exit(0); /* only if execv fails */
    }
    else { /* pid!=0; parent process */
        waitpid(pid,0,0); /* wait for child to exit */
        t_end=rtclock();
        printf("Parallel time: %0.6lfs\n", t_end - t_start);
    }

    pid=fork();
    if (pid==0) { /* child process */
        printf("Running sequential version of Doom\n");
        //static char *argv[]={"echo","-iwad doom1.wad -timedemo demo3 -window",NULL};
        t_start = rtclock();
        int status =system("../input/chocolate-doom-serial -iwad ../input/doom1.wad -timedemo demo3 -window");
        //execv("./chocolate-doom",argv);
        exit(0); /* only if execv fails */
    }
    else { /* pid!=0; parent process */
        waitpid(pid,0,0); /* wait for child to exit */
        t_end=rtclock();
        printf("Sequential time: %0.6lfs\n", t_end - t_start);
    }

    return 0;
}
