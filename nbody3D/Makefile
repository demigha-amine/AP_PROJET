
OFLAGS=-Ofast -fopenmp 


CFLAGS=-march=native 


all: nbody3D.g nbody3D1.g nbody3D2.g nbody3D3.g nbody3D4.g nbody3D5.g nbody3D6.g nbody3D7.g nbody3D.cl nbody3D1.cl nbody3D2.cl nbody3D3.cl nbody3D4.cl nbody3D5.cl nbody3D6.cl nbody3D7.cl

nbody3D.g: nbody.c
	gcc $(CFLAGS) -g3 $(OFLAGS) -fopt-info-all=nbody.gcc.optrpt $< -o $@ -lm

nbody3D1.g: nbody1.c 
	gcc $(CFLAGS) $(OFLAGS) -fopt-info-all=nbody1.gcc.optrpt $< -o $@ -lm

nbody3D2.g: nbody2.c
	gcc $(CFLAGS) $(OFLAGS) -fopt-info-all=nbody2.gcc.optrpt $< -o $@ -lm

nbody3D3.g: nbody3.c
	gcc -static $(CFLAGS) $(OFLAGS) -fopt-info-all=nbody3.gcc.optrpt $< -o $@ -lm

nbody3D4.g: nbody4.c
	gcc $(CFLAGS) -funroll-loops $(OFLAGS) -fopt-info-all=nbody4.gcc.optrpt $< -o $@ -lm

nbody3D5.g: nbody5.c
	gcc $(CFLAGS) -mavx -finline-functions -funroll-loops $(OFLAGS) -fopt-info-all=nbody5.gcc.optrpt $< -o $@ -lm

nbody3D6.g: nbody6.c
	gcc $(CFLAGS) -msse -finline-functions -funroll-loops $(OFLAGS) -fopt-info-all=nbody6.gcc.optrpt $< -o $@ -lm

nbody3D7.g: nbody7.c
	gcc $(CFLAGS) -msse -finline-functions $(OFLAGS) -fopt-info-all=nbody7.gcc.optrpt $< -o $@ -lm

############################################################################################################################

nbody3D.cl: nbody.c
	clang $(CFLAGS) -g3 $(OFLAGS) $< -o $@ -lm

nbody3D1.cl: nbody1.c
	clang $(CFLAGS) $(OFLAGS)  $< -o $@ -lm

nbody3D2.cl: nbody2.c
	clang $(CFLAGS) $(OFLAGS)  $< -o $@ -lm

nbody3D3.cl: nbody3.c
	clang $(CFLAGS) $(OFLAGS)  $< -o $@ -lm

nbody3D4.cl: nbody4.c
	clang $(CFLAGS) -funroll-loops -Wno-pass-failed $(OFLAGS)  $< -o $@ -lm

nbody3D5.cl: nbody5.c
	clang $(CFLAGS) -mavx -finline-functions -funroll-loops  -Wno-pass-failed $(OFLAGS)  $< -o $@ -lm

nbody3D6.cl: nbody6.c
	clang $(CFLAGS) -msse -finline-functions -funroll-loops -Wno-pass-failed $(OFLAGS)  $< -o $@ -lm

nbody3D7.cl: nbody7.c
	clang $(CFLAGS) -msse -finline-functions -Wno-pass-failed $(OFLAGS)  $< -o $@ -lm


run: 0 1 2 3 4 5 6 7 


0: 
	taskset -c 3 ./nbody3D.g $(np)
	taskset -c 3 ./nbody3D.cl $(np)
1:
	taskset -c 3 ./nbody3D1.g $(np)
	taskset -c 3 ./nbody3D1.cl $(np)
2:	
	taskset -c 3 ./nbody3D2.g $(np)
	taskset -c 3 ./nbody3D2.cl $(np)
3:	
	taskset -c 3 ./nbody3D3.g $(np)
	taskset -c 3 ./nbody3D3.cl $(np)
4:	
	taskset -c 3 ./nbody3D4.g $(np)
	taskset -c 3 ./nbody3D4.cl $(np)
5:	
	taskset -c 3 ./nbody3D5.g $(np)
	taskset -c 3 ./nbody3D5.cl $(np)

6:	
	taskset -c 3 ./nbody3D6.g $(np)
	taskset -c 3 ./nbody3D6.cl $(np)

7:	
	./nbody3D7.g $(np)
	./nbody3D7.cl $(np)


clean:
	rm -Rf *.g *.cl *.optrpt


