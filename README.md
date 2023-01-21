# AP_PROJET
## Projet Archis Parallèles - nbody 3D simulation

Afin d'améliorer la performance, on a implémenté 7 versions d'optimisation avec `gcc` et `clang` :

* nbody.c : version AOS.

* nbody1.c : version SOA.

* nbody2.c : version Memory alignment.

* nbody3.c : version Removing costly instructions: sqrt, pow, division.

* nbody4.c : version Unroll.

* nbody5.c : version Vectorization AVX for x86_64 architectures.

* nbody6.c : version Vectorization SSE for x86_64 architectures

* nbody7.c : version Parallelization using OpenMP.

## Compilation

      $ make

## Exécution

Pour exécuter toutes les versions à la fois :

      $ make run
      
OR

      $ make run np=<numbers of particles>

Pour exécuter l'une des versions {0,....,7} :

      $ make <numéro de version>
      
OR

      $ make <numéro de version> np=<numbers of particles>
      
* Les exécutables *.g = `GCC`

* Les exécutables *.cl = `CLANG`

    
