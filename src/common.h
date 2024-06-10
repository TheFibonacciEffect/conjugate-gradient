#ifndef COMMON_H
#define COMMON_H

// Präprozessor Makros und Definitionen
// (https://de.wikipedia.org/wiki/C-Pr%C3%A4prozessor#Definition_und_Ersetzung_von_Makros)

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

#ifdef MAIN_PROGRAM
   #define EXTERN
#else
   #define EXTERN extern
#endif

#endif // COMMON_H

