#include <stdio.h>
#include <math.h>

int main() {
    int i, j;
    double step = 2.0 / 9.0;
    double x, y;

    for (i = 0; i < 10; i++) {
        x = -1.0 + i * step;
        for (j = 0; j < 10; j++) {
            y = -1.0 + j * step;
            printf("(%lf, %lf)\n", x, y);
        }
    }
    return 0;
}
