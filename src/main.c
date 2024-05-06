#include <stdio.h>
#include <math.h>

float f(float x, float y) {
    return (x * x + y * y)/2;
}

int main() {
    int i, j;
    int n = 10;
    double step = 2.0 / (n - 1);
    double x, y;

    for (i = 0; i < n; i++) {
        x = -1.0 + i * step;
        for (j = 0; j < n; j++) {
            y = -1.0 + j * step;
            printf("(%lf, %lf) -> %lf\n", x, y, f(x, y));
        }
    }
    return 0;
}
