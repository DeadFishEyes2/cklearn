#include <stdio.h>
#include <stdlib.h>

void plotScatter(float *x, float *y, int size) {
    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp == NULL) {
        fprintf(stderr, "Could not open gnuplot\n");
        return;
    }

    fprintf(gp, "set title 'Scatter Plot'\n");
    fprintf(gp, "plot '-' with points pointtype 7 pointsize 1 lc rgb 'blue'\n");

    for (int i = 0; i < size; i++) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }
    fprintf(gp, "e\n"); // End of data

    fflush(gp);
    pclose(gp);
}

int main() {
    float x[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float y[] = {2.1, 4.2, 6.1, 7.9, 10.2};
    int size = sizeof(x) / sizeof(x[0]);

    plotScatter(x, y, size);

    return 0;
}
