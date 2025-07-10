#include <stdio.h>
#include <stdlib.h>

void plotScatter(float *x, float *y, int size) {
    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp == NULL) {
        fprintf(stderr, "Could not open gnuplot\n");
        return;
    }

    // Set plot style and formatting
    fprintf(gp, "set title 'Elegant Scatter Plot'\n");
    fprintf(gp, "set xlabel 'X-axis'\n");
    fprintf(gp, "set ylabel 'Y-axis'\n");
    fprintf(gp, "set grid\n");  // Enable grid
    fprintf(gp, "set key off\n");  // Disable legend if not needed

    // Customize point appearance
    fprintf(gp, "set style line 1 lc rgb '#0060ad' pt 7 ps 1.5 lw 1\n");  // Blue points
    fprintf(gp, "plot '-' with points ls 1\n");  // Use the defined style

    // Send data points
    for (int i = 0; i < size; i++) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }
    fprintf(gp, "e\n");  // End of data

    fflush(gp);
    pclose(gp);
}

int main() {
    float x[] = {1.0, 2.0, 3.0, 4.0, 6.0};
    float y[] = {2.1, 4.2, 6.1, 7.9, 10.2};
    int size = sizeof(x) / sizeof(x[0]);

    plotScatter(x, y, size);

    return 0;
}
