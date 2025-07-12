#include <stdio.h>
#include <stdlib.h>

void plotScatter(float *x, float *y, int size) {
    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp == NULL) {
        fprintf(stderr, "Could not open gnuplot\n");
        return;
    }

    // Set plot style and formatting
    fprintf(gp, "set title 'Scatter Plot with Regression Line'\n");
    fprintf(gp, "set xlabel 'X-axis'\n");
    fprintf(gp, "set ylabel 'Y-axis'\n");
    fprintf(gp, "set grid\n");  // Enable grid
    fprintf(gp, "set key off\n");  // Disable legend if not needed

    // Send data to a temporary file for fitting
    fprintf(gp, "$data << EOD\n");
    for (int i = 0; i < size; i++) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }
    fprintf(gp, "EOD\n");

    // Define a linear model and fit it to the data
    fprintf(gp, "f(x) = m*x + b\n");
    fprintf(gp, "fit f(x) '$data' via m, b\n");

    // Customize appearance
    fprintf(gp, "set style line 1 lc rgb '#0060ad' pt 7 ps 1.5 lw 1\n");  // Scatter points
    fprintf(gp, "set style line 2 lc rgb '#dd181f' lw 2\n");             // Regression line (red)

    // Plot scatter points and regression line
    fprintf(gp, "plot '$data' with points ls 1, f(x) with lines ls 2\n");

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
