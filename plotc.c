#include <stdio.h>
#include <stdlib.h>

void scatterplot(float *x, float *y, int size) {
    FILE *gp = popen("gnuplot -persistent > /dev/null 2>&1", "w");  // Silence all output
    if (gp == NULL) {
        fprintf(stderr, "Could not open gnuplot\n");
        return;
    }

    fprintf(gp, "set title 'Simple Scatter Plot'\n");
    fprintf(gp, "set xlabel 'X-axis'\n");
    fprintf(gp, "set ylabel 'Y-axis'\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key off\n");

    fprintf(gp, "$data << EOD\n");
    for (int i = 0; i < size; i++) {
        fprintf(gp, "%f %f\n", x[i], y[i]);
    }
    fprintf(gp, "EOD\n");

    fprintf(gp, "plot '$data' with points pt 7 ps 1.5 lc rgb '#0060ad'\n");

    fflush(gp);
    pclose(gp);
}


void regplot(float *x, float *y, int size) {
    FILE *gp = popen("gnuplot -persistent > /dev/null 2>&1", "w");

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

void hueplot(float *x, float *y, int *hue, int size) {
    FILE *gp = popen("gnuplot -persistent", "w");
    if (gp == NULL) {
        fprintf(stderr, "Could not open gnuplot\n");
        return;
    }

    fprintf(gp, "set title 'Scatter Plot with Hue'\n");
    fprintf(gp, "set xlabel 'X-axis'\n");
    fprintf(gp, "set ylabel 'Y-axis'\n");
    fprintf(gp, "set grid\n");
    fprintf(gp, "set key outside\n"); // legend outside

    // Find unique hue categories (assuming small number)
    int unique[10], uniqueCount = 0;
    for (int i = 0; i < size; i++) {
        int found = 0;
        for (int j = 0; j < uniqueCount; j++) {
            if (hue[i] == unique[j]) {
                found = 1;
                break;
            }
        }
        if (!found && uniqueCount < 10) {
            unique[uniqueCount++] = hue[i];
        }
    }

    // Define some line styles for up to 10 hues
    fprintf(gp, "set style line 1 lc rgb '#1f77b4' pt 7 ps 1.5\n"); // blue
    fprintf(gp, "set style line 2 lc rgb '#ff7f0e' pt 7 ps 1.5\n"); // orange
    fprintf(gp, "set style line 3 lc rgb '#2ca02c' pt 7 ps 1.5\n"); // green
    fprintf(gp, "set style line 4 lc rgb '#d62728' pt 7 ps 1.5\n"); // red
    fprintf(gp, "set style line 5 lc rgb '#9467bd' pt 7 ps 1.5\n"); // purple
    fprintf(gp, "set style line 6 lc rgb '#8c564b' pt 7 ps 1.5\n"); // brown
    fprintf(gp, "set style line 7 lc rgb '#e377c2' pt 7 ps 1.5\n"); // pink
    fprintf(gp, "set style line 8 lc rgb '#7f7f7f' pt 7 ps 1.5\n"); // gray
    fprintf(gp, "set style line 9 lc rgb '#bcbd22' pt 7 ps 1.5\n"); // olive
    fprintf(gp, "set style line 10 lc rgb '#17becf' pt 7 ps 1.5\n"); // cyan

    // Send each hue group data as a separate datablock
    for (int u = 0; u < uniqueCount; u++) {
        fprintf(gp, "$data%d << EOD\n", u+1);
        for (int i = 0; i < size; i++) {
            if (hue[i] == unique[u]) {
                fprintf(gp, "%f %f\n", x[i], y[i]);
            }
        }
        fprintf(gp, "EOD\n");
    }

    // Build plot command with all groups
    fprintf(gp, "plot ");
    for (int u = 0; u < uniqueCount; u++) {
        fprintf(gp, "$data%d with points ls %d title 'Hue %d'", u+1, u+1, unique[u]);
        if (u < uniqueCount - 1)
            fprintf(gp, ", ");
    }
    fprintf(gp, "\n");

    fflush(gp);
    pclose(gp);
}
