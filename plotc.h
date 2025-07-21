#ifndef PLOT_H
#define PLOT_H

#include <stdio.h>

// Function that creates a simple 2d scatter plot
void scatterplot(float *x, float *y, int size);

// Function to plot a scatter plot with linear regression
void regplot(float *x, float *y, int size);

// Function to plot a scatter plot with different hues (categories)
void hueplot(float *x, float *y, int *hue, int size);

#endif // PLOT_H
