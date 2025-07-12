#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "pandac.h"

float euclideanDistance(float *a, float *b, int num_dimensions) {
    double sum = 0.0;
    for (int i = 0; i < num_dimensions; i++) {
        double diff = a[i] - b[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

int findNearestCentroid(dataFrame *df, float **centroids, float *nearest_centroid, int num_clusters) {
    int i, j;
    int changed = 0;
    float min_distance, distance;

    for (i = 0; i < df->num_rows; i++) {
        float old = nearest_centroid[i];
        nearest_centroid[i] = 0; //asume the nearest centroid is the first one
        min_distance = euclideanDistance(df->data[i], centroids[0], df->num_columns);
        for (j = 1; j < num_clusters; j++) {
            distance = euclideanDistance(df->data[i], centroids[j], df->num_columns);
            if (distance < min_distance) {
                min_distance = distance;
                nearest_centroid[i] = j;
            }
        }
        if (nearest_centroid[i] != old) {
            changed = 1;
        }
    }

    return changed; // 1 if any assignment changed
}

void updateCentroids(dataFrame *df, float **centroids, float *nearest_centroid, int num_clusters) {
    int i, j;
    int *counts = (int*)calloc(num_clusters, sizeof(int)); // count points per cluster

    // Reset centroids to 0
    for (i = 0; i < num_clusters; i++) {
        for (j = 0; j < df->num_columns; j++) {
            centroids[i][j] = 0.0;
        }
    }

    // Sum all points in each cluster
    for (i = 0; i < df->num_rows; i++) {
        int cluster = nearest_centroid[i];
        counts[cluster]++;
        for (j = 0; j < df->num_columns; j++) {
            centroids[cluster][j] += df->data[i][j];
        }
    }

    // Divide to get mean
    for (i = 0; i < num_clusters; i++) {
        if (counts[i] == 0) continue; // avoid divide-by-zero if cluster is empty
        for (j = 0; j < df->num_columns; j++) {
            centroids[i][j] /= counts[i];
        }
    }

    free(counts);
}

void kmeans(dataFrame *df, int num_clusters, int num_iterations){
    if (num_clusters == 0){
        printf("Cannot make 0 clusters");
        return;
    }

    srand(time(NULL));
    
    //allocating space for the centroids and randomly initializing them
    float **centroids = (float**)malloc(num_clusters * sizeof(float*));
    int i, j;
    float column_min, column_max;
    for (i = 0; i < num_clusters; i++){
        centroids[i] = (float*)malloc((df->num_columns) * sizeof(float));
        for (j = 0; j < df->num_columns; j++){
            column_min = getColumnMin(df, df->columns[j]);
            column_max = getColumnMax(df, df->columns[j]);
            //the code above forces a search for the column name, not ideal, could be fixed by having a funciton that gets the column by index instead
            //centroids are set to random values determined by the max and min elements from each column
            centroids[i][j] = column_min + ((float)rand() / RAND_MAX) * (column_max - column_min);
        }
    }

    //find the centroid positions
    float* nearest_centroid = (float*)malloc(df->num_rows * sizeof(float));
    int iter = 0;
    while (findNearestCentroid(df, centroids, nearest_centroid, num_clusters) && (iter < num_iterations)){
        updateCentroids(df, centroids, nearest_centroid, num_clusters);
        iter++;
    }

    //Trasposing a row into a column
    float** centroid_column = (float**)malloc(df->num_rows * sizeof(float*));
    for (i = 0; i < df->num_rows; i++){
        centroid_column[i] = (float*)malloc(sizeof(float));
        centroid_column[i][0] = nearest_centroid[i]; 
    }

    addColumns(df, 1, centroid_column, (char *[]){"Cluster"});
}

int main() {

    dataFrame *df = readCSV("cluster_data.csv");
    printDataFrame(df);
    printf("\n\n");

    // normalizeColumnMinMax(df, "X");
    // normalizeColumnMinMax(df, "Y");
    kmeans(df, 3, 10);
    printDataFrame(df);
    
    return 0;
}