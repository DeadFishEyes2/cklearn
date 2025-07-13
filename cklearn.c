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

float computeMSE(dataFrame *df, float **centroids, float *nearest_centroid) {
    double total_error = 0.0;

    for (int i = 0; i < df->num_rows; i++) {
        int cluster = nearest_centroid[i];
        double distance = euclideanDistance(df->data[i], centroids[cluster], df->num_columns);
        total_error += distance * distance;
    }

    return total_error / df->num_rows;
}

float kmeans(dataFrame *df, int num_clusters, float **final_centroids){
    if (num_clusters == 0){
        printf("Cannot make 0 clusters");
        return INFINITY;
    }
    
    // Initialize the final_centroids with random values
    int i, j;
    float column_min, column_max;
    for (i = 0; i < num_clusters; i++){
        for (j = 0; j < df->num_columns; j++){
            column_min = getColumnMin(df, df->columns[j]);
            column_max = getColumnMax(df, df->columns[j]);
            //the code above forces a search for the column name, not ideal, could be fixed by having a funciton that gets the column by index instead
            //centroids are set to random values determined by the max and min elements from each column
            final_centroids[i][j] = column_min + ((float)rand() / RAND_MAX) * (column_max - column_min);
        }
    }

    //find the centroid positions
    float* nearest_centroid = (float*)malloc(df->num_rows * sizeof(float));
    int iter = 0;
    while (findNearestCentroid(df, final_centroids, nearest_centroid, num_clusters)){
        updateCentroids(df, final_centroids, nearest_centroid, num_clusters);
        iter++;
    }

    // Compute MSE
    float mse = computeMSE(df, final_centroids, nearest_centroid);

    // Clean up only the locally allocated memory
    free(nearest_centroid);
    
    return mse;
}

float** kmeansFit(dataFrame *df, int num_clusters, int num_iterations) {
    float **best_centroids = (float**)malloc(num_clusters * sizeof(float*));
    for (int i = 0; i < num_clusters; i++) {
        best_centroids[i] = (float*)malloc(df->num_columns * sizeof(float));
    }

    float best_mse = INFINITY;

    float **current_centroids = (float**)malloc(num_clusters * sizeof(float*));
    for (int i = 0; i < num_clusters; i++) {
        current_centroids[i] = (float*)malloc(df->num_columns * sizeof(float));
    }

    float mse;

    for (int iter = 0; iter < num_iterations; iter++) {
        mse = kmeans(df, num_clusters, current_centroids);
        printf("Iteration %d: MSE = %f\n", iter + 1, mse);

        if (mse < best_mse) {
            best_mse = mse;
            // Copy current centroids to best_centroids
            for (int i = 0; i < num_clusters; i++) {
                for (int j = 0; j < df->num_columns; j++) {
                    best_centroids[i][j] = current_centroids[i][j];
                }
            }
        }
    }

    for (int i = 0; i < num_clusters; i++)
        free(current_centroids[i]);
    free(current_centroids);

    printf("Best MSE after %d iterations: %f\n", num_iterations, best_mse);
    return best_centroids;
}

void kmeansPredict(dataFrame *df, int num_clusters, float **centroids){
    float *nearest_centroid = (float*)malloc((df->num_rows) * sizeof(float));
    findNearestCentroid(df, centroids, nearest_centroid, num_clusters);
    
    //Trasposing a row into a column
    int i;
    float** centroid_column = (float**)malloc(df->num_rows * sizeof(float*));
    for (i = 0; i < df->num_rows; i++){
        centroid_column[i] = (float*)malloc(sizeof(float));
        centroid_column[i][0] = nearest_centroid[i]; 
    }

    addColumns(df, 1, centroid_column, (char *[]){"Cluster"});
    
    // Free the allocated memory
    free(nearest_centroid);
    for (i = 0; i < df->num_rows; i++){
        free(centroid_column[i]);
    }
    free(centroid_column);
}

int main() {

    srand(time(NULL));

    dataFrame *df = readCSV("cluster_data.csv");
    printDataFrame(df);
    printf("\n\n");

    // normalizeColumnMinMax(df, "X");
    // normalizeColumnMinMax(df, "Y");
    float** centroids = kmeansFit(df, 3, 10);
    kmeansPredict(df, 3, centroids);
    printDataFrame(df);
    
    // Free the centroids memory
    for (int i = 0; i < 3; i++) {
        free(centroids[i]);
    }
    free(centroids);
    
    freeDataFrame(df);
    
    return 0;
}