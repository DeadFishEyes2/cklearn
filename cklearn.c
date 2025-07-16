#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include "pandac.h"

float euclideanDistance(float *a, float *b, int num_dimensions) {
    double sum_sq_diff = 0.0;
    int valid_dimensions_count = 0;

    for (int i = 0; i < num_dimensions; i++) {
        // Only include dimensions where both values are valid (not NaN)
        if (!isnan(a[i]) && !isnan(b[i])) {
            double diff = a[i] - b[i];
            sum_sq_diff += diff * diff;
            valid_dimensions_count++;
        } else {
            return FLT_MAX/2;
        }
    }

    // If no common non-NaN dimensions were found, return a very large distance.
    // This prevents division by zero and correctly indicates that the points
    // cannot be meaningfully compared by this metric.
    if (valid_dimensions_count == 0) {
        return FLT_MAX; // Use FLT_MAX from float.h
    }

    // Normalize the sum of squared differences by the number of valid dimensions.
    return sqrt(sum_sq_diff / valid_dimensions_count);
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

void insertSortedNeighbor(float *neighbor_indices, float *neighbor_distances, int k, int candidate_index, float candidate_distance) {
    int i;

    // If the candidate is farther than the farthest neighbor, ignore it
    if (candidate_distance >= neighbor_distances[k - 1]) {
        return;
    }

    // Find position where candidate should be inserted
    for (i = k - 2; i >= 0 && neighbor_distances[i] > candidate_distance; i--) {
        // Shift neighbor to the right
        neighbor_distances[i + 1] = neighbor_distances[i];
        neighbor_indices[i + 1] = neighbor_indices[i];
    }

    // Insert the new neighbor (cast index to float)
    neighbor_distances[i + 1] = candidate_distance;
    neighbor_indices[i + 1] = (float)candidate_index;
}

dataFrame *KNN(dataFrame *df, int k, int num_features, char **feature_names) {

    if (k > df->num_rows){
        printf("More neighbors than rows");
        return NULL;
    }

    // Reduce dimensionality to selected features
    dataFrame *reduced_df = selectColumns(df, num_features, feature_names);

    int num_rows = reduced_df->num_rows;

    // Allocate float arrays for neighbors and distances
    float **neighbors = (float **)malloc(num_rows * sizeof(float *));
    float **distances = (float **)malloc(num_rows * sizeof(float *));

    int i, j;
    for (i = 0; i < num_rows; i++) {
        neighbors[i] = (float *)malloc(k * sizeof(float));
        distances[i] = (float *)malloc(k * sizeof(float));

        // Initialize all distances to a large value
        for (j = 0; j < k; j++) {
            neighbors[i][j] = -1.0f;          // placeholder for invalid index
            distances[i][j] = FLT_MAX;        // initialize distances to "infinity"
        }
    }

    // Compute pairwise distances
    for (i = 0; i < num_rows; i++) {
        for (j = i + 1; j < num_rows; j++) {
            float d = euclideanDistance(reduced_df->data[i], reduced_df->data[j], num_features);

            // Update neighbor lists for both points
            insertSortedNeighbor(neighbors[i], distances[i], k, j, d);
            insertSortedNeighbor(neighbors[j], distances[j], k, i, d);
        }
    }

    // Build column names: "neighbor 1", "neighbor 2", ...
    char **column_names = (char **)malloc(k * sizeof(char *));
    for (i = 0; i < k; i++) {
        column_names[i] = (char *)malloc(20 * sizeof(char));
        snprintf(column_names[i], 20, "neighbor %d", i + 1);
    }

    // Create dataFrame with neighbors
    dataFrame *neighbor_df = createDataFrame(k, df->num_rows, neighbors, column_names);

    // Clean up
    for (i = 0; i < num_rows; i++) {
        free(distances[i]);
        free(neighbors[i]); // free because createDataFrame copied data
    }
    free(distances);
    free(neighbors);

    for (i = 0; i < k; i++) {
        free(column_names[i]);
    }
    free(column_names);

    freeDataFrame(reduced_df);

    return neighbor_df;
}

float nnMean(dataFrame *df, dataFrame *nn_df, int row, int column) {
    float sum = 0.0f;
    int count = 0;

    // Iterate over neighbors
    for (int i = 0; i < nn_df->num_columns; i++) {
        int neighbor_idx = (int)nn_df->data[row][i];

        // Skip invalid neighbors
        if (neighbor_idx < 0 || neighbor_idx >= df->num_rows) continue;
        if (neighbor_idx == row) continue; // Skip self

        float val = df->data[neighbor_idx][column];
        if (!isnan(val)) {
            sum += val;
            count++;
        }
    }

    // If no valid neighbors, return NaN
    return (count > 0) ? (sum / count) : NAN;
}

void fillNaN(dataFrame *df, dataFrame *nn_df){
    int num_rows = df->num_rows;
    int num_columns = df->num_columns;
    
    int i, j;
    for (i = 0; i < num_rows; i++){
        for (j = 0; j < num_columns; j++)
            if (isnan(df->data[i][j])){
                df->data[i][j] = nnMean(df, nn_df, i, j);
            }
    }
}

int main() {

    srand(time(NULL));

    dataFrame *df = readCSV("cluster_data.csv");
    printDataFrameWithIndex(df);
    printf("\n\n");

    //normalizeColumnMinMax(df, "X");
    //normalizeColumnMinMax(df, "Y");
    dataFrame* neighbor_df = KNN(df, 4, 4, (char*[]){"X", "Y", "Z", "W"});
    printDataFrameWithIndex(neighbor_df);
    printf("\n\n");

    fillNaN(df, neighbor_df);
    printDataFrameWithIndex(df);
    printf("\n\n");
    
    freeDataFrame(neighbor_df);
    freeDataFrame(df);
    
    
    return 0;
}