#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <float.h>
#include "pandac.h"
#include "plotc.h"

typedef struct {
    double distance;
    int index;
} DistanceInfo;

typedef struct {
    float *weights;
    int num_weights;
    float bias;
} LiniarRegression;

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

void dataFrameScatterplot(dataFrame *df, const char *X, const char *Y) {

    int x_col = getColumnIndex(df, X);
    int y_col = getColumnIndex(df, Y);

    if (x_col == -1 || y_col == -1)
        return;

    if (x_col >= df->num_columns || y_col >= df->num_columns) {
        fprintf(stderr, "Invalid column index\n");
        return;
    }

    int size = df->num_rows;
    float *x = malloc(size * sizeof(float));
    float *y = malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        x[i] = df->data[i][x_col];
        y[i] = df->data[i][y_col];
    }

    // Call existing scatter plot function
    scatterplot(x, y, size);

    free(x);
    free(y);
}

void dataFrameRegplot(dataFrame *df, const char *X, const char *Y) {

    int x_col = getColumnIndex(df, X);
    int y_col = getColumnIndex(df, Y);

    if (x_col == -1 || y_col == -1)
        return;

    if (x_col >= df->num_columns || y_col >= df->num_columns) {
        fprintf(stderr, "Invalid column index\n");
        return;
    }

    int size = df->num_rows;
    float *x = malloc(size * sizeof(float));
    float *y = malloc(size * sizeof(float));

    for (int i = 0; i < size; i++) {
        x[i] = df->data[i][x_col];
        y[i] = df->data[i][y_col];
    }

    // Call existing scatter plot function
    regplot(x, y, size);

    free(x);
    free(y);
}

void dataFrameHueplot(dataFrame *df, const char* X, const char *Y, const char *HUE) {

    int x_col = getColumnIndex(df, X);
    int y_col = getColumnIndex(df, Y);
    int hue_col = getColumnIndex(df, HUE);

    if (x_col >= df->num_columns || y_col >= df->num_columns || hue_col >= df->num_columns) {
        fprintf(stderr, "Invalid column index\n");
        return;
    }

    int size = df->num_rows;
    float *x = malloc(size * sizeof(float));
    float *y = malloc(size * sizeof(float));
    int *hue = malloc(size * sizeof(int));

    for (int i = 0; i < size; i++) {
        x[i] = df->data[i][x_col];
        y[i] = df->data[i][y_col];
        hue[i] = (int)df->data[i][hue_col];  // Assuming hue is stored as int
    }

    // Call existing scatter plot with hue function
    hueplot(x, y, hue, size);

    free(x);
    free(y);
    free(hue);
}

int compareDistanceInfo(const void *a, const void *b) {
    DistanceInfo *da = (DistanceInfo*)a;
    DistanceInfo *db = (DistanceInfo*)b;
    if (da->distance < db->distance) return -1;
    if (da->distance > db->distance) return 1;
    return 0;
}

void LocalOutlierFactor(dataFrame *df, int k){
    int n = df->num_rows;
    if (k >= n) {
        fprintf(stderr, "k must be smaller than the number of rows for LOF.\n");
        return;
    }

    double *k_distances = malloc(n * sizeof(double));
    int **neighbors_indices = malloc(n * sizeof(int*));

    for (int i = 0; i < n; i++) {
        neighbors_indices[i] = malloc(k * sizeof(int));

        // Create an array of distances from point 'i' to all other points.
        // We use (n-1) because we exclude the point itself.
        DistanceInfo *distances = malloc((n - 1) * sizeof(DistanceInfo));
        int d_idx = 0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            distances[d_idx].distance = euclideanDistance(df->data[i], df->data[j], df->num_columns);
            distances[d_idx].index = j;
            d_idx++;
        }

        // Sort the distances to find the nearest neighbors.
        qsort(distances, n - 1, sizeof(DistanceInfo), compareDistanceInfo);

        // The k-distance is the distance to the k-th neighbor (at index k-1).
        k_distances[i] = distances[k - 1].distance;

        // Store the indices of the k-nearest neighbors.
        for (int j = 0; j < k; j++) {
            neighbors_indices[i][j] = distances[j].index;
        }

        free(distances);
    }

    double *lrd = calloc(n, sizeof(double));
    for (int i = 0; i < n; i++) {
        double reach_dist_sum = 0.0;
        // For each neighbor of point 'i'...
        for (int j = 0; j < k; j++) {
            int neighbor_idx = neighbors_indices[i][j];
            
            // Get the actual distance from 'i' to its neighbor.
            double dist_to_neighbor = euclideanDistance(df->data[i], df->data[neighbor_idx], df->num_columns);
            
            // Reachability distance is the max of (actual distance) and (k-distance of the neighbor).
            double reach_dist = fmax(dist_to_neighbor, k_distances[neighbor_idx]);
            reach_dist_sum += reach_dist;
        }
        // LRD is the inverse of the average reachability distance.
        lrd[i] = (reach_dist_sum > 0) ? (k / reach_dist_sum) : 0.0;
    }

    float **lof_col_data = malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        lof_col_data[i] = malloc(sizeof(float));
        double lrd_ratio_sum = 0.0;
        // For each neighbor of point 'i'...
        for (int j = 0; j < k; j++) {
            int neighbor_idx = neighbors_indices[i][j];
            // Sum the ratios of the neighbor's LRD to point 'i's LRD.
            // Add a small epsilon to avoid division by zero for very dense points.
            if (lrd[i] > 1e-9) { 
                lrd_ratio_sum += lrd[neighbor_idx] / lrd[i];
            }
        }
        // The LOF is the average of these ratios.
        lof_col_data[i][0] = (float)(lrd_ratio_sum / k);
    }

    addColumns(df, 1, lof_col_data, (char*[]){"LOF"});

    for (int i = 0; i < n; i++) {
        free(neighbors_indices[i]);
        free(lof_col_data[i]);
    }
    free(neighbors_indices);
    free(lof_col_data);
    free(k_distances);
    free(lrd);
}

float randf(void){
    return (float)rand()/RAND_MAX;
}

float liniarMSE(dataFrame *train, LiniarRegression *model){
    float d, y, result;
    result = 0;

    int i, j;
    for (i = 0; i < train->num_rows; i++){
        y = 0;
        for (j = 0; j < train->num_columns-1; j++){
            y += train->data[i][j]*(model->weights[j]);
        }
        y += model->bias;
        d = y - train->data[i][train->num_columns-1];
        result += d*d;
    }
    
    return result;
}

void initializeModel(LiniarRegression *model, int num_weights){
    
    model->weights = (float*)malloc(num_weights*sizeof(float));

    model->num_weights = num_weights;
    
    int i;
    for (i = 0; i < num_weights; i++){
        model->weights[i] = randf()*10.0f;
    }
    model->bias = randf()*10.0f;
}

void freeLiniarModel(LiniarRegression *model){
    free(model->weights);
    free(model);
}

LiniarRegression *liniarRegressionFit(dataFrame *train, float epsilon, float learning_rate){
    
    LiniarRegression *model = (LiniarRegression*)malloc(sizeof(LiniarRegression));
    initializeModel(model, train->num_columns-1);

    int i;
    float *dcost = (float*)malloc(train->num_columns*sizeof(float));

    float c;
    int num_iterations;
    for (num_iterations = 0; num_iterations < 100000; num_iterations++){
        c = liniarMSE(train, model);
        for (i = 0; i < train->num_columns-1; i++){
            model->weights[i] += epsilon;
            dcost[i] = (liniarMSE(train, model) - c)/epsilon;
            model->weights[i] -= epsilon;
        }
        model->bias += epsilon;
        dcost[train->num_columns-1] = (liniarMSE(train, model) - c)/epsilon;
        model->bias -= epsilon;
        
        for (i = 0; i < train->num_columns-1; i++){
            model->weights[i] -= learning_rate*dcost[i];
        }
        model->bias -= learning_rate*dcost[train->num_columns-1];
    }

    free(dcost);

    return model;
}

void liniarRegressionPredict(dataFrame *test, LiniarRegression *model){
    
    float *y = (float*)malloc(test->num_rows*sizeof(float));

    int i, j;
    for (i = 0; i < test->num_rows; i++){
        y[i] = 0;
        for (j = 0; j < model->num_weights; j++){
            y[i] += test->data[i][j] * model->weights[j];
        }
        y[i] += model->bias;
    }

    //Trasposing a row into a column
    float** y_column = (float**)malloc(test->num_rows * sizeof(float*));
    for (i = 0; i < test->num_rows; i++){
        y_column[i] = (float*)malloc(sizeof(float));
        y_column[i][0] = y[i]; 
    }

    addColumns(test, 1, y_column, (char *[]){"prediction"});
    
    // Free the allocated memory
    free(y);
    for (i = 0; i < test->num_rows; i++){
        free(y_column[i]);
    }
    free(y_column);

}

int main() {

    srand(time(NULL));

    dataFrame *train = readCSV("train.csv");
    dataFrame *test = readCSV("test.csv");

    LiniarRegression *model = liniarRegressionFit(train, 1e-3, 1e-4);
    liniarRegressionPredict(test, model);

    printDataFrame(test);

    freeLiniarModel(model);
    freeDataFrame(train);
    freeDataFrame(test);

    return 0;
}