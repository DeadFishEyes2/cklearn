#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef struct dataFrame{
    int num_columns;
    int num_rows;
    float **data;
    char **columns;
} dataFrame;

dataFrame* createDataFrame (int num_columns, int num_rows, float** data, char **columns){
    dataFrame* df = (dataFrame*)malloc(sizeof(dataFrame));
    if (df == NULL) {
        fprintf(stderr, "Memory allocation failed for dataFrame\n");
        return NULL;
    }

    df->num_columns = num_columns;
    df->num_rows = num_rows;

    // Allocate and copy column names
    df->columns = (char**)malloc(num_columns * sizeof(char*));
    if (df->columns == NULL) {
        fprintf(stderr, "Memory allocation failed for columns\n");
        free(df);
        return NULL;
    }
    for (int i = 0; i < num_columns; ++i) {
        df->columns[i] = strdup(columns[i]); // copies the string
    }

    // Allocate and copy data
    df->data = (float**)malloc(num_rows * sizeof(float*));
    if (df->data == NULL) {
        fprintf(stderr, "Memory allocation failed for data\n");
        for (int i = 0; i < num_columns; ++i) 
            free(df->columns[i]);
        free(df->columns);
        free(df);
        return NULL;
    }

    for (int i = 0; i < num_rows; ++i) {
        df->data[i] = (float*)malloc(num_columns * sizeof(float));
        for (int j = 0; j < num_columns; ++j) {
            df->data[i][j] = data[i][j]; // copy each element
        }
    }

    return df;
}

void freeDataFrame(dataFrame* df) {
    for (int i = 0; i < df->num_rows; ++i) {
        free(df->data[i]);
    }
    free(df->data);

    for (int i = 0; i < df->num_columns; ++i) {
        free(df->columns[i]);
    }
    free(df->columns);

    free(df);
}

dataFrame* readCSV(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Could not open file %s\n", filename);
        return NULL;
    }

    char line[1024];
    int num_columns = 0;
    int num_rows = 0;

    // Read header
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "Failed to read header\n");
        fclose(fp);
        return NULL;
    }

    // Count columns
    char *token = strtok(line, ",\n");
    char **columns = NULL;
    while (token) {
        columns = (char**)realloc(columns, (num_columns + 1) * sizeof(char*));
        columns[num_columns] = strdup(token);
        num_columns++;
        token = strtok(NULL, ",\n");
    }

    // Read rows
    float **data = NULL;
    while (fgets(line, sizeof(line), fp)) {
        data = (float**)realloc(data, (num_rows + 1) * sizeof(float*));
        data[num_rows] = (float*)malloc(num_columns * sizeof(float));

        token = strtok(line, ",\n");
        for (int i = 0; i < num_columns; i++) {
            if (token) {
                data[num_rows][i] = strtof(token, NULL); // convert string to float
                token = strtok(NULL, ",\n");
            } else {
                data[num_rows][i] = 0.0; // fill missing values with 0
            }
        }
        num_rows++;
    }

    fclose(fp);

    dataFrame *df = createDataFrame(num_columns, num_rows, data, columns);

    // Free temporary arrays (data + columns were copied into df)
    for (int i = 0; i < num_rows; i++) free(data[i]);
    free(data);
    for (int i = 0; i < num_columns; i++) free(columns[i]);
    free(columns);

    return df;
}

void printDataFrame(dataFrame *df) {
    int col_width = 12; // Set width for each column

    // Print column headers
    for (int i = 0; i < df->num_columns; i++) {
        printf("%-*s", col_width, df->columns[i]); // Left-align
    }
    printf("\n");

    // Print rows
    for (int i = 0; i < df->num_rows; i++) {
        for (int j = 0; j < df->num_columns; j++) {
            printf("%-*.2f", col_width, df->data[i][j]); // Left-align numbers (to right-align "%*-.2f")
        }
        printf("\n");
    }
}

void addRows(dataFrame* df, int num_new_rows, float **data){
    int prev_rows = df->num_rows;
    int i, j;
    
    //Resize the data container
    df->num_rows += num_new_rows;
    float **temp = (float**)realloc(df->data, df->num_rows * sizeof(float*)); //using a temp pointer so data isn't lost in case of a failed allocation
    if (!temp) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    df->data = temp;
    for (i = 0; i < num_new_rows; i++)
        df->data[i + prev_rows] = (float*)malloc(df->num_columns * sizeof(float));
    
    //Copy the new rows
    for (i = 0; i < num_new_rows; i++){
        
        for (j = 0; j < df->num_columns; j++){
            df->data[i+prev_rows][j] = data[i][j];
        }    
    }
}

void addColumns(dataFrame* df, int num_new_columns, float** data, char **columns){
    int prev_columns = df->num_columns;
    int i, j;
    
    //Resize the data container
    df->num_columns += num_new_columns;
    char **temp_columns = (char**)realloc(df->columns, df->num_columns * sizeof(char*));
    if (!temp_columns) {
        fprintf(stderr, "Failed to allocate memory\n");
        exit(EXIT_FAILURE);
    }
    df->columns = temp_columns;

    //Copying the name of the new columns into the dataFrame
    for (i = 0; i < num_new_columns; i++) {
        df->columns[prev_columns + i] = strdup(columns[i]);
        if (!df->columns[prev_columns + i]) {
            fprintf(stderr, "Failed to allocate memory for column name\n");
            exit(EXIT_FAILURE);
        }
    }

    //Copying the data from the new columns into the dataFrame
    for (i = 0; i < df->num_rows; i++) {
        float *temp_row = (float*)realloc(df->data[i], df->num_columns * sizeof(float));
        if (!temp_row) {
            fprintf(stderr, "Failed to allocate memory for row %d\n", i);
            exit(EXIT_FAILURE);
        }
        df->data[i] = temp_row;

        for (j = 0; j < num_new_columns; j++) {
            df->data[i][prev_columns + j] = data[i][j];
        }
    }
}

int getColumnIndex(dataFrame *df, const char *column_name){
    int i;
    for (i = 0; i < df->num_columns; i++){
        if (strcmp(df->columns[i], column_name) == 0)
            return i;
    }
    return -1;
}

float getColumnMean(dataFrame* df, const char *column_name){
    int column_index = getColumnIndex(df, column_name);
    int i;
    double sum = 0;
    for (i = 0; i < df->num_rows; i++){
        sum += df->data[i][column_index];
    }
    return sum/(df->num_rows);
}

float getColumnMax(dataFrame* df, const char *column_name){
    int column_index = getColumnIndex(df, column_name);
    int i;
    float max = df->data[0][column_index];
    for (i = 1; i < df->num_rows; i++){
        if(df->data[i][column_index] > max)
            max = df->data[i][column_index];
    }
    return max;
}

float getColumnMin(dataFrame* df, const char *column_name){
    int column_index = getColumnIndex(df, column_name);
    int i;
    float min = df->data[0][column_index];
    for (i = 1; i < df->num_rows; i++){
        if(df->data[i][column_index] < min)
            min = df->data[i][column_index];
    }
    return min;
}

float getColumnStd(dataFrame *df, const char *column_name) {
    int col_idx = getColumnIndex(df, column_name);
    if (col_idx == -1) {
        fprintf(stderr, "Column not found: %s\n", column_name);
        exit(EXIT_FAILURE);
    }

    float mean = getColumnMean(df, column_name);
    float sum_sq_diff = 0.0;

    for (int i = 0; i < df->num_rows; i++) {
        float diff = df->data[i][col_idx] - mean;
        sum_sq_diff += diff * diff;
    }

    return sqrt(sum_sq_diff / df->num_rows);
}

int main() {
    dataFrame *df1 = readCSV("data1.csv");
    if (df1) {
        printDataFrame(df1);
    }
    printf("\n\n");
    dataFrame *df2 = readCSV("data2.csv");
    if (df2) {
        printDataFrame(df2);
    }
    printf("\n\n");
    addColumns(df1, df2->num_columns, df2->data, df2->columns);
    if (df1) {
        printDataFrame(df1);
    }
    printf("\n\n");
    printf("%f\n%f\n%f\n%f", getColumnMean(df1, "income"), getColumnMin(df1, "income"), getColumnMax(df1, "income"), getColumnStd(df1, "id"));
    return 0;
}
