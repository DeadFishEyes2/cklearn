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

void trim(char *str) {
    // Utility: trim trailing newline or whitespace
    size_t len = strlen(str);
    while (len > 0 && (str[len - 1] == '\n' || str[len - 1] == '\r' || str[len - 1] == ' ')) {
        str[--len] = '\0';
    }
}

int splitCSVLine(char *line, char ***out_tokens) {
    // Utility: split a CSV line safely
    int count = 0;
    int capacity = 10;
    char **tokens = malloc(capacity * sizeof(char *));
    if (!tokens) return -1;

    char *start = line;
    char *end;
    while (1) {
        end = strchr(start, ',');
        if (!end) { // last token
            trim(start);
            tokens[count++] = strdup(start);
            break;
        }

        *end = '\0'; // null-terminate token
        trim(start);
        tokens[count++] = strdup(start);
        start = end + 1;

        // grow array if needed
        if (count >= capacity) {
            capacity *= 2;
            tokens = realloc(tokens, capacity * sizeof(char *));
            if (!tokens) return -1;
        }
    }

    *out_tokens = tokens;
    return count; // return number of tokens
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
    char **columns = NULL;
    float **data = NULL;

    // Read header
    if (!fgets(line, sizeof(line), fp)) {
        fprintf(stderr, "Failed to read header\n");
        fclose(fp);
        return NULL;
    }
    trim(line);
    num_columns = splitCSVLine(line, &columns);

    // Read rows
    while (fgets(line, sizeof(line), fp)) {
        trim(line);

        // Grow rows
        data = realloc(data, (num_rows + 1) * sizeof(float *));
        data[num_rows] = malloc(num_columns * sizeof(float));

        char **row_tokens = NULL;
        int token_count = splitCSVLine(line, &row_tokens);

        // Fill row
        for (int i = 0; i < num_columns; i++) {
            if (i < token_count) {
                if (strlen(row_tokens[i]) == 0 || strcasecmp(row_tokens[i], "NaN") == 0) {
                    data[num_rows][i] = NAN; // missing value
                } else {
                    data[num_rows][i] = strtof(row_tokens[i], NULL);
                }
            } else {
                data[num_rows][i] = NAN; // missing trailing column
            }
            free(row_tokens[i]);
        }
        free(row_tokens);
        num_rows++;
    }
    fclose(fp);

    // Build dataFrame
    dataFrame *df = malloc(sizeof(dataFrame));
    df->num_rows = num_rows;
    df->num_columns = num_columns;
    df->data = data;
    df->columns = columns;

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
    int i, num_nans = 0;
    double sum = 0;
    for (i = 0; i < df->num_rows; i++){
        if (isnan(df->data[i][column_index])){
            num_nans++;
        } else {
            sum += df->data[i][column_index];
        }
    }
    return sum/(df->num_rows - num_nans);
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

    int num_nans = 0;
    float mean = getColumnMean(df, column_name);
    float sum_sq_diff = 0.0;

    for (int i = 0; i < df->num_rows; i++) {
        if (isnan(df->data[i][col_idx])){
            num_nans++;
        } else {
            float diff = df->data[i][col_idx] - mean;
            sum_sq_diff += diff * diff;
        }
    }

    return sqrt(sum_sq_diff / (df->num_rows - num_nans));
}

void normalizeColumnMinMax(dataFrame *df, char *column_name) {
    int col_idx = getColumnIndex(df, column_name);
    if (col_idx == -1) {
        fprintf(stderr, "Column not found\n");
        return;
    }
    
    // Find min and max
    float min = getColumnMin(df, column_name);
    float max = getColumnMax(df, column_name);
    
    // Normalize
    for (int i = 0; i < df->num_rows; i++) {
        df->data[i][col_idx] = (df->data[i][col_idx] - min) / (max - min);
    }
}

void normalizeColumnZScore(dataFrame *df, char *column_name) {
    int col_idx = getColumnIndex(df, column_name);
    if (col_idx == -1) {
        fprintf(stderr, "Column not found\n");
        return;
    }
    
    float mean = getColumnMean(df, column_name);
    float std = getColumnStd(df, column_name);
    
    if (std == 0) {
        fprintf(stderr, "Standard deviation is zero, cannot normalize\n");
        return;
    }
    
    // Normalize
    for (int i = 0; i < df->num_rows; i++) {
        df->data[i][col_idx] = (df->data[i][col_idx] - mean) / std;
    }
}

dataFrame* selectColumns(dataFrame *df, int num_selected_columns,  char **selected_columns){

    int i, j;

    //allocate data container
    dataFrame* selected_df = (dataFrame*)malloc(sizeof(dataFrame));
    if (selected_df == NULL){
        fprintf(stderr, "Failed to allocate memory for selected dataFrame\n");
        return NULL;
    }
    selected_df->num_rows = df->num_rows;
    selected_df->num_columns = num_selected_columns;
    selected_df->data = (float**)malloc((selected_df->num_rows) * sizeof(float*));
    if (selected_df->data == NULL){
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
        return NULL;
    }
    selected_df->columns = (char**)malloc(num_selected_columns * sizeof(char*));
    if (selected_df->columns == NULL){
        free(selected_df->data);
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
    }
    for (i = 0; i < selected_df->num_rows; i++){
        selected_df->data[i] = (float*)malloc(num_selected_columns * sizeof(float));
        if (selected_df->data[i] == NULL){ //in case of a failed allocation, the selected_df is freed from memory
            for (j = 0; j < i; j++){
                free(selected_df->data[j]);
            }
            free(selected_df->columns);
            free(selected_df);
            fprintf(stderr, "Failed to allocate memory\n");
            return NULL;
        }
    }

    //copy selected columns into the new dataFrame
    int* column_indexes = (int*)malloc(num_selected_columns*sizeof(int));
    for (j = 0; j < num_selected_columns; j++){
        column_indexes[j] = getColumnIndex(df, df->columns[j]);
        selected_df->columns[j] = strdup(df->columns[column_indexes[j]]);
    }
    for (i = 0; i < selected_df->num_rows; i++){
        for (j = 0; j < selected_df->num_columns; j++){
            selected_df->data[i][j] = df->data[i][column_indexes[j]];
        }
    }
    return selected_df;
}

dataFrame* selectTopRows(dataFrame *df, int num_top_rows){
    
    int i, j;

    //allocate data container
    dataFrame* selected_df = (dataFrame*)malloc(sizeof(dataFrame));
    if (selected_df == NULL){
        fprintf(stderr, "Failed to allocate memory for selected dataFrame\n");
        return NULL;
    }
    selected_df->num_rows = num_top_rows;
    selected_df->num_columns = df->num_columns;
    selected_df->data = (float**)malloc((selected_df->num_rows) * sizeof(float*));
    if (selected_df->data == NULL){
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
        return NULL;
    }
    selected_df->columns = (char**)malloc((selected_df->num_columns) * sizeof(char*));
    if (selected_df->columns == NULL){
        free(selected_df->data);
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
    }
    for (i = 0; i < selected_df->num_rows; i++){
        selected_df->data[i] = (float*)malloc((selected_df->num_columns) * sizeof(float));
        if (selected_df->data[i] == NULL){ //in case of a failed allocation, the selected_df is freed from memory
            for (j = 0; j < i; j++){
                free(selected_df->data[j]);
            }
            free(selected_df->columns);
            free(selected_df);
            fprintf(stderr, "Failed to allocate memory\n");
            return NULL;
        }
    }

    //copy the column names
    for (j = 0; j < selected_df->num_columns; j++)
        selected_df->columns[j] = strdup(df->columns[j]);

    //copy the first rows from the original dataFrame to the selected dataFrame
    for (i = 0; i < selected_df->num_rows; i++){
        for (j = 0; j < selected_df->num_columns; j++){
            selected_df->data[i][j] = df->data[i][j];
        }
    }

    return selected_df;
}

dataFrame* selectBottomRows(dataFrame *df, int num_bottom_rows){
    int i, j;

    //allocate data container
    dataFrame* selected_df = (dataFrame*)malloc(sizeof(dataFrame));
    if (selected_df == NULL){
        fprintf(stderr, "Failed to allocate memory for selected dataFrame\n");
        return NULL;
    }
    selected_df->num_rows = num_bottom_rows;
    selected_df->num_columns = df->num_columns;
    selected_df->data = (float**)malloc((selected_df->num_rows) * sizeof(float*));
    if (selected_df->data == NULL){
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
        return NULL;
    }
    selected_df->columns = (char**)malloc((selected_df->num_columns) * sizeof(char*));
    if (selected_df->columns == NULL){
        free(selected_df->data);
        free(selected_df);
        fprintf(stderr, "Failed to allocate memory\n");
    }
    for (i = 0; i < selected_df->num_rows; i++){
        selected_df->data[i] = (float*)malloc((selected_df->num_columns) * sizeof(float));
        if (selected_df->data[i] == NULL){ //in case of a failed allocation, the selected_df is freed from memory
            for (j = 0; j < i; j++){
                free(selected_df->data[j]);
            }
            free(selected_df->columns);
            free(selected_df);
            fprintf(stderr, "Failed to allocate memory\n");
            return NULL;
        }
    }

    //copy the column names
    for (j = 0; j < selected_df->num_columns; j++)
        selected_df->columns[j] = strdup(df->columns[j]);

    //copy the first rows from the original dataFrame to the selected dataFrame
    int num_skipped_rows = (df->num_rows) - (selected_df->num_rows);
    for (i = 0; i < selected_df->num_rows; i++){
        for (j = 0; j < selected_df->num_columns; j++){
            selected_df->data[i][j] = df->data[i+num_skipped_rows][j];
        }
    }
    return selected_df;
}

void addNans(dataFrame* small_df, dataFrame* big_df){
    int diff = big_df->num_rows - small_df->num_rows;

    //allocating memory for the NaNs
    float **nan_buffer = (float**)malloc(diff * sizeof(float*));
    if (nan_buffer == NULL){
        fprintf(stderr, "Memory error when allocatin NaNs");
        exit(EXIT_FAILURE);
    }
    int num_columns = small_df->num_columns;
    int i, j;
    for (i = 0; i < diff; i++){
        nan_buffer[i] = (float*)malloc(num_columns * sizeof(float));
        if (nan_buffer[i] == NULL){
            for (j = 0; j < i; j++){
                free(nan_buffer[j]);
            }
            free(nan_buffer);
            fprintf(stderr, "Memory error when allocatin NaNs");
            exit(EXIT_FAILURE);
        }
        
    }

    //adding NaNs to the nan_buffer
    for (i = 0; i < diff; i++){
        for (j = 0; j < small_df->num_columns; j++){
            nan_buffer[i][j] = NAN;
        }
    }

    addRows(small_df, diff, nan_buffer);
}

void equalizeRows(dataFrame *df1, dataFrame *df2, char* method){
    /*
        Methods for equalizing rows
        "add" - 'a'
        Adds NaNs to the dataFrame with fewer rows to match the dimensions of the bigger one
        "cut" - 'c'
        Removes the last rows of the bigger dataFrame to match the dimesions of the smaller one
    */
    
    if (method[0] == 'a'){
        if (df1->num_rows < df2->num_rows){
            addNans(df1, df2);
        } else {
            addNans(df2, df1);
        }
    } else {

    }
}

dataFrame* innerJoin(dataFrame *df1, dataFrame *df2) {
    int i, j, k;

    // 1. Find common columns
    int *df1_common_idx = malloc(df1->num_columns * sizeof(int));
    int *df2_common_idx = malloc(df2->num_columns * sizeof(int));
    int num_common_columns = 0;

    for (i = 0; i < df1->num_columns; i++) {
        for (j = 0; j < df2->num_columns; j++) {
            if (strcmp(df1->columns[i], df2->columns[j]) == 0) {
                df1_common_idx[num_common_columns] = i;
                df2_common_idx[num_common_columns] = j;
                num_common_columns++;
            }
        }
    }

    if (num_common_columns == 0) {
        fprintf(stderr, "No common columns found for join\n");
        return NULL;
    }

    // 2. Determine result schema
    int result_columns = df1->num_columns + (df2->num_columns - num_common_columns);
    char **result_col_names = malloc(result_columns * sizeof(char*));

    // Copy df1 columns
    for (i = 0; i < df1->num_columns; i++) {
        result_col_names[i] = strdup(df1->columns[i]);
    }

    // Copy df2 non-common columns
    int offset = df1->num_columns;
    for (j = 0; j < df2->num_columns; j++) {
        int is_common = 0;
        for (k = 0; k < num_common_columns; k++) {
            if (df2_common_idx[k] == j) {
                is_common = 1;
                break;
            }
        }
        if (!is_common) {
            result_col_names[offset++] = strdup(df2->columns[j]);
        }
    }

    // 3. Compare rows and build result data
    float **result_data = NULL;
    int result_rows = 0;

    for (i = 0; i < df1->num_rows; i++) {
        for (j = 0; j < df2->num_rows; j++) {
            int match = 1;
            for (k = 0; k < num_common_columns; k++) {
                float val1 = df1->data[i][df1_common_idx[k]];
                float val2 = df2->data[j][df2_common_idx[k]];
                if (val1 != val2 && !(isnan(val1) && isnan(val2))) {
                    match = 0;
                    break;
                }
            }

            if (match) {
                result_data = realloc(result_data, (result_rows + 1) * sizeof(float*));
                result_data[result_rows] = malloc(result_columns * sizeof(float));

                // Copy df1 row
                for (k = 0; k < df1->num_columns; k++) {
                    result_data[result_rows][k] = df1->data[i][k];
                }

                // Copy df2 non-common columns
                int col_offset = df1->num_columns;
                for (k = 0; k < df2->num_columns; k++) {
                    int is_common = 0;
                    for (int m = 0; m < num_common_columns; m++) {
                        if (df2_common_idx[m] == k) {
                            is_common = 1;
                            break;
                        }
                    }
                    if (!is_common) {
                        result_data[result_rows][col_offset++] = df2->data[j][k];
                    }
                }

                result_rows++;
            }
        }
    }

    free(df1_common_idx);
    free(df2_common_idx);

    // 4. Build dataframe
    dataFrame *result_df = createDataFrame(result_columns, result_rows, result_data, result_col_names);

    // Free temporary allocations
    for (i = 0; i < result_rows; i++) free(result_data[i]);
    free(result_data);
    for (i = 0; i < result_columns; i++) free(result_col_names[i]);
    free(result_col_names);

    return result_df;
}


