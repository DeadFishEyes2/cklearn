#ifndef PANDAC_H
#define PANDAC_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

// Define the structure for the DataFrame
typedef struct dataFrame {
    int num_columns;
    int num_rows;
    float **data;
    char **columns;
} dataFrame;

/**
 * @brief Creates and allocates memory for a new dataFrame.
 * @param num_columns The number of columns.
 * @param num_rows The number of rows.
 * @param data A 2D array of float data.
 * @param columns An array of strings for column names.
 * @return A pointer to the newly created dataFrame, or NULL on failure.
 */
dataFrame* createDataFrame(int num_columns, int num_rows, float** data, char **columns);

/**
 * @brief Frees all memory associated with a dataFrame.
 * @param df The dataFrame to free.
 */
void freeDataFrame(dataFrame* df);

/**
 * @brief Utility function to trim trailing whitespace and newlines from a string.
 * @param str The string to trim.
 */
void trim(char *str);

/**
 * @brief Utility function to split a single line of a CSV file into tokens.
 * @param line The line to split.
 * @param out_tokens A pointer to an array of strings that will store the tokens.
 * @return The number of tokens found, or -1 on failure.
 */
int splitCSVLine(char *line, char ***out_tokens);

/**
 * @brief Reads a CSV file and creates a dataFrame.
 * @param filename The path to the CSV file.
 * @return A pointer to the dataFrame, or NULL if the file cannot be opened or read.
 */
dataFrame* readCSV(const char *filename);

/**
 * @brief Prints the content of a dataFrame to the console.
 * @param df The dataFrame to print.
 */
void printDataFrame(dataFrame *df);

/**
 * @brief Adds new rows to an existing dataFrame.
 * @param df The dataFrame to modify.
 * @param num_new_rows The number of rows to add.
 * @param data A 2D array containing the new row data.
 */
void addRows(dataFrame* df, int num_new_rows, float **data);

/**
 * @brief Adds new columns to an existing dataFrame.
 * @param df The dataFrame to modify.
 * @param num_new_columns The number of columns to add.
 * @param data A 2D array containing the new column data.
 * @param columns An array of strings for the new column names.
 */
void addColumns(dataFrame* df, int num_new_columns, float** data, char **columns);

/**
 * @brief Gets the index of a column by its name.
 * @param df The dataFrame to search in.
 * @param column_name The name of the column to find.
 * @return The index of the column, or -1 if not found.
 */
int getColumnIndex(dataFrame *df, const char *column_name);

/**
 * @brief Calculates the mean of a specific column.
 * @param df The dataFrame.
 * @param column_name The name of the column.
 * @return The mean of the column.
 */
float getColumnMean(dataFrame* df, const char *column_name);

/**
 * @brief Finds the maximum value in a specific column.
 * @param df The dataFrame.
 * @param column_name The name of the column.
 * @return The maximum value in the column.
 */
float getColumnMax(dataFrame* df, const char *column_name);

/**
 * @brief Finds the minimum value in a specific column.
 * @param df The dataFrame.
 * @param column_name The name of the column.
 * @return The minimum value in the column.
 */
float getColumnMin(dataFrame* df, const char *column_name);

/**
 * @brief Calculates the standard deviation of a specific column.
 * @param df The dataFrame.
 * @param column_name The name of the column.
 * @return The standard deviation of the column.
 */
float getColumnStd(dataFrame *df, const char *column_name);

/**
 * @brief Normalizes a column using Min-Max scaling.
 * @param df The dataFrame.
 * @param column_name The name of the column to normalize.
 */
void normalizeColumnMinMax(dataFrame *df, char *column_name);

/**
 * @brief Normalizes a column using Z-score standardization.
 * @param df The dataFrame.
 * @param column_name The name of the column to normalize.
 */
void normalizeColumnZScore(dataFrame *df, char *column_name);

/**
 * @brief Creates a new dataFrame containing only selected columns from an existing one.
 * @param df The original dataFrame.
 * @param num_selected_columns The number of columns to select.
 * @param selected_columns An array of strings with the names of the columns to select.
 * @return A new dataFrame with only the selected columns.
 */
dataFrame* selectColumns(dataFrame *df, int num_selected_columns,  char **selected_columns);

/**
 * @brief Creates a new dataFrame containing the top N rows of an existing one.
 * @param df The original dataFrame.
 * @param num_top_rows The number of rows to select from the top.
 * @return A new dataFrame with the top N rows.
 */
dataFrame* selectTopRows(dataFrame *df, int num_top_rows);

/**
 * @brief Creates a new dataFrame containing the bottom N rows of an existing one.
 * @param df The original dataFrame.
 * @param num_bottom_rows The number of rows to select from the bottom.
 * @return A new dataFrame with the bottom N rows.
 */
dataFrame* selectBottomRows(dataFrame *df, int num_bottom_rows);

/**
 * @brief Adds NaN rows to a smaller dataFrame to match the row count of a larger one.
 * @param small_df The dataFrame with fewer rows.
 * @param big_df The dataFrame with more rows.
 */
void addNans(dataFrame* small_df, dataFrame* big_df);

/**
 * @brief Equalizes the number of rows between two dataFrames.
 * @param df1 The first dataFrame.
 * @param df2 The second dataFrame.
 * @param method The method to use: 'a' for adding NaNs, 'c' for cutting rows.
 */
void equalizeRows(dataFrame *df1, dataFrame *df2, char* method);

/**
 * @brief Performs an inner join on two dataFrames based on common columns.
 * @param df1 The first dataFrame.
 * @param df2 The second dataFrame.
 * @return A new dataFrame that is the result of the inner join.
 */
dataFrame* innerJoin(dataFrame *df1, dataFrame *df2);

/**
 * @brief Joins two dataFrames based on specified columns.
 * @param df1 The first dataFrame.
 * @param df2 The second dataFrame.
 * @param num_selected_columns The number of columns to join on.
 * @param selected_columns The names of the columns to join on.
 * @return A new dataFrame that is the result of the join.
 */
dataFrame* columnJoin(dataFrame *df1, dataFrame *df2, int num_selected_columns,  char **selected_columns);

void replaceColumns(dataFrame *df, int num_replaced, char **old_columns, char **new_columns, float **data);

dataFrame* copyDataFrame (dataFrame* df);

#endif // PANDAC_H