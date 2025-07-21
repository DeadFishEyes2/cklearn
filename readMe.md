cklearn: Machine Learning in C
==============================

`cklearn` is a machine learning library written in pure `C`. It’s designed to be lightweight and fast, with tools for data manipulation, analysis, and common ML algorithms. Inspired by scikit-learn and pandas, it gives C programmers the ability to do ML without relying on Python, making it great for `embedded systems` or cases where you need raw speed and low-level control.

## 📖 Table of Contents

- [Features](#features)
- [Auxiliary Libraries](#auxiliary-libraries)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Building the Project](#building-the-project)
- [Usage](#usage)
  - [Data Loading and Manipulation (pandac)](#data-loading-and-manipulation-pandac)
  - [Machine Learning (cklearn)](#machine-learning-cklearn)
  - [Plotting (plotc)](#plotting-plotc)
- [Examples](#examples)
- [Contributing](#contributing)
    

# Features
--------

`cklearn` currently offers collection of functionalities, focusing on foundational machine learning tasks and data preprocessing. Each component is designed for efficiency and ease of integration into C projects.

*   **Clustering:**
    
    *   **K-Means Clustering:** This unsupervised learning algorithm is a cornerstone of data analysis, used for partitioning `n` observations into `k` clusters where each observation belongs to the cluster with the nearest mean (centroid). `cklearn` provides a robust implementation with functions for:
        
        *   `kmeans`: Performs a single run of the K-Means algorithm, iteratively refining cluster assignments and centroid positions until convergence.
            
        *   `kmeansFit`: Executes K-Means multiple times with different initial centroid configurations to mitigate the impact of local optima, returning the best set of centroids based on a chosen metric (e.g., Mean Squared Error). This is crucial for achieving more stable and optimal clustering results.
            
        *   `kmeansPredict`: Assigns new or existing data points to their closest cluster, based on a pre-computed set of centroids. This allows for the classification of new data after the model has been trained.
            
*   **Nearest Neighbors:**
    
    *   **K-Nearest Neighbors (KNN):** A simple yet powerful non-parametric algorithm used for both classification and regression. In `cklearn`, its primary utility is currently for finding the `k` closest data points to any given observation within a dataset. This fundamental capability can be extended for various applications, including data imputation and anomaly detection, by analyzing the characteristics of these neighbors.
        
*   **Data Imputation:**
    
    *   **NaN (Not a Number) filling using the mean of nearest neighbors (`fillNaN`):** Missing data is a common challenge in real-world datasets. `cklearn` addresses this by providing a method to impute (fill in) missing values. The `fillNaN` function leverages the K-Nearest Neighbors approach by replacing a missing value with the mean of that specific feature from its `k` nearest neighbors. This is often a more sophisticated imputation strategy than simple mean or median imputation, as it considers the local structure of the data.
        
*   **Distance Metrics:**
    
    *   **Euclidean Distance calculation:** The Euclidean distance is a fundamental metric used extensively throughout `cklearn`, particularly in clustering and nearest neighbor algorithms. It quantifies the straight-line distance between two points in Euclidean space. This core function is optimized for performance and forms the basis for many other algorithms within the library. Future versions may include other distance metrics like Manhattan or Cosine similarity to cater to different data types and analytical needs.
        

# Auxiliary Libraries
-------------------

`cklearn` is built upon and complements two dedicated auxiliary libraries, `pandac` for efficient data handling and `plotc` for basic data visualization. These libraries are integral to providing a comprehensive data science toolkit in C.

### `pandac`: Data Manipulation

Inspired by the powerful `pandas` library in Python, `pandac` provides a robust `dataFrame` structure and a suite of functions designed for efficient data manipulation and preprocessing. It aims to offer a familiar interface for users accustomed to tabular data operations.

*   **Reading and writing CSV files (`readCSV`, `writeCSV`):** These functions provide the essential capability to load data from standard Comma Separated Value (CSV) files into a `dataFrame` and to export `dataFrame` content back into CSV format. `readCSV` includes basic error handling for file access and robust parsing of numerical and NaN values.
    
*   **Basic data frame operations:**
    
    *   **Creation and freeing of DataFrames (`createDataFrame`, `freeDataFrame`):** These core functions manage the memory allocation and deallocation for `dataFrame` structures, ensuring efficient resource management. `createDataFrame` allows for programmatic construction of DataFrames from existing C arrays.
        
    *   **Printing DataFrames (`printDataFrame`, `printDataFrameWithIndex`):** Essential for inspecting data, these functions provide formatted output of the `dataFrame` content to the console, with `printDataFrameWithIndex` adding row indices for easier reference.
        
    *   **Adding rows and columns (`addRows`, `addColumns`):** These functions allow for the dynamic expansion of a `dataFrame` by appending new rows or columns of data, facilitating iterative data construction or feature engineering.
        
    *   **Column indexing (`getColumnIndex`):** Provides a convenient way to retrieve a column's numerical index by its string name, making column-based operations more intuitive.
        
    *   **Statistical summaries (`getColumnMean`, `getColumnMax`, `getColumnMin`, `getColumnStd`):** These functions compute fundamental descriptive statistics for individual columns, providing quick insights into data distribution and central tendencies. They are designed to handle NaN values gracefully, excluding them from calculations.
        
    *   **Data normalization (Min-Max and Z-score scaling: `normalizeColumnMinMax`, `normalizeColumnZScore`):** Data normalization is a critical preprocessing step for many machine learning algorithms. Min-Max scaling rescales features to a fixed range (typically 0 to 1), while Z-score standardization transforms data to have a mean of 0 and a standard deviation of 1. Both methods help prevent features with larger numerical ranges from dominating the learning process.
        
    *   **Selecting specific columns or rows (`selectColumns`, `selectTopRows`, `selectBottomRows`):** These functions enable the creation of new `dataFrame` subsets based on column names or by selecting a specified number of rows from the beginning or end of the DataFrame, useful for data exploration and feature selection.
        
    *   **Handling missing values (adding NaNs: `addNans`):** This utility function can pad a smaller DataFrame with NaN values to match the row count of a larger one, which is useful in certain data alignment scenarios.
        
    *   **Equalizing row counts between DataFrames (`equalizeRows`):** Facilitates operations between DataFrames of different sizes by either adding NaNs or truncating rows to achieve consistent dimensions.
        
    *   **Performing inner joins (`innerJoin`):** Implements a relational database-like inner join operation, combining rows from two DataFrames where there are matching values in common columns. This is powerful for integrating data from multiple sources.
        
    *   **Replacing columns (`replaceColumns`):** Allows for the in-place modification of existing columns within a DataFrame, either by updating their values or replacing them entirely with new data.
        

### `plotc`: Data Visualization

Similar in spirit to `matplotlib` but tailored for C, `plotc` provides essential plotting capabilities by interfacing with `gnuplot`. This allows for direct visualization of `cklearn`'s results and `pandac`'s data structures without needing to export data to external plotting environments. It's crucial to have `gnuplot` installed and accessible in your system's PATH for these functions to work.

*   **Simple scatter plots (`scatterplot`):** Generates a basic 2D scatter plot, ideal for visualizing the relationship between two numerical variables. It provides a quick visual check of data distribution.
    
*   **Scatter plots with linear regression lines (`regplot`):** Extends the scatter plot by adding a calculated linear regression line, helping to identify and visualize linear trends within the data. This is useful for understanding correlations and potential predictive relationships.
    
*   **Scatter plots with different hues/categories (`hueplot`):** Enables the visualization of data points colored by a categorical variable (hue). This is particularly effective for observing clusters or group separations, such as the output of K-Means clustering, where different colors represent different clusters.
    
*   **DataFrame-integrated plotting functions (`dataFrameScatterplot`, `dataFrameRegplot`, `dataFrameHueplot`):** These convenience functions directly accept `dataFrame` objects and column names, simplifying the plotting process by abstracting away the need to manually extract data arrays. They seamlessly integrate `pandac`'s data structures with `plotc`'s visualization capabilities.
    

# Getting Started
---------------

To begin working with `cklearn`, follow these steps to set up your development environment and build the project.

### Prerequisites

Before you can build and run `cklearn`, ensure you have the following software installed on your system:

*   **A C compiler (e.g., GCC):** GCC (GNU Compiler Collection) is the standard compiler for C projects on Linux and macOS, and can be installed on Windows via MinGW or Cygwin. It is essential for compiling the `.c` source files into executable binaries.
    
*   **`make`:** This build automation tool is used to manage the compilation process, especially for projects with multiple source files and dependencies. The provided `Makefile` simplifies the compilation steps.
    
*   **`gnuplot`:** This command-line-driven graphing utility is required by the `plotc` library to generate visualizations. Ensure it's installed and its executable is discoverable in your system's PATH environment variable. You can test its installation by typing `gnuplot` in your terminal.
    

### Building the Project

1.  **Clone the repository (or ensure all `.c` and `.h` files are in the same directory):** If your project is hosted on a Git repository, use the following command to clone it to your local machine. Otherwise, ensure all source (`.c`) and header (`.h`) files are present in your working directory.
    
        git clone https://github.com/your-repo/cklearn.git # Replace with your actual repository URL
        cd cklearn
        
    
2.  **Compile the source files:** The project includes a `Makefile` that automates the compilation process. This `Makefile` defines the compiler (`CC`), compilation flags (`CFLAGS`), source files (`SRCS`), object files (`OBJS`), and the final executable target (`TARGET`). The flags `-Wall -Wextra` enable extensive warnings, `-std=c99` ensures C99 standard compliance, `-g` includes debugging information, and `-lm` links the math library.
    
        CC = gcc
        CFLAGS = -Wall -Wextra -std=c99 -g -lm
        
        SRCS = cklearn.c pandac.c plotc.c
        OBJS = $(SRCS:.c=.o)
        TARGET = cklearn_app
        
        all: $(TARGET)
        
        $(TARGET): $(OBJS)
        	$(CC) $(OBJS) -o $(TARGET) $(CFLAGS)
        
        %.o: %.c
        	$(CC) -c $< -o $@ $(CFLAGS)
        
        clean:
        	rm -f $(OBJS) $(TARGET)
        
    
    To compile the project, navigate to the project directory in your terminal and simply run:
    
        make
        
    
    Upon successful compilation, this command will create an executable file named `cklearn_app` (or whatever name you've specified for `TARGET` in your `Makefile`) in the current directory. The `clean` target can be used to remove compiled object files and the executable.
    

# Usage
-----

This section provides practical examples demonstrating how to integrate and utilize the `pandac`, `cklearn`, and `plotc` libraries within your C applications.

### Data Loading and Manipulation (`pandac`)

The `pandac` library is designed to simplify common data handling tasks, making it intuitive to load, inspect, and preprocess tabular data.

    #include "pandac.h" // Include the pandac header for DataFrame operations
    
    int main() {
        // Read a CSV file into a DataFrame. The readCSV function handles parsing headers,
        // numerical data, and recognizing "NaN" for missing values.
        dataFrame *df = readCSV("your_data.csv");
        if (df == NULL) {
            fprintf(stderr, "Error: Could not load data from your_data.csv\n");
            return 1; // Indicate an error
        }
    
        // Print the DataFrame to the console with row indices. This provides a clear
        // overview of the loaded data, similar to pandas' .head() method.
        printf("Original DataFrame:\n");
        printDataFrameWithIndex(df);
        printf("\n");
    
        // Get the mean of a specific column. This function automatically handles
        // NaN values by excluding them from the calculation.
        float mean_val = getColumnMean(df, "ColumnName");
        printf("Mean of 'ColumnName': %f\n", mean_val);
        printf("\n");
    
        // Normalize a column using Min-Max scaling. This transforms the values in
        // 'AnotherColumn' to a range between 0 and 1, which is often beneficial
        // for many machine learning algorithms.
        printf("DataFrame after Min-Max normalization on 'AnotherColumn':\n");
        normalizeColumnMinMax(df, "AnotherColumn");
        printDataFrame(df); // Print without index for conciseness after modification
        printf("\n");
    
        // It's crucial to free the memory allocated for the DataFrame once it's no longer needed
        // to prevent memory leaks.
        freeDataFrame(df);
        return 0; // Indicate successful execution
    }
    

### Machine Learning (`cklearn`)

The `cklearn` library provides implementations of core machine learning algorithms. Here's how you might use K-Means clustering.

    #include "cklearn.h" // Include the cklearn header for ML algorithms
    #include "pandac.h"   // Also include pandac for DataFrame structure and operations
    #include <time.h>     // For srand(time(NULL)) to ensure different centroid initialization
    
    int main() {
        srand(time(NULL)); // Initialize random seed for K-Means centroid initialization
    
        // Load your dataset for clustering. This CSV should contain numerical features.
        dataFrame *df = readCSV("clusters_data.csv");
        if (df == NULL) {
            fprintf(stderr, "Error: Could not load data from clusters_data.csv\n");
            return 1;
        }
    
        int num_clusters = 3;   // Define the number of clusters you want to find
        int num_iterations = 10; // Number of times to run K-Means with different initial centroids
    
        // Allocate memory for the centroids that will be returned by kmeansFit.
        // This 2D array will hold the coordinates of the final cluster centers.
        float **centroids = (float**)malloc(num_clusters * sizeof(float*));
        if (centroids == NULL) {
            fprintf(stderr, "Memory allocation failed for centroids.\n");
            freeDataFrame(df);
            return 1;
        }
        for (int i = 0; i < num_clusters; i++) {
            centroids[i] = (float*)malloc(df->num_columns * sizeof(float));
            if (centroids[i] == NULL) {
                fprintf(stderr, "Memory allocation failed for centroid %d.\n", i);
                // Clean up already allocated centroid rows before exiting
                for (int j = 0; j < i; j++) free(centroids[j]);
                free(centroids);
                freeDataFrame(df);
                return 1;
            }
        }
    
        // Fit the K-Means model to your data. kmeansFit runs the algorithm multiple times
        // and returns the set of centroids that resulted in the lowest Mean Squared Error (MSE).
        printf("Fitting K-Means model with %d clusters and %d iterations...\n", num_clusters, num_iterations);
        float **best_centroids = kmeansFit(df, num_clusters, num_iterations);
    
        // Predict clusters for the DataFrame. This function adds a new column named "Cluster"
        // to your DataFrame, indicating which cluster each row belongs to.
        printf("Predicting clusters and adding 'Cluster' column to DataFrame:\n");
        kmeansPredict(df, num_clusters, best_centroids);
        printDataFrameWithIndex(df);
        printf("\n");
    
        // After use, it's vital to free the memory allocated for the centroids.
        for (int i = 0; i < num_clusters; i++) {
            free(best_centroids[i]);
        }
        free(best_centroids);
    
        // Finally, free the DataFrame memory.
        freeDataFrame(df);
        return 0;
    }
    

### Plotting (`plotc`)

The `plotc` library provides convenient functions to visualize your data directly from `dataFrame` objects using `gnuplot`.

    #include "plotc.h"  // Include the plotc header for plotting functions
    #include "pandac.h" // Include pandac for DataFrame structure
    
    int main() {
        // Load your dataset for visualization.
        dataFrame *df = readCSV("your_data.csv");
        if (df == NULL) {
            fprintf(stderr, "Error: Could not load data from your_data.csv\n");
            return 1;
        }
    
        // Create a simple scatter plot from DataFrame columns "X_Column" and "Y_Column".
        // This opens a gnuplot window displaying the relationship between these two variables.
        printf("Generating simple scatter plot for 'X_Column' vs 'Y_Column'...\n");
        dataFrameScatterplot(df, "X_Column", "Y_Column");
        printf("Scatter plot generated. Check gnuplot window.\n\n");
    
        // Create a regression plot for "Feature1" and "Target". This plot will show
        // the scatter points along with a calculated linear regression line,
        // indicating potential linear relationships.
        printf("Generating regression plot for 'Feature1' vs 'Target'...\n");
        dataFrameRegplot(df, "Feature1", "Target");
        printf("Regression plot generated. Check gnuplot window.\n\n");
    
        // Create a hue plot. This assumes your DataFrame has a "Cluster" column
        // (e.g., from a K-Means prediction) that contains integer categories.
        // Each unique integer in the "Cluster" column will be represented by a different color,
        // allowing for visual separation of groups.
        printf("Generating hue plot for 'x' vs 'y' colored by 'Cluster'...\n");
        dataFrameHueplot(df, "x", "y", "Cluster");
        printf("Hue plot generated. Check gnuplot window.\n\n");
    
        // Free the DataFrame memory after all operations are complete.
        freeDataFrame(df);
        return 0;
    }
    

# Examples
--------

The `main` function within `cklearn.c` serves as a comprehensive demonstration of how to integrate and utilize the K-Means clustering algorithm with data loading, manipulation, and visualization capabilities provided by `pandac` and `plotc`. It showcases a typical workflow for clustering analysis.

    int main() {
        srand(time(NULL)); // Initialize random seed for K-Means to ensure different centroid initializations across runs.
    
        // Load two distinct datasets, "clear_clusters.csv" and "fuzzy_clusters.csv",
        // which likely represent data with well-defined and less-defined clusters, respectively.
        dataFrame *df_1 = readCSV("clear_clusters.csv");
        if (df_1 == NULL) {
            fprintf(stderr, "Error loading clear_clusters.csv\n");
            return 1;
        }
        printf("DataFrame 1 (clear_clusters.csv):\n");
        printDataFrameWithIndex(df_1);
        printf("\n\n");
    
        dataFrame *df_2 = readCSV("fuzzy_clusters.csv");
        if (df_2 == NULL) {
            fprintf(stderr, "Error loading fuzzy_clusters.csv\n");
            freeDataFrame(df_1); // Free df_1 if df_2 fails to load
            return 1;
        }
        printf("DataFrame 2 (fuzzy_clusters.csv):\n");
        printDataFrameWithIndex(df_2);
        printf("\n\n");
    
        int num_cluster = 3; // Define the target number of clusters for both datasets.
    
        // --- K-Means on df_1 (clear_clusters.csv) ---
        printf("Performing K-Means on df_1 (clear_clusters.csv) with %d clusters...\n", num_cluster);
        // kmeansFit runs the K-Means algorithm multiple times (10 iterations in this case)
        // with different random initializations and returns the best set of centroids found.
        float **centroids_1 = kmeansFit(df_1, num_cluster, 10);
        if (centroids_1 == NULL) {
            fprintf(stderr, "K-Means fit failed for df_1.\n");
            freeDataFrame(df_1);
            freeDataFrame(df_2);
            return 1;
        }
    
        // kmeansPredict assigns each data point in df_1 to its nearest centroid,
        // adding a new column named "Cluster" to the DataFrame.
        kmeansPredict(df_1, num_cluster, centroids_1);
        printf("DataFrame 1 after K-Means prediction:\n");
        printDataFrame(df_1); // Print the DataFrame including the new "Cluster" column.
    
        // dataFrameHueplot visualizes the clustered data. It plots 'x' against 'y'
        // and uses the "Cluster" column to color-code the data points,
        // making the cluster assignments visually apparent.
        printf("Generating hue plot for df_1...\n");
        dataFrameHueplot(df_1, "x", "y", "Cluster");
        printf("Hue plot for df_1 generated. Check gnuplot window.\n\n");
    
        // --- K-Means on df_2 (fuzzy_clusters.csv) ---
        printf("Performing K-Means on df_2 (fuzzy_clusters.csv) with %d clusters...\n", num_cluster);
        float **centroids_2 = kmeansFit(df_2, num_cluster, 10);
        if (centroids_2 == NULL) {
            fprintf(stderr, "K-Means fit failed for df_2.\n");
            freeDataFrame(df_1);
            freeDataFrame(df_2);
            // Also free centroids_1 if it was successfully allocated
            for (int i = 0; i < num_cluster; i++) free(centroids_1[i]);
            free(centroids_1);
            return 1;
        }
    
        kmeansPredict(df_2, num_cluster, centroids_2);
        printf("DataFrame 2 after K-Means prediction:\n");
        printDataFrame(df_2); // Print the DataFrame including the new "Cluster" column.
    
        printf("Generating hue plot for df_2...\n");
        dataFrameHueplot(df_2, "x", "y", "Cluster");
        printf("Hue plot for df_2 generated. Check gnuplot window.\n\n");
    
        // --- Memory Cleanup ---
        // It is crucial to free all dynamically allocated memory to prevent memory leaks.
        // This includes the DataFrames themselves and the centroids returned by kmeansFit.
        printf("Freeing allocated memory...\n");
        freeDataFrame(df_1);
        freeDataFrame(df_2);
    
        // Free the centroid arrays. Note that `kmeansFit` returns a new allocation,
        // so these must be freed separately.
        for (int i = 0; i < num_cluster; i++) {
            free(centroids_1[i]);
            free(centroids_2[i]);
        }
        free(centroids_1);
        free(centroids_2);
        printf("Memory freed successfully.\n");
    
        return 0; // Indicate successful program execution.
    }
    

# Contributing
------------

We welcome contributions from the community to enhance `cklearn`! Whether you're interested in implementing new machine learning algorithms, improving the efficiency or robustness of existing functionalities, fixing bugs, or expanding the documentation, your efforts are highly appreciated. Please feel free to open issues to report bugs or suggest new features, and submit pull requests with your proposed changes. We encourage adherence to standard C coding practices and clear, concise commenting within the code.

One exceptional addition to cklearn would be multithreadding since finding neighbors and calculating means could benefit from a paralel processing aproach.