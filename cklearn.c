#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "pandac.h"

int main() {

    dataFrame *df1 = readCSV("data1.csv");
    printDataFrame(df1);
    printf("\n\n");

    dataFrame *df2 = readCSV("data2.csv");
    printDataFrame(df2);
    printf("\n\n");

    dataFrame *df3 = innerJoin(df1, df2);
    printDataFrame(df3);
    
    return 0;
}