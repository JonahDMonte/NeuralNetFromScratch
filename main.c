#include <stdio.h>
#include <stdlib.h>
#include "string.h"
#include <time.h>
#include <math.h>
#include <unistd.h>  //Header file for sleep(). man 3 sleep for details.
#include <pthread.h>
#define TRAIN_ROWS 60000
#define TEST_ROWS 10000
#define COLS 769


#include <stdio.h>
#include <math.h>

// Softmax function
void softmax(long double* z, long double* y, int length) {
    long double max_val = z[0];
    long double sum = 0.0;

    // Find the maximum value in z
    for (int i = 1; i < length; i++) {
        if (z[i] > max_val) {
            max_val = z[i];
        }
    }

    // Calculate softmax values and sum
    for (int i = 0; i < length; i++) {
        y[i] = expl(z[i] - max_val);
        sum += y[i];
    }

    // Normalize the values
    for (int i = 0; i < length; i++) {
        y[i] /= sum;
    }
}




long double relu(long double x)
{

    if (x <= 0)
    {
        return(0.0);
    }
    else
    {
        return(x);
    }
}

int main() {

    //Initializing data arrays in memory
    printf("Initializing\n");
    int (*trainload)[COLS] = malloc(sizeof(int[TRAIN_ROWS][COLS]));
    int (*testload)[COLS] = malloc(sizeof(int[TEST_ROWS][COLS]));
    long double (*train)[COLS] = malloc(sizeof(long double[TRAIN_ROWS][COLS]));
    long double (*test)[COLS] = malloc(sizeof(long double[TEST_ROWS][COLS]));

    //Import the data from the Excel sheet into the arrays
    char line[9999];
    FILE *file;
    file = fopen("C:\\Users\\Jonah\\CLionProjects\\NeuralNet\\mnist_test.txt", "r");

    int num;
    int z = 0;
    printf("Loading Data\n");
    while (z < TEST_ROWS) {

        while (fgets(line, 9999, file)) // while the line exists
        {
            printf("Test Data: %.2Lf%   \r",  100.0*(long double)(z+1)/(long double)TEST_ROWS);
            char *token = strtok(line, ","); // split it based on commas

            for (int j = 0; j < 769; j++) {

                sscanf(token, "%d", &num);
                testload[z][j] = num;

                token = strtok(NULL, ",");

            }
            z++;

        }
    }

    fclose(file);

    FILE *file2 = fopen("C:\\Users\\Jonah\\CLionProjects\\NeuralNet\\mnist_train.txt", "r");

    z = 0;
    printf("\n");
    while (z < TRAIN_ROWS) {

        while (fgets(line, 9999, file2)) // while the line exists
        {
            printf("Train Data: %.2Lf%   \r",  100.0*(long double)(z+1)/(long double)TRAIN_ROWS);
            char *token = strtok(line, ","); // split it based on commas

            for (int j = 0; j < 769; j++) {

                sscanf(token, "%d", &num);
                trainload[z][j] = num;

                token = strtok(NULL, ",");
            }
            z++;

        }
    }
    fclose(file2);

    //Normalize the data between 0 and 1
    printf("Normalizing data\n");
    for (int i = 0; i < TRAIN_ROWS; i++) {
        train[i][0] = (long double) trainload[i][0];
        for (int j = 1; j < 769; j++) {
            train[i][j] = ((long double)trainload[i][j]) / (long double)255.0;

        }

    }
    free(trainload);
    for (int i = 0; i < TEST_ROWS; i++) {
        test[i][0] = (long double) testload[i][0];
        for (int j = 1; j < 769; j++) {
            test[i][j] = (long double) testload[i][j] / (long double) 255.0;
        }

    }

    free(testload);
    printf("Making neural net\n");

    //initialize weight matrices
    long double (*w1)[128] = malloc(sizeof(long double[768][128]));
    long double (*w2)[16] = malloc(sizeof(long double[16][128]));
    long double (*w3)[10] = malloc(sizeof(long double[10][16]));

    //initialize actuation vectors
    long double *a1 = malloc(sizeof(long double[768]));
    long double *a2 = malloc(sizeof(long double[16]));
    long double *a3 = malloc(sizeof(long double[10]));
    long double *z1 = malloc(sizeof(long double[768]));
    long double *z2 = malloc(sizeof(long double[16]));
    long double *z3 = malloc(sizeof(long double[10]));

    //fill weight vectors with random values from 0-1
    srand(time(NULL));
    for (int i = 0; i < 768; i++) {
        for (int j = 0; j < 128; j++) {
            w1[i][j] = (long double) rand() / RAND_MAX;
        }
    }
    srand(time(NULL));

    for (int i = 0; i < 128; i++) {
        for (int j = 0; j < 16; j++) {
            w2[i][j] = (long double) rand() / RAND_MAX;
        }
    }
    srand(time(NULL));

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 10; j++) {
            w3[i][j] = (long double) rand() / RAND_MAX;
        }
    }





    //epoch loop
    int epochs = 10;
    for (int e = 1; e <= epochs; e++) {
        printf("Epoch: %d ", e);
        for (int training_row = 0; training_row < TRAIN_ROWS; training_row++) {
            printf("Training Progress: %.2Lf   ",  100.0*(long double)training_row/(long double)TRAIN_ROWS);
            long double label = train[training_row][0]; //set label

            long double *a0 = malloc(sizeof(long double[768])); //instantiate 0th actuation
            for (int j = 0; j < 768; j++) {
                a0[j] = train[training_row][j + 1]; //fill 0th actuation with current image

            }

            // multiply by weight matrix 1
            for (int k = 0; k < 128; k++) {
                long double val = 0;
                for (int j = 0; j < 768; j++) {
                    val = val + a0[j] * w1[k][j];
                }
                z1[k] = val;

            }
            for (int v = 0; v < 128; v++) // normalize the values
            {
                a1[v] = relu(a1[v]);
            }
            // multiply by weight matrix 2
            for (int k = 0; k < 16; k++) {
                long double val = 0;
                for (int j = 0; j < 128; j++) {
                    val = val + a1[j] * w2[k][j];
                }
                z2[k] = val;
            }
            for (int v = 0; v < 16; v++) // normalize the values
            {
                a2[v] = relu(a2[v]);
            }
            //multiply by weight matrix 3
            for (int k = 0; k < 10; k++) {
                long double val = 0;
                for (int j = 0; j < 16; j++) {
                    val = val + a2[j] * w3[k][j];
                }
                z3[k] = val;
            }




            long double ideal[10];
            long double cost = 0.0;
            for (int h = 0; h < 10; h++) {
                if (h == (int)label) {
                    ideal[h] = 1.0;
                } else {
                    ideal[h] = 0.0;
                }
            }
            long double y[10];

            long double max_val = z3[0];
            long double sum = 0.0;

            // Find the maximum value in z
            for (int i = 1; i < 10; i++) {
                if (z3[i] > max_val) {
                    max_val = z3[i];
                }
            }

            // Calculate softmax values and sum
            for (int i = 0; i < 10; i++) {
                y[i] = expl(max_val - z3[i]);
                sum += y[i];
            }

            // Normalize the values
            for (int i = 0; i < 10; i++) {
                y[i] /= sum;
            }
            printf("Loss: %Lf \r" , cost);


            //free(y);
            // dC/dz3 = a3 - ideal
            long double dCdz3[10];
            for (int i = 0; i < 10; i++) {
                dCdz3[i] = a3[i] - ideal[i];
            }
            //dC/da2 = (W3)^T * dC/dz3
            long double dCda2[16];
            for (int i = 0; i < 16; i++) {
                long double val = 0;
                for (int j = 0; j < 10; j++) {
                    val += w3[j][i] * dCdz3[j];
                }
                dCda2[i] = val;
            }

            // dC/dz2 = dC/da_2 * (1 if zi > 0, else 0)
            long double dCdz2[16];
            for (int i = 0; i < 16; i++) {
                if (dCda2[i] > 0) {
                    dCdz2[i] = 1;
                } else {
                    dCdz2[i] = 0;
                }
            }

            // dC/dW3 = dC/dz3 * a2^T
            long double dCdw3[10][16];
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 16; j++) {
                    dCdw3[i][j] = dCdz3[i] * a2[j];
                }
            }

            // dC/da1 [128] = (W2 [16x128])^T * dC/dz2 [16]
            long double dCda1[128];
            for (int i = 0; i < 128; i++) {
                long double val = 0;
                for (int j = 0; j < 16; j++) {
                    val += w2[j][i] * dCdz2[j];
                }
                dCda1[i] = val;
            }

            // dC/dz1 [128] = dC/da1 [128] * (1 if z1 > 0, else 0)

            long double dCdz1[128];
            for (int i = 0; i < 128; i++) {
                if (dCda1[i] > 0) {
                    dCdz1[i] = 1;
                } else {
                    dCdz1[i] = 0;
                }
            }

            // dC/dW2 [16x128] = dC/dz2 [16] * a1^T [128x1]

            long double dCdw2[16][128];
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 128; j++) {
                    dCdw2[i][j] = dCdz2[i] * a1[j];
                }
            }

            // dC/dW1 [128x768] = dC/dz1 [128] * input^T [768x1]
            long double dCdw1[128][768];
            for (int i = 0; i < 128; i++) {
                for (int j = 0; j < 768; j++) {
                    dCdw1[i][j] = dCdz1[i] * a0[j];
                }
            }



            long double lr = 0.1;
            // W1_new [128x768] = W1 [128x768] - lr * dC/dW1 [128x768]
            for (int i = 0; i < 128; i++) {
                for (int j = 0; j < 768; j++) {
                    w1[i][j] -= dCdw1[i][j] * lr;
                }
            }

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 128; j++) {
                    w2[i][j] -= dCdw2[i][j] * lr;
                }
            }
            for (int i = 0; i < 10; i++) {
                for (int j = 0; j < 16; j++) {
                    w3[i][j] -= dCdw3[i][j] * lr;
                }
            }


        }

        //testing loop
//        printf("Beginning Testing Loop\n");
//        for (int i = 0; i < TEST_ROWS; i++) {
//            long double label = test[i][0];
//            long double pass[768];
//
//            for (int j = 0; j < 768; j++) {
//                pass[j] = test[i][j + 1];
//            }
//
//
//            long double *a0 = malloc(sizeof(long double[768])); //instantiate 0th actuation
//
//
//                // multiply by weight matrix 1
//                for (int k = 0; k < 128; k++) {
//                    long double val = 0;
//                    for (int j = 0; j < 768; j++) {
//                        val = val + a0[j] * w1[k][j];
//                    }
//                    z1[k] = val;
//
//                }
//                for (int v = 0; v < 128; v++) // normalize the values
//                {
//                    a1[v] = relu(a1[v]);
//                }
//                // multiply by weight matrix 2
//                for (int k = 0; k < 16; k++) {
//                    long double val = 0;
//                    for (int j = 0; j < 128; j++) {
//                        val = val + a1[j] * w2[k][j];
//                    }
//                    z2[k] = val;
//                }
//                for (int v = 0; v < 16; v++) // normalize the values
//                {
//                    a2[v] = relu(a2[v]);
//                }
//                //multiply by weight matrix 3
//                for (int k = 0; k < 10; k++) {
//                    long double val = 0;
//                    for (int j = 0; j < 16; j++) {
//                        val = val + a2[j] * w3[k][j];
//                    }
//                    z3[k] = val;
//                }
//
//
//                long double *y = softmax(z3, 10);
//
//                long double ideal[10];
//                long double cost = 0;
//                for (int h = 0; h < 10; h++) {
//                    if (h == (int) label) {
//                        ideal[h] = 1.0;
//                    } else {
//                        ideal[h] = 0.0;
//                    }
//                }
//
//                for (int h = 0; h < 10; h++) {
//                    cost += powl((y[h] - ideal[h]), 2);
//                }
//                printf("Loss: %Lf \r", cost);
//
//
//                //todo calculate loss
//                //todo get a percentage?
//
//            }
        printf("\n");
        }
    }





