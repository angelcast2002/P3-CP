#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <vector>
#include "pgm.h"
#include <opencv2/opencv.hpp>

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const double radInc = degreeInc * M_PI / 180;

// Estructura para almacenar los parámetros de las líneas detectadas
struct Line {
    double r;
    double theta;
};

// Declaración de memoria constante
__constant__ double d_Cos[degreeBins];
__constant__ double d_Sin[degreeBins];

//*****************************************************************
// La función CPU_HoughTran es la misma que antes
void CPU_HoughTran(unsigned char *pic, int w, int h, int **acc)
{
    double rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2.0;
    *acc = new int[rBins * degreeBins];
    memset(*acc, 0, sizeof(int) * rBins * degreeBins);
    int xCent = w / 2;
    int yCent = h / 2;
    double rScale = (2.0 * rMax) / rBins;

    // Precompute theta values
    double *thetaValues = new double[degreeBins];
    double theta = 0.0;
    for (int tIdx = 0; tIdx < degreeBins; tIdx++)
    {
        thetaValues[tIdx] = theta;
        theta += radInc;
    }

    for (int i = 0; i < w; i++)
    {
        for (int j = 0; j < h; j++)
        {
            int idx = j * w + i;
            if (pic[idx] > 0)
            {
                int xCoord = i - xCent;
                int yCoord = yCent - j;
                for (int tIdx = 0; tIdx < degreeBins; tIdx++)
                {
                    double r = xCoord * cos(thetaValues[tIdx]) + yCoord * sin(thetaValues[tIdx]);
                    int rIdx = (int)((r + rMax) / rScale + 0.5);
                    if (rIdx >= 0 && rIdx < rBins)
                    {
                        (*acc)[rIdx * degreeBins + tIdx]++;
                    }
                }
            }
        }
    }

    delete[] thetaValues;
}

// GPU kernel. Un hilo por píxel de imagen.
__global__ void GPU_HoughTran(unsigned char *pic, int w, int h, int *acc, double rMax, double rScale)
{
    int gloID = blockIdx.x * blockDim.x + threadIdx.x;
    int locID = threadIdx.x;

    if (gloID >= w * h) return;

    int xCent = w / 2;
    int yCent = h / 2;

    int xCoord = gloID % w - xCent;
    int yCoord = yCent - gloID / w;

    // Definir acumulador local en memoria compartida
    extern __shared__ int localAcc[];

    // Inicializar acumulador local a 0
    for (int idx = locID; idx < degreeBins * rBins; idx += blockDim.x) {
        localAcc[idx] = 0;
    }
    __syncthreads(); // Barrera para asegurar que todos los hilos hayan completado la inicialización

    if (pic[gloID] > 0)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            double r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
            int rIdx = (int)((r + rMax) / rScale + 0.5);
            if (rIdx >= 0 && rIdx < rBins)
            {
                atomicAdd(&localAcc[rIdx * degreeBins + tIdx], 1);
            }
        }
    }
    __syncthreads(); // Barrera para asegurar que todos los hilos hayan completado el incremento del acumulador local

    // Sumar valores del acumulador local al acumulador global
    for (int idx = locID; idx < degreeBins * rBins; idx += blockDim.x) {
        if (localAcc[idx] > 0) { // Solo si hay algo que sumar
            atomicAdd(&acc[idx], localAcc[idx]);
        }
    }
}

// Las funciones drawLine y savePPM son las mismas que antes
void drawLine(unsigned char *image, int w, int h, double r, double theta)
{
    int xCent = w / 2;
    int yCent = h / 2;

    double cosT = cos(theta);
    double sinT = sin(theta);

    if (fabs(sinT) > 0.5)
    {
        // Recorremos en x
        for (int x = 0; x < w; x++)
        {
            double y = (r - (x - xCent) * cosT) / sinT;
            int yInt = yCent - (int)(y + 0.5);
            if (yInt >= 0 && yInt < h)
            {
                int idx = yInt * w + x;
                // Dibujar en rojo (R,G,B)
                image[3 * idx] = 255;     // R
                image[3 * idx + 1] = 0;   // G
                image[3 * idx + 2] = 0;   // B
            }
        }
    }
    else
    {
        // Recorremos en y
        for (int y = 0; y < h; y++)
        {
            double x = (r - (yCent - y) * sinT) / cosT;
            int xInt = (int)(x + xCent + 0.5);
            if (xInt >= 0 && xInt < w)
            {
                int idx = y * w + xInt;
                // Dibujar en rojo (R,G,B)
                image[3 * idx] = 255;     // R
                image[3 * idx + 1] = 0;   // G
                image[3 * idx + 2] = 0;   // B
            }
        }
    }
}

void savePPM(const char *filename, unsigned char *image, int w, int h)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        printf("Error al abrir el archivo para escribir: %s\n", filename);
        return;
    }
    fprintf(fp, "P6\n%d %d\n255\n", w, h);
    fwrite(image, sizeof(unsigned char), w * h * 3, fp);
    fclose(fp);
}

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        printf("Uso: %s <imagen.pgm>\n", argv[0]);
        return -1;
    }

    int i;

    PGMImage inImg(argv[1]);

    int *cpuht;
    int w = inImg.x_dim;
    int h = inImg.y_dim;

    // Cálculo en CPU
    CPU_HoughTran(inImg.pixels, w, h, &cpuht);

    // Precalcular valores de theta, coseno y seno
    double *thetaValues = (double *)malloc(sizeof(double) * degreeBins);
    double *pcCos = (double *)malloc(sizeof(double) * degreeBins);
    double *pcSin = (double *)malloc(sizeof(double) * degreeBins);
    double theta = 0.0;
    for (i = 0; i < degreeBins; i++)
    {
        thetaValues[i] = theta;
        pcCos[i] = cos(theta);
        pcSin[i] = sin(theta);
        theta += radInc;
    }

    double rMax = sqrt(1.0 * w * w + 1.0 * h * h) / 2.0;
    double rScale = (2.0 * rMax) / rBins;

    // Copiar valores precalculados a memoria constante
    cudaMemcpyToSymbol(d_Cos, pcCos, sizeof(double) * degreeBins);
    cudaMemcpyToSymbol(d_Sin, pcSin, sizeof(double) * degreeBins);

    // Configurar y copiar datos del host al dispositivo
    unsigned char *d_in;
    int *d_hough, *h_hough;

    h_hough = (int *)malloc(degreeBins * rBins * sizeof(int));

    cudaMalloc((void **)&d_in, sizeof(unsigned char) * w * h);
    cudaMalloc((void **)&d_hough, sizeof(int) * degreeBins * rBins);
    cudaMemcpy(d_in, inImg.pixels, sizeof(unsigned char) * w * h, cudaMemcpyHostToDevice);
    cudaMemset(d_hough, 0, sizeof(int) * degreeBins * rBins);

    // Configuración de la ejecución: 1-D grid de 1-D blocks, cada uno con 256 threads
    int threadsPerBlock = 256;
    int blockNum = (w * h + threadsPerBlock - 1) / threadsPerBlock;

    // Calcular el tamaño de memoria compartida necesaria
    size_t sharedMemSize = sizeof(int) * degreeBins * rBins;

    // Ejecutar el kernel
    GPU_HoughTran<<<blockNum, threadsPerBlock, sharedMemSize>>>(d_in, w, h, d_hough, rMax, rScale);

    // Obtener resultados del dispositivo
    cudaMemcpy(h_hough, d_hough, sizeof(int) * degreeBins * rBins, cudaMemcpyDeviceToHost);

    // Liberar memoria en el dispositivo
    cudaFree(d_in);
    cudaFree(d_hough);

    // Comparar resultados CPU y GPU
    int discrepancies = 0;
    for (i = 0; i < degreeBins * rBins; i++)
    {
        if (cpuht[i] != h_hough[i])
        {
            discrepancies++;
            printf("Calculation mismatch at : %i CPU=%i GPU=%i\n", i, cpuht[i], h_hough[i]);
        }
    }
    if (discrepancies == 0)
        printf("Los resultados de CPU y GPU coinciden.\n");
    else
        printf("Se encontraron %d discrepancias.\n", discrepancies);

    // Detectar líneas significativas
    // Calculamos el umbral basado en el promedio y desviación estándar
    double sum = 0.0;
    double sumSq = 0.0;
    int total = degreeBins * rBins;
    for (i = 0; i < total; i++)
    {
        sum += h_hough[i];
        sumSq += h_hough[i] * h_hough[i];
    }
    double mean = sum / total;
    double variance = (sumSq / total) - (mean * mean);
    double stddev = sqrt(variance);
    double threshold = (mean + 2 * stddev) * 1.20;

    printf("Umbral establecido en: %f\n", threshold);

    // Almacenar líneas con peso mayor al umbral
    std::vector<Line> lines;
    for (int rIdx = 0; rIdx < rBins; rIdx++)
    {
        for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {
            int idx = rIdx * degreeBins + tIdx;
            if (h_hough[idx] > threshold)
            {
                double r = rIdx * rScale - rMax;
                double theta = thetaValues[tIdx];
                Line line = {r, theta};
                lines.push_back(line);
            }
        }
    }

    printf("Número de líneas detectadas: %lu\n", lines.size());

    // Crear una imagen en color para dibujar las líneas
    unsigned char *resultImage = (unsigned char *)malloc(w * h * 3 * sizeof(unsigned char));
    // Inicializar la imagen resultante con la imagen original en escala de grises
    for (int idx = 0; idx < w * h; idx++)
    {
        unsigned char pixel = inImg.pixels[idx];
        resultImage[3 * idx] = pixel;     // R
        resultImage[3 * idx + 1] = pixel; // G
        resultImage[3 * idx + 2] = pixel; // B
    }

    // Dibujar las líneas sobre la imagen
    for (size_t idx = 0; idx < lines.size(); idx++)
    {
        drawLine(resultImage, w, h, lines[idx].r, lines[idx].theta);
    }

    // Guardar la imagen resultante en formato PPM
    savePPM("results/outputShared.ppm", resultImage, w, h);
    printf("Imagen con líneas guardada en 'outputShared.ppm'\n");

    // Guardar la imagen resultante en formato PNG usando OpenCV
    cv::Mat imgMat(h, w, CV_8UC3, resultImage);
    cv::imwrite("results/outputShared.png", imgMat);
    printf("Imagen con líneas guardada en 'outputShared.png'\n");

    // Liberar memoria en el host
    free(h_hough);
    delete[] cpuht;
    free(pcCos);
    free(pcSin);
    free(thetaValues);
    free(resultImage);

    return 0;
}
