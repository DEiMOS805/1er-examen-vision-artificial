#define _USE_MATH_DEFINES

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

using namespace cv;
using namespace std;

vector<vector<float>> createGaussianMask(float sigma, int n)
{
    int dif = (n - 1) / 2; // Cantidad de pixeles desde el centro hasta el borde del kernel

    float s = 2 * sigma * sigma;
    float r = 0.0, z = 0.0;

    vector<vector<float>> matrix(n, vector<float>(n, 0));

    for (int x = -dif; x <= dif; x++)
    {
        for (int y = -dif; y <= dif; y++)
        {
            // Formula de valores gaussianos divida en partes para mejor lectura
            r = sqrt(x * x + y * y);
            z = (exp(-(r * r) / s)) / (M_PI * s);

            matrix[x + dif][y + dif] = (exp(-(r * r) / s)) / (M_PI * s);
        }
    }

    return matrix;
}

vector<vector<float>> createGxMask()
{
    vector<vector<float>> mask(3, vector<float>(3, 0));

    mask[0][0] = -1;
    mask[0][1] = -2;
    mask[0][2] = -1;

    mask[1][0] = 0;
    mask[1][1] = 0;
    mask[1][2] = 0;

    mask[2][0] = 1;
    mask[2][1] = 2;
    mask[2][2] = 1;

    return mask;
}

vector<vector<float>> createGyMask()
{
    vector<vector<float>> mask(3, vector<float>(3, 0));

    mask[0][0] = -1;
    mask[0][1] = 0;
    mask[0][2] = 1;

    mask[1][0] = -2;
    mask[1][1] = 0;
    mask[1][2] = 2;

    mask[2][0] = -1;
    mask[2][1] = 0;
    mask[2][2] = 1;

    return mask;
}

void printMatrix(vector<vector<float>> matrix, int n)
{
    cout << "\n\n";

    // Recorrer matriz e imprimir valores [i][j]
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            cout << "\t";
            cout << matrix[i][j];
            cout << " ";
        }
        cout << "\n";
    }

    cout << "\n\n";
}

Mat normalize(Mat original, int newMin, int newMax)
{
    int rows = original.rows;
    int cols = original.cols;
    float constant = 0.0;
    int min = original.at<uchar>(Point(0, 0));
    int max = original.at<uchar>(Point(0, 0));

    Mat result(rows, cols, CV_8UC1);

    // Obtener la intensidad maxima y minima de la imagen de entrada de esta funcion
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (max < original.at<uchar>(Point(j, i)))
            {
                max = original.at<uchar>(Point(j, i));
            }

            if (min > original.at<uchar>(Point(j, i)))
            {
                min = original.at<uchar>(Point(j, i));
            }
        }
    }

    /*cout << "El valor maximo es: " << max << endl;
    cout << "El valor minimo es: " << min << endl;*/

    // Aplicar formula de normalizacion de imagenes
    constant = (newMax - newMin) / (max - min);
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            result.at<uchar>(Point(j, i)) = (original.at<uchar>(Point(j, i)) - min) * constant + newMin;
        }
    }

    return result;
}

Mat createBorderedImage(int rows, int cols, int n)
{
    // Diferencia desde el centro del kernel hasta un borde para aumentar el tamaño de la matriz con bordes
    int dif = n - 1;

    // Crear matriz con (n - 1) pixeles más por cada lado
    Mat matrix(rows + dif, cols + dif, CV_8UC1);

    // Rellenar matriz con 0s
    for (int i = 0; i < rows + dif; i++)
    {
        for (int j = 0; j < cols + dif; j++)
        {
            matrix.at<uchar>(Point(j, i)) = uchar(0);
        }
    }

    return matrix;
}

Mat adaptBorderedImage(Mat bordered, Mat original, int n)
{
    // Cantidad de pixeles desde el centro hasta el borde del kernel
    int dif = (n - 1) / 2;

    // Tomando en cuenta unicamente los pixeles desde la diferencia de pixeles de la matriz
    // con bordes y la matriz original, se recorre la matriz con bordes y se sustituyen las
    // intensidades de la matriz original en la de los bordes
    for (int i = dif; i < bordered.rows - dif; i++)
    {
        for (int j = dif; j < bordered.cols - dif; j++)
        {
            bordered.at<uchar>(Point(j, i)) = original.at<uchar>(Point(j - dif, i - dif));
        }
    }

    return bordered;
}

double convolution(Mat bordered, vector<vector<float>> mask, int n, int x, int y)
{
    float sum = 0.0;
    int dif = (n - 1) / 2;

    // Se realizan las operaciones de suma de productos de los vecinos cercanos
    for (int i = -dif; i <= dif; i++)
    {
        for (int j = -dif; j <= dif; j++)
        {
            int coordX = x + i + dif;
            int coordY = y + j + dif;

            float maskValue = mask[i + dif][j + dif];
            float imageValue = bordered.at<uchar>(coordY, coordX);

            sum += maskValue * imageValue;
        }
    }

    return sum;
}

Mat applyMask(Mat original, Mat bordered, vector<vector<float>> mask, int n)
{
    int rows = original.rows;
    int cols = original.cols;

    Mat result(rows, cols, CV_8UC1);

    // Se recorre la imagen con bordes y se le asigna el resultado de las operaciones
    // suma de productos a la matriz de resultado
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float val = abs(static_cast<int>(convolution(bordered, mask, n, j, i)));
            result.at<uchar>(Point(j, i)) = val;
        }
    }

    return result;
}

Mat sobelFilter(Mat gx, Mat gy)
{
    int rows = gx.rows;
    int cols = gx.cols;
    double intensity = 0.0;
    double valGx = 0.0, valGy = 0.0;

    Mat result(rows, cols, CV_8UC1);

    // Se realizan las operaciones pitagoricas sobre las imagenes de las mascaras
    // Gx y Gy y se binariza el resultado
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Se obtienen la intensidad de cada pixel de las imagenes creadas con los kernels Gx y Gy
            valGx = gx.at<uchar>(Point(j, i));
            valGy = gy.at<uchar>(Point(j, i));

            // Se aplica distancia euclidiana
            intensity = static_cast<int>(sqrt(pow(valGx, 2) + pow(valGy, 2)));

            result.at<uchar>(Point(j, i)) = uchar(intensity);
        }
    }

    return result;
}

vector<vector<float>> getGradient(Mat gx, Mat gy) {
    int rows = gx.rows;
    int cols = gx.cols;
    double valGx = 0.0, valGy = 0.0;

    vector<vector<float>> result(rows, vector<float>(cols, 0));

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            // Se obtienen la intensidad de cada pixel de las imagenes creadas con los kernels Gx y Gy
            valGx = gx.at<uchar>(Point(j, i));
            valGy = gy.at<uchar>(Point(j, i));

            // Se obtiene el angulo de cada pixel
            result[i][j] = atan(abs(valGy) / abs(valGx));

            // Se pasa su valor a valores sexagesimales
            result[i][j] = (result[i][j] * 180) / M_PI;

            // Validar que no haya angulos negativos
            if (result[i][j] < 0) {
                result[i][j] += 180;
            }
        }
    }

    return result;
}

Mat nonMaxSupression(Mat original, vector<vector<float>> gradient) {
    int rows = original.rows;
    int cols = original.cols;

    Mat result(rows, cols, CV_8UC1);

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            int first = 255, second = 255;

            // Si el valor del gradiente esta entre 0 y 22.5, obtiene los valores de izquierda y derecha
            if ((0 <= gradient[i][j] < 22.5) || (157.5 <= gradient[i][j] <= 180)) { // angulo 0
                first = original.at<uchar>(Point(j + 1, i));
                second = original.at<uchar>(Point(j - 1, i));
            }
            // Si el valor del gradiente esta entre 22.5 y 67.5, obtiene los valores de las esquinas
            else if (22.5 <= gradient[i][j] < 67.5) {           // angulo 45
                first = original.at<uchar>(Point(j - 1, i + 1));
                second = original.at<uchar>(Point(j + 1, i - 1));
            }
            // Si el valor del gradiente esta entre 67.5 y 112.5, obtiene los valores de arriba y abajo
            else if (67.5 <= gradient[i][j] < 112.5) {          // angulo 90
                first = original.at<uchar>(Point(j, i + 1));
                second = original.at<uchar>(Point(j, i - 1));
            }
            // Si el valor del gradiente esta entre 112.5 y 157.5, obtiene los valores de las esquinas contrarias
            else if (112.5 <= gradient[i][j] < 157.5) {         // angulo 135
                first = original.at<uchar>(Point(j - 1, i - 1));
                second = original.at<uchar>(Point(j + 1, i + 1));
            }

            // Se realiza la reduccion de bordes
            // Si los valores de first y second son menores a los de la imagen original, se conservan
            if (original.at<uchar>(Point(j, i)) >= first && original.at<uchar>(Point(j, i)) >= second) {                     // angle 90
                result.at<uchar>(Point(j, i)) = original.at<uchar>(Point(j, i));
            }
            // En caso contrario se vuelven 0
            else {
                result.at<uchar>(Point(j, i)) = 0;
            }
        }
    }

    return result;
}

Mat hysteresis(Mat original, float low, float high) {
    int rows = original.rows;
    int cols = original.cols;
    float highThreshold = 0.0, lowThreshold = 0.0;
    int max = 0;

    Mat result(rows, cols, CV_8UC1);

    // Obtener el valor maximo dentro de la imagen de entrada (original)
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (original.at<uchar>(Point(j, i)) > max) {
                max = original.at<uchar>(Point(j, i));
            }
        }
    }

    highThreshold = max * high;
    lowThreshold = highThreshold * low;

    int strong = 255;
    int weak = lowThreshold;
    int irrelevant = 0;

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (original.at<uchar>(Point(j, i)) >= highThreshold) {
                result.at<uchar>(Point(j, i)) = strong;
            }
            /*else if (lowThreshold < original.at<uchar>(Point(i, j)) < highThreshold) {
                result.at<uchar>(Point(j, i)) = weak;
            }*/
            else {
                result.at<uchar>(Point(j, i)) = irrelevant;
            }
        }
    }

    return result;
}

int main()
{
    // Declaracion de variables a usar
    Mat original, normalized, gaussianImage, equalized;
    char imageName[] = "lena.png";
    int newMin = 0, newMax = 0;
    double red = 0.0, green = 0.0, blue = 0.0;
    float sigma = 0.0;
    int n = 0;

    // Pedir valores iniciales al usuario
    cout << "Digite el largo y ancho de la mascara: ";
    cin >> n;
    cout << "Digite el valor de 'sigma': ";
    cin >> sigma;
    cout << "Digite el nuevo valor minimo para la normalizacion: ";
    cin >> newMin;
    cout << "Digite el nuevo valor maximo para la normalizacion: ";
    cin >> newMax;

    // Validaciones en la entrada de datos del usuario
    if ((n % 2) == 0)
    {
        cout << "\n\tLa mascara debe ser de longitud impar" << endl;
        exit(1);
    }

    if (sigma < 0 || sigma > 2)
    {
        cout << "\n\tEl valor de sigma debe estar entre 0 y 2" << endl;
        exit(1);
    }

    // Leer imagen y validar que se haya cargado correctamente
    original = imread(imageName);
    if (!original.data)
    {
        cout << "\n\tError al cargar la imagen: " << imageName << endl;
        exit(1);
    }

    Mat grayscaled(original.rows, original.cols, CV_8UC1);

    // Transformacion de la imagen original a escala de grises
    for (int i = 0; i < original.rows; i++)
    {
        for (int j = 0; j < original.rows; j++)
        {
            // Obtener valores de RGB de la imagen original
            blue = original.at<Vec3b>(Point(j, i)).val[0];
            green = original.at<Vec3b>(Point(j, i)).val[1];
            red = original.at<Vec3b>(Point(j, i)).val[2];

            // Crear imagen en escala de grises usando el metodo NTSC
            grayscaled.at<uchar>(Point(j, i)) = uchar((blue * 0.299) + (green * 0.587) + (red * 0.114));
        }
    }

    // Normalizar imagen en escala de grises con los valores que digito el usuario
    normalized = normalize(grayscaled, newMin, newMax);

    // Crear la matriz para almacenar la imagen con bordes y adaptarla a la imagen en escala de grises
    Mat borderedImage = createBorderedImage(normalized.rows, normalized.cols, n);
    borderedImage = adaptBorderedImage(borderedImage, normalized, n);

    // Crear y rellenar el kernel gaussiano de dimensiones n x n
    vector<vector<float>> gaussianMask = createGaussianMask(sigma, n);

    // Imprimir valores del kernel gaussiano en consola
    cout << "\nValores calculados para el kernel gaussiano de " << n << " * " << n << ":" << endl;
    printMatrix(gaussianMask, n);

    // Aplicar operaciones con el kernel gaussiano a la imagen normalizada para suavizarla
    gaussianImage = applyMask(normalized, borderedImage, gaussianMask, n);

    equalizeHist(gaussianImage, equalized);

    // Crear kernels para el filtro sobel
    vector<vector<float>> gx = createGxMask();
    vector<vector<float>> gy = createGyMask();

    // Crear matrices aplicandoles ya los kernels Gx y Gy
    Mat gxImage = applyMask(equalized, borderedImage, gx, 3);
    Mat gyImage = applyMask(equalized, borderedImage, gy, 3);

    // Crear matriz para almacenar el resultado de aplicar el filtro sobel
    Mat sobelImage = sobelFilter(gxImage, gyImage);

    // Se obtiene el gradiente de la imagen normalizada
    vector<vector<float>> gradient = getGradient(gxImage, gyImage);

    // Se disminuye el ancho de los bordes obtenidos con el filtro sobel
    Mat nonMaxImage = nonMaxSupression(sobelImage, gradient);

    // Se realiza la hysteresis
    Mat cannyImage = hysteresis(nonMaxImage, 0.35, 0.9);

    // Imprimir tamaños de imagenes en consola
    cout << "\nTamanio imagen original:" << endl;
    cout << "\tFilas: " << original.rows << endl;
    cout << "\tColumnas: " << original.cols << endl;

    cout << "\nTamanio imagen en escala de grises:" << endl;
    cout << "\tFilas: " << grayscaled.rows << endl;
    cout << "\tColumnas: " << grayscaled.cols << endl;

    cout << "\nTamanio imagen normalizada:" << endl;
    cout << "\tFilas: " << normalized.rows << endl;
    cout << "\tColumnas: " << normalized.cols << endl;

    cout << "\nTamanio imagen con filtro gaussiano:" << endl;
    cout << "\tFilas: " << gaussianImage.rows << endl;
    cout << "\tColumnas: " << gaussianImage.cols << endl;

    cout << "\nTamanio imagen con filtro gaussiano ecualizada:" << endl;
    cout << "\tFilas: " << equalized.rows << endl;
    cout << "\tColumnas: " << equalized.cols << endl;

    cout << "\nTamanio imagen con filtro de sobel:" << endl;
    cout << "\tFilas: " << sobelImage.rows << endl;
    cout << "\tColumnas: " << sobelImage.cols << endl;

    cout << "\nTamanio imagen con filtro de canny (hysteresis):" << endl;
    cout << "\tFilas: " << cannyImage.rows << endl;
    cout << "\tColumnas: " << cannyImage.cols << endl;

    // Mostrar imagenes en pantalla
    namedWindow("Imagen original", WINDOW_AUTOSIZE);
    imshow("Imagen original", original);

    namedWindow("Imagen en escala de grises", WINDOW_AUTOSIZE);
    imshow("Imagen en escala de grises", grayscaled);

    namedWindow("Imagen en escala de grises normalizada", WINDOW_AUTOSIZE);
    imshow("Imagen en escala de grises normalizada", normalized);

    /*namedWindow("Imagen con bordes", WINDOW_AUTOSIZE);
    imshow("Imagen con bordes", borderedImage);*/

    namedWindow("Imagen con filtro gaussiano", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro gaussiano", gaussianImage);

    namedWindow("Imagen con filtro gaussiano ecualizada", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro gaussiano ecualizada", equalized);

    namedWindow("Imagen con mascara Gx", WINDOW_AUTOSIZE);
    imshow("Imagen con mascara Gx", gxImage);

    namedWindow("Imagen con mascara Gy", WINDOW_AUTOSIZE);
    imshow("Imagen con mascara Gy", gyImage);

    namedWindow("Imagen con filtro de Sobel", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro de Sobel", sobelImage);

    namedWindow("Imagen con filtro de nonMax", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro de nonMax", nonMaxImage);

    namedWindow("Imagen con filtro de canny (hysteresis)", WINDOW_AUTOSIZE);
    imshow("Imagen con filtro de canny (hysteresis)", cannyImage);

    waitKey(0);
    return 1;
}