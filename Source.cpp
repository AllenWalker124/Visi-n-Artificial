/* Alvarado Aguilar Jesús Antonio */
/* Grupo 5BM1 */
/* Examen práctico - 1er Parcial*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <math.h>

#define PI 3.14159265358979323846

using namespace cv;
using namespace std;


double gradosARadianes(double grados)
{
	return grados * PI / 180;
}

double radianesAGrados(double radianes)
{
	return radianes * 180 / PI;
}

double encontrarAngulo(double numerador, double denominador) {
	double angulo;
	double aux;
	//double num = 12;
	//double denom = 26;
	aux = numerador / denominador;
	angulo = atan(aux);
	angulo = radianesAGrados(angulo); // Convertimos los radianes a grados

	return angulo;
}

/* Funciones para convertir Mat tipo uchar a float y viceversa */
Mat matrizF2U(Mat f) {
	Mat matrizUchar;
	f.convertTo(matrizUchar, CV_8UC1);
	return matrizUchar;
}

Mat matrizU2F(Mat u) {
	Mat matrizFloat;
	u.convertTo(matrizFloat, CV_64F);
	return matrizFloat;
}

Mat filtroGaussiano(Mat matrizAumentada, Mat kernel) {
	int aum = (kernel.rows - 1) / 2;
	int r = matrizAumentada.rows - aum * 2;
	int c = matrizAumentada.cols - aum * 2;

	Mat matrizFiltro(r, c, CV_8UC1);

	for (int i = 0; i < r; i++) {
		for (int j = 0; j < c; j++) {
			double auxSuma = 0.0f;
			for (int iKernel = 0; iKernel < kernel.rows; iKernel++) {
				for (int jKernel = 0; jKernel < kernel.cols; jKernel++) {
					int xAux = iKernel - aum;
					int yAux = -jKernel + aum;

					float auxKernel = kernel.at<float>(iKernel, jKernel);
					float auxPixel = static_cast<float>(matrizAumentada.at<uchar>(i + xAux + aum, j + yAux + aum));
					auxSuma = auxSuma + (auxKernel * auxPixel);
				}
			};
			matrizFiltro.at<uchar>(i, j) = (uchar)((int)abs(auxSuma));
		}
	}
	return matrizFiltro;
}

Mat llenarKernel2(Mat kernel, int sigma) {
	Mat KernelLleno = Mat::zeros(kernel.rows, kernel.cols, CV_32F);
	float numerador, s;
	s = 2 * sigma * sigma;
	float normalizacion = 0.0;

	int a, b;
	a = floor(kernel.rows / 2);
	b = floor(kernel.cols / 2);

	// Llenando el kernel
	for (int x = -a; x <= a; x++) {
		for (int y = -b; y <= b; y++) {
			numerador = sqrt(x * x + y * y);
			KernelLleno.at<float>(x + a, y + b) = (exp(-(numerador * numerador) / s)) / (3.1416 * s);
			//matriz[x + a][y + b] = (exp(-(numerador * numerador) / s)) / (3.1416 * s);
			normalizacion += KernelLleno.at<float>(x + a, y + b);
		}
	}

	// Para normalizar la matriz (kernel)
	for (int i = 0; i < kernel.rows; i++) {
		for (int j = 0; j < kernel.cols; j++) {
			KernelLleno.at<float>(i, j) /= normalizacion;
		}
	}
	return KernelLleno;
}

void guardarExcel(Mat imagenAumentada) {
	FILE* i_orig = fopen("Gradiente.xls", "w"); //archivo excel
	for (int i = 0; i < imagenAumentada.rows; i++)
	{
		for (int j = 0; j < imagenAumentada.cols; j++) fprintf(i_orig, "%d \t", static_cast<int>(imagenAumentada.at<uchar>(i, j)));
		fprintf(i_orig, "\n");
	}
	fclose(i_orig);
}

Mat kernelSobelGx() {
	Mat SobelGx = Mat::zeros(3, 3, CV_32F);

	SobelGx.at<float>(0, 0) = -1;
	SobelGx.at<float>(0, 1) = 0;
	SobelGx.at<float>(0, 2) = 1;
	SobelGx.at<float>(1, 0) = -2;
	SobelGx.at<float>(1, 1) = 0;
	SobelGx.at<float>(1, 2) = 2;
	SobelGx.at<float>(2, 0) = -1;
	SobelGx.at<float>(2, 1) = 0;
	SobelGx.at<float>(2, 2) = 1;

	/*cout << "Sobel Gx: " << endl;
	cout << SobelGx << endl;*/
	return SobelGx;
}

Mat kernelSobelGy() {
	Mat SobelGy = Mat::zeros(3, 3, CV_32F);

	SobelGy.at<float>(0, 0) = -1;
	SobelGy.at<float>(0, 1) = -2;
	SobelGy.at<float>(0, 2) = -1;
	SobelGy.at<float>(1, 0) = 0;
	SobelGy.at<float>(1, 1) = 0;
	SobelGy.at<float>(1, 2) = 0;
	SobelGy.at<float>(2, 0) = 1;
	SobelGy.at<float>(2, 1) = 2;
	SobelGy.at<float>(2, 2) = 1;

	/*cout << "Sobel Gy: " << endl;
	cout << SobelGy << endl;*/
	return SobelGy;
}

Mat SobelG(Mat imagenGx, Mat imagenGy) {
	int f = imagenGx.rows;
	int c = imagenGx.cols;
	int x, y;
	double sumaCuadrados, aux1, aux2;
	double auxRaiz = 0.0f;

	Mat imagenSobelG = Mat::zeros(f, c, CV_8UC1);

	for (x = 0; x < f; x++) {
		for (y = 0; y < c; y++) {
			float valPixelGx = static_cast<float>(imagenGx.at<uchar>(x, y));
			float valPixelGy = static_cast<float>(imagenGy.at<uchar>(x, y));
			aux1 = pow(valPixelGx, 2);
			aux2 = pow(valPixelGy, 2);
			sumaCuadrados = aux1 + aux2;
			auxRaiz = sqrt(sumaCuadrados);
			imagenSobelG.at<uchar>(x, y) = (uchar)((double)floor(auxRaiz));
		}
	}

	return imagenSobelG;

}

Mat calcularAnguloGradiente(Mat imagenGx, Mat imagenGy) {
	int f = imagenGx.rows;
	int c = imagenGx.cols;
	int x, y;
	double angulo = 0.0f;

	Mat matrizGradiente = Mat::zeros(f, c, CV_64F);

	for (x = 0; x < f; x++) {
		for (y = 0; y < c; y++) {
			double valPixelGx = static_cast<float>(imagenGx.at<uchar>(x, y));
			double valPixelGy = static_cast<float>(imagenGy.at<uchar>(x, y));

			angulo = encontrarAngulo(valPixelGy, valPixelGx);
			matrizGradiente.at<double>(x, y) = (double)((double)angulo);
		}
	}

	return matrizGradiente;
}

int clasificarAngulo(int anguloOriginal) {
	int anguloAproximado;
	//anguloOriginal = 84;

	if (anguloOriginal < 23 || 158 <= anguloOriginal <= 180) {
		anguloAproximado = 0;
	}
	else if (anguloOriginal >= 23 && anguloOriginal < 68) {
		anguloAproximado = 45;
	}
	else if (anguloOriginal >= 68 && anguloOriginal < 113) {
		anguloAproximado = 90;
	}
	else if (anguloOriginal >= 113 && anguloOriginal < 158) {
		anguloAproximado = 135;
	}

	/*cout << "Angulo original: " << anguloOriginal << endl;
	cout << "Angulo aproximado: " << anguloAproximado << endl;*/

	return anguloAproximado;
}


Mat supresionNoMaxima(Mat magnitudGradiente, Mat anguloGradiente) {
	Mat I = Mat::zeros(magnitudGradiente.rows, magnitudGradiente.cols, CV_8UC1);

	int valorEm, valorEo, valorAnguloOriginal, valorAnguloAprox, aux1, aux2, valorDestino;	// Em = valor del pixel de la matriz magnitudGradiente,  Eo = valor del pixel de la matriz anguloGradiente

	for (int i = 1; i < magnitudGradiente.rows - 1; i++) {
		for (int j = 1; j < magnitudGradiente.cols - 1; j++) {
			valorAnguloOriginal = static_cast<int>(anguloGradiente.at<uchar>(i, j));
			valorAnguloAprox = clasificarAngulo(valorAnguloOriginal);
			valorEm = static_cast<int>(magnitudGradiente.at<uchar>(i, j));

			if (valorAnguloAprox == 0) {
				aux1 = static_cast<int>(magnitudGradiente.at<uchar>(i, j + 1));
				aux2 = static_cast<int>(magnitudGradiente.at<uchar>(i, j - 1));
				if (valorEm < aux1 || valorEm < aux2) {
					valorDestino = 0;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
				else {
					valorDestino = valorEm;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
			}
			else if (valorAnguloAprox == 45) {
				aux1 = static_cast<int>(magnitudGradiente.at<uchar>(i + 1, j - 1));
				aux2 = static_cast<int>(magnitudGradiente.at<uchar>(i - 1, j + 1));
				if (valorEm < aux1 || valorEm < aux2) {
					valorDestino = 0;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
				else {
					valorDestino = valorEm;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
			}
			else if (valorAnguloAprox == 90) {
				aux1 = static_cast<int>(magnitudGradiente.at<uchar>(i + 1, j));
				aux2 = static_cast<int>(magnitudGradiente.at<uchar>(i - 1, j));
				if (valorEm < aux1 || valorEm < aux2) {
					valorDestino = 0;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
				else {
					valorDestino = valorEm;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
			}
			else if (valorAnguloAprox == 135) {
				aux1 = static_cast<int>(magnitudGradiente.at<uchar>(i - 1, j - 1));
				aux2 = static_cast<int>(magnitudGradiente.at<uchar>(i + 1, j + 1));
				if (valorEm < aux1 || valorEm < aux2) {
					valorDestino = 0;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
				else {
					valorDestino = valorEm;
					I.at<uchar>(i, j) = uchar(valorDestino);
				}
			}
		}
	}

	return I;
}

Mat calcularUmbrales(Mat SupNoMax) {
	int f, c;
	f = SupNoMax.rows;
	c = SupNoMax.cols;
	double valUmbralAlto =  100;
	double valUmbralBajo =  220;

	Mat umbralizada = Mat::zeros(f, c, CV_8UC1);

	/* Obtenemos el valor más grande de nuestra matríz */

	float max = 0;

	for (int x = 0; x < f; x++)
	{
		for (int y = 0; y < c; y++)
		{
			float valor = static_cast<float>(SupNoMax.at<uchar>(x, y));

			if (valor > max) {
				max = valor;
			}
		}
	}

	for (int i = 0; i < f; i++) {
		for (int j = 0; j < c; j++) {
			float auxPos = static_cast<float>(SupNoMax.at<uchar>(i, j));
			//cout << auxPos << endl;
			if (auxPos >= valUmbralAlto) {
				umbralizada.at<uchar>(i, j) = static_cast<uchar>(255);
			}
			else if ((auxPos < valUmbralAlto) && (auxPos >= valUmbralBajo)) {
				umbralizada.at<uchar>(i, j) = static_cast<uchar>(25);
			}
			else if (auxPos < valUmbralBajo) {
				umbralizada.at<uchar>(i, j) = static_cast<uchar>(0);
			}
		}
	}

	return umbralizada;
}

Mat histeresis(Mat imagen) {
	int f = imagen.rows;
	int c = imagen.cols;
	int i, j;
	float fuerte = 255;
	float debil = 25;
	Mat aux = Mat::zeros(f, c, CV_64F);
	aux = matrizU2F(imagen);

	for (i = 1; i < f; i++) {
		for (j = 1; j < c; j++) {
			if (aux.at<float>(i, j) == debil) {
				if ((aux.at<float>(i + 1, j - 1) == fuerte) || (aux.at<float>(i + 1, j) == fuerte) || (aux.at<float>(i + 1, j + 1) == fuerte) || (aux.at<float>(i, j - 1) == fuerte) || (aux.at<float>(i, j + 1) == fuerte) || (aux.at<float>(i - 1, j - 1) == fuerte) || (aux.at<float>(i - 1, j) == fuerte) || (aux.at<float>(i - 1, j + 1) == fuerte)) {
					aux.at<float>(i, j) = fuerte;
				}
				else {
					aux.at<float>(i, j) = debil;
				}
			}
		}
	}

	return matrizF2U(aux);
} 

void principal(int size, int sigma) {
	int i, j, q, k;

	// Abrimos una imagen para aplicarle el filtro gaussiano con el Kernel:
	char imageName[] = "C:/Users/caran/OneDrive/Documentos/Tareas 5to Semestre/Visión Artificial/Prácticas/Práctica 2/Practica2/lena.png";
	Mat image;
	image = imread(imageName);

	// Convertimos a imagen original a color a escala de grises:
	Mat imGrises(image.rows, image.cols, CV_8UC1);
	cvtColor(image, imGrises, COLOR_RGB2GRAY);

	// Obtenemos la nueva imagen con bordes aumentados
	Mat imagenAumentada = Mat::zeros(image.rows + size - 1, image.cols + size - 1, CV_8UC1);
	Mat kernel = Mat::zeros(size, size, CV_32F);

	kernel = llenarKernel2(kernel, sigma);
	cout << "Kernel calculado: " << endl;
	cout << kernel << endl << endl;

	// Imagen con bordes aumentados

	for (i = int(size / 2), q = 0; i < imagenAumentada.rows - int(size / 2); i++, q++) {
		for (j = int(size / 2), k = 0; j < imagenAumentada.cols - int(size / 2); j++, k++) {
			imagenAumentada.at<uchar>(i, j) = imGrises.at<uchar>(q, k);
		}
	}

	// guardarExcel(imagenAumentada);


	// Aplicamos Filtro Gaussiano

	Mat imFiltroGauss = Mat::zeros(image.rows, image.cols, CV_8UC1);
	imFiltroGauss = filtroGaussiano(imagenAumentada, kernel);


	cout << "Tamaño imagen original: \t Rows: " << image.rows << "     Cols: " << image.cols << endl;
	cout << "Tamaño imagen escala de grises: \t Rows: " << imGrises.rows << "     Cols: " << imGrises.cols << endl;
	//cout << "Tamaño imagen con bordes: \t Rows: " << imagenAumentada.rows << "     Cols: " << imagenAumentada.cols << endl;
	cout << "Tamaño imagen filtro Gaussiano: \t Rows: " << imFiltroGauss.rows << "     Cols: " << imFiltroGauss.cols << endl;


	namedWindow("Imagen original", WINDOW_AUTOSIZE);
	imshow("Imagen original", image);
	imshow("Imagen en escala de grises", imGrises);
	//imshow("Imagen con bordes aumentados", imagenAumentada);
	imshow("Imagen con filtro gaussiano", imFiltroGauss);


	/* Hasta aquí termina la aplicación del filtro Gaussiano y comienza Sobel */


	/* Antes de aplicar Sobel, ecualizamos nuestra imagen */

	Mat imEcualizada = Mat::zeros(image.rows, image.cols, CV_8UC1);
	equalizeHist(imFiltroGauss, imEcualizada);
	cout << "Tamaño imagen ecualizada: \t Rows: " << imEcualizada.rows << "     Cols: " << imEcualizada.cols << endl;
	imshow("Imagen ecualizada", imEcualizada);

	/* Continuamos con Sobel */

	Mat SobelGx = kernelSobelGx();
	Mat SobelGy = kernelSobelGy();

	Mat imagenAumentadaSobelGx = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);  // Se le suman 2 (o lo que es igual a 3 - 1) por que son los bordes que aumentan al operar con las matrices de tamaño 3x3 de SobelGx y SobelGy
	Mat imagenAumentadaSobelGy = Mat::zeros(image.rows + 2, image.cols + 2, CV_8UC1);

	// Copiamos los valores de la imagen con filtro gaussiano a las nuevas matrices para aplicar Sobel:

	for (i = 1, q = 0; i < imagenAumentadaSobelGx.rows - 1; i++, q++) {
		for (j = 1, k = 0; j < imagenAumentadaSobelGx.cols - 1; j++, k++) {
			imagenAumentadaSobelGx.at<uchar>(i, j) = static_cast<float>(imEcualizada.at<uchar>(q, k));
		}
	}

	for (i = 1, q = 0; i < imagenAumentadaSobelGy.rows - 1; i++, q++) {
		for (j = 1, k = 0; j < imagenAumentadaSobelGy.cols - 1; j++, k++) {
			imagenAumentadaSobelGy.at<uchar>(i, j) = static_cast<float>(imEcualizada.at<uchar>(q, k));
		}
	}


	/* Para evitar crear otra función que aplique el operador Sobel sobre la imagen ya con el filtro de Gauss, se utilizará la misma función de filtroGaussiano() pero ahora se le pasaran
	como parámetros la imagen con el filtro y los kernels de Gx y Gy correspondientes a Sobel */

	Mat imOperadorSobelGx = Mat::zeros(image.rows, image.cols, CV_64F);
	imOperadorSobelGx = filtroGaussiano(imagenAumentadaSobelGx, SobelGx);

	Mat imOperadorSobelGy = Mat::zeros(image.rows, image.cols, CV_64F);
	imOperadorSobelGy = filtroGaussiano(imagenAumentadaSobelGy, SobelGy);


	/* Ya que tenemos ambas Gx y Gy, debemos obtener la imagen |G|*/

	Mat magnitudGradiente = Mat::zeros(image.rows, image.cols, CV_8UC1);
	magnitudGradiente = SobelG(imOperadorSobelGx, imOperadorSobelGy);         

	/*Mat magnitudGradienteMostrar;
	magnitudGradienteMostrar = matrizF2U(magnitudGradiente);*/

	cout << "Tamaño imagen |G|: \t Rows: " << magnitudGradiente.rows << "     Cols: " << magnitudGradiente.cols << endl;
	imshow("Imagen Magnitud Gradiente |G|", magnitudGradiente);

	/* Falta calcular el ángulo */

	Mat anguloGradiente = Mat::zeros(image.rows, image.cols, CV_64F);
	anguloGradiente = calcularAnguloGradiente(imOperadorSobelGx, imOperadorSobelGy);					

	//guardarExcel(anguloGradiente);

	//Mat anguloGradienteMostrar;
	//anguloGradienteMostrar = matrizF2U(anguloGradiente);	// Esta solo es una conversión para mostrar la imagen 


	/* Aplicamos Supresión No Máxima */
	Mat SupNoMax = Mat::zeros(image.rows, image.cols, CV_8UC1);
	SupNoMax = supresionNoMaxima(magnitudGradiente, anguloGradiente);		

	/*cout << "Tamaño imagen gradiente: \t Rows: " << SupNoMax.rows << "     Cols: " << SupNoMax.cols << endl;
	imshow("Imagen Supresión No Máxima", SupNoMax);*/

	/* Calculamos los umbrales */

	Mat Umbrales = Mat::zeros(image.rows, image.cols, CV_8UC1);
	Umbrales = calcularUmbrales(SupNoMax);
	//imshow("Imagen Threshold", Umbrales);
	

	/* Calculamos la histéresis */

	Mat ImFinal = Mat::zeros(image.rows, image.cols, CV_8UC1);
	ImFinal = histeresis(Umbrales);

	cout << "Tamaño imagen filtro Canny: \t Rows: " << ImFinal.rows << "     Cols: " << ImFinal.cols << endl;
	imshow("Imagen Filtro Canny", ImFinal);
}


int main()
{
	float sigma;
	int k;

	cout << "Kernel size: ";
	cin >> k;
	cout << "Sigma: ";
	cin >> sigma;
	cout << endl << endl;

	principal(k, sigma);

	waitKey(0);
	return 0;
}