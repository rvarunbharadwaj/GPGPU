#include "RGBtoGrayScale.h"

int main()
{
	double cpuComputeTime = 0.0f;
	double gpuComputeTime = 0.0f;
	
	cout << "Loading Images" << endl;
	//Load a RGB image
	//CImg<unsigned char> imgRGB = CImg<>("D:\\Varun_GPGPU\\Images\\Images\\lena.jpg");
	//CImg<unsigned char> imgRGB = CImg<>("D:\\Varun_GPGPU\\Images\\Images\\cat.png");
	CImg<unsigned char> imgRGB = CImg<>("D:\\Varun_GPGPU\\Images\\Images\\mountain-landscape-reflection.jpg");
	
	//Display Image
	CImgDisplay dispRGB(imgRGB, "Color Image");//pass the image object and a name
	//Display Image Size
	cout << "Image Height: " << imgRGB.height()<<" pixels"<<endl; //pixels in y direction
	cout << "Image Width: " << imgRGB.width() << " pixels" << endl; //pixels in x direction
	
	//Store RGB image size in bytes
	unsigned int rgbSize = imgRGB.width() * imgRGB.height() * 3*sizeof(unsigned char); //Size of image, 3 corresponds to channels

	//Initialize a pointer to the RGB image data stored by CImg
	unsigned char* ptrRGB = imgRGB.data();

	//Create an empty image with a single channel - GrayScale //for color image specify the channels// for video specify the frames after width and height
	CImg<unsigned char>imgGrayScale(imgRGB.width(), imgRGB.height());
	//Store GrayScale image size in bytes
	unsigned int graySize = imgRGB.width() * imgRGB.height() * 1 * sizeof(unsigned char); //Grayscale has single channel

	//Initialize a pointer to the GrayScale image data stored by CImg
	unsigned char* ptrGray = imgGrayScale.data();

	//CPU Version 0
	cpuComputeTime = cpu__RGBtoGrayScale_Ver0(imgRGB, imgGrayScale);
	
	cout << "RGB to GrayScale Conversion CPU Compute Time (Ver0): " << cpuComputeTime << " usecs" << endl;
	//Display the GrayScale Image
	CImgDisplay dispGray1(imgGrayScale, "GrayScale Image CPU version 0");
	
	//CPU Version 1
	cpuComputeTime = cpu__RGBtoGrayScale_Ver1(ptrRGB, ptrGray, imgRGB.height(), imgRGB.width());
	
	cout << "RGB to GrayScale Conversion CPU Compute Time (Ver1): " << cpuComputeTime << " usecs" << endl;
	//Display the GrayScale Image
	CImgDisplay dispGray2(imgGrayScale, "GrayScale Image CPU Ver 1");

	//RGB to Grayscale Conversion using GPU
	CImg<unsigned char> gpu_imgGrayScale0(imgRGB.width(), imgRGB.height());
	unsigned char* gpu_ptrGray = gpu_imgGrayScale0.data();
	
	//GPU Versions 0
	gpuComputeTime = gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 0);
	cout << "RGB to GrayScale Conversion GPU Compute Time (Ver0): " << gpuComputeTime << " usecs" << endl;
	CImgDisplay gpu_dispGray3(gpu_imgGrayScale0, "GrayScale Image - GPU - Ver0");
	
	//GPU Versions 1
	CImg<unsigned char> gpu_imgGrayScale1(imgRGB.width(), imgRGB.height());
	gpu_ptrGray = gpu_imgGrayScale1.data();
	gpuComputeTime = gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 1);
	
	cout << "RGB to GrayScale Conversion GPU Compute Time (Ver1): " << gpuComputeTime << " usecs" << endl;
	CImgDisplay gpu_dispGray4(gpu_imgGrayScale1, "GrayScale Image - GPU - Ver1");

	//GPU Version 2
	CImg<unsigned char> gpu_imgGrayScale2(imgRGB.width(), imgRGB.height());
	gpu_ptrGray = gpu_imgGrayScale2.data();
	gpuComputeTime = gpu_RGBtoGrayScaleHelper(ptrRGB, gpu_ptrGray, rgbSize, graySize, imgRGB.height(), imgRGB.width(), 2);

	cout << "RGB to GrayScale Conversion GPU Compute Time (Ver1): " << gpuComputeTime << " usecs" << endl;
	CImgDisplay gpu_dispGray5(gpu_imgGrayScale1, "GrayScale Image - GPU - Ver2");

	cin.get();
	return 0;
}