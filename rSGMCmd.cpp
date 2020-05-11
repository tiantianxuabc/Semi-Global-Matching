// Copyright ?Robert Spangenberg, 2014.
// See license.txt for more details

#include "iostream"
#include <bitset>
#include <smmintrin.h> // intrinsics
#include <emmintrin.h>
#include<fstream>
#include<stdio.h>
#include "StereoBMHelper.h"
#include "FastFilters.h"
#include "StereoSGM.h"


#include <cmath>
#include <opencv.hpp>

#include<time.h>

#include <vector>
#include <list>
#include <algorithm>
#include <numeric>
#include <iostream>

#include "MyImage.h"

#if _DEBUG
#pragma  comment(lib, "opencv_world430d.lib")

#else
#pragma comment(lib, "opencv_world430.lib")

#endif

const int dispRange = 256;


void correctEndianness(uint16* input, uint16* output, uint32 size)
{
    uint8* outputByte = (uint8*)output;
    uint8* inputByte = (uint8*)input;

    for (uint32 i=0; i < size; i++) 
	{
        *(outputByte+1) = *inputByte;
        *(outputByte) = *(inputByte+1);
        outputByte+=2;
        inputByte+=2;
    }
}

template<typename T>
void processCensus5x5SGM(T* leftImg, T* rightImg, float32* output, float32* dispImgRight,
    int width, int height, uint16 paths, const int dispCount)
{
    const int maxDisp = dispCount - 1;

    std::cout << std::endl <<paths << ", " << dispCount << std::endl;

    // get memory and init sgm params
    uint32* leftImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);
    uint32* rightImgCensus = (uint32*)_mm_malloc(width*height*sizeof(uint32), 16);

    StereoSGMParams_t params;
    params.lrCheck = true;
    params.MedianFilter = true;
    params.Paths = paths;
  
    params.NoPasses = 2;    

	//为dis分配存储空间，（width*height*(64)*sizeof(uint16)）//存储Data Cost Calculation
	uint16* dsi = (uint16*)_mm_malloc(width*height*(maxDisp + 1)*sizeof(uint16), 32);
	StereoSGM<T> m_sgm16(width, height, maxDisp, params);
   
	/*求取左图像的Census结果，结果存入leftImgCensus*/
	census5x5_16bit_SSE(leftImg, leftImgCensus, width, height);
	/*求取右图像的Census结果，结果存入rightImgCensus*/
	census5x5_16bit_SSE(rightImg, rightImgCensus, width, height);
	//得到视差空间dsi[X*Y*D]
	costMeasureCensus5x5_xyd_SSE(leftImgCensus, rightImgCensus, height, width, dispCount, params.InvalidDispCost, dsi);
	

	m_sgm16.process(dsi, leftImg, output, dispImgRight);      
	_mm_free(dsi);    
}

void onMouse(int event, int x, int y, int flags, void *param)
{
	cv::Mat *im = reinterpret_cast<cv::Mat *>(param);
	switch(event)
	{
	case cv::EVENT_LBUTTONDBLCLK:
		std::cout<<"at ("<<std::setw(3)<<x<<","<<std::setw(3)<<y<<") value is: "
			<<static_cast<int>(im->at<uchar>(cv::Point(x,y)))<<std::endl;
		break;
	}
}

void jet(float x, int& r, int& g, int& b)
{
	if (x < 0) x = -0.05;
	if (x > 1) x = 1.05;
	x = x / 1.15 + 0.1; // use slightly asymmetric range to avoid darkest shades of blue.
	r = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .75))))));
	g = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .5))))));
	b = __max(0, __min(255, (int)(round(255 * (1.5 - 4 * fabs(x - .25))))));
}

cv::Mat Float2ColorJet(cv::Mat &fimg, float dmin, float dmax)
{

	int width = fimg.cols, height = fimg.rows;
	cv::Mat img(height, width, CV_8UC3);

	float scale = 1.0 / (dmax - dmin);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float f = fimg.at<float>(y, x);
			int r = 0;
			int g = 0;
			int b = 0;

			/*if (f != INFINITY)*/ {
				float val = scale * (f - dmin);
				jet(val, r, g, b);
			}

			img.at<cv::Vec3b>(y, x) = cv::Vec3b(b, g, r);
		}
	}

	return img;
}


int formatJPG()
{
	
	cv::Mat imgL = cv::imread("D:\\L0.png", cv::IMREAD_GRAYSCALE);
	if(imgL.empty())
	{
		std::cout<<"Left image does not exist!";
		system("pause");
		return 0;
	}

	
	
	
// 	cv::transpose(imgL, imgL);
// 	cv::flip(imgL, imgL, 1);
// 	cv::flip(imgL, imgL, 1);


	

	cv::Mat imgR = cv::imread("D:\\R0.png", cv::IMREAD_GRAYSCALE);
	if(imgR.empty())
	{
		std::cout<<"Right image does not exist!";
		system("pause");
		return 0;
	}

	/*cv::imwrite("E:/000000_L_1.png", imgR);*/

// 	cv::transpose(imgR, imgR);
// 	cv::flip(imgR, imgR, 1);
// 	cv::flip(imgR, imgR, 1);

	int cols_ = 0;
	int rows_ = 0;

	if(imgL.cols != imgR.cols && imgL.rows != imgL.rows)
	{
		std::cout<<"The size of left and right is not same!"<<std::endl;
		system("pause");
	}
	if( imgL.cols % 16 != 0 || cols_ % 16 != 0)
	{
		cols_ = imgL.cols - imgL.cols % 16;
		rows_ = imgL.rows;
		std::cout<<"Modify Image width be a multiple of 16 to calculate!"<<std::endl;		
		std::cout<<"image 1(rows X cols): "<<rows_<<" X "<<cols_<<std::endl;
		std::cout<<"image 2(rows X cols): "<<rows_<<" X "<<cols_<<std::endl;
	}
	else
	{
		cols_ = imgL.cols;
		rows_ = imgL.rows;
		std::cout<<"image 1(rows X cols): "<<rows_<<" X "<<cols_<<std::endl;
		std::cout<<"image 2(rows X cols): "<<rows_<<" X "<<cols_<<std::endl;
	}

	cv::Mat imgDisp(cv::Size(cols_, rows_), CV_8U);
	
	uint16* leftImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	uint16* rightImg = (uint16*)_mm_malloc(rows_*cols_*sizeof(uint16), 16);
	for(int i = 0; i < rows_; i++)
	{
		for(int j = 0; j < cols_; j++)
		{
			leftImg[i * cols_ + j ] = *(imgL.data + i*imgL.step + j * imgL.elemSize());
			rightImg[i * cols_ + j ] = *(imgR.data + i*imgR.step + j * imgR.elemSize());
		}
	}

	//左右图像的视差图分配存储空间（width*height*sizeof(float32)）
	float32* dispImg      = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);
	float32* dispImgRight = (float32*)_mm_malloc(rows_*cols_*sizeof(float32), 16);

	
	const int numPaths = 8;		
	
	clock_t start = clock();

	processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, cols_, rows_, numPaths, dispRange);

	clock_t end = clock();

	std::cout << "The execute time of SGM is: " << (double)(end - start) / CLOCKS_PER_SEC << std::endl;

	

	cv::Mat imgFloat(cv::Size(cols_, rows_), CV_32FC1);
	int test = 0;
	for(int i = 0; i < rows_; i++)
	{
		for(int j = 0; j < cols_; j++)
		{
			if(dispImg[i * cols_ + j] >= 0)
			{
				if (/*(uint8)*/dispImg[i * cols_ + j] > dispRange - 1)
				{
					imgDisp.at<uchar>(i, j) = dispRange - 1;
				}
				else
				{
					imgDisp.at<uchar>(i, j) = (uint8)dispImg[i * cols_ + j];
				}
				imgFloat.at<float>(i, j) = dispImg[i * cols_ + j];
				
			}
			else
			{
				imgDisp.at<uchar>(i, j) = 0;
				imgFloat.at<float>(i, j) = 0;
			}
		}
	}


	
	/************************************************************************/
/*                                                                      */
/************************************************************************/

	/*************************v-disparity****************************/
// 	cv::Mat vDisp(cv::Size(dispRange, imgL.rows), CV_8U);
// 	uint8 tmp = 0;
// 	int dispMax = 0;
// 	for(int i = 0; i < imgL.rows; i++)
// 	{
// 		uint8 a[dispRange] = {0};
// 		for (int j = 0; j <  cols_; j++)
// 		{
// 			if( (tmp = *(imgDisp.data + i*imgDisp.step + j*imgDisp.elemSize())) < dispRange)
// 			{
// 				if(a[tmp] < 255)
// 				{
// 					a[tmp] = ++a[tmp];
// 				}				
// 			}
// 			else
// 			{
// 				if(a[dispRange - 1] < 255)
// 				{
// 					a[dispRange - 1] = ++a[dispRange - 1];
// 				}
// 			}			
// 		}
// 
// 		for(int k = 0; k < dispRange; k++)
// 		{
// 			*(vDisp.data + i*vDisp.step + k*vDisp.elemSize()) = a[k];			
// 		}
// 	}
// 	cv::imwrite("E:\\vDisp.png", vDisp);
	/**************************************************************/

	/*************************u-disparity*************************/
//  	cv::Mat uDisp(cv::Size(cols_, dispRange), CV_8U);
//  	
//  	for(int i = 0; i < cols_; i++)
//  	{
//  		int a[dispRange] = {0};
//  		tmp = 0;
//  		for (int j = 0; j <  imgL.rows; j++)
//  		{
//  			if( (tmp = *(imgDisp.data + j*imgDisp.step + i*imgDisp.elemSize())) < dispRange)
//  			{
//  				if(a[tmp] < 255)
//  				{
//  					a[tmp] = ++a[tmp];
//  				}				
//  			}
//  			else
//  			{
//  				if(a[dispRange - 1] < 255)
//  				{
//  					a[dispRange - 1] = ++a[dispRange - 1];
//  				}				
//  			}
//  		}
//  		for(int k = 0; k < dispRange; k++)
//  		{
//  			*(uDisp.data + k*uDisp.step + i*uDisp.elemSize()) = a[k];			
//  		}		
//  	}
 	//cv::imwrite("E:\\uDisp.png",uDisp);
	/**************************************************************/
	

	
	
	//cv::imwrite("E:\\view_3.png",imgDisp);
	cv::namedWindow("LeftImg");
	cv::imshow("LeftImg",imgL);
	

// 	cv::namedWindow("RightImg");
// 	cv::imshow("RightImg",imgR);
	

	cv::namedWindow("DispImg");
	cv::imshow("DispImg",imgDisp);	

	double minFloat, maxFloat;
	

	cv::namedWindow("ColorDispImg");
	cv::imshow("ColorDispImg", Float2ColorJet(imgFloat, 0, dispRange));
	
	cv::imwrite("D:/Disp.jpg", imgDisp);
	cv::imwrite("D:/ColorDispImg.jpg", Float2ColorJet(imgFloat, 0, dispRange));

	while(1)
	{
		cv::setMouseCallback("DispImg", onMouse, reinterpret_cast<void*>(&imgDisp));
		if(cv::waitKey(0) == 27)
			break;
	}
	
	
	
	_mm_free(leftImg);
	_mm_free(rightImg);
	
	return 0;
}

int formatPGM()
{
	char *im1name  = "E:\\PGMImg\\CarTruckaloe_left\\000015_0.pgm";
	char *im2name  = "E:\\PGMImg\\CarTruckaloe_left\\000015_1.pgm";
	char *dispname = "E:\\000015_0.jpg";

	fillPopCount16LUT();
	
	MyImage<uint16> myImg1, myImg2;
	readPGM(myImg1, im1name);
	readPGM(myImg2, im2name);
	 
	std::cout << "image 1 " << myImg1.getWidth() << "x" << myImg1.getHeight() << std::endl;
	std::cout << "image 2 " << myImg2.getWidth() << "x" << myImg2.getHeight() << std::endl;


	if (myImg1.getWidth() % 16 != 0) 
	{
		std::cout << "Image width must be a multiple of 16" << std::endl;
		return 0;
	}

	
	//分别为leftImg 和 rightImg分配 存储空间（width*height*sizeof(uint16)）
	uint16* leftImg = (uint16*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(uint16), 16);
	uint16* rightImg = (uint16*)_mm_malloc(myImg2.getWidth()*myImg2.getHeight()*sizeof(uint16), 16);

	//纠正字节顺序
	correctEndianness((uint16*)myImg1.getData(), leftImg, myImg1.getWidth()*myImg1.getHeight());
	correctEndianness((uint16*)myImg2.getData(), rightImg, myImg1.getWidth()*myImg1.getHeight());

	clock_t start = clock();
	//左右图像的视差图分配存储空间（width*height*sizeof(float32)）
	float32* dispImg = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);
	float32* dispImgRight = (float32*)_mm_malloc(myImg1.getWidth()*myImg1.getHeight()*sizeof(float32), 16);
	
	const int dispRange = 128;	
	const int numPaths = 8;		
	processCensus5x5SGM(leftImg, rightImg, dispImg, dispImgRight, myImg1.getWidth(), myImg1.getHeight(), numPaths, dispRange);

	

	cv::Mat imgDisp(cv::Size(myImg1.getWidth(),  myImg1.getHeight()), CV_8U);

	for(unsigned int i = 0; i <  myImg1.getHeight(); i++)
	{
		for(unsigned int j = 0; j < myImg1.getWidth(); j++)
		{
			if(dispImg[i * myImg1.getWidth() + j] > 0)
			{
				*(imgDisp.data + i*imgDisp.step + j * imgDisp.elemSize()) =  dispImg[i * myImg1.getWidth() + j];
			}
			else
			{
				*(imgDisp.data + i*imgDisp.step + j * imgDisp.elemSize()) = 0;
			}
		}
	}
	clock_t endtime = clock();
	std::cout<<std::endl;
	std::cout<<"time is "<<(double)(endtime - start)/CLOCKS_PER_SEC<<std::endl;

	

	 cv::namedWindow("DispResult");
	 cv::imshow("DispResult",imgDisp);
	 cv::imwrite(dispname,imgDisp);
	 cv::setMouseCallback("DispResult", onMouse, reinterpret_cast<void*>(&imgDisp));
	 cv::waitKey(0);
	
	_mm_free(leftImg);
	_mm_free(rightImg);
	_mm_free(dispImg);
	_mm_free(dispImgRight);	

	return 0;
}
int main()
{	
	return formatJPG();
	/*return formatPGM();*/
}