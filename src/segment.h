#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "segm/msImageProcessor.h"
#include <iostream>

#ifndef SEGMENT_H
#define SEGMENT_H

using namespace std;
using namespace cv;

Mat doGrabCut(Mat inImg, cv::Rect rectangle)
{   
    /* segmentation result (4 possible values) */
    Mat result;
    /* the models (internally used) */
    Mat bgModel,fgModel;

    cv::rectangle(inImg, rectangle, Scalar(255,255,255),1);
    /* Perform GrabCut segmentation */
    grabCut(inImg, result, rectangle, bgModel, fgModel, 1, GC_INIT_WITH_RECT);
    
    /* Get the foreground */
    compare(result,GC_PR_FGD,result,CMP_EQ);
    /* Put foreground pixels in the result */
    Mat foreground(inImg.size(),CV_8UC3,Scalar(255,255,255));
    inImg.copyTo(foreground,result);

    return foreground;
}

/* Image Segmentation functions */
Mat doMeanShift(Mat inImg, int sigmaR, double sigmaS, int minRegion, SpeedUpLevel speedup)
{
    /* Convert image into 1-dim array of bytes in RGB */
    unsigned char *img_array = new unsigned char[inImg.rows * inImg.cols * 3];
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) {
            for( int z = 0; z < 3; z++) {
                img_array[3*(x + y*inImg.cols) + z] = inImg.at<Vec3b>(y,x)[2-z];
            }
        }
    }

    /* Create meanshift object and put in image data */
    msImageProcessor meanshift;
    meanshift.DefineImage(img_array, COLOR, inImg.rows, inImg.cols);
    
    /* Perform meanshift segmentation */
    meanshift.Segment(sigmaR, sigmaS, minRegion, speedup);

    /* Get the resulting data and convert into Mat */
    unsigned char *out_array = new unsigned char[inImg.rows * inImg.cols * 3];
    meanshift.GetResults(out_array);
    
    Mat new_image( inImg.size(), inImg.type() );
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) { 
            new_image.at<Vec3b>(y,x)[2] = out_array[3*(x + y*inImg.cols) + 0];
            new_image.at<Vec3b>(y,x)[1] = out_array[3*(x + y*inImg.cols) + 1];
            new_image.at<Vec3b>(y,x)[0] = out_array[3*(x + y*inImg.cols) + 2];
        }
    }

    delete[] img_array;
    delete[] out_array;

    return new_image;
}

#endif
