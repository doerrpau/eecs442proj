#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "segment.h"

using namespace std;
using namespace cv;

/* Global Constants */
const double IMAGE_WIDTH = 360.0;
const double IMAGE_DISPLAY_WIDTH = 720.0;
const SpeedUpLevel ms_speedup = HIGH_SPEEDUP;

/* Global Variables */
bool drawing_box;
cv::Rect box;
Mat orig;

char* filename;

int sigmaR = 5;
double sigmaS = 7.0;
int minRegion = 100;
double scale;

/* Function Prototypes */
Mat doMeanShift(Mat, int, double, int, SpeedUpLevel);
Mat doKMeans(Mat);
Mat doGrabCut(Mat, cv::Rect);

void callMouse(int event,int x,int y,int flags,void* param)
{

    switch( event )
    {
        case CV_EVENT_LBUTTONDOWN:
        {
            drawing_box=true;
            box = cvRect( x, y, 0, 0 );
        }
            break;
        case CV_EVENT_MOUSEMOVE:
        {
            if( drawing_box )
            {
                box.width = x-box.x;
                box.height = y-box.y;
            }
            break;
        }
        case CV_EVENT_LBUTTONUP:
        {   
            drawing_box=false;
            if( box.width < 0 )
            {
                box.x += box.width;
                box.width *= -1;
            }

            if( box.height < 0 )
            {
                box.y += box.height;
                box.height *= -1;
            }

            // Cast input image from void* to Mat*
            cv::Mat* image  = static_cast<cv::Mat *>(param);
            image->copyTo(orig);

            Mat grabCutImg = doGrabCut(*image, box);
             
            // display result
            namedWindow("GrabCut Dog");
            imshow("GrabCut Dog",grabCutImg);

            orig.copyTo(*image);

            break;
        }
        default:
            break;
    }
}

int main(int argc, char** argv)
{
    /* Get file to process from arguments */
    if (argc >= 2) {
        filename = argv[1];
    } else {
        cout << "No filename provided" << endl;
        return 1;
    }

    /* Load the image */
    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    Mat s_image;
    double scale = IMAGE_WIDTH/image.cols;
    resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 

    /* kmeans segmentation on the image */
    Mat kMeansImg = doKMeans(s_image);
    /* mean shift segmentation on the image */
    Mat meanShiftImg = doMeanShift(s_image, sigmaR, sigmaS, minRegion, ms_speedup);
    /* Upscale images */
    scale = IMAGE_DISPLAY_WIDTH/meanShiftImg.cols;
    resize(kMeansImg, kMeansImg, Size(0,0), scale, scale);
    resize(meanShiftImg, meanShiftImg, Size(0,0), scale, scale);
    resize(s_image, s_image, Size(0,0), scale, scale);

    /* Display the original image and the segmented images */
    namedWindow("Dog");
    imshow("Dog",s_image);
    namedWindow("K Means Dog");
    imshow( "K Means Dog", kMeansImg);
    namedWindow("Mean Shift Dog");
    imshow( "Mean Shift Dog", meanShiftImg); 
    
    setMouseCallback("Dog",callMouse,&s_image);

    //End program by hitting any key
    waitKey();
    return 0;
}

Mat doKMeans(Mat inImg)
{
    /* Format image array for kmeans */
    Mat samples(inImg.rows * inImg.cols, 3, CV_32F);
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) {
            for( int z = 0; z < 3; z++) {
                samples.at<float>(y + x*inImg.rows, z) = inImg.at<Vec3b>(y,x)[z];
            }
        }
    }

    /* Setup kmeans parameters */
    int clusterCount = 5;
    Mat labels;
    int attempts = 5;
    Mat centers;

    /* kmeans clustering */
    kmeans(samples, clusterCount, labels, TermCriteria(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001), 
        attempts, KMEANS_PP_CENTERS, centers );

    /* Get kmeans image */
    Mat new_image( inImg.size(), inImg.type() );
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) { 
            int cluster_idx = labels.at<int>(y + x*inImg.rows,0);
            new_image.at<Vec3b>(y,x)[0] = centers.at<float>(cluster_idx, 0);
            new_image.at<Vec3b>(y,x)[1] = centers.at<float>(cluster_idx, 1);
            new_image.at<Vec3b>(y,x)[2] = centers.at<float>(cluster_idx, 2);
        }
    }

    return new_image;
}
