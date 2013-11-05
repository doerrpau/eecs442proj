#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "segm/msImageProcessor.h"

using namespace std;
using namespace cv;

/* Global Constants */
const double IMAGE_WIDTH = 360.0;
const double IMAGE_DISPLAY_WIDTH = 720.0;
const int sigmaR_max = 100;
const int sigmaS_max = 100;
const int minRegion_max = 1000;
const SpeedUpLevel ms_speedup = HIGH_SPEEDUP;

/* Global Variables */
char* filename;

int sigmaR = 5;
int sigmaS_slider = 7;
double sigmaS = 7.0;
int minRegion = 100;
double scale;
Mat image;
Mat s_image;

/* Function Prototypes */
Mat doMeanShift(Mat, int, double, int, SpeedUpLevel);
Mat doKMeans(Mat);
Mat doGrabCut(Mat, cv::Rect);
void do_trackbars(int, void*);

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
    image = imread(filename, CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    scale = IMAGE_WIDTH/image.cols;
    resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 

    /* Display the original image and the segmented images */
    //namedWindow("Dog");
    //imshow("Dog",image);
    namedWindow("Mean Shift");

    createTrackbar("SigmaR", "Mean Shift", &sigmaR, sigmaR_max, do_trackbars);
    createTrackbar("SigmaS", "Mean Shift", &sigmaS_slider, sigmaS_max, do_trackbars);
    createTrackbar("Min Region", "Mean Shift", &minRegion, minRegion_max, do_trackbars);

    /* Inital display */
    do_trackbars(sigmaR, 0);

    waitKey();
    return 0;
}

/* Trackbar callback function */
void do_trackbars(int, void*)
{
    /* Calculate parameters */
    sigmaS = 100.0 * (double)sigmaS_slider / (double)sigmaS_max;

    /* mean shift segmentation on the image */
    Mat meanShiftImg = doMeanShift(s_image, sigmaR, sigmaS, minRegion, ms_speedup);

    /* Upscale image */
    scale = IMAGE_DISPLAY_WIDTH/meanShiftImg.cols;
    resize(meanShiftImg, meanShiftImg, Size(0,0), scale, scale);

    imshow( "Mean Shift", meanShiftImg);
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
