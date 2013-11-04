#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "segm/msImageProcessor.h"
#include <iostream>

using namespace std;
using namespace cv;

#define IMAGE_WIDTH 480.0

Mat doMeanShift(Mat);
Mat doKMeans(Mat);
Mat doGrabCut(Mat, cv::Rect);

bool drawing_box;
cv::Rect box;
Mat orig;

void drawTextBox(Mat img, String text, Scalar bgColor,Scalar fgColor, Point coords)
{
    int scale = 1;
    // Draw a text box on image
    // Note about locations: putText draws from lower left corner, while rectangle does verticies
    // Note about color: it's BGR, not RGB

    rectangle(img, coords,Point(coords.x+text.length()*19.5*scale,coords.y+50*scale), bgColor, -1, 8, 0);
    putText(img, text, Point(coords.x+5,coords.y+25), FONT_HERSHEY_TRIPLEX, scale, fgColor,2, 8,false);
}

void callMouse(int event,int x,int y,int flags,void* param)
{

    switch( event )
    {
        // Event handling
        case CV_EVENT_LBUTTONDOWN:
        {
            // Start drawing box
            drawing_box=true;
            box = cvRect( x, y, 0, 0 );
            break;
        }

        case CV_EVENT_MOUSEMOVE:
        {
            // Alter box box params
            if( drawing_box )
            {
                box.width = x-box.x;
                box.height = y-box.y;
            }
            break;
        }
        case CV_EVENT_LBUTTONUP:
        {   
            // Finish drawing box
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

            // Draw & show
            rectangle(*image, box,Scalar(255,255,255),1);

            // Reshow image
            imshow("Dog",*image);

            Mat grabCutImg = doGrabCut(*image, box);
            
            // Adding text above segmentation
            drawTextBox(grabCutImg, "I CAN HAZ FRIZBEE?", Scalar(000,000,000),Scalar(255,255,255), Point(box.x, box.height));

            // display result
            namedWindow("GrabCut Dog");
            imshow("GrabCut Dog",grabCutImg);

            orig.copyTo(*image);

            break;
        }
        default:
            drawing_box=false;
            box = cvRect( x, y, 0, 0 );
            break;
    }
}

int main(int argc, char** argv)
{
    /* Load the image */
    Mat image = imread("../images/dog.jpg", CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    double scale = IMAGE_WIDTH/image.cols;
    resize(image, image, Size(0,0), scale, scale, INTER_AREA); 

    /* kmeans segmentation on the image */
    Mat kMeansImg = doKMeans(image);
    /* mean shift segmentation on the image */
    Mat meanShiftImg = doMeanShift(image);

    /* Display the original image and the segmented images */
    namedWindow("Dog");
    imshow("Dog",image);
    namedWindow("K Means Dog");
    imshow( "K Means Dog", kMeansImg);
    namedWindow("Mean Shift Dog");
    imshow( "Mean Shift Dog", meanShiftImg); 
    
    setMouseCallback("Dog",callMouse,&image);

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
Mat doMeanShift(Mat inImg)
{
    /* Convert image into 1-dim array of bytes in RGB */
    unsigned char *img_array = new unsigned char[inImg.rows * inImg.cols * 3];
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) {
            for( int z = 0; z < 3; z++) {
                img_array[y + x*inImg.rows + z] = inImg.at<Vec3b>(y,x)[2-z];
            }
        }
    }

    /* Create meanshift object and put in image data */
    msImageProcessor meanshift;
    meanshift.DefineImage(img_array, COLOR, inImg.rows, inImg.cols);
    
    /* Mean Shift parameters */
    int sigmaR = 7;
    double sigmaS = 12.0;
    int minRegion = 100;
    SpeedUpLevel speedup = HIGH_SPEEDUP;

    /* Perform meanshift segmentation */
    meanshift.Segment(sigmaR, sigmaS, minRegion, speedup);

    /* Get the resulting data and convert into Mat */
    unsigned char *out_array = new unsigned char[inImg.rows * inImg.cols * 3];
    meanshift.GetResults(out_array);
    
    Mat new_image( inImg.size(), inImg.type() );
    for( int y = 0; y < inImg.rows; y++ ) {
        for( int x = 0; x < inImg.cols; x++ ) { 
            new_image.at<Vec3b>(y,x)[2] = out_array[y + x*inImg.rows + 0];
            new_image.at<Vec3b>(y,x)[1] = out_array[y + x*inImg.rows + 1];
            new_image.at<Vec3b>(y,x)[0] = out_array[y + x*inImg.rows + 2];
        }
    }

    delete[] img_array;
    delete[] out_array;

    return new_image;
}
