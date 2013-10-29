#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    /* Load the image */
    Mat image = imread( "../images/dog.jpg" );

    // define bounding rectangle
    int border = 75;
    int border2 = border + border;
    cv::Rect rectangle(border,border,image.cols-border2,image.rows-border2);
 
    Mat result; // segmentation result (4 possible values)
    Mat bgModel,fgModel; // the models (internally used)
 
    // Perform GrabCut segmentation
    grabCut(image,    // input image
        result,   // segmentation result
        rectangle,// rectangle containing foreground 
        bgModel,fgModel, // models
        1,        // number of iterations
        GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground
    compare(result,GC_PR_FGD,result,CMP_EQ);
    // Generate output image
    Mat foreground(image.size(),CV_8UC3,Scalar(255,255,255));
    image.copyTo(foreground,result); // bg pixels not copied
 
    // draw rectangle on original image
    cv::rectangle(image, rectangle, Scalar(255,255,255),1);
    namedWindow("Dog");
    imshow("Dog",image);
 
    // display result
    namedWindow("Segmented Dog");
    imshow("Segmented Dog",foreground);

    waitKey();
    return 0;
}
