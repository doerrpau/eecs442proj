#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
    /* Load the image */
    IplImage* img = cvLoadImage( "../images/dog.jpg" );
    /* Create the window */
    cvNamedWindow( "Dog", CV_WINDOW_AUTOSIZE );
    /* Display the image */
    cvShowImage("Dog", img);
    cvWaitKey(0);
    cvReleaseImage( &img );
    cvDestroyWindow( "Dog" );
    return 0;
}
