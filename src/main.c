#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
 
int main(int argc, char** argv)
{
    IplImage* img = cvLoadImage( "../images/dog.jpg" );
    cvNamedWindow( "Example1", CV_WINDOW_AUTOSIZE );
    cvShowImage("Example1", img);
    cvWaitKey(0);
    cvReleaseImage( &img );
    cvDestroyWindow( "Example1" );
    return 0;
}
