#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool drawing_box;
cv::Rect box;

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

            // Draw & show
            rectangle(*image, box,Scalar(255,255,255),1);
            imshow("Dog",*image);
            
            break;
        }
        default:
            break;
    }
}

int main(int argc, char** argv)
{
    /* Load the image */
    Mat image = imread( "../images/dog.jpg" );

    namedWindow("Dog");
    imshow("Dog",image);

    // define bounding rectangle
    int border = 75;
    int border2 = border + border;
    cv::Rect rectangle(border,border,image.cols-border2,image.rows-border2);
    
    setMouseCallback("Dog",callMouse,&image);

    Mat result; // segmentation result (4 possible values)
    Mat bgModel,fgModel; // the models (internally used)
 
    waitKey();

    // Perform GrabCut segmentation
    grabCut(image,    // input image
        result,   // segmentation result
        box/*rectangle*/,// rectangle containing foreground 
        bgModel,fgModel, // models
        1,        // number of iterations
        GC_INIT_WITH_RECT); // use rectangle
    // Get the pixels marked as likely foreground
    compare(result,GC_PR_FGD,result,CMP_EQ);
    // Generate output image
    Mat foreground(image.size(),CV_8UC3,Scalar(255,255,255));
    image.copyTo(foreground,result); // bg pixels not copied
 
    // draw rectangle on original image
    //cv::rectangle(image, rectangle, Scalar(255,255,255),1);
    imshow("Dog",image);

    // display result
    namedWindow("Segmented Dog");
    imshow("Segmented Dog",foreground);

    waitKey();
    return 0;
}
