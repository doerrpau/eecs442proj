#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

bool drawing_box;
cv::Rect box;
Mat orig;

void drawTextBox(Mat img, String text, Scalar bgColor,Scalar fgColor, Point coords)
{
    // Draw a text box on image
    // Note about locations: putText draws from lower left corner, while rectangle does verticies
    // Note about color: it's BGR, not RGB
    rectangle(img, coords,Point(coords.x+text.length()*19.5,coords.y+50), bgColor, -1, 8, 0);
    putText(img, text, Point(coords.x+5,coords.y+25), FONT_HERSHEY_TRIPLEX, 1, fgColor,2, 8,false);
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

            Mat result; // segmentation result (4 possible values)
            Mat bgModel,fgModel; // the models (internally used)

            // Perform GrabCut segmentation
            grabCut(*image,    // input image
                result,   // segmentation result
                box/*rectangle*/,// rectangle containing foreground 
                bgModel,fgModel, // models
                1,        // number of iterations
                GC_INIT_WITH_RECT); // use rectangle

            // Get the pixels marked as likely foreground
            compare(result,GC_PR_FGD,result,CMP_EQ);

            // Generate output image
            Mat foreground(image->size(),CV_8UC3,Scalar(255,255,255));
            image->copyTo(foreground,result); // bg pixels not copied

            // Adding text above segmentation
            drawTextBox(foreground, "I CAN HAZ FRIZBEE?", Scalar(000,000,000),Scalar(255,255,255), Point(box.x, box.height));

            // display result
            namedWindow("Segmented Dog");
            imshow("Segmented Dog",foreground);

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
    Mat image = imread( "../images/dog.jpg" );

    namedWindow("Dog");
    imshow("Dog",image);
  
    setMouseCallback("Dog",callMouse,&image);

    //End program by hitting any key
    waitKey();
    return 0;
}
