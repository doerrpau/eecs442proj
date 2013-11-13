#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <cstdio>
#include "segment.h"

using namespace std;
using namespace cv;

/* Global Constants */
const double IMAGE_WIDTH = 360.0;
const double IMAGE_DISPLAY_WIDTH = 720.0;

/* Efficient graph cut parameters */
double gs_sigma = 0.5;
double gs_k = 300.0;
int gs_min = 400;

void drawTextBox(Mat img, String text, Scalar bgColor,Scalar fgColor, Point coords)
{
    int scale = 1;
    // Draw a text box on image
    // Note about locations: putText draws from lower left corner, while rectangle does verticies
    // Note about color: it's BGR, not RGB

    rectangle(img, coords,Point(coords.x+text.length()*19.5*scale,coords.y+50*scale), bgColor, -1, 8, 0);
    putText(img, text, Point(coords.x+5,coords.y+25), FONT_HERSHEY_TRIPLEX, scale, fgColor,2, 8,false);
}

Mat label(char* filename, char* prob_table)
{
    /* Load the image */
    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    Mat s_image;
    double scale = IMAGE_WIDTH/image.cols;
    resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 

    /* graph cut segmentation on the image */
    Mat graphCutImg = doGraphCut(s_image, gs_sigma, gs_k, gs_min);
    /* Upscale images */
    scale = IMAGE_DISPLAY_WIDTH/graphCutImg.cols;
    resize(graphCutImg, graphCutImg, Size(0,0), scale, scale);
    resize(s_image, s_image, Size(0,0), scale, scale);

    return graphCutImg;
}

int main(int argc, char** argv)
{
    char *in_prob_table = NULL;
    char *in_image = NULL;
    char *out_image = NULL;
    int c;
    int index;
 
    opterr = 0;
 
    /* Process Arguments */
    while ((c = getopt(argc, argv, "hp:i:o:")) != -1) {
        switch (c) {
           case 'h':
               printf("Help/usage\n");
               break;
           case 'p':
               in_prob_table = optarg;
               break;
           case 'i':
               in_image = optarg;
               break;
           case 'o':
               out_image = optarg;
               break;
           case '?':
               if (optopt == 'p')
                   fprintf(stderr, "Option -%c requires probability table \
                       location as argument.\n", optopt);
               if (optopt == 'o')
                   fprintf(stderr, "Option -%c requires output image location \
                       as argument.\n", optopt);
               if (optopt == 'i')
                   fprintf(stderr, "Option -%c requires input image location \
                       as argument.\n", optopt);
               else if (isprint (optopt))
                   fprintf (stderr, "Unknown option `-%c'.\n", optopt);
               else
                   fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
               return 1;
           default:
               abort ();
        }
    }

    for (index = optind; index < argc; index++) {
        printf ("Non-option argument %s\n", argv[index]);
        return 1;
    }

    if (*in_prob_table == NULL) {
        printf("No probability table, quitting.\n");
        printf("Please generate the probability table using train.\
                Provide the table as an argument to this program with -p\n");
        return 1;
    }

    if (*in_image == NULL) {
        printf("No input image, quitting.\n");
        printf("Please provide an input image to label with -i\n");
        return 1;
    }

    /* Process probability table */

    /* Label the image */
    Mat l_img = label(in_image, in_prob_table);

    /* Save labelled image */
    if (out_image != NULL) {
        imwrite(out_image, l_img);
    }

    /* Display image */
    namedWindow("Labelled Image");
    imshow("Labelled Image", l_img); 
    waitKey();
    
    return 0;
}
