#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <vector>
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
//double gs_sigma = 0.7;
//double gs_k = 600.0;
//int gs_min = 600;

/* Feature computation parameters */
int seg_min_size = 150;

/* Structure for an image segment blob */
/* Contains set of useful features for distinguishing blobs */
struct BlobFeat {
    /* Features */
    int size;
    double centroid[2];
    double euc_distance; /* Average Euclidean distance */
    double avg_col[3]; /* BGR color (0-255) */
    double std_col[3]; /* Standard Deviation of BGR color */
    double compactness;
    double first_moment;
    double avg_orien[12];

    /* Coordinates of every pixel in segment */
    vector<int> x_cor;
    vector<int> y_cor;
};

/* Compute feature values for all the segmented in the input image */
/* Returns array of blob feature structures */
vector<BlobFeat*> getBlobFeatures(Mat &orig_img, Mat &seg_img, int min_size)
{
    /* List of the unique segment values */
    /* The segmented image uses a unique (but unknown) value for each segement */
    vector<Vec3b> seg_val;

    /* Blob features */
    vector<BlobFeat*> features;

    /* Iterate through the the image pixels and add them to blob structure */
    for (int y = 0; y < seg_img.rows; y++ ) {
        for (int x = 0; x < seg_img.cols; x++ ) {
            bool val_seen = false;
            for (int i = 0; i != seg_val.size(); i++) {
                /* If the pixel is part of already-seen segment, use the new pixel */
                /* to compute feature information for that blob */
                if (seg_val[i][0] == seg_img.at<Vec3b>(y,x)[0] &&
                        seg_val[i][1] == seg_img.at<Vec3b>(y,x)[1] &&
                        seg_val[i][2] == seg_img.at<Vec3b>(y,x)[2]) {
                    
                    val_seen = true;
                    features[i]->size++;
                    features[i]->x_cor.push_back(x);
                    features[i]->y_cor.push_back(y);
                }
            }
            /* If it is a new segment */
            if (val_seen == false) {
                seg_val.push_back(seg_img.at<Vec3b>(y,x));
                features.push_back(new BlobFeat);
                features.back()->size = 1;
                features.back()->x_cor.push_back(x);
                features.back()->y_cor.push_back(y);
            }
        }
    }

    /* Erase blobs below min size */
    vector<BlobFeat*>::iterator it = features.begin();
    while (it != features.end()) {
        if ((*it)->size < min_size) {
            features.erase(it);
        } else {
            it++;
        }
    }

    /* Calculate segment features */
    for (int i = 0; i != features.size(); i++) {

        /* Compute region centroid */
        for (int k = 0; k < features[i]->size; k++) {
            features[i]->centroid[0] += (double)features[i]->x_cor[k] / (double)features[i]->size;
            features[i]->centroid[1] += (double)features[i]->y_cor[k] / (double)features[i]->size;
        }

        /* Compute avg Euclidean distance */
        for (int k = 0; k < features[i]->size; k++) {
            features[i]->euc_distance += sqrt( 
                    pow((double)features[i]->x_cor[k] - features[i]->centroid[0], 2.0) +
                    pow((double)features[i]->y_cor[k] - features[i]->centroid[1], 2.0)
                    ) / features[i]->size;
        }

        /* Compute average color of the region */
        Vec3b pix_col;

        for (int k = 0; k < features[i]->size; k++) {
            pix_col = orig_img.at<Vec3b>(features[i]->y_cor[k], features[i]->x_cor[k]);
            features[i]->avg_col[0] += (double)pix_col[0] / (double)features[i]->size;
            features[i]->avg_col[1] += (double)pix_col[1] / (double)features[i]->size;
            features[i]->avg_col[2] += (double)pix_col[2] / (double)features[i]->size; 
        }
        
        /* Now that we have avg color, compute color std */
        for (int k = 0; k < features[i]->size; k++) {
            pix_col = orig_img.at<Vec3b>(features[i]->y_cor[k], features[i]->x_cor[k]);
            features[i]->std_col[0] += pow((double)features[i]->avg_col[0]-(double)pix_col[0], 2.0);
            features[i]->std_col[1] += pow((double)features[i]->avg_col[1]-(double)pix_col[1], 2.0);
            features[i]->std_col[2] += pow((double)features[i]->avg_col[2]-(double)pix_col[2], 2.0);
        }
        features[i]->std_col[0] = sqrt(features[i]->std_col[0]/features[i]->size);
        features[i]->std_col[1] = sqrt(features[i]->std_col[1]/features[i]->size);
        features[i]->std_col[2] = sqrt(features[i]->std_col[2]/features[i]->size);

        /* Compute ratio of region size to boundary length squared (compactness) */
        /* Pseudo-perimeter using a rectangle */
        int x_min = orig_img.cols;
        int x_max = 0;
        int y_min = orig_img.rows;
        int y_max = 0;
        for (int k = 0; k < features[i]->size; k++) {
            if (features[i]->x_cor[k] < x_min)
                x_min = features[i]->x_cor[k];
            else if (features[i]->x_cor[k] > x_max)
                x_min = features[i]->x_cor[k];
            if (features[i]->y_cor[k] < y_min)
                y_min = features[i]->y_cor[k];
            else if (features[i]->y_cor[k] > y_max)
                y_min = features[i]->y_cor[k];
        }
        int perimeter = 2*(x_max-x_min) + 2*(y_max-y_min);
        features[i]->compactness = features[i]->size / pow((double)perimeter,2.0);

        /* Average orientational energy */

        /* First moment */
    }

    return features;
    
}

Mat train(char* img_dir)
{
    /* Load the image */
    Mat image = imread(img_dir, CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    Mat s_image;
    double scale = IMAGE_WIDTH/image.cols;
    resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 

    /* graph cut segmentation on the image */
    Mat graphCutImg = doGraphCut(s_image, gs_sigma, gs_k, gs_min);

    /* Get segment features using segmented image */
    vector<BlobFeat*> features = getBlobFeatures(s_image, graphCutImg, seg_min_size);

    /* Upscale images */
    scale = IMAGE_DISPLAY_WIDTH/graphCutImg.cols;
    resize(graphCutImg, graphCutImg, Size(0,0), scale, scale);
    resize(s_image, s_image, Size(0,0), scale, scale);

    /* Display the original image and the segmented images */
    namedWindow("Image");
    imshow("Image",s_image);
    namedWindow("Efficient Graph Cut");
    imshow( "Efficient Graph Cut", graphCutImg); 
    
    /* End program by hitting any key */
    waitKey();
    return graphCutImg;
}

int main(int argc, char** argv)
{
    int flag_train = 0;
    char *img_dir = NULL;
    char *prob_table = NULL;
    int c;
    int index;
 
    opterr = 0;
 
    /* Process Arguments */
    while ((c = getopt(argc, argv, "hd:p:")) != -1) {
        switch (c) {
           case 'h':
               printf("Help/usage\n");
               break;
           case 'd':
               img_dir = optarg;
               break;
           case 'p':
               prob_table = optarg;
               break;
           case '?':
               if (optopt == 'd')
                   fprintf (stderr, "Option -%c requires directory path argument.\n", optopt);
               if (optopt == 'p')
                   fprintf (stderr, "Option -%c requires output file path argument.\n", optopt);
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

    if (*img_dir == NULL) {
        printf("No image directory specified, quitting.\n");
        printf("Specify training image directory using -d option\n");
        return 1;
    }    

    if (*prob_table == NULL) {
        printf("No probability table output location specified, quitting.\n");
        printf("Specify location of the outputted probability table using -p option\n");
        return 1;
    }

    /* Use training images to train the algorithm */
    Mat ptable = train(img_dir);

    /* Save the probability table */

    return 0;
}
