#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <cstdio>
#include "segment.h"
#include "table_io.h"
#include "blob_token.h"

using namespace std;
using namespace cv;

/* Global Constants */
const double IMAGE_WIDTH = 360.0;
const double IMAGE_DISPLAY_WIDTH = 720.0;

/* Efficient graph cut parameters */
double gs_sigma = 0.5;
double gs_k = 300.0;
int gs_min = 400;

/* Vector quantization parameters */
int vq_k = 100;

/* Feature computation parameters */
int seg_min_size = 150;

/* Word Dictionary learned from training data set */
vector<string> img_labels;

void drawTextBox(Mat &img, String text, Scalar bgColor,Scalar fgColor, Point coords)
{
    double scale = 0.5;
    // Draw a text box on image
    // Note about locations: putText draws from lower left corner, while rectangle does verticies
    // Note about color: it's BGR, not RGB

    //rectangle(img, coords,Point(coords.x+text.length()*19.5*scale,coords.y+50*scale), bgColor, -1, 8, 0);
    putText(img, text, Point(coords.x+5,coords.y+25), FONT_HERSHEY_TRIPLEX, scale, fgColor, 1, 8, false);
}

Mat label(char* filename, Mat prob_table, Mat centers)
{
    /* Store highest probability word for each blob type for labeling use */
    vector<int> blob_words;
    for (int b = 0; b < prob_table.cols; b++) {
        double max_prob = 0.0;
        int best_word = -1;
        for (int w = 0; w < prob_table.rows; w++) {
            if (prob_table.at<float>(w, b) > max_prob) {
                max_prob = prob_table.at<float>(w, b);
                best_word = w;
            }
        }
        blob_words.push_back(best_word);
    }

    for (int i = 0; i < blob_words.size(); i++) {
        cout << blob_words[i] << endl;
    }
    cout << img_labels.size() << endl;

    /* Load the image */
    Mat image = imread(filename, CV_LOAD_IMAGE_COLOR);

    /* Scale the image proportionally to standardize processing time */
    /* Probably downscaling, so use pixel area relation interpolation */
    Mat s_image;
    double scale = IMAGE_WIDTH/image.cols;
    resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 
    
    /* graph cut segmentation on the image */
    Mat graphCutImg = doGraphCut(s_image, gs_sigma, gs_k, gs_min);

    /* Get segment features using segmented image */
    vector<BlobFeat*> features = 
        getBlobFeatures(s_image, graphCutImg, seg_min_size, vector<int>(), 0);

    /* Put blob features into matrix form */
    Mat blobFeat = vectorizeFeatures(features);

    /* Use K-D tree / nearest neighbors to vector quantize features based on training data */
    cv::flann::Index kdtree(centers, flann::KDTreeIndexParams(4));
    Mat matches;
    Mat distances;
    kdtree.knnSearch(blobFeat, matches, distances, 1, flann::SearchParams(32));

    cout << matches << endl; 
    
    /* Iterate through the image segments, look up probability, and label the image */
    for (int i = 0; i < features.size(); i++) {
        int word_id = blob_words[matches.at<int>(0,i)];
        String word = "null";
        if (word_id >= 0.0 && word_id < img_labels.size()) {
            word = img_labels[word_id];
        }
        int x_cen = features[i]->centroid[0];
        int y_cen = features[i]->centroid[1];
        drawTextBox(s_image, word, Scalar(255,255,255),Scalar(0,0,0), Point(x_cen,y_cen));
    }
    
    /* Scale up image */
    scale = IMAGE_DISPLAY_WIDTH/image.cols;
    resize(s_image, image, Size(0,0), scale, scale, INTER_CUBIC); 

    return image;
}

int main(int argc, char** argv)
{
    char *in_prob_table = NULL;
    char *in_image = NULL;
    char *center_mat = NULL;
    char *out_image = NULL;
    int c;
    int index;
 
    opterr = 0;
 
    /* Process Arguments */
    while ((c = getopt(argc, argv, "hp:i:c:o:")) != -1) {
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
           case 'c':
               center_mat = optarg;
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
               if (optopt == 'c')
                   fprintf(stderr, "Option -%c requires centers mat location \
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

    if (in_prob_table == NULL) {
        printf("No probability table, quitting.\n");
        printf("Please generate the probability table using train.\
                Provide the table as an argument to this program with -p\n");
        return 1;
    }

    if (in_image == NULL) {
        printf("No input image, quitting.\n");
        printf("Please provide an input image to label with -i\n");
        return 1;
    }
    
    if (center_mat == NULL) {
        printf("No kmeans centers, quitting.\n");
        printf("Please provide file with -c\n");
        return 1;
    }

    /* Process probability table */
    Mat probTable;
    readTable(probTable, img_labels, in_prob_table);
 
    /* Load kmeans center table */
    Mat centers;
    readMat(centers, center_mat);

    /* Label the image */
    Mat l_img = label(in_image, probTable, centers);

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
