#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include "segment.h"

#ifndef BLOB_TOKEN_H
#define BLOB_TOKEN_H

/* Number of values of a byte */
#define BYTE_VAL 256
/* Size of each of the 3 dimensions of the color histogram */
#define CHIST_SIZE 4
/* Number of bins of the gradient histogram */
#define GHIST_SIZE 16

/* Structure containing image data for EM */
struct imgData {
    vector<int> blobs; /* blob IDs */
    vector<int> words; /* label IDs */
    double** pTable; /* Probability table for this image */
    double** tTable;
};

/* Structure for an image segment blob */
/* Contains set of useful features for distinguishing blobs */
struct BlobFeat {
    /* number of features in the BlobFeat structure */
    const static int num_feat = 1 + 2 + 1 + 1 + (CHIST_SIZE*CHIST_SIZE*CHIST_SIZE) + GHIST_SIZE;

    /* Features */
    int size;
    float centroid[2];
    float euc_distance; /* Average Euclidean distance */
    float compactness;
    float col_hist[CHIST_SIZE][CHIST_SIZE][CHIST_SIZE]; /* divide RGB into 4x4x4 and make histogram of colors */
    float grad_hist[GHIST_SIZE];

    /* Coordinates of every pixel in segment */
    vector<int> x_cor;
    vector<int> y_cor;

    /* Labels associated with the blob */
    vector<int> blob_labels;

    /* Image this blob is from */
    int imgId;
};

/* Weights for all of the image region features for kmeans calculation */
/* If left unweighted, features with larger magnitudes, like size, have far too much weight */
/* Weights are in the order that features appear in kmeans vector */
static float feat_weight[] = {1.0/8704.0, /* size */
                              2.5/178.2,  /* centroid */
                              1.0/44.8,  /* euc dist */
                              1.0/0.0276, /* compactness */
                              50.0, /* color hist */
                              50.0}; /* gradient hist */


/* Format the vector of BlobFeat into a matrix for processing */
Mat vectorizeFeatures(vector<BlobFeat*> &in) 
{
    Mat blobFeat = Mat::zeros(in.size(), BlobFeat::num_feat, CV_32F);
    for (int i = 0; i < in.size(); i++) {
        blobFeat.at<float>(i,0) = in[i]->size * feat_weight[0];
        blobFeat.at<float>(i,1) = in[i]->centroid[0] * feat_weight[1];
        blobFeat.at<float>(i,2) = in[i]->centroid[1] * feat_weight[1];
        blobFeat.at<float>(i,3) = in[i]->euc_distance * feat_weight[2];
        blobFeat.at<float>(i,4) =in[i]->compactness * feat_weight[3];
        for (int b = 0; b < CHIST_SIZE; b++) {
            for (int g = 0; g < CHIST_SIZE; g++) {
                for (int r = 0; r < CHIST_SIZE; r++) {
                    blobFeat.at<float>(i,5+b*16+g*4+r) = in[i]->col_hist[b][g][r] * feat_weight[4];
                }
            }
        }
        for (int g = 0; g < GHIST_SIZE; g++) {
            blobFeat.at<float>(i,5+CHIST_SIZE*CHIST_SIZE*CHIST_SIZE+g) = 
                in[i]->grad_hist[g] * feat_weight[5];
        }
    }
    
    return blobFeat;
}

/* Compute image derivative using scharr kernel */
Mat scharrImage(Mat in)
{
    Mat blur;

    /* Scharr operator */
    GaussianBlur(in, blur, Size(3,3), 0, 0, BORDER_DEFAULT);
    cvtColor(blur, blur, CV_RGB2GRAY);
    Mat grad_x, grad_y;
    Mat abs_grad_x, abs_grad_y;
    Scharr(blur, grad_x, CV_16S, 1, 0, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    Scharr(blur, grad_y, CV_16S, 0, 1, 1, 0, BORDER_DEFAULT);
    convertScaleAbs(grad_y, abs_grad_y);
    Mat grad;
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    return grad;
}


/* Use k-means to vector quantize the segment features into k groups */
/* The k groups can then be used in a probability table */
void blobVectorization(vector<BlobFeat*> &in, int k, Mat &labels, Mat &centers)
{
    /* Format the segment features for k-means */
    Mat blobFeat = vectorizeFeatures(in);

    /* k-means to vector quantize */  
    int attempts = 5;
    TermCriteria tc(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
    kmeans(blobFeat, k, labels, tc, attempts, KMEANS_PP_CENTERS, centers);

    /* return blob categorization */
    return;
};

/* Compute feature values for all the segmented in the input image */
/* Returns array of blob feature structures */
vector<BlobFeat*> getBlobFeatures(Mat orig_img, Mat seg_img, int min_size, vector<int> l_ids, int imgId)
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
                BlobFeat *bf = new BlobFeat;
                features.push_back(bf);
                features.back()->blob_labels = l_ids;
                features.back()->imgId = imgId;
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
        
        features[i]->centroid[0] = 0.0;
        features[i]->centroid[1] = 0.0;
       
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
 
        /* Compute color histogram of the region - normalized by size */
        /* initialize to zero */
        for (int b = 0; b < CHIST_SIZE; b++) {
            for (int g = 0; g < CHIST_SIZE; g++) {
                for (int r = 0; r < CHIST_SIZE; r++) {
                    features[i]->col_hist[b][g][r] = 0.0;
                }
            }
        }
        Vec3b pix_col;
        int bin_s = BYTE_VAL / CHIST_SIZE; /* size of each color bin */
        for (int k = 0; k < features[i]->size; k++) {
            pix_col = orig_img.at<Vec3b>(features[i]->y_cor[k], features[i]->x_cor[k]);
            features[i]->col_hist[pix_col[0]/bin_s][pix_col[1]/bin_s][pix_col[2]/bin_s] += 
                1.0 / features[i]->size;
        }

        /* Compute gradient histogram of the region - normalized by size */
        Mat grad = scharrImage(orig_img);
        /* initialize to zero */
        for (int g = 0; g < GHIST_SIZE; g++) {
                    features[i]->grad_hist[g] = 0.0;
        }
        unsigned char pix_grad;
        bin_s = BYTE_VAL / GHIST_SIZE;
        for (int k = 0; k < features[i]->size; k++) {
            pix_grad = grad.at<unsigned char>(features[i]->y_cor[k], features[i]->x_cor[k]);
            features[i]->grad_hist[pix_grad/bin_s] += 1.0 / features[i]->size;
        }
   }

    return features; 
}

#endif
