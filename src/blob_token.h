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
    const static int num_feat = 24;

    /* Features */
    int size;
    float centroid[2];
    float euc_distance; /* Average Euclidean distance */
    float avg_col[3]; /* BGR color (0-255) */
    float std_col[3]; /* Standard Deviation of BGR color */
    float compactness;
    float first_moment;
    float avg_orien[12];

    /* Coordinates of every pixel in segment */
    vector<int> x_cor;
    vector<int> y_cor;

    /* Labels associated with the blob */
    vector<int> blob_labels;

    /* Image this blob is from */
    int imgId;
};

/* Format the vector of BlobFeat into a matrix for processing */
Mat vectorizeFeatures(vector<BlobFeat*> &in) 
{
    Mat blobFeat = Mat::zeros(in.size(), BlobFeat::num_feat, CV_32F);
    for (int i = 0; i < in.size(); i++) {
        blobFeat.at<float>(i,0) = (in[i]->size);
        blobFeat.at<float>(i,1) = (in[i]->centroid[0]);
        blobFeat.at<float>(i,2) = (in[i]->centroid[1]);
        blobFeat.at<float>(i,3) = (in[i]->euc_distance);
        blobFeat.at<float>(i,4) = (in[i]->avg_col[0]);
        blobFeat.at<float>(i,5) = (in[i]->avg_col[1]);
        blobFeat.at<float>(i,6) = (in[i]->avg_col[2]);
        blobFeat.at<float>(i,7) = (in[i]->std_col[0]);
        blobFeat.at<float>(i,8) = (in[i]->std_col[1]);
        blobFeat.at<float>(i,9) = (in[i]->std_col[2]);
        blobFeat.at<float>(i,10) =(in[i]->compactness);
        blobFeat.at<float>(i,11) = (float)(in[i]->first_moment);
        for (int j = 0; j < 12; j++) {
            blobFeat.at<float>(i,12+j) = (float)(in[i]->avg_orien[j]);
        }
    }
    
    return blobFeat;
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
vector<BlobFeat*> getBlobFeatures(Mat &orig_img, Mat &seg_img, int min_size, vector<int> l_ids, int imgId)
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
        features[i]->avg_col[0] = 0.0;
        features[i]->avg_col[1] = 0.0;
        features[i]->avg_col[2] = 0.0;
        features[i]->std_col[0] = 0.0;
        features[i]->std_col[1] = 0.0;
        features[i]->std_col[2] = 0.0;

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
            features[i]->avg_col[0] += (float)pix_col[0] / (float)features[i]->size;
            features[i]->avg_col[1] += (float)pix_col[1] / (float)features[i]->size;
            features[i]->avg_col[2] += (float)pix_col[2] / (float)features[i]->size; 
        }
        
        /* Now that we have avg color, compute color std */
        for (int k = 0; k < features[i]->size; k++) {
            pix_col = orig_img.at<Vec3b>(features[i]->y_cor[k], features[i]->x_cor[k]);
            features[i]->std_col[0] += pow((float)features[i]->avg_col[0]-(float)pix_col[0], 2.0);
            features[i]->std_col[1] += pow((float)features[i]->avg_col[1]-(float)pix_col[1], 2.0);
            features[i]->std_col[2] += pow((float)features[i]->avg_col[2]-(float)pix_col[2], 2.0);
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
        for (int k = 0; k < 12; k++) {
            features[i]->avg_orien[k] = 1.0;
        }

        /* First moment */
        features[i]->first_moment = 1.0;
    }

    return features; 
}

#endif
