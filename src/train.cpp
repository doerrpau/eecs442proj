#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <list>
#include <string>
#include "segment.h"
#include "table_io.h"
#include "blob_token.h"

using namespace std;
using namespace cv;
using namespace boost::filesystem;

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

/* Vector quantization parameters */
int vq_k = 150;

/* Feature computation parameters */
int seg_min_size = 150;

/* Vector of each seen image label */
/* Location in the array determines numerical ID of label */
vector<string> img_labels;

/* Number of images */
int num_images = 0;

/* Iterate through the images in the directory and do segmentation */
/* Get blob features for each segment */
/* k-means to vector quantize the segments into smaller set of blobs */
/* use keywords recorded for each blob to find probability table */
// Also does EM to get probability table
Mat train(char* img_dir)
{

    /* Vector of all segment features */
    vector<BlobFeat*> all_feats;
    path p(img_dir);
    
    try {
        if (exists(p) && is_directory(p)) {

            /* Vector will contain the labelled image directorys */
            vector<path> label_path;
            copy(directory_iterator(p), directory_iterator(), back_inserter(label_path));

            /* Iterate through the labelled image directories to process images */
            for (int i = 0; i != label_path.size(); i++) {

                if (is_directory(label_path[i])) {

                    /* Get the labels from current label path */
                    /* Get second-to-last element (the label directory name) */
                    path::iterator it(label_path[i].end());
                    it--;
                    string label_str = it->string();
                    /* Split directory name into the labels */
                    vector<string> labels;
                    int pos = 0;
                    string delimiter = "_";
                    string label;
                    while ((pos = label_str.find(delimiter)) != string::npos) {
                        label = label_str.substr(0, pos);
                        labels.push_back(label);
                        label_str.erase(0, pos + delimiter.length());
                    }
                    labels.push_back(label_str);

                    /* Check if labels new and get label IDs */
                    vector<int> label_ids;
                    for (vector<string>::iterator l_it = labels.begin(); l_it != labels.end(); l_it++) {
                        bool label_seen = false;
                        for (int p = 0; p != img_labels.size(); p++) {
                            /* If the pixel is part of already-seen segment, use the new pixel */
                            /* to compute feature information for that blob */
                            if (img_labels[p] == *l_it) {
                                label_seen = true;
                                label_ids.push_back(p);
                            }
                        }
                        /* If it is a new segment */
                        if (label_seen == false) {
                            label_ids.push_back(img_labels.size());
                            img_labels.push_back(*l_it);
                        }
                    }

                    /* Vector to contain image paths */
                    vector<path> img_path;
                    copy(directory_iterator(label_path[i]), directory_iterator(), back_inserter(img_path));
                    
                    /* Iterate through the images (image paths) */
                    for (vector<path>::iterator pit(img_path.begin()); pit != img_path.end(); pit++) {
                         /* Load the image */
                        Mat image = imread(pit->string(), CV_LOAD_IMAGE_COLOR);

                        /* Scale the image proportionally to standardize processing time */
                        /* Probably downscaling, so use pixel area relation interpolation */
                        Mat s_image;
                        double scale = IMAGE_WIDTH/image.cols;
                        resize(image, s_image, Size(0,0), scale, scale, INTER_AREA); 

                        /* graph cut segmentation on the image */
                        Mat graphCutImg = doGraphCut(s_image, gs_sigma, gs_k, gs_min);

                        /* Get segment features using segmented image */
                        vector<BlobFeat*> features = 
                            getBlobFeatures(s_image, graphCutImg, seg_min_size, label_ids, num_images++);

                        /* Save the segment features */
                        all_feats.insert(all_feats.end(), features.begin(), features.end());
                    }
                }
            }

        } 
        else {
            cout << p << " is not a valid image directory" << endl;
        }
    }

    catch (const filesystem_error& ex) {
        cout << ex.what() << endl;
        exit(1);
    }

    /* Do vector quantization on all_feats */
    Mat labels = blobVectorization(all_feats, vq_k); /* labels is CV_32S */

    /* Use features and labels to create image data structures */
    vector<imgData> img_data;
    for (int i = 0; i < num_images; i++) {
        img_data.push_back(imgData());
        for (int f = 0; f < all_feats.size(); f++) {
            /* If the segment is in this image */
            if (all_feats[f]->imgId == i) {
                /* add blob ID to current image */
                img_data[i].blobs.push_back(labels.at<int>(f,0));
                /* add word IDs to current image */
                if (img_data[i].words.empty()) {
                    img_data[i].words.insert(img_data[i].words.end(), 
                                            all_feats[f]->blob_labels.begin(),
                                            all_feats[f]->blob_labels.end());
                }
            }
        }

        /* Allocate the probability tables */
        img_data[i].pTable[p] = new double*[img_data[i].words.size()];
        img_data[i].tTable[p] = new double*[img_data[i].words.size()];
        for (int p = 0; p < img_data[i].words.size(); p++) {
            img_data[i].pTable[p] = new double[img_data[i].blobs.size()];
            img_data[i].tTable[p] = new double[img_data[i].blobs.size()];
        }
    }


     //=======================||[ EM Algorithm ]||============================//
    //Mat result;
    Mat probTable = ones( img_labels.size(), vq_k,CV_32F);
    double smallChange = 10^(-4);
    double change = 10000;
    
    int iter,N,l,m = 0;
    double sumP,sumT=0;

    // Get words and blobs for each image
    N =  img_data.size()
   
    //============================ Init =======================================
    // Go over each image and fill in initial probabilities
    
    for(int n=0; n<N; n++)
    {
        for(int j=0;j<img_data[n].words.size();j++)
        {
            for(int i=0;i<img_data[n].blobs.size();i++)
            {
                img_data[n].pTable[j][i] = img_data[n].words.size()/img_data[n].blobs.size();
                img_data[n].tTablep[j][i] = 1/img_data[n].words.size();
            }
        }
    }
    //============================ End Init ====================================

    //================================ EM ======================================
    while(smallChange < change && iter <2)
    {
        //CONVERGENCE OR #ITER?

        // E Step
        // Calculate p_tild(a_nj|w_nj,b_nj,old params) for each image over all
        // words and blobs

        for(n=0;n<N;n++)
        {
            sumP = 0;//for each image

            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            for(j=0;j<m;j++)
            {
                for(i=0;i<l;i++)
                {
                    //for each word/blob
                    pTemp[n][j][i] = img_data[n].pTable[j][i]*img_data[n].tTable[j][i];
                    sumP+=pTemp[n][j][i];
                }
            }

            // Normalize p_tild(a_nj|w_nj,b_nj,old params)
            pTemp[n][j][i]/=sumP;
        }
        //============================ End E Step =================================//

        //============================== M Step ===================================//
        // | M.1 |
        // Get mixing probablilities by looking over each word for each image
        // of the same size

        //for each image set
        for(n=0;n<N;n++)
        {
            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            for(j=0;j<m;j++)
            {
                for(i=0;i<l;i++)
                {
                    //Go over each image with same size
                    countNlm = 0;
                    sumP = 0;
                    for(int nn =0;nn<N;nn++)
                    {

                        for(int jj=0;jj<img_data[nn].words.size();jj++)
                        {
                            if(img_data[nn].words.size()!=m)
                            {
                                break;
                            }

                            for(int ii=0;ii<img_data[nn].blobs.size();ii++)
                            {
                                //loop over conditions
                                if(img_data[nn].blobs.size()!=l)
                                {
                                    continue;
                                }
                                
                                sumP+=pTemp[nn][jj][ii];    //if same number of words&blobs
                                countNlm++;             //for each image in set
                                
                            }
                        }
                    }
                    img_data[n].pTable[j][i] = sumP/countNlm;
                }
            }
            
        }

        // |M.2 & M.3|
        // Get t_tild(w_nj=w*|b_ni=b*) by looing for pairs (w*,b*) that appear
        // in same image...may just be unique assignment
        for(n=0;n<N;n++)
        {
            sumT=0;
            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            //for each word/blob pair
            for(j=0;j<m;j++)
            {
                for(i=0;i<l;i++)
                {

                    sumP=0;
                    //loop over all images to look for same blob pair
                    for(int nn =0;nn<N;nn++)
                    {
                        for(int jj=0;jj<img_data[nn].words.size();jj++)
                        {
                            // See if word is in image
                            if(img_data[nn].words[jj]!=img_data[n].words[j])
                            {
                                break;
                            }

                            for(int ii=0;ii<img_data[nn].blobs.size();ii++)
                            {
                                // See if blob is in image
                                if(img_data[nn].blobs[jii]!=img_data[n].blobs[i])
                                {
                                    continue;
                                }

                                sumP+=pTemp[nn][jj][ii];//if word&blob are in image
                            }
                        }
                        img_data[n].tTable[j][i]=sumP;//at end of images loop
                    }
                    sumT+=sumP;
                }
            }

            //for each word/blob pair
            for(j=0;j<img_data[n].words.size();j++)
            {
                for(i=0;i<img_data[n].blobs.size();i++)
                {
                    // Normalize t_tild(w_nj=w*|b_ni=b*) over each image
                    img_data[n].tTable[j][i]/=sumT;
                }
            }
            
        }
        //================================ End M Step ===============================//
        iter++;
    }

    cout << "Change after iteration" <<change <<"\n";

    // Calculate final probability table
    double prod1,prod2 = 1;
    double sum1 = 0;

    for(int w=0;w<totNumWords;w++)
    {
        for(int b=0;b<totNumBlobs;b++)
        {
            prod1=1;
            for(int n=0;n<numImages;n++)
            {
                prod2=1;
                for(int j=0;j<images[n].words.size;j++)
                {
                    sum1=0;
                    for(int i=0;j<images[n].blobs.size;i++)
                    {
                        //If the blob b is in the image, add it to the sum
                        // (it's zero otherwise)
                        if(b = images[n].blobs[j])
                        {
                            sum1+=images[n].pTable[j][i]*images[n].tTable[j][i];
                        }
                    }

                    prod2*=sum1;
                }
                prod1*=prod2;
            }
            probTable.at<double>[w][b]=prod1;   //Write to probTable matrix
        }
    }


    // Output probabilty table probTable to a file

    return probTable;
    //return result;
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

    if (img_dir == NULL) {
        printf("No image directory specified, quitting.\n");
        printf("Specify training image directory using -d option\n");
        return 1;
    }    

    if (prob_table == NULL) {
        printf("No probability table output location specified, quitting.\n");
        printf("Specify location of the outputted probability table using -p option\n");
        return 1;
    }

    /* Use training images to train the algorithm */
    Mat ptable = train(img_dir);

    /* Save the probability table */

    return 0;
}
