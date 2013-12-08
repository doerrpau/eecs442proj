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
// Also return feature kmeans centers to classify test image blobs
Mat train(char* img_dir, Mat &probTable, Mat &centers)
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
    cout << "Vector Quantization"<<endl;
    /* Do vector quantization on all_feats */
    Mat labels;
    blobVectorization(all_feats, vq_k, labels, centers); /* labels is CV_32S */

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
        img_data[i].pTable = new double*[img_data[i].words.size()];
        img_data[i].tTable = new double*[img_data[i].words.size()];
        for (int p = 0; p < img_data[i].words.size(); p++) {
            img_data[i].pTable[p] = new double[img_data[i].blobs.size()];
            img_data[i].tTable[p] = new double[img_data[i].blobs.size()];
        }
    }


     //=======================||[ EM Algorithm ]||============================//
    //Mat result;
    probTable = Mat::zeros( img_labels.size(), vq_k,CV_32F);
    double smallChange = 10^(-4);
    double change = 10000;
    
    int iter=0,N=0,l=0,m=0;
    double sumP=0,sumT=0;

    // Get words and blobs for each image
    N =  img_data.size();
   
    //============================ Init =======================================
    // Go over each image and fill in initial probabilities
    cout << "Initializing EM" << endl;
    for(int n=0; n<N; n++)
    {
        for(int j=0;j<img_data[n].words.size();j++)
        {
            for(int i=0;i<img_data[n].blobs.size();i++)
            {
                img_data[n].pTable[j][i] =(double)img_data[n].words.size()/(double)img_data[n].blobs.size();
                img_data[n].tTable[j][i] = 1.0/(double)img_data[n].words.size();
            }
        }
    }

    vector<double**> pTemp;
    pTemp.resize(N);
    
    /* Allocate pTemp */
    for(int n=0;n<N;n++)
    {
        m = img_data[n].words.size();
        l = img_data[n].blobs.size();

        pTemp[n] = new double*[m];

        for(int j=0;j<m;j++)
        {
            pTemp[n][j] = new double[l];
        }
    }

    //============================ End Init ====================================



    //================================ EM ======================================
    while(smallChange < change && iter < 100)
    {
        //CONVERGENCE OR #ITER?

        // E Step
        // Calculate p_tild(a_nj|w_nj,b_nj,old params) for each image over all
        // words and blobs
        cout << "E Step" << endl;
        for(int n=0;n<N;n++)
        {
            
            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            for(int j=0;j<m;j++)
            {
                sumP = 0;//for each word
                for(int i=0;i<l;i++)
                {
                    //for each word/blob
                    pTemp[n][j][i] = img_data[n].pTable[j][i]*img_data[n].tTable[j][i];
                    sumP+=pTemp[n][j][i];
                }

                // Normalize p_tild(a_nj|w_nj,b_nj,old params)
                // for each word
                for(int i=0;i<l;i++)
                {
                    pTemp[n][j][i]/=sumP;
                    //cout << pTemp[n][j][i] << endl;
                }   
            }
        }
        //============================ End E Step =================================//

        cout << "M Step" << endl;

        //============================== M Step ===================================//
        // | M.1 |
        // Get mixing probablilities by looking over each word for each image
        // of the same size

        //for each image set
        for(int n=0;n<N;n++)
        {
            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            for(int j=0;j<m;j++)
            {
                for(int i=0;i<l;i++)
                {
                    //Go over each image with same size
                    int countNlm = 0;
                    sumP = 0;
                    for(int nn =0;nn<N;nn++)
                    {

                        //Image size check
                        if(img_data[nn].words.size()!=m || img_data[nn].blobs.size()!=l)
                        {
                            continue;
                        }

                        for(int jj=0;jj<img_data[nn].words.size();jj++)
                        {
                            if(img_data[n].words[j] == img_data[nn].words[jj])
                            {
                               continue;
                            }
                            for(int ii=0;ii<img_data[nn].blobs.size();ii++)
                            {
                                if(img_data[n].blobs[i] == img_data[nn].blobs[ii])
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
        for(int n=0;n<N;n++)
        {
            
            m = img_data[n].words.size();
            l = img_data[n].blobs.size();

            //for each word/blob pair
            for(int j=0;j<m;j++)
            {
                for(int i=0;i<l;i++)
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
                                if(img_data[nn].blobs[ii]!=img_data[n].blobs[i])
                                {
                                    continue;
                                }
                                sumP+=pTemp[nn][jj][ii];//if word&blob are in image
                            }
                        }
                    }
                    /****************************************************/
                    img_data[n].tTable[j][i]+=sumP;//at end of images loop
                }
            }

            //for each word/blob pair
            // Normalize t_tild(w_nj=w*|b_ni=b*) over each blob in image
            // Hold the blob and iterate over the words
            for(int i=0;i<img_data[n].blobs.size();i++)
            {
                sumT=0;
                for(int j=0;j<img_data[n].words.size();j++)    
                {
                    sumT+=img_data[n].tTable[j][i];
                }

                //Normalize
                for(int j=0;j<img_data[n].words.size();j++)    
                {
                    img_data[n].tTable[j][i]/=sumT;
                }
            }
            
        }
        //================================ End M Step ===============================//
        iter++;
    }

    /* Delete pTemp */
    for(int n=0;n<N;n++)
    {
        m = img_data[n].words.size();
        l = img_data[n].blobs.size();

        for(int j=0;j<m;j++)
        {
            delete[] pTemp[n][j];

        }

        delete[] pTemp[n];
    }

    cout << "Change after iteration: " <<change <<"\n";

    // Calculate final probability table
    double prod1=1,prod2 = 1;
    double sum1 = 0;
    bool pairFound = false;

    for(int w=0;w<img_labels.size();w++)
    {
        for(int b=0;b<vq_k;b++)
        {

            bool pairFound = false;
            prod1=1;
            sum1=0;
            for(int n=0;n<N;n++)
            {
                prod2=1;
                for(int j=0;j<img_data[n].words.size();j++)
                {
                    for(int i=0;i<img_data[n].blobs.size();i++)
                    {
                        //If the blob b is in the image, add it to the sum
                        // (it's zero otherwise)
                        if(b == img_data[n].blobs[i] && w == img_data[n].words[j])
                        {
                            //cout << img_labels[w] << " image:" << n << " blob:" << i << " prob:" << 
                            //    img_data[n].pTable[j][i]*img_data[n].tTable[j][i] << endl;
                            sum1+=img_data[n].pTable[j][i]*img_data[n].tTable[j][i];
                            pairFound = true;
                        }
                    }

                    // Ignore images that don't have the word/blob
                    if(pairFound)
                    {
                        //cout << img_labels[w] << ": " << sum1 << endl;
                        //prod2*=sum1;
                    }
                }
                prod1*=prod2;
            }

            // If an instance of a word and a blob were found in an image together
            if(pairFound)
            {
                probTable.at<float>(w,b)=sum1;   //Write to probTable matrix
            }
        }
    }

    // Normailize probability table....needed?
    cout <<"Normalizing"<<endl;
    for(int b=0;b<vq_k;b++)
    {
        double sum = 0;
        for(int w=0;w<img_labels.size();w++)
        {
            sum+=probTable.at<float>(w,b);
        }
        for(int w=0;w<img_labels.size();w++)
        {
            if (sum != 0) {
                probTable.at<float>(w,b)/=sum;
            }
        }
    }
    cout << "writing to file" << endl;
    //cout << probTable << endl;
    return probTable;
    //return result;
}

int main(int argc, char** argv)
{
    int flag_train = 0;
    char *img_dir = NULL;
    char *prob_table = NULL;
    char *center_table = NULL;
    int c;
    int index;
 
    opterr = 0;
 
    /* Process Arguments */
    while ((c = getopt(argc, argv, "hd:p:c:")) != -1) {
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
           case 'c':
               center_table = optarg;
               break;
           case '?':
               if (optopt == 'd')
                   fprintf (stderr, "Option -%c requires directory path argument.\n", optopt);
               if (optopt == 'p')
                   fprintf (stderr, "Option -%c requires output file path argument.\n", optopt);
               if (optopt == 'c')
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
    
    if (center_table == NULL) {
        printf("No kmeans centers output location specified, quitting.\n");
        printf("Specify location using -c option\n");
        return 1;
    }

    Mat centers, ptable;

    /* Use training images to train the algorithm */
    train(img_dir, ptable, centers);

    /* Save the probability table and kmeans centers*/
    storeTable(ptable, img_labels, prob_table);
    storeMat(centers, center_table);

    return 0;
}
