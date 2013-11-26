#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <boost/filesystem.hpp>
#include <unistd.h>
#include <cstdio>
#include <cmath>
#include <vector>
<<<<<<< HEAD
=======
#include <list>
>>>>>>> fc8b4ced0b6f4eda81c93168683c6cfb7b8d715b
#include <string>
#include "segment.h"

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
/* number of features in the BlobFeat structure */
const int num_feat = 24;

/* Use k-means to vector quantize the segment features into k groups */
/* The k groups can then be used in a probability table */
Mat blobVectorization(vector<BlobFeat*> &in, int k)
{
    /* Format the segment features for k-means */
    Mat blobFeat = Mat::zeros(in.size(), num_feat, CV_32F);
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
    
    /* k-means to vector quantize */  
    Mat labels;
    int attempts = 5;
    Mat centers;
    TermCriteria tc(CV_TERMCRIT_ITER|CV_TERMCRIT_EPS, 10000, 0.0001);
    kmeans(blobFeat, k, labels, tc, attempts, KMEANS_PP_CENTERS, centers);

    /* return blob categorization */
    return labels;
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

<<<<<<< HEAD
void storeTable(Mat pTable, vector<string> words)
{
  // Store proabaility data to a csv file
  // pTable is a matrix with all probabilites
  // Each row corresponds to one word
  // words is a vector of strings that holds the words
  
  int i,j,numCols,numRows = 0;
  ofstream table("probTable.csv");
  
  pTable.size(numCols,numRows);

  table << numRows << "," << numCols <<"\n";

  // Save each word
  for(i=0;i<words.size();i++)
  {
    table << words[i] << ",";

    // Save each conditional probabililty p(w|b)
    for(j=0;j<numCols;j++)
    {
      table << pTable[i][j] <<",";
    }
      table<<"\n";
  }

  table.close();
}

void readTable(Mat pTable, string words[])
{
  //Read csv file for words and blob probabilities

  string nums;  // String that holds numbers separated by commas
  int i,j,k,len,numWords,numBlobs=0;
  ifstream table("probTable.csv");
  bool foundSize,foundWord = false;

  // First line has number of words, then number of blobs
  getline(table,nums);
  k = nums.at(',');
  numWords = atoi(nums.substr(0,k-1));

  //Remove data at front of string
  nums = nums.substr(k+1,len-1);
  len = nums.length();
  
  numBlobs = atoi(nums.substr(0,len-2));

  // Parse rest of table for words & blobs
  // FIX ME?
  while(getline(table,nums))
  {  
    len= nums.length(); //Get line's length
    j=0;                //Blob counter

    //Parse line
    while(j<numBlobs) 
    {
      // Get the word at beginning of line
      if(!foundWord)
      {
        k=nums.at(',');
        words[0] = nums.substr(0,k);
        nums = nums.substr(k+1,len-1);
        len = nums.length();
        foundWord=true;
      }
      else
      {
        k = nums.at(',');
        pTable[i][j] = atof(nums.substr(j,k-j));
        nums = nums.substr(k+1,len-1);
        len = nums.length();
        j++;
      }
    }
    i++;
    foundWord=false;
  }
  table.close();

  return;
}

=======
/* Iterate through the images in the directory and do segmentation */
/* Get blob features for each segment */
/* k-means to vector quantize the segments into smaller set of blobs */
/* use keywords recorded for each blob to find probability table */
>>>>>>> fc8b4ced0b6f4eda81c93168683c6cfb7b8d715b
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
    imgData* img_data = new imgData[num_images];
    for (int i = 0; i < num_images; i++) {
        for (int f = 0; f < all_feats.size(); f++) {
            /* If the segment is in this image */
            if (all_feats[f]->imgId = i) {
                /* add blob ID to current image */
                img_data[i]->blobs.push_back(labels.at<int>(f,0));
                /* add word IDs to current image */
                if (img_data[i]->words.empty()) {
                    img_data[i]->words.insert(img_data[i]->words.end(), 
                                            all_feats[f]->blob_labels.begin(),
                                            all_feats[f]->blob_labels.end());
                }
            }
        }

        /* Allocate the probability tables */
        img_data[i].pTable = new double[vq_k][img_labels.size()];
        img_data[i].tTable = new double[vq_k][img_labels.size()];
    }

    Mat result;
    return result;
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
