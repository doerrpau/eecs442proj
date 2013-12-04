#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#ifndef TABLE_IO_H
#define TABLE_IO_H

using namespace std;
using namespace cv;

void storeTable(Mat pTable, vector<string> words, const char* filename)
{
  // Store proabaility data to a csv file
  // pTable is a matrix with all probabilites
  // Each row corresponds to one word
  // words is a vector of strings that holds the words
  
  int i,j = 0;
  ofstream table(filename, ofstream::out);
  
  int numCols = pTable.cols;
  int numRows = pTable.rows;

  table << numRows << "," << numCols <<"\n";

  // Save each word
  for(i=0;i<words.size();i++)
  {
    table << words[i] << ",";

    // Save each conditional probabililty p(w|b)
    for(j=0;j<numCols;j++)
    {
      table << pTable.at<float>(i,j) <<",";
    }
      table<<"\n";
  }

  table.close();
}

void storeMat(Mat cTable, const char* filename)
{
  // Store kmeans centers to a csv file
  // cTable is a matrix
  // filename is the filename to store in

  int i,j = 0;
  ofstream table(filename, ofstream::out);
  
  int numCols = cTable.cols;
  int numRows = cTable.rows;

  table << numRows << "," << numCols <<"\n";

  // Save each word
  for(i=0;i<numRows;i++)
  {
    // Save each conditional probabililty p(w|b)
    for(j=0;j<numCols;j++)
    {
      table << cTable.at<float>(i,j) <<",";
    }
      table<<"\n";
  }

  table.close();
}
void readTable(Mat &pTable, vector<string> &words, const char* filename)
{
  //Read csv file for words and blob probabilities

  string nums;  // String that holds numbers separated by commas
  int i = 0;
  int j = 0;
  int k = 0;
  int len = 0;
  int numWords = 0;
  int numBlobs=0;
  ifstream table(filename, ifstream::in);
  bool foundSize,foundWord = false;

  // First line has number of words, then number of blobs
  getline(table,nums);
  k = nums.find(",");
  numWords = atoi(nums.c_str());

  //Remove data at front of string
  len = nums.length();
  nums = nums.substr(k+1,len-1);
  len = nums.length();
  
  numBlobs = atoi(nums.c_str());

  pTable = Mat::zeros(numWords, numBlobs, CV_32F);

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
        k=nums.find(",");
        words.push_back(nums.substr(0,k));
        nums = nums.substr(k+1,len-1);
        len = nums.length();
        foundWord=true;
      }
      else
      {
        k = nums.find(",");
        string temp = nums.substr(0,k);
        pTable.at<float>(i,j) = atof(temp.c_str());
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

void readMat(Mat &cTable, const char* filename)
{
  //Read csv file for words and blob probabilities

  string nums;  // String that holds numbers separated by commas
  int i = 0;
  int j = 0;
  int k = 0;
  int len = 0;
  int numWords = 0;
  int numBlobs=0;
  ifstream table(filename, ifstream::in);
  bool foundSize = false;

  // First line has number of words, then number of blobs
  getline(table,nums);
  k = nums.find(",");
  numWords = atoi(nums.c_str());

  //Remove data at front of string
  len = nums.length();
  nums = nums.substr(k+1,len-1);
  len = nums.length();
  
  numBlobs = atoi(nums.c_str());

  cTable = Mat::zeros(numWords, numBlobs, CV_32F);

  // Parse rest of table for words & blobs
  // FIX ME?
  while(getline(table,nums))
  {  
    len= nums.length(); //Get line's length
    j=0;                //Blob counter

    //Parse line
    while(j<numBlobs) 
    {
        k = nums.find(",");
        string temp = nums.substr(0,k);
        cTable.at<float>(i,j) = atof(temp.c_str());
        nums = nums.substr(k+1,len-1);
        len = nums.length();
        j++;
    }
    i++;
  }
  table.close();

  return;
}
#endif
