#include <fstream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <string>

#ifndef TABLE_IO_H
#define TABLE_IO_H

void storeTable(Mat pTable, vector<string> words)
{
  // Store proabaility data to a csv file
  // pTable is a matrix with all probabilites
  // Each row corresponds to one word
  // words is a vector of strings that holds the words
  
  int i,j = 0;
  ofstream table("probTable.csv",ofstream::out);
  
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
      table << pTable.at<double>(i,j) <<",";
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
  ifstream table("probTable.csv", ifstream::in);
  bool foundSize,foundWord = false;

  // First line has number of words, then number of blobs
  getline(table,nums);
  k = nums.at(',');
  numWords = atoi(nums.substr(0,k-1).c_str());

  //Remove data at front of string
  nums = nums.substr(k+1,len-1);
  len = nums.length();
  
  numBlobs = atoi(nums.substr(0,len-2).c_str());

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
        pTable.at<double>(i,j) = atof(nums.substr(j,k-j).c_str());
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

#endif
