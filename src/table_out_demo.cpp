#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>
#include <cstdio>
#include <string>
#include <vector>
#include "table_io.h"

using namespace std;
using namespace cv;


int main(int argc, char** argv)
{
    const char file[] = "probTable.csv";

    vector<string> words;
    words.push_back("ant");
    words.push_back("bat");
    words.push_back("cat");
    words.push_back("dog");
    words.push_back("eel");
    words.push_back("fox");
    words.push_back("gnat");
    words.push_back("hill");
    words.push_back("imp");
    words.push_back("joe");

    Mat outMat = Mat::eye(10, 10, CV_64F); 

    storeTable(outMat, words, file);

    return 0;
}
