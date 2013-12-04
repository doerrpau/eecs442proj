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
    
    Mat inMat;
    vector<string> words; 

    readTable(inMat, words, file);

    for (int i = 0; i < words.size(); i++) {
        cout << words[i] << endl;
    }

    cout << inMat << endl;

    return 0;
}
