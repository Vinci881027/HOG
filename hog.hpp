#define PI 3.1415926535
#define PIANGLE 180

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

class HOG
{
public:
	HOG(cv::Mat &img, int cellSize, int binSize);
	std::vector<std::vector<double> > run(std::string outputFile);

private:
	cv::Mat _img;
	int _cellSize, _binSize, _binRange;
	double _maxMag;

	std::vector<double> cellGrad(cv::Mat &cellMag, cv::Mat &cellAngle);
	std::vector<double> blockNorm(std::vector<double> &blockVec);
	cv::Mat renderGrad(cv::Mat &img, std::vector<std::vector<std::vector<double> > > &cellGradVector);
};
