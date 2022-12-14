#define PI 3.1415926535
#define PIANGLE 180

#include <opencv2/opencv.hpp>

class HOG
{
public:
	HOG(cv::Mat &img, int cellSize, int binSize);
	cv::Mat run();

private:
	cv::Mat _img;
	int _cellSize, _binSize, _binRange;
	double _maxMag;
	std::vector<double> cellGrad(cv::Mat &cellMag, cv::Mat &cellAngle);
	cv::Mat renderGrad(cv::Mat &img, std::vector<std::vector<std::vector<double> > > &cellGradVector);
};
