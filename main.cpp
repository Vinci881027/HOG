#include "hog.hpp"

int main(int argc, char **argv)
{
	std::vector<std::vector<double> > featureVec1;
	cv::Mat img1 = cv::imread("images/pedestrian1.jpg");
	HOG hog1(img1, 8, 9);
	featureVec1 = hog1.run("images/pedestrian1_result.jpg");
	std::cout << "HOG1 feature vector size = (" << featureVec1.size() << ", " << featureVec1[0].size() << ")" << std::endl;

	std::vector<std::vector<double> > featureVec2;
	cv::Mat img2 = cv::imread("images/pedestrian2.jpg");
	HOG hog2(img2, 8, 9);
	featureVec2 = hog2.run("images/pedestrian2_result.jpg");
	std::cout << "HOG2 feature vector size = (" << featureVec2.size() << ", " << featureVec2[0].size() << ")" << std::endl;

	std::vector<std::vector<double> > featureVec3;
	cv::Mat img3 = cv::imread("images/pedestrian3.jpg");
	HOG hog3(img3, 8, 9);
	featureVec3 = hog3.run("images/pedestrian3_result.jpg");
	std::cout << "HOG3 feature vector size = (" << featureVec3.size() << ", " << featureVec3[0].size() << ")" << std::endl;
}
