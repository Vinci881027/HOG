#include <iostream>
#include "hog.hpp"

int main(int argc, char **argv)
{
	cv::Mat img1 = cv::imread("images/pedestrian1.jpg");
	HOG hog1(img1, 8, 9);
	cv::imwrite("images/pedestrian1_result.jpg", hog1.run());

	cv::Mat img2 = cv::imread("images/pedestrian2.jpg");
	HOG hog2(img2, 8, 9);
	cv::imwrite("images/pedestrian2_result.jpg", hog2.run());

	cv::Mat img3 = cv::imread("images/pedestrian3.jpg");
	HOG hog3(img3, 8, 9);
	cv::imwrite("images/pedestrian3_result.jpg", hog3.run());
}
