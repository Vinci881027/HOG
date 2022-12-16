#include <cmath>
#include "hog.hpp"

HOG::HOG(cv::Mat &img, int cellSize, int binSize)
{
	double minVal, maxVal;
	cv::Point minLoc, maxLoc;

	_img = img;
	cv::cvtColor(_img, _img, CV_BGR2GRAY);
	minMaxLoc(_img, &minVal, &maxVal, &minLoc, &maxLoc);
	_img.convertTo(_img, CV_64F, 1.0 / maxVal, 0);
	cv::sqrt(_img, _img);
	_img = 255 * _img;
	_cellSize = cellSize;
	_binSize = binSize;
	_binRange = 180 / _binSize;
	_maxMag = 0;
}

std::vector<std::vector<double> > HOG::run(std::string outputFile)
{
	int height = _img.rows;
	int width = _img.cols;
	int cellHeight = height / _cellSize;
	int cellWidth = width / _cellSize;

	cv::Mat gx, gy, gradMag, gradAngle, cellMag, cellAngle;
	cv::Mat img = cv::Mat(height, width, CV_64F, double(0));

	std::vector<std::vector<std::vector<double> > > cellGradVec(cellHeight, std::vector<std::vector<double> >(cellWidth, std::vector<double>(_binSize, 0)));
	std::vector<std::vector<double> > featureVec;
	std::vector<double> blockVec;

	// Calculate the gradient of images
	cv::Sobel(_img, gx, CV_64F, 1, 0, 5);
	cv::Sobel(_img, gy, CV_64F, 0, 1, 5);
	cv::cartToPolar(gx, gy, gradMag, gradAngle, 1);
	gradMag = abs(gradMag);

	// Change angle to unsigned gradient (0~180)
	for (size_t i = 0; i < gradAngle.rows; i++)
	{
		for (size_t j = 0; j < gradAngle.cols; j++)
		{
			gradAngle.at<double>(i, j) = std::fmod(gradAngle.at<double>(i, j), static_cast<double>(PIANGLE));
		}
	}

	// Calculate histogram of gradients in cells
	for (size_t i = 0; i < cellGradVec.size(); i++)
	{
		for (size_t j = 0; j < cellGradVec[i].size(); j++)
		{
			cellMag = gradMag(cv::Range(i * _cellSize, (i + 1) * _cellSize), cv::Range(j * _cellSize, (j + 1) * _cellSize));
			cellAngle = gradAngle(cv::Range(i * _cellSize, (i + 1) * _cellSize), cv::Range(j * _cellSize, (j + 1) * _cellSize));
			cellGradVec[i][j] = cellGrad(cellMag, cellAngle);
		}
	}
	
	// Block normalization
	for (size_t i = 0; i < cellGradVec.size() - 1; i++)
	{
		for (size_t j = 0; j < cellGradVec[i].size() - 1; j++)
		{
			blockVec.clear();
			for (size_t k = 0; k < cellGradVec[i][j].size(); k++)
			{
				blockVec.push_back(cellGradVec[i][j][k]);
			}
			for (size_t k = 0; k < cellGradVec[i][j].size(); k++)
			{
				blockVec.push_back(cellGradVec[i][j + 1][k]);
			}
			for (size_t k = 0; k < cellGradVec[i][j].size(); k++)
			{
				blockVec.push_back(cellGradVec[i + 1][j][k]);
			}
			for (size_t k = 0; k < cellGradVec[i][j].size(); k++)
			{
				blockVec.push_back(cellGradVec[i + 1][j + 1][k]);
			}

			blockVec = blockNorm(blockVec);
			featureVec.push_back(blockVec);
		}
	}

	// Render the gradient
	img = renderGrad(img, cellGradVec);
	cv::imwrite(outputFile, img);

	return featureVec;
}

std::vector<double> HOG::cellGrad(cv::Mat &cellMag, cv::Mat &cellAngle)
{
	int binIdx = 0;
	double mod = 0;
	std::vector<double> bins(_binSize, 0);

	for (size_t i = 0; i < cellMag.rows; i++)
	{
		for (size_t j = 0; j < cellMag.cols; j++)
		{
			binIdx = cellAngle.at<double>(i, j) / _binRange;
			mod = std::fmod(cellAngle.at<double>(i, j), static_cast<double>(_binRange));
			bins[binIdx] += cellMag.at<double>(i, j) * (1 - (mod / static_cast<double>(_binRange)));
			bins[(binIdx + 1) % _binSize] += cellMag.at<double>(i, j) * (mod / static_cast<double>(_binRange));
		}
	}

	// Find the max magnitude in the whole image
	for (size_t i = 0; i < bins.size(); i++)
	{
		if (_maxMag < bins[i])
		{
			_maxMag = bins[i];
		}
	}

	return bins;
}

std::vector<double> HOG::blockNorm(std::vector<double> &blockVec)
{
	double mag = 0;

	// Count the magnitude of a block
	for (size_t i = 0; i < blockVec.size(); i++)
	{
		mag += pow(blockVec[i], 2);
	}
	mag = sqrt(mag);

	// Normalization
	if (mag > 0)
	{
		for (size_t i = 0; i < blockVec.size(); i++)
		{
			blockVec[i] /= mag;
		}
	}

	return blockVec;
}

cv::Mat HOG::renderGrad(cv::Mat &img, std::vector<std::vector<std::vector<double> > > &cellGradVec)
{
	int angle = 0;
	int x1 = 0, y1 = 0, x2 = 0, y2 = 0;
	double angleRadian = 0;

	for (size_t i = 0; i < cellGradVec.size(); i++)
	{
		for (size_t j = 0; j < cellGradVec[i].size(); j++)
		{
			angle = 0;
			for (size_t k = 0; k < cellGradVec[i][j].size(); k++)
			{
				cellGradVec[i][j][k] /= _maxMag;
				angleRadian = PI * (angle / PIANGLE);
				x1 = static_cast<int>(j * _cellSize - _cellSize / 2 * cellGradVec[i][j][k] * cos(angleRadian));
				y1 = static_cast<int>(i * _cellSize - _cellSize / 2 * cellGradVec[i][j][k] * sin(angleRadian));
				x2 = static_cast<int>(j * _cellSize + _cellSize / 2 * cellGradVec[i][j][k] * cos(angleRadian));
				y2 = static_cast<int>(i * _cellSize + _cellSize / 2 * cellGradVec[i][j][k] * sin(angleRadian));
				cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), static_cast<int>(255 * cellGradVec[i][j][k]));

				angle += _binRange;
			}
		}
	}

	return img;
}
