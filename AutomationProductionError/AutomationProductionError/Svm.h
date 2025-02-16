#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/ml/ml.hpp>

using namespace std;
using namespace cv;
using namespace cv::ml;

class Svm
{
public:
	double C;
	cv::Ptr<cv::ml::SVM> svm_model;

public:

	/// <summary>
	/// Trích chọn đặc trưng Hog
	/// </summary>
	/// <param name="img"> ảnh cần trích chọn đặc trưng </param>
	/// <param name="hogFeatures"> trả về vector đặc trưng </param>
	void extractHOGFeatures(const Mat& img, vector<float>& hogFeatures)
	{
		HOGDescriptor hog;

		Size winSize(64, 128);  // window size for HOG extraction
		Size blockSize(16, 16);
		Size blockStride(8, 8);
		Size cellSize(8, 8);
		int nbins = 9;  // number of orientation bins

		// Set the parameters for HOG
		hog.winSize = winSize;  // resize ảnh với kích thước 64x128
		hog.blockSize = blockSize; // chia ảnh thành các block có kích thước 16x16
		hog.blockStride = blockStride; // chia 1 block thành 4 ô 8x8
		hog.cellSize = cellSize;
		hog.nbins = nbins; // tính biểu đồ histogram chứa 9 bin với các góc 0, 20, 40 ... 160

		// Compute the HOG features for the image
		hog.compute(img, hogFeatures);
	}

	void prepareTrainingData(std::vector<cv::Mat>& images, std::vector<int>& labels, cv::Mat& trainingData, cv::Mat& trainingLabels)
	{
		std::vector<float> features;
		for (size_t i = 0; i < images.size(); ++i)
		{
			extractHOGFeatures(images[i], features);
			trainingData.push_back(cv::Mat(features).t()); // Add feature vector as row
			trainingLabels.push_back(labels[i]); // Add corresponding label
		}
	}

	cv::Ptr<cv::ml::SVM> trainSVM(cv::Mat& trainingData, cv::Mat& trainingLabels)
	{
		cout << "Start training...";
		// Set up SVM parameters
		cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
		svm->setKernel(cv::ml::SVM::LINEAR); // Use linear kernel
		svm->setType(cv::ml::SVM::C_SVC);
		svm->setC(C); // Regularization parameter

		// Train the SVM model
		svm->train(trainingData, cv::ml::ROW_SAMPLE, trainingLabels);
		svm->save("svm_model.xml");  // Save model to a file for later use
		cout << " completed." << endl;
		return svm;
	}

	int classifyImage(cv::Ptr<cv::ml::SVM>& svm, cv::Mat& image)
	{
		// Extract HOG features from the image
		std::vector<float> features;
		extractHOGFeatures(image, features);
		cv::Mat featureMat = cv::Mat(features).t();

		// Predict the label
		return (int)svm->predict(featureMat);
	}

};



