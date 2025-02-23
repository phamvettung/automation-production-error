#include <opencv2/opencv.hpp>
#include <vector>
#include <random>
#include <iostream>

#include "KNN.h"
#include "Svm.h"
#include "OtsuThresholding.h"

using namespace std;
using namespace cv;
using namespace cv::ml;

vector<Mat> flawlessImg_Full, pressedImg_Full, flawlessImg_32, pressedImg_32;



/// <summary>
/// Đọc ảnh với kích thước đầy đủ và với kích thước 32x32.
/// </summary>
/// <param name="flawlessImgs"></param>
/// <param name="pressedImgs"></param>
void readImageFolder(vector<Mat>& flawlessImgFull, vector<Mat>& pressedImgFull, vector<Mat>& flawlessImg32, vector<Mat>& pressedImg32) {
	vector<String> fn_flawless, fn_pressed;
	glob("Dataset/flawless/*.jpg", fn_flawless, false);
	glob("Dataset/pressed/*.jpg", fn_pressed, false);

	//Trộn ngẫu nhiên
	random_device rd;
	mt19937 g(rd());
	shuffle(flawlessImgFull.begin(), flawlessImgFull.end(), g);
	shuffle(pressedImgFull.begin(), pressedImgFull.end(), g);

	shuffle(flawlessImg32.begin(), flawlessImg32.end(), g);
	shuffle(pressedImg32.begin(), pressedImg32.end(), g);

	Mat src;
	size_t count_flawless = fn_flawless.size();
	for (size_t i = 0; i < count_flawless; i++) {
		src = cv::imread(fn_flawless[i]);
		flawlessImgFull.push_back(src);
		resize(src, src, Size(32, 32));
		flawlessImg32.push_back(src);
	}
	size_t count_pressed = fn_pressed.size();
	for (size_t i = 0; i < count_pressed; i++) {
		src = cv::imread(fn_pressed[i]);
		pressedImgFull.push_back(src);
		resize(src, src, Size(32, 32));
		pressedImg32.push_back(src);
	}
}


/// <summary>
/// 
/// </summary>
/// <param name="Data"></param>
/// <param name="k"></param>
/// <param name="ratio"></param>
/// <returns></returns>
vector<vector<Mat>> splitData(const vector<Mat>& Data, int k, double trainingRate = 0) {
	vector<vector<Mat>> result;
	int totalSize = Data.size();

	if (k == 0 || totalSize == 0 || k > totalSize) {
		std::cerr << "Invalid input values for k or vector size." << std::endl;
		return result;
	}

	if (trainingRate == 0) {
		int partSize = totalSize / k;
		for (int i = 0; i < k; ++i) {
			int start = i * partSize;
			int end = (i == k - 1) ? totalSize : (i + 1) * partSize;
			std::vector<Mat> part(Data.begin() + start, Data.begin() + end);
			result.push_back(part);
		}
	}
	else
	{
		int trainingSize = totalSize * trainingRate;
		for (int i = 0; i < k; ++i) {
			int start = i * trainingSize;
			int end = (i == k - 1) ? totalSize : (i + 1) * trainingSize;
			std::vector<Mat> part(Data.begin() + start, Data.begin() + end);
			result.push_back(part);
		}
	}

	return result;
}

/// <summary>
/// 
/// </summary>
/// <param name="class1"></param>
/// <param name="class2"></param>
/// <param name="k_fold"></param>
void crossValidationCombineStratifiedSampling(KNN& knn, const vector<Mat>& class1, const vector<Mat>& class2, int k_fold = 5) {
	//1. Chia tập tập D thành k phần bằng nhau và không giao nhau.
	vector<vector<Mat>> D_class1 = splitData(class1, k_fold);
	vector<vector<Mat>> D_class2 = splitData(class2, k_fold);

	//Tập các quan sát và nhãn đã được phân tầng.
	vector<vector<Mat>> D_stratified;
	vector <vector<int>> Lb_stratified;
	for (int i = 0; i < k_fold; i++) {

		vector<Mat> observations; vector<int> labels;
		copy(D_class1[i].begin(), D_class1[i].end(), back_inserter(observations));
		copy(D_class2[i].begin(), D_class2[i].end(), back_inserter(observations));

		vector<int> Lb_class1(D_class1[i].size(), 0); vector<int> Lb_class2(D_class2[i].size(), 1);
		copy(Lb_class1.begin(), Lb_class1.end(), back_inserter(labels));
		copy(Lb_class2.begin(), Lb_class2.end(), back_inserter(labels));

		D_stratified.push_back(observations);
		Lb_stratified.push_back(labels);
	}

	//2. Thực hiện k lần chạy, mỗi lần chạy lấy 1 phần dùng để test, còn lại dùng để huấn luyện.
	vector<Mat> D_train, D_test;
	vector<int> Lb_train, Lb_test;
	vector<double> results;
	for (int i = 0; i < k_fold; ++i) {

		//Xóa vector trước khi gán lại phần tử cho vector
		D_test.clear(); D_train.clear();

		//Lấy 1 phần của tập D để test
		D_test = D_stratified[i];
		Lb_test = Lb_stratified[i];

		//Phần còn lại dùng để train
		for (int j = 0; j < D_stratified.size(); j++) {
			if (j != i) {
				copy(D_stratified[j].begin(), D_stratified[j].end(), back_inserter(D_train));
				copy(Lb_stratified[j].begin(), Lb_stratified[j].end(), back_inserter(Lb_train));
			}
		}

		//Huấn luyện mô hình
		knn.train(D_train, Lb_train);
		//Dùng độ đo accuracy để đánh giá mô hình
		int numberOfCorrectPredictions = 0, TotalNumberOfPredictions = D_test.size();
		for (int i = 0; i < D_test.size(); i++) {
			Mat queryImage = D_test[i];
			int predictLabel = knn.predict(queryImage);
			if (predictLabel == Lb_test[i])
				numberOfCorrectPredictions++;
		}
		double accuracy = (double)numberOfCorrectPredictions / (double)TotalNumberOfPredictions;
		results.push_back(accuracy);
		cout << "k = " << i + 1 << ", Number of correct predictions: " << numberOfCorrectPredictions << ", Accuracy: " << accuracy << endl;
	}

	//3. Lấy trung bình kết quả đánh giá từ k lần chạy.
	double sumAccuracy = 0;
	for (int i = 0; i < results.size(); i++)
		sumAccuracy += results[i];
	double averageAccuracy = sumAccuracy / results.size();
	cout << "Average accuracy = " << averageAccuracy << endl;
}


void modelSelectionUsingHoldout(Svm& svm, const vector<Mat>& class1, const vector<Mat>& class2, int k = 2) {
	//1. Chia tập tập D thành 2 phần D_train, T_valid
	vector<vector<Mat>> D_class1 = splitData(class1, k, 0.1 /*lấy ra 1/10 ảnh để train*/);
	vector<vector<Mat>> D_class2 = splitData(class2, k, 0.1);

	//Tập các quan sát và nhãn của 2 class.
	vector<vector<Mat>> D; //[0] = D train, [1] = T valid
	vector <vector<int>> Lb;
	for (int i = 0; i < k; i++) {

		vector<Mat> observations; vector<int> labels;
		copy(D_class1[i].begin(), D_class1[i].end(), back_inserter(observations));
		copy(D_class2[i].begin(), D_class2[i].end(), back_inserter(observations));

		vector<int> Lb_class1(D_class1[i].size(), 0); vector<int> Lb_class2(D_class2[i].size(), 1);
		copy(Lb_class1.begin(), Lb_class1.end(), back_inserter(labels));
		copy(Lb_class2.begin(), Lb_class2.end(), back_inserter(labels));

		D.push_back(observations);
		Lb.push_back(labels);
	}

	// Tập S chứa các giá trị C tiềm năng
	vector<double> S = { 1 };
	vector<Mat> D_train, T_valid;
	vector<int> Lb_train, Lb_valid;
	double C_optimal = 0, Pc = 0;
	//Với mỗi C thuộc S, train với D_train, đo hiệu quả với T_valid và lấy kết quả Pc.
	for (int i = 0; i < S.size(); i++) {
		D_train = D[0]; T_valid = D[1];
		Lb_train = Lb[0]; Lb_valid = Lb[1];

		svm.C = S[i];
		// Prepare training data
		cv::Mat trainingData, trainingLabels;
		svm.prepareTrainingData(D_train, Lb_train, trainingData, trainingLabels);
		// Train the SVM
		svm.svm_model = svm.trainSVM(trainingData, trainingLabels);
		// Classify a new image
		int numberOfCorrectPredictions = 0, TotalNumberOfPredictions = T_valid.size();
		for (int i = 0; i < T_valid.size(); i++) {
			cv::Mat queryImage;
			cvtColor(T_valid[i], queryImage, COLOR_BGR2GRAY);
			int predictLabel = svm.classifyImage(svm.svm_model, queryImage);
			if (predictLabel == Lb_valid[i])
				numberOfCorrectPredictions++;
		}
		double accuracy = (double)numberOfCorrectPredictions / (double)TotalNumberOfPredictions;
		if (accuracy > Pc) { //Chọn ra C tốt nhất tương ứng với Pc lớn nhất
			Pc = accuracy;
			C_optimal = S[i];
		}
		cout << "C = " << S[i] << ", Number of correct predictions: " << numberOfCorrectPredictions << ", Accuracy: " << accuracy << endl;
	}
	cout << "The best C value is " << C_optimal << endl;
}


Mat gradient(Mat img) {
	Mat m = Mat::zeros(Size(img.rows, img.cols), CV_8UC1);
	for (int x = 0; x < img.rows - 1; x++)
		for (int y = 0; y < img.cols - 1; y++) {
			int gx = (int)img.at<uchar>(Point(x + 1, y)) - (int)img.at<uchar>(Point(x, y));
			int gy = (int)img.at<uchar>(Point(x, y + 1)) - (int)img.at<uchar>(Point(x, y));
			m.at<uchar>(Point(x, y)) = abs(gx) + abs(gy);
		}

	OtsuThresholding ots;
	Mat matThreshed;
	int thresh = ots.computeThreshold(m);
	threshold(m, matThreshed, thresh, 255, THRESH_BINARY);

	return matThreshed;
}

Mat getContours(Mat imgOriginal, Mat img, string labels) {
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(img, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

	vector<vector<Point>> conPoly(contours.size());
	vector<Rect> boundRect(contours.size());

	//Chuyển ảnh mức xám sang ảnh grb
	Mat grbImage, grayImage;
	cvtColor(imgOriginal, grayImage, COLOR_RGB2GRAY);
	cvtColor(grayImage, grbImage, COLOR_GRAY2RGB);

	for (int i = 0; i < contours.size(); i++) {
		int area = contourArea(contours[i]);
		cout << "area: " << area << endl;

		if (area > 1000) {
			float peri = arcLength(contours[i], true);
			approxPolyDP(contours[i], conPoly[i], 0.02 * peri, true);
			//cout << "conPoly: " << conPoly[i].size() << endl;
			boundRect[i] = boundingRect(conPoly[i]);

			if (labels == "OK") {
				rectangle(grbImage, boundRect[i].tl(), boundRect[i].br(), Scalar(0, 255, 0), 3);
				putText(grbImage, labels, { boundRect[i].x + 20, boundRect[i].y + 20 }, FONT_HERSHEY_COMPLEX_SMALL, 0.95, Scalar(0, 255, 0));
			}
			else
			{
				rectangle(grbImage, boundRect[i].tl()/*top left*/, boundRect[i].br()/*bottom right*/, Scalar(0, 0, 255), 3);
				putText(grbImage, labels, { boundRect[i].x + 20, boundRect[i].y + 20 }, FONT_HERSHEY_COMPLEX_SMALL, 0.95, Scalar(0, 0, 255));
			}
		}
	}
	return grbImage;
}


Mat detectProduct(Mat inputImg, int label) {

	Mat hsvImage, mask;
	int hmin = 10, smin = 0, vmin = 0;
	int hmax = 43, smax = 255, vmax = 255;

	cvtColor(inputImg, hsvImage, cv::COLOR_BGR2HSV);

	Scalar lower(hmin, smin, vmin);
	Scalar upper(hmax, smax, vmax);

	inRange(hsvImage, lower, upper, mask);

	Mat imageNegative = 255 - mask;

	Mat gradientImage = gradient(mask);
	Mat kernel = getStructuringElement(MORPH_RECT, Size(5, 5));
	Mat dilatedImage, erodedImage;
	dilate(gradientImage, dilatedImage, kernel);
	erode(dilatedImage, erodedImage, kernel);

	/*imshow("mask", mask);
	imshow("img negative", imageNegative);
	imshow("gradient", gradientImage);
	imshow("dilate img", dilatedImage);
	imshow("erode", erodedImage);*/

	Mat contours;
	if (label == 0) {
		contours = getContours(inputImg, imageNegative, "OK");
	}
	else {
		contours = getContours(inputImg, imageNegative, "NG");
	}

	return contours;
}

void testModel(KNN& knn, Svm& svm, vector<Mat> flawlessImg, vector<Mat> pressedImg, int models = 0) {

	int predictLabel;
	Mat resizedImage, outputImage, queryImage;

	for (int i = 0; i < 5; ++i) { //test 5 ảnh đầu tiên
		if (models == 1) {
			resize(flawlessImg[i], resizedImage, Size(32, 32));
			predictLabel = knn.predict(resizedImage);
			outputImage = detectProduct(flawlessImg[i], predictLabel);
			imshow("k-NN 1", outputImage);

			resize(pressedImg[i], resizedImage, Size(32, 32));
			predictLabel = knn.predict(resizedImage);
			outputImage = detectProduct(pressedImg[i], predictLabel);
			imshow("k-NN 2", outputImage);

		}
		else if (models == 2)
		{
			cvtColor(flawlessImg[i], queryImage, COLOR_BGR2GRAY);
			predictLabel = svm.classifyImage(svm.svm_model, queryImage);
			outputImage = detectProduct(flawlessImg[i], predictLabel);
			imshow("SVM 1", outputImage);


			cvtColor(pressedImg[i], queryImage, COLOR_BGR2GRAY);
			predictLabel = svm.classifyImage(svm.svm_model, queryImage);
			outputImage = detectProduct(pressedImg[i], predictLabel);
			imshow("SVM 2", outputImage);
		}

		waitKey(0);
	}
}


int main() {

	int models = 0;
	cout << "Choose a classification model: \n1. k-Nearest Neighbors.\n2. Support Vector Machines." << endl; cin >> models;

	KNN knn(3);
	Svm svm;

	switch (models)
	{
	case 1:
		cout << "-- k-Nearest Neighbors using Cross validation combines Stratified sampling --" << endl;
		readImageFolder(flawlessImg_Full, pressedImg_Full, flawlessImg_32, pressedImg_32);
		crossValidationCombineStratifiedSampling(knn, flawlessImg_32, pressedImg_32);
		break;
	case 2:
		cout << "-- Support Vector Machines with Model selection using holdout --" << endl;
		readImageFolder(flawlessImg_Full, pressedImg_Full, flawlessImg_32, pressedImg_32);
		modelSelectionUsingHoldout(svm, flawlessImg_Full, pressedImg_Full);
		break;
	default:
		break;
	}

	testModel(knn, svm, flawlessImg_Full, pressedImg_Full, models);
	return 0;

}