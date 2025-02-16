#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;
using namespace cv;

class KNN
{
	int k;
	vector<Mat> trainingImages;
	vector<int> trainingLabels;


public:
	KNN(int k) : k(k) {}

	void train(const vector<Mat>& images, const vector<int>& labels) {
		trainingImages = images;
		trainingLabels = labels;
	}

	int predict(const Mat& queryImage) {

		//tính khoảng cách của ảnh test với tất cả các ảnh trong tập train
		vector<float> distances;
		for (const auto& image : trainingImages) {
			float distance = calculateDistance(queryImage, image);
			distances.push_back(distance);
		}

		std::vector<int> indices(distances.size());
		for (int i = 0; i < distances.size(); ++i) {
			indices[i] = i;
		}

		/*
		* Sắp xếp các index của tập trainingLabels có khoảng cách gần với ảnh test.
		* indices[0] = index có khoảng cách gần với ảnh test nhất
		* indices[1] = index có khoảng cách gần thứ 2
		* indices[2] = index có khoảng cách gần thứ 3
		* indices[3] = index có khoảng cách gần thứ 4
		* ...
		*/
		std::sort(indices.begin(), indices.end(), [&](int a, int b) {
			return distances[a] < distances[b];
			});

		//Lấy ra k láng giềng gần nhất lưu vào tập kNearestLabels
		std::vector<int> kNearestLabels;
		for (int i = 0; i < k; ++i) {
			kNearestLabels.push_back(trainingLabels[indices[i]]);
		}

		return majorityVote(kNearestLabels);
	}

private:
	float calculateDistance(const Mat& img1, const Mat& img2) {
		return norm(img1 - img2);
	}

	int majorityVote(const vector<int>& labels) {
		//Đếm số lượng của từng nhãn có trong tập labels
		std::map<int, int> labelCounts;
		for (int label : labels) {
			labelCounts[label]++;
		}

		//Lấy ra nhãn có số lượng lớn nhất
		int maxCount = 0;
		int predictedLabel = -1;
		for (const auto& pair : labelCounts) {
			if (pair.second > maxCount) {
				maxCount = pair.second;
				predictedLabel = pair.first;
			}
		}

		return predictedLabel;
	}

};



