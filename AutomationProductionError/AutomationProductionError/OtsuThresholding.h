#pragma once
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class OtsuThresholding
{
public:
	long double computeThreshold(Mat src) {
		long double hist[256];
		calculateHistogram(src, hist);

		long double maxVariance = LDBL_MIN; int maxIndex = 0;
		long double Wb, Wf, Mb, Mf, currentVariance; //Weight background-foreground, Mean background-foreground

		for (int i = 0; i < 256; i++) {
			//Tìm giá trị phương sai với sự tương đồng trong 1 lớp lớn nhất.
			Wb = calculateWeight(0, i + 1, hist);
			Wf = calculateWeight(i + 1, 256, hist);
			Mb = calculateMean(0, i + 1, hist);
			Mf = calculateMean(i + 1, 256, hist);
			currentVariance = Wb * Wf * pow((Mb - Mf), 2);
			if (currentVariance > maxVariance) {
				maxVariance = currentVariance;
				maxIndex = i;
			}
		}

		return maxIndex;
	}

private:
	/// <summary>
	/// Tính biểu đồ histogram
	/// </summary>
	/// <param name="src"></param>
	/// <param name="hist"></param>
	/// <returns></returns>
	long double* calculateHistogram(Mat src, long double hist[]) {
		//Khởi tạo mảng histogram
		for (int i = 0; i < 256; i++)
			hist[i] = 0;
		//tính toán histogram
		for (int y = 0; y < src.rows; y++)
			for (int x = 0; x < src.cols; x++)
				hist[(int)src.at<uchar>(y, x)]++;
		return hist;
	}

	/// <summary>
	/// Tính tổng số mức xám
	/// </summary>
	/// <param name="hist"></param>
	/// <returns></returns>
	int calculateN(long double* hist) {
		int n = 0;
		for (int i = 0; i < 256; i++) {
			n += hist[i];
		}
		return n;
	}

	/// <summary>
	/// Tính trọng số từ hist[s] -> hist[e]
	/// </summary>
	/// <param name="s"></param>
	/// <param name="e"></param>
	/// <param name="hist"></param>
	/// <returns></returns>
	double calculateWeight(int s, int e, long double* hist) {
		long double weight = 0, ni = 0; int n = 0;
		for (int i = s; i < e; i++) {
			ni += hist[i];
		}
		n = calculateN(hist);
		weight = ni / n;
		return weight;
	}

	/// <summary>
	/// Tính trung bình mức xám từ hist[s] -> hist[e]
	/// </summary>
	/// <param name="s"></param>
	/// <param name="e"></param>
	/// <param name="hist"></param>
	/// <returns></returns>
	double calculateMean(int s, int e, long double* hist) {
		long double mean = 0;  long double ipi = 0, weight = 0;
		for (int i = s; i < e; i++) {
			ipi += hist[i] * i;
			weight += hist[i];
		}
		//weight = calWeight(s, e, hist);
		mean = ipi / weight;
		return mean;
	}
};



