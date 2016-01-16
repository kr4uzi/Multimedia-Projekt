#pragma once
#include <deque>
#include <opencv2/core/core.hpp>	// Mat
#include <svm_light/svm.h>			// linear_model, sparse_vector

namespace mmp
{
	class inria_cfg;

	class classifier
	{
	private:
		const inria_cfg& cfg;
		svm::linear_model * model;

		std::deque<svm::sparse_vector> positives;
		std::deque<svm::sparse_vector> negatives;

	private:
		static svm::sparse_vector features_to_svector(const cv::Mat& mat);

	public:
		classifier(classifier&& rhs);
		classifier(const inria_cfg& cfg);
		~classifier();
		
		void train();
		void load(bool hard = false);

		double classify(const cv::Mat& mat) const;
	};
}
