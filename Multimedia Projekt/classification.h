#pragma once
#include <deque>
#include <opencv2/core/core.hpp>	// Mat
#include <svm_light/svm.h>			// linear_model
#include "inria.h"					// inria_cfg
#include "extraction.h"				// hog::vector_type

namespace mmp
{
	class classifier
	{
	private:
		svm::linear_model * model;
		std::deque<svm::sparse_vector> positives;
		std::deque<svm::sparse_vector> negatives;
		const inria_cfg& cfg;

	public:
		static svm::sparse_vector mat_to_svector(const cv::Mat& mat);

	public:
		classifier(classifier&& rhs);
		classifier(const inria_cfg& cfg);
		~classifier();
		
		void train();
		void load(bool hard = false);

		double classify(const cv::Mat& mat) const;
	};
}
