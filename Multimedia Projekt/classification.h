#pragma once
#include <cstddef>						// size_t
#include <string>
#include <deque>
#include <opencv2/core/core.hpp>		// Mat
#include "inria.h"						// inria_cfg
#include "svm_light_wrapper.h"

namespace mmp
{
	class classifier
	{
	private:
		svm::model model;

		std::deque<svm::sparse_vector> positives;
		std::deque<svm::sparse_vector> negatives;
		const inria_cfg& cfg;

	public:
		classifier(classifier&& rhs);
		classifier(const inria_cfg& cfg);
		~classifier();
		
		void train();
		void load(bool hard = false);

		double classify(const cv::Mat& mat) const;
	};
}
