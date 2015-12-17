#pragma once
#include <string>
#include <vector>
#include <deque>
#include <svm_light/svm_light_util.h>	// SvmLightUtil, SparseVector
#include <opencv2/core/core.hpp>		// Mat
#include "inria.h"						// inria_cfg

namespace mmp
{
	class classifier
	{
	public: 
		static const unsigned hogs_per_negative = 10;

	private:
		SvmLightUtil svm;
		const joachims::model * model;

		SvmLightUtil::Parameters parameters;
		std::vector<SparseVector> positives;
		std::vector<SparseVector> negatives;
		std::deque<SparseVector> false_positives;
		const inria_cfg& cfg;

	private:
		static SparseVector convert_mat(const cv::Mat& mat);

	public:
		classifier(classifier&& rhs);
		classifier(const inria_cfg& cfg);
		~classifier();
		
		void train();
		void load(bool hard = false);

		double classify(const SparseVector& feature) const;
		double classify(const cv::Mat& mat) const;
	};
}
