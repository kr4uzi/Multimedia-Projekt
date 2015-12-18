#pragma once
#include <string>
#include <deque>
#include <opencv2/core/core.hpp>		// Mat
#include "inria.h"						// inria_cfg
#include "sparse_vector.h"

namespace mmp
{
	class classifier
	{
	private:
		// returns mat converted to array of style <index>:<weight> format (WORD *)
		static void * mat_to_svector(const cv::Mat& mat);
		static void * train(const std::deque<void *>& positives, const std::deque<void *>& negatives, long vec_size);
		static double classify(void * model, const cv::Mat& mat, void ** sparse_vec);
		static void save_model_to_file(void * model, const std::string& filename);
		static void * load_model_from_file(const std::string& filename);

	private:
		void * model;

		std::deque<void *> positives;
		std::deque<void *> negatives;
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
