#pragma once
#include <string>

namespace mmp
{
	class inria_cfg
	{
	private:
		std::string root;
		std::string svm_path_normal;
		std::string svm_path_hard;
		std::string eval_file;
		std::string eval_file_hard;
		double _svm_c;
		unsigned num_rngs;
		unsigned num_fps;

	public:
		inria_cfg() = default;
		inria_cfg(const std::string& root, 
			const std::string& svm_file, const std::string& svm_file_hard, 
			const std::string& eval_file, const std::string& eval_file_hard, 
			double svm_c,
			unsigned num_rng_windows_per_neg_sample,
			unsigned num_false_positives_training);

		std::string svm_file() const;
		std::string svm_file_hard() const;
		std::string evaluation_file() const;
		std::string evaluation_file_hard() const;
		std::string root_path() const;

		std::string normalized_positive_train_path() const;
		unsigned normalized_positive_training_y_offset() const;
		unsigned normalized_positive_training_x_offset() const;
		std::string train_annotation_path() const;
		std::string positive_train_path() const;
		std::string negative_train_path() const;
		unsigned random_windows_per_negative_training_sample() const;
		unsigned num_hard_false_positive_retrain() const;

		std::string normalized_positive_test_path() const;
		unsigned normalized_positive_test_y_offset() const;
		unsigned normalized_positive_test_x_offset() const;
		std::string test_annotation_path() const;
		std::string positive_test_path() const;
		std::string negative_test_path() const;

		double svm_c() const;
		std::string training_file() const;
		std::string training_hard_file() const;
	};
}