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

	public:
		inria_cfg(const std::string& root, const std::string& svm_file, const std::string& svm_file_hard, const std::string& eval_file, const std::string& eval_file_hard);

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

		std::string normalized_positive_test_path() const;
		unsigned normalized_positive_test_y_offset() const;
		unsigned normalized_positive_test_x_offset() const;
		std::string test_annotation_path() const;
		std::string positive_test_path() const;
		std::string negative_test_path() const;

		bool debug() const { return true; }
		std::string training_file() const { return root + "/training_normal.dat"; }
	};
}