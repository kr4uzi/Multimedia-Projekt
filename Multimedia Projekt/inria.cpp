#include "inria.h"
using namespace mmp;

inria_cfg::inria_cfg(const std::string& r, const std::string& s, const std::string& sh, const std::string& ev, const std::string& evh)
	: root(r), svm_path_normal(s), svm_path_hard(sh), eval_file(ev), eval_file_hard(evh)
{

}

std::string inria_cfg::root_path() const { return root; }
std::string inria_cfg::svm_file() const { return svm_path_normal; }
std::string inria_cfg::svm_file_hard() const { return svm_path_hard; }
std::string inria_cfg::evaluation_file() const { return eval_file; }
std::string inria_cfg::evaluation_file_hard() const { return eval_file_hard; }
std::string inria_cfg::negative_test_path() const { return root + "/Test/neg/"; }
std::string inria_cfg::negative_train_path() const { return root + "/Train/neg/"; }
std::string inria_cfg::normalized_positive_test_path() const { return root + "/70X134H96/Test/pos/"; }
std::string inria_cfg::normalized_positive_train_path() const { return root + "/96X160H96/Train/pos/"; }
std::string inria_cfg::positive_test_path() const { return root + "/Test/pos/"; }
std::string inria_cfg::positive_train_path() const { return root + "/Train/pos/"; }
std::string inria_cfg::test_annotation_path() const { return root + "/Test/annotations/"; }
std::string inria_cfg::train_annotation_path() const { return root + "/Train/annotations/"; }
unsigned inria_cfg::normalized_positive_training_y_offset() const { return 16; }
unsigned inria_cfg::normalized_positive_training_x_offset() const { return 16; }
unsigned inria_cfg::normalized_positive_test_y_offset() const { return 3; }
unsigned inria_cfg::normalized_positive_test_x_offset() const { return 3; }
unsigned inria_cfg::random_windows_per_negative_training_sample() const { return 10; }
