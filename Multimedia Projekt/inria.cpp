#include "inria.h"
using namespace mmp;

inria_cfg::inria_cfg(const std::string& r, const std::string& s, const std::string& sh, const std::string& ev, const std::string& evh, double c, unsigned num_rng_windows_per_neg_sample, unsigned num_false_positives_training)
	: root(r), svm_path_normal(s), svm_path_hard(sh), eval_file(ev), eval_file_hard(evh), _svm_c(c), num_rngs(num_rng_windows_per_neg_sample), num_fps(num_false_positives_training)
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
unsigned inria_cfg::random_windows_per_negative_training_sample() const { return num_rngs; }
double inria_cfg::svm_c() const { return _svm_c; }
std::string inria_cfg::training_file() const { return root + "/training_normal.dat"; }
std::string inria_cfg::training_hard_file() const { return root + "/training_hard.dat"; }
unsigned inria_cfg::num_hard_false_positive_retrain() const { return num_fps; }
