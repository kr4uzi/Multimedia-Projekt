#pragma once

#ifndef _SVM_LIGHT_UTIL_SVM_LIGHT_UTIL
#define _SVM_LIGHT_UTIL_SVM_LIGHT_UTIL

#include <svm_light_old/svm_common.h>
#include <svm_light_old/svm_learn.h>
#include <string>
#include <vector>
#include <fstream>
#include <deque>
#include <svm_light_old/sparse_vector.h>
#include <opencv2/core/core.hpp>

namespace svm
{
	class sparse_vector
	{
	public:
		typedef std::size_t size_type;

		class const_iterator
		{
		private:
			void * _word;

		private:
			friend class sparse_vector;
			const_iterator(void * _word);

		public:
			sparse_vector::size_type index() const;
			float operator*() const;
			const_iterator& operator++();
			bool operator==(const const_iterator& rhs) const;
			bool operator!=(const const_iterator& rhs) const { return !(*this == rhs); }
		};

	private:
		size_type _size;
		void * _vec;
		void * _word_end;

	public:
		sparse_vector(const sparse_vector& rhs);
		sparse_vector(sparse_vector&& rhs);

		sparse_vector(const cv::Mat& mat);
		~sparse_vector();

		float operator[](size_type i) const;
		size_type size() const { return _size; }
		void * data() { return _vec; }
		const void * data() const { return _vec; }

		const_iterator begin() const;
		const_iterator end() const;
	};

	std::string to_string(const sparse_vector& svec);
}

enum EXAMPLE_LABEL
{
	NEG_EXAMPLE = -1,
	POS_EXAMPLE = 1
};

// @TODO: 
// - refactor name 'vocabulary_size'
// - clean up code
// - since SvmLightUtil does not have any members(?) it should be rather a 
//   namespace than a class
/// C++ wrapper for SVM^Light. For more details, refer to http://www.cs.cornell.edu/People/tj/svm_light/ 
class SvmLightUtil
{
public:
	SvmLightUtil();
	~SvmLightUtil();

	static const int DEFAULT_VERBOSITY = 0;

	/// This class stores all information required for training a model
	///
	/// Usage example:
	/// - Ininitialize default parameters for a binary classification problem with a linear model, and C=1 
	/// \code
	///	SvmLightUtil::Parameters svmParameters;
	///	svmParameters.kernel_type = SVM_LIGHT_LINEAR; 
	///	svmParameters.model_type = CLASSIFICATION; 
	///	svmParameters.svm_c = 1; 
	/// \endcode
	class Parameters
	{
	public:
		Parameters();

		int kernel_type;
		double kernel_param;
		int model_type;
		double svm_c;
		bool compute_loo;
		int biased_hyperplane;
	};

	/// Reads in a model from file svm_model_file (absolute path)
	static joachims::MODEL const* getModel(std::string const& svm_model_file);
	/// Deletes model svm_model
	static void deleteModel(joachims::MODEL const* svm_model);

	// Convert a JWORD sparse representation to a c-array (dense)
	static void convert(joachims::JWORD const* words, int size, double* dense_arr, int offset);
	// Convert a JWORD sparse representation to a SparseVector
	static void convert(joachims::JWORD const* words, int size, SparseVector& sparse_vector, int offset);
	// Convert a SparseVector to JWORD sparse representation
	static void convert(SparseVector const& sparse_vector, int size, joachims::JWORD* words, int offset);
	static void convert(SparseVector const& sparse_vector, int size, joachims::JWORD* words);
	// Convert a c-array (dense) to JWORD sparse representation
	static void convert(double const* dense_arr, int size, joachims::JWORD* words, int offset);
	static void convert(double const* dense_arr, int size, joachims::JWORD* words);
	// Convert a c-array (dense) to SparseVector
	static void convert(double const* dense_arr, SparseVector& sparse_vector, int size);

	static joachims::JWORD* allocateSparseVector(int size_n);

	/// Sets the verbosity level. Set value to: 0 disabled output, 1 enabled (basic), 2 verbose
	void setVerbosity(int value) const;
	
	/// Reads in all parameters of a previously trained model from file svm_model_file (absolute path).
	/// 
	/// - b is the hyperplane bias.
	/// - support_vectors is filled with training examples corresponding to non-zero Lagrange Multipliers
	/// - alphas is filled with the (signed) Lagrange Multipliers. For instance,
	/// a positive sign indicates a support vector that corresponds to a positive training example.
	/// \attention Both, support_vectors and alphas, must be empty vectors
	void getDecisionFunctionParameters(std::string const& svm_model_file, int size_n, double& b, std::vector<SparseVector>& support_vectors, std::vector<double>& alphas) const;
	void getDecisionFunctionParameters(joachims::MODEL const* svm_model, int size_n, double& b, std::vector<SparseVector>& support_vectors, std::vector<double>& alphas) const;
	
	void train(std::string const& svm_data_file, std::string const& svm_model_file, double cost_ratio, Parameters const& parameters) const;
	void train(std::vector<SparseVector> const& examples, std::vector<double> const& targets, int size_n, std::string const& svm_model_file, Parameters const& parameters) const;
	joachims::MODEL const* train(std::vector<SparseVector> const& examples, std::vector<double> const& targets, std::vector<double> const& costfactors, int size_n, double cost_ratio, Parameters const& parameters) const;
	joachims::MODEL const* train(std::vector<SparseVector> const& examples, std::vector<double> const& targets, int size_n, Parameters const& parameters) const;	
	void train(std::vector<SparseVector> const& examples, std::vector<double> const& targets, std::vector<double> const& costfactors, int size_n, std::string const& svm_model_file, double cost_ratio, Parameters const& parameters) const;
	/// Trains a model with the supplied positive and negative examples as training data.
	/// The resulting model is store to file svm_model_file (absolute path).
	/// - cost_ratio should be set to the default (1.0), i.e. if the ratio of positive and negative examples is balanced
	/// - parameters stores all information about the training parameters, such as C, the kernel type and the model type 
	void train(std::vector<SparseVector> const& pos_examples, std::vector<SparseVector> const& neg_examples, int size_n, std::string const& svm_model_file, double cost_ratio, Parameters const& parameters) const;
	//added
	void train(std::deque<svm::sparse_vector> const& pos_examples, std::deque<svm::sparse_vector> const& neg_examples, int size_n, const std::string& model_file, double cost_ratio, Parameters const& parameters) const;

	joachims::MODEL const* train(std::vector<SparseVector> const& pos_examples, std::vector<SparseVector> const& neg_examples, int size_n, double cost_ratio, Parameters const& parameters) const;
	joachims::MODEL const* train(std::vector<double const*> const& examples, std::vector<double> const& targets, int size_n, Parameters const& parameters) const;

	//joachims::MODEL const* train(std::vector<SparseVector> const& pos_examples, std::vector<SparseVector> const& neg_examples, int size_n, double cost_ratio, Parameters const& parameters) const;
	
	// @TODO: return a std::vector instead?
	double const* classify(std::string const& svm_data_file, std::string const& svm_model_file) const;
	double const* classify(std::vector<SparseVector> const& examples, int size_n, std::string const& svm_model_file) const;
	double const* classify(std::vector<SparseVector> const& pos_examples, std::vector<SparseVector> const& neg_examples, int size_n, std::string const& svm_model_file) const;
	double const* classify(std::string const& svm_data_file, joachims::MODEL const* svm_model) const;
	
	/// Classifies a single example of dimension size_n, given a model svm_model. Returns the signed distance from hyperplane
	double classify(SparseVector const& example, int size_n, joachims::MODEL const* svm_model) const;
	double const* classify(std::vector<SparseVector> const& examples, int size_n, joachims::MODEL const* svm_model) const;
	double const* classify(std::vector<SparseVector> const& pos_examples, std::vector<SparseVector> const& neg_examples, int size_n, joachims::MODEL const* svm_model) const;
	double classify(double const* example, int size_n, joachims::MODEL const* svm_model) const;
	//added
	double classify(const svm::sparse_vector& svec, int size_n, joachims::MODEL const * svm_model) const;


	void readFromFile(std::string const& file, std::vector<SparseVector>& examples, std::vector<double>& targets, int size_n) const;
	void readFromFile(std::string const& file, std::vector<SparseVector>& examples, EXAMPLE_LABEL example_label, int size_n) const;
	void writeToFile(std::string const& file, std::vector<SparseVector> const& examples, EXAMPLE_LABEL example_label) const;
	void writeToFile(std::string const& file, std::vector<SparseVector> const& examples, std::vector<double> const& targets) const;
	void writeToStream(std::ofstream& output_stream, SparseVector const& example, double target) const;

private:
	void initialize(joachims::LEARN_PARM *learn_parm, joachims::KERNEL_PARM *kernel_parm) const;
	void train(std::string const& svm_model_file, joachims::DOC **docs, long totwords, long totdoc, double *target, double cost_ratio, Parameters const& parameters) const;
	joachims::MODEL const* train(joachims::DOC **docs, long totwords, long totdoc, double *target, double cost_ratio, Parameters const& parameters) const;
	void classify(joachims::MODEL const* svm_model, joachims::DOC **docs, long totwords, long totdoc, double* distances) const;
};

#endif
