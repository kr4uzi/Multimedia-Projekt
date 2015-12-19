#include "svm_light_wrapper.h"
#include <svm_light/svm.h>
#include <cmath>
using namespace mmp::svm;

namespace
{
	void params_init(LEARN_PARM * learn_parm, KERNEL_PARM * kernel_parm)
	{
		learn_parm->biased_hyperplane = 1;
		learn_parm->sharedslack = 0;
		learn_parm->remove_inconsistent = 0;
		learn_parm->skip_final_opt_check = 0;
		learn_parm->svm_maxqpsize = 10;
		learn_parm->svm_newvarsinqp = 0;
		learn_parm->svm_iter_to_shrink = -9999;
		learn_parm->maxiter = 100000;
		learn_parm->kernel_cache_size = 40;
		learn_parm->svm_c = 0.0;
		learn_parm->eps = 0.1;
		learn_parm->transduction_posratio = -1.0;
		learn_parm->svm_costratio = 1.0;
		learn_parm->svm_costratio_unlab = 1.0;
		learn_parm->svm_unlabbound = 1E-5;
		learn_parm->epsilon_crit = 0.001;
		learn_parm->epsilon_a = 1E-15;
		learn_parm->compute_loo = 0;
		learn_parm->rho = 1.0;
		learn_parm->xa_depth = 0;

		kernel_parm->kernel_type = 0;
		kernel_parm->poly_degree = 3;
		kernel_parm->rbf_gamma = 1.0;
		kernel_parm->coef_lin = 1;
		kernel_parm->coef_const = 1;
	}
}

sparse_vector::sparse_vector(const cv::Mat& mat)
	: _size(mat.rows * mat.cols * mat.channels())
{
	const auto channels = mat.channels();
	std::vector<WORD> words(mat.rows * mat.cols * channels + 1);

	unsigned i = 0;
	for (int y = 0; y < mat.rows; y++)
	{
		auto row = mat.ptr<float>(y);

		for (int x = 0; x < mat.cols; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				if (std::fabs(row[c]))
				{
					words[i].wnum = y * mat.cols + x * channels + c + 1;
					words[i].weight = row[c];
					i++;
				}
			}

			row += channels;
		}
	}
	words[i].wnum = 0;

	_vec = create_svector(words.data(), "", 1);
}

sparse_vector::sparse_vector(sparse_vector&& rhs)
	: _vec(rhs._vec), _size(rhs._size)
{
	rhs._vec = nullptr;
}

sparse_vector::~sparse_vector()
{
	free_svector((SVECTOR *)_vec);
}

float sparse_vector::operator[](size_type idx) const
{
	auto vec = (SVECTOR *)_vec;
	for (std::size_t i = 0; i < _size; i++)
	{
		if (vec->words[i].wnum == idx)
			return vec->words[i].weight;
	}

	return 0;
}

void mmp::svm::model::push_back_example(const sparse_vector& svec, char sign)
{
	docs.push_back(create_example((long)docs.size(), 0, 0, 1, (SVECTOR *)svec.data()));
	targets.push_back(sign);
}

mmp::svm::model& mmp::svm::model::operator=(mmp::svm::model&& rhs)
{
	assert(this != &rhs);

	if (_model)
		free_model((MODEL *)_model, 0);

	_model = rhs._model;
	rhs._model = nullptr;

	docs = std::move(rhs.docs);
	targets = std::move(rhs.targets);
	vec_size = rhs.vec_size;

	return *this;
}

mmp::svm::model::model(mmp::svm::model&& rhs)
	: _model(rhs._model)
{
	rhs._model = nullptr;
}

mmp::svm::model::model(const std::string& filename)
{
	_model = read_model(const_cast<char *>(filename.c_str()));
}

void mmp::svm::model::learn()
{
	LEARN_PARM learn_param;
	KERNEL_PARM kernel_param;
	params_init(&learn_param, &kernel_param);
	learn_param.type = CLASSIFICATION;
	learn_param.svm_c = 0.01;
	kernel_param.kernel_type = LINEAR;

	auto temp = new MODEL;
	svm_learn_classification((DOC **)docs.data(), targets.data(), (long)docs.size(), (long)vec_size, &learn_param, &kernel_param, nullptr, temp, nullptr);
	add_weight_vector_to_linear_model(temp);

	_model = copy_model(temp);
	free_model(temp, 0);	// dont delete docs
}

mmp::svm::model::~model()
{
	if (_model)
		free_model((MODEL *)_model, 1);

	for (auto& doc : docs)
		free_example((DOC *)doc, 0);	// dont delete vectors
}

double mmp::svm::model::classify(const svm::sparse_vector& vec) const
{
	auto doc = create_example(0, 0, 0, 0, (SVECTOR *)vec.data());
	auto dist = classify_example_linear((MODEL *)_model, doc);
	free_example(doc, 0);
	return dist;
}

void mmp::svm::model::save(const std::string& filename) const
{
	write_model(const_cast<char *>(filename.c_str()), (MODEL *)_model);
}