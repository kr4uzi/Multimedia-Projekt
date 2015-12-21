#include "svm_light_wrapper.h"
#include <svm_light/svm.h>
#include <cmath>		// fabs
#include <type_traits>	// underlying_type
#include <cstring>		// strcpy
#include <fstream>
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
		std::strncpy(kernel_parm->custom, "empty", sizeof(KERNEL_PARM::custom));
	}
}

std::string mmp::svm::to_string(const sparse_vector& svec)
{
	std::string str;
	//for (sparse_vector::size_type i = 1; i <= svec.size(); i++)
	//{
	//	auto val = svec[i];
	//	if (std::fabs(val) > 0)
	//	{
	//		str += ' ';
	//		str += std::to_string(i);
	//		str += ':';
	//		str += std::to_string(val);
	//	}
	//}
	for (auto i = svec.begin(); i != svec.end(); ++i)
	{
		str += std::to_string(i.index());
		str += ':';
		str += std::to_string(*i);
		str += ' ';
	}

	return str;
}

sparse_vector::const_iterator sparse_vector::begin() const
{
	return const_iterator(((SVECTOR *)_vec)->words);
}

sparse_vector::const_iterator sparse_vector::end() const
{
	return const_iterator(_word_end);
}

sparse_vector::const_iterator::const_iterator(void * word)
	: _word(word)
{

}

bool sparse_vector::const_iterator::operator==(const const_iterator& rhs) const
{
	return _word == rhs._word;
}

sparse_vector::const_iterator& sparse_vector::const_iterator::operator++()
{
	auto word = (WORD *)_word;
	_word = ++word;
	return *this;
}

float sparse_vector::const_iterator::operator*() const
{
	return ((WORD *)_word)->weight;
}

sparse_vector::size_type sparse_vector::const_iterator::index() const
{
	return ((WORD *)_word)->wnum;
}

sparse_vector::sparse_vector(const sparse_vector& rhs)
	: _size(rhs._size)
{
	_vec = copy_svector((SVECTOR *)rhs._vec);
}

sparse_vector::sparse_vector(const cv::Mat& mat)
	: _size(mat.rows * mat.cols * mat.channels())
{
	//const auto channels = mat.channels();
	//std::vector<WORD> words(mat.rows * mat.cols * channels + 1);

	//unsigned i = 0;
	//for (int y = 0; y < mat.rows; y++)
	//{
	//	auto row = mat.ptr<float>(y);

	//	for (int x = 0; x < mat.cols; x++)
	//	{
	//		for (int c = 0; c < channels; c++)
	//		{
	//			if (std::fabs(row[c]) > 0)
	//			{
	//				words[i].wnum = y * mat.cols * channels + x * channels + c + 1;
	//				words[i].weight = row[c];
	//				i++;					
	//			}
	//		}

	//		row += channels;
	//	}
	//}
	//words[i].wnum = 0;

	//_vec = create_svector(words.data(), "", 1);
	const auto channels = mat.channels();
	auto vec = new SVECTOR;
	vec->words = new WORD[mat.rows * mat.cols * channels + 1];

	unsigned i = 0;
	for (int y = 0; y < mat.rows; y++)
	{
		auto row = mat.ptr<float>(y);

		for (int x = 0; x < mat.cols; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				if (std::fabs(row[c]) > 0)
				{
					vec->words[i].wnum = y * mat.cols * channels + x * channels + c + 1;
					vec->words[i].weight = row[c];
					i++;
				}
			}

			row += channels;
		}
	}
	vec->words[i].wnum = 0;
	_word_end = &vec->words[i];

	vec->twonorm_sq = sprod_ss(vec, vec);
	vec->userdefined = new char[1];
	vec->userdefined[0] = 0;
	vec->kernel_id = 0;
	vec->next = nullptr;
	vec->factor = 1;
	_vec = vec;
}

sparse_vector::sparse_vector(sparse_vector&& rhs)
	: _vec(rhs._vec), _size(rhs._size), _word_end(rhs._word_end)
{
	rhs._vec = nullptr;
}

sparse_vector::~sparse_vector()
{
	//free_svector((SVECTOR *)_vec);
	if (_vec)
	{
		auto vec = (SVECTOR *)_vec;
		delete[] vec->words;
		delete[] vec->userdefined;
		delete vec;
	}
}

float sparse_vector::operator[](size_type idx) const
{
	auto start = ((SVECTOR *)_vec)->words;
	while (start->wnum && start->wnum <= idx)
	{
		if (start->wnum == idx)
			return start->weight;

		start++;
	}

	return 0;
}

void mmp::svm::model::push_back_example(const svm::sparse_vector& svec, example_type sign)
{
	typedef std::underlying_type<example_type>::type etype;
	docs.push_back(create_example((long)docs.size(), 0, 0, 1, (SVECTOR *)svec.data()));
	targets.push_back(static_cast<etype>(sign));
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
	add_weight_vector_to_linear_model((MODEL *)_model);
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
	free_model(temp, 0);	// we're not owner of the docs (they get deleted when this object is beeing destructed)
}

mmp::svm::model::~model()
{
	if (_model)
		free_model((MODEL *)_model, 1);	// delete docs (added via constructor or via learn [copy])

	for (auto& doc : docs)
		free_example((DOC *)doc, 0);	// dont delete sparse_vectors
}

double mmp::svm::model::classify(const svm::sparse_vector& vec) const
{
	if (!_model) throw std::exception("model not loaded!");

	auto doc = create_example(0, 0, 0, 0, (SVECTOR *)vec.data());
	auto dist = classify_example_linear((MODEL *)_model, doc);
	free_example(doc, 0);
	return dist;
}

void mmp::svm::model::save(const std::string& filename) const
{
	if (!_model) throw std::exception("model not loaded!");

	write_model(filename.c_str(), (MODEL *)_model);
}