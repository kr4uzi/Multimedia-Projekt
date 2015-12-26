#include "svm.h"
extern "C" {
#include "svm_common.h"
#include "svm_learn.h"
}
#include <cstring>		// strcpy
#include <algorithm>	// swap
using namespace svm;

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
		strncpy(learn_parm->alphafile, "", sizeof(LEARN_PARM::alphafile));

		kernel_parm->kernel_type = 0;
		kernel_parm->poly_degree = 3;
		kernel_parm->rbf_gamma = 1.0;
		kernel_parm->coef_lin = 1;
		kernel_parm->coef_const = 1;
		std::strncpy(kernel_parm->custom, "empty", sizeof(KERNEL_PARM::custom));
	}
}

void sparse_vector::const_iterator::operator++()
{
	auto word = (WORD *)_ptr;
	_ptr = ++word;
}

float sparse_vector::const_iterator::operator*() const
{
	// even though FVAL might be double a conversation to float
	// does not result in information loss because sparse_vector
	// doesn't fill it with doubles in the first place
	return (float)((WORD *)_ptr)->weight;
}

sparse_vector::size_type sparse_vector::const_iterator::index() const
{
	return ((WORD *)_ptr)->wnum;
}

void sparse_vector::svector_init(void ** svector, const std::vector<sparse_vector::word>& words, void ** words_end)
{
	std::vector<WORD> _words;
	_words.reserve(words.size() + 1);
	for (auto& word : words)
		_words.push_back(WORD{ word.first, word.second });
	_words.push_back(WORD{ 0 });

	auto vec = create_svector(_words.data(), "", 1);
	*svector = vec;

	WORD * end = vec->words;
	while (end->wnum)
		++end;
	*words_end = end;
}

sparse_vector& sparse_vector::operator=(sparse_vector&& rhs)
{
	_size = rhs._size;
	_svector = rhs._svector;
	_words_end = rhs._words_end;
	rhs._svector = rhs._words_end = nullptr;
	return *this;
}

sparse_vector::sparse_vector(const sparse_vector& rhs)
	: _size(rhs._size)
{
	auto svector = copy_svector((SVECTOR *)rhs._svector);
	WORD * end = svector->words;
	while (end->wnum)
		++end;

	_svector = svector;
	_words_end = end;
}

sparse_vector::sparse_vector(sparse_vector&& rhs)
	: _svector(rhs._svector), _words_end(rhs._words_end), _size(rhs._size)
{
	rhs._svector = nullptr;
	rhs._words_end = nullptr;
}

sparse_vector::const_iterator sparse_vector::begin() const
{
	return const_iterator(((SVECTOR *)_svector)->words);
}

sparse_vector::~sparse_vector()
{
	free_svector((SVECTOR *)_svector);
}

std::string svm::to_string(const sparse_vector& svec)
{
	std::string str;
	for (auto i = svec.begin(); i != svec.end(); ++i)
	{
		str += std::to_string(i.index());
		str += ':';
		str += std::to_string(*i);
		str += ' ';
	}

	return str;
}

void * svm::linear_model::create_doc(sparse_vector::size_type index, const sparse_vector& svec, double costfactor) const
{
	return create_example(index, 0, 0, costfactor, const_cast<SVECTOR *>(static_cast<const SVECTOR *>(svec.c_ptr())));
}

svm::linear_model::linear_model(const std::string& filename)
{
	_model = read_model(const_cast<char *>(filename.c_str()));
	add_weight_vector_to_linear_model((MODEL *)_model);
}

void svm::linear_model::model_init(void ** model, std::vector<void *>& docs, std::vector<double>& targets, sparse_vector::size_type vec_size, double c)
{
	LEARN_PARM learn_param;
	KERNEL_PARM kernel_param;
	params_init(&learn_param, &kernel_param);
	learn_param.type = CLASSIFICATION;
	learn_param.svm_c = c;
	learn_param.svm_iter_to_shrink = 2;
	learn_param.skip_final_opt_check = 0;
	kernel_param.kernel_type = LINEAR;

	MODEL * mod = (MODEL *)malloc(sizeof(MODEL));
	svm_learn_classification((DOC **)docs.data(), targets.data(), (long)docs.size(), vec_size, &learn_param, &kernel_param, nullptr, mod, nullptr);
	add_weight_vector_to_linear_model(mod);
	// copy the model so we have our own copy of the support vectors
	*model = copy_model(mod);

	free_model(mod, 0);
	for (auto& doc : docs)
		free_example((DOC *)doc, 0); // we are not owning the svectors
}

svm::linear_model::~linear_model()
{
	// we always are owner of the svectors
	free_model((MODEL *)_model, 1);
}

void svm::linear_model::save(const std::string& filename) const
{
	write_model(const_cast<char *>(filename.c_str()), (MODEL *)_model);
}

double svm::linear_model::classify(const sparse_vector& svec) const
{
	auto doc = create_doc(-1, svec, 0);
	double value = classify_example_linear((MODEL *)_model, (DOC *)doc);
	free_example((DOC *)doc, 0);
	return value;
}