
#include "svm_light_util.h"
#include "shared/Macro.h"

using namespace std;
using namespace joachims;

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

svm::sparse_vector::const_iterator svm::sparse_vector::begin() const
{
	return const_iterator(((SVECTOR *)_vec)->words);
}

svm::sparse_vector::const_iterator svm::sparse_vector::end() const
{
	return const_iterator(_word_end);
}

svm::sparse_vector::const_iterator::const_iterator(void * word)
	: _word(word)
{

}

bool svm::sparse_vector::const_iterator::operator==(const const_iterator& rhs) const
{
	return _word == rhs._word;
}

svm::sparse_vector::const_iterator& svm::sparse_vector::const_iterator::operator++()
{
	auto word = (JWORD *)_word;
	_word = ++word;
	return *this;
}

float svm::sparse_vector::const_iterator::operator*() const
{
	return ((JWORD *)_word)->weight;
}

svm::sparse_vector::size_type svm::sparse_vector::const_iterator::index() const
{
	return ((JWORD *)_word)->wnum;
}

svm::sparse_vector::sparse_vector(const svm::sparse_vector& rhs)
	: _size(rhs._size)
{
	_vec = copy_svector((SVECTOR *)rhs._vec);
}

svm::sparse_vector::sparse_vector(const cv::Mat& mat)
	: _size(mat.rows * mat.cols * mat.channels())
{
	const auto channels = mat.channels();
	std::vector<JWORD> words(mat.rows * mat.cols * channels + 1);
	std::vector<JWORD>::size_type i = 0;

	for (int y = 0; y < mat.rows; y++)
	{
		auto row = mat.ptr<float>(y);

		for (int x = 0; x < mat.cols; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				if (std::fabs(row[c]) > 0)
				{
					words[i].wnum = y * mat.cols * channels + x * channels + c + 1;
					words[i].weight = row[c];
					i++;
				}
			}

			row += channels;
		}
	}
	words[i].wnum = 0;

	auto vec = create_svector(words.data(), "", 1);
	_vec = vec;

	// get the end() - iterator
	i = 0;
	for (auto iter = vec->words; iter->wnum != 0; iter++) i++;
	_word_end = &vec->words[i];
}

svm::sparse_vector::sparse_vector(svm::sparse_vector&& rhs)
	: _vec(rhs._vec), _size(rhs._size), _word_end(rhs._word_end)
{
	rhs._vec = nullptr;
}

svm::sparse_vector::~sparse_vector()
{
	free_svector((SVECTOR *)_vec);
}

float svm::sparse_vector::operator[](size_type idx) const
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

SvmLightUtil::SvmLightUtil()
{
	setVerbosity(DEFAULT_VERBOSITY);
}

SvmLightUtil::~SvmLightUtil()
{
}

SvmLightUtil::Parameters::Parameters() : 
	kernel_type(UNDEFINED_KERNEL_TYPE), 
	kernel_param(0.0),
	model_type(UNDEFINED_MODEL_TYPE),
	svm_c(0.0),
	compute_loo(0),
	biased_hyperplane(1)
{
}

void SvmLightUtil::setVerbosity(int value) const
{
	verbosity = value;
}

void SvmLightUtil::getDecisionFunctionParameters(string const& svm_model_file, int size_n, double& b, vector<SparseVector>& support_vectors, vector<double>& alphas) const
{
	MODEL const* svm_model = getModel(svm_model_file);

	getDecisionFunctionParameters(svm_model, size_n, b, support_vectors, alphas);

	deleteModel(svm_model);
}

void SvmLightUtil::getDecisionFunctionParameters(MODEL const* svm_model, int size_n, double& b, vector<SparseVector>& support_vectors, vector<double>& alphas) const
{
	b = svm_model->b;
	DOC** docs = svm_model->supvec;

	for(int ii = 1; ii < svm_model->sv_num; ++ii)
	{
		DOC* doc = docs[ii];
		JWORD* words = doc->fvec->words;
		SparseVector sv(size_n);

		for(int jj = 0; (words[jj]).wnum != 0; ++jj) 
		{ 
			int word_id = (words[jj]).wnum - 1;
			_check(word_id >= 0);
			_check(word_id < size_n);
			sv[word_id] = (words[jj]).weight;
		}

		support_vectors.push_back(sv);
		alphas.push_back(svm_model->alpha[ii]);
	}
}

JWORD* SvmLightUtil::allocateSparseVector(int size_n)
{
	return (JWORD *)my_malloc(sizeof(JWORD)*(size_n+1));
}

void SvmLightUtil::train(std::deque<svm::sparse_vector> const& pos_examples, std::deque<svm::sparse_vector> const& neg_examples, int size_n, const std::string& svm_file, double cost_ratio, Parameters const& parameters) const
{
	vector<SparseVector> examples;
	vector<double> targets;
	vector<double> costfactors;

	for (auto& vec : pos_examples)
	{
		SparseVector svec(size_n);
		for (auto i = vec.begin(); i != vec.end(); ++i)
			svec[i.index()] = *i;
		examples.push_back(std::move(svec));
	}

	for (auto& svec : neg_examples)
	{
		SparseVector svec(size_n);
		for (auto i = svec.begin(); i != svec.end(); ++i)
			svec[i.index()] = *i;
		examples.push_back(std::move(svec));
	}

	targets.insert(targets.end(), pos_examples.size(), POS_EXAMPLE);
	targets.insert(targets.end(), neg_examples.size(), NEG_EXAMPLE);
	costfactors.insert(costfactors.end(), pos_examples.size(), 1);
	costfactors.insert(costfactors.end(), neg_examples.size(), 1);

	train(examples, targets, costfactors, size_n, svm_file, cost_ratio, parameters);
}

double SvmLightUtil::classify(const svm::sparse_vector& vec, int size_n, joachims::MODEL const * svm_model) const
{
	SparseVector svec(size_n);
	for (auto i = vec.begin(); i != vec.end(); ++i)
		svec[i.index()] = *i;

	return classify(svec, size_n, svm_model);
}

MODEL const* SvmLightUtil::train(DOC **docs, long totwords, long totdoc, double *target, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	MODEL const* result;

	//char restartfile[200]; // file with initial alphas

	long i;
	double *alpha_in=NULL;
	kernel_cache *kernel_cache;
	LEARN_PARM learn_parm;
	KERNEL_PARM kernel_parm;	
	MODEL* svm_model = (MODEL*)my_malloc(sizeof(MODEL));

	initialize(&learn_parm, &kernel_parm);
	_check(parameters.kernel_type == SVM_LIGHT_LINEAR || parameters.kernel_type == SVM_LIGHT_POLY || parameters.kernel_type == SVM_LIGHT_RBF || parameters.kernel_type == SVM_LIGHT_CUSTOM);
	_check(parameters.model_type == CLASSIFICATION || parameters.model_type == REGRESSION);
	kernel_parm.kernel_type = parameters.kernel_type;
	learn_parm.type = parameters.model_type;
	learn_parm.svm_c = parameters.svm_c;
	learn_parm.compute_loo = parameters.compute_loo;
	learn_parm.svm_costratio = cost_ratio;
	_check(parameters.biased_hyperplane == 0 || parameters.biased_hyperplane == 1);
	learn_parm.biased_hyperplane = parameters.biased_hyperplane;

	//if(restartfile[0]) alpha_in=read_alphas(restartfile,totdoc);

	if(kernel_parm.kernel_type == SVM_LIGHT_LINEAR) { // don't need the cache 
		kernel_cache=NULL;
	}
	else 
	{
		// Always get a new kernel cache. It is not possible to use the same cache for two different training runs
		kernel_cache=kernel_cache_init(totdoc,learn_parm.kernel_cache_size);

		if (kernel_parm.kernel_type == SVM_LIGHT_RBF)
		{
			printf("\nnonlinear kernel not supported");
			exit(0);
		}
		else if (kernel_parm.kernel_type == SVM_LIGHT_POLY)
		{
			_error("not implemented yet");
		}
	}

	if(learn_parm.type == CLASSIFICATION) {
		svm_learn_classification(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,svm_model,alpha_in);
	}
	else if(learn_parm.type == REGRESSION) {
		svm_learn_regression(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,svm_model);
	}
	else if(learn_parm.type == RANKING) {
		svm_learn_ranking(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,&kernel_cache,svm_model);
	}
	else if(learn_parm.type == OPTIMIZATION) {
		svm_learn_optimization(docs,target,totdoc,totwords,&learn_parm,
			&kernel_parm,kernel_cache,svm_model,alpha_in);
	}

	// compute weight vector if a linear kernel was used
	if (kernel_parm.kernel_type == SVM_LIGHT_LINEAR) 
	{ 
		add_weight_vector_to_linear_model(svm_model);
	}

	if (kernel_cache) 
	{
		// Free the memory used for the cache. 
		kernel_cache_cleanup(kernel_cache);
	}

	result = copy_model(svm_model);

	free(alpha_in);
	// we must not free the model data, since the support vectors point to the 
	// documents which will be deleted already
	free_model(svm_model,0);
	for(i=0;i<totdoc;i++) 
		free_example(docs[i],1);
	free(docs);
	free(target);

	return result;
}

void SvmLightUtil::train(string const& svm_model_file, DOC **docs, long totwords, long totdoc, double *target, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	MODEL const* svm_model = train(docs, totwords, totdoc, target, cost_ratio, parameters);

	write_model(const_cast<char*>(svm_model_file.c_str()), const_cast<MODEL*>(svm_model));

	// we must free the model data, since it has no dependency to the training 
	// data any longer
	free_model(const_cast<MODEL*>(svm_model), 1);
}

void SvmLightUtil::train(string const& svm_data_file, string const& svm_model_file, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	//char restartfile[200]; // file with initial alphas

  DOC **docs; // training examples 
  long totwords,totdoc;
  double *target;

  read_documents(const_cast<char*>(svm_data_file.c_str()),&docs,&target,&totwords,&totdoc);
  //if(restartfile[0]) alpha_in=read_alphas(restartfile,totdoc);

	train(svm_model_file, docs, totwords, totdoc, target, cost_ratio, parameters);
}

void SvmLightUtil::train(vector<SparseVector> const& examples, vector<double> const& targets, int size_n, string const& svm_model_file, SvmLightUtil::Parameters const& parameters) const
{
	double cost_ratio = 1;
	vector<double> cost_factors(examples.size(), 1);

	train(examples, targets, cost_factors, size_n, svm_model_file, cost_ratio, parameters);
}

MODEL const* SvmLightUtil::train(vector<SparseVector> const& examples, vector<double> const& targets, vector<double> const& costfactors, int size_n, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	_check(examples.size() == targets.size());
	_check(costfactors.size() == targets.size());

	MODEL const* result;

	DOC** docs;
	long totwords,totdoc;
	double *target;

	int num_examples = examples.size();

	// @TODO: we manually set the highest number of words, which can differ from
	// the (true) highest feature index. this should have no effect though
	totwords = size_n; 
	totdoc = num_examples;
	if (totdoc == 0)
	{
		_warn("found no documents");
	}

	docs = (DOC **)my_malloc(sizeof(DOC *)*num_examples);   
	target = (double *)my_malloc(sizeof(double)*num_examples); 
	JWORD* words = allocateSparseVector(size_n);

	char* comment = NULL;

	for(int ii = 0; ii < num_examples; ++ii)
	{
		//_check(examples.at(ii).nnz() > 0);
		target[ii] = targets.at(ii);
		convert(examples.at(ii), size_n, words);
		docs[ii] = create_example(ii,0,0,costfactors.at(ii),create_svector(words,comment,1.0));
	}		

	result = train(docs, totwords, totdoc, target, cost_ratio, parameters);

	free(words);

	return result;
}

void SvmLightUtil::train(vector<SparseVector> const& examples, vector<double> const& targets, vector<double> const& costfactors, int size_n, string const& svm_model_file, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	MODEL const* svm_model = train(examples, targets, costfactors, size_n, cost_ratio, parameters);

	write_model(const_cast<char*>(svm_model_file.c_str()), const_cast<MODEL*>(svm_model));

	// we must free the model data, since it has no dependency to the training 
	// data any longer
	free_model(const_cast<MODEL*>(svm_model), 1);
}

MODEL const* SvmLightUtil::train(vector<SparseVector> const& examples, vector<double> const& targets, int size_n, SvmLightUtil::Parameters const& parameters) const
{
	vector<double> costfactors;
	costfactors.insert(costfactors.end(), examples.size(), 1.0);
	double cost_ratio = 1.0;

	return train(examples, targets, costfactors, size_n, cost_ratio, parameters);
}

MODEL const* SvmLightUtil::train(vector<SparseVector> const& pos_examples, vector<SparseVector> const& neg_examples, int size_n, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	MODEL const* result;
	
	vector<SparseVector> examples;
	vector<double> targets;
	vector<double> costfactors;

	examples.insert(examples.end(), pos_examples.begin(), pos_examples.end());
	examples.insert(examples.end(), neg_examples.begin(), neg_examples.end());
	targets.insert(targets.end(), pos_examples.size(), POS_EXAMPLE);
	targets.insert(targets.end(), neg_examples.size(), NEG_EXAMPLE);
	costfactors.insert(costfactors.end(), pos_examples.size(), 1);
	costfactors.insert(costfactors.end(), neg_examples.size(), 1);

	result = train(examples, targets, costfactors, size_n, cost_ratio, parameters);

	return result;
}

void SvmLightUtil::train(vector<SparseVector> const& pos_examples, vector<SparseVector> const& neg_examples, int size_n, string const& svm_model_file, double cost_ratio, SvmLightUtil::Parameters const& parameters) const
{
	MODEL const* svm_model = train(pos_examples, neg_examples, size_n, cost_ratio, parameters);

	write_model(const_cast<char*>(svm_model_file.c_str()), const_cast<MODEL*>(svm_model));

	// we must free the model data, since it has no dependency to the training 
	// data any longer
	free_model(const_cast<MODEL*>(svm_model), 1);
}

MODEL const* SvmLightUtil::train(vector<double const*> const& examples, vector<double> const& targets, int size_n, SvmLightUtil::Parameters const& parameters) const
{
  _check(examples.size() == targets.size());

  MODEL const* result;

  DOC** docs;
  long totwords,totdoc;
  double *target;

  int num_examples = examples.size();

  // @TODO: we manually set the highest number of words, which can differ from
  // the (true) highest feature index. this should have no effect though
  totwords = size_n; 
  totdoc = num_examples;
  if (totdoc == 0)
  {
    _warn("found no documents");
  }

  docs = (DOC **)my_malloc(sizeof(DOC *)*num_examples);   
  target = (double *)my_malloc(sizeof(double)*num_examples); 
  JWORD* words = allocateSparseVector(size_n);

  char* comment = NULL;

  for(int ii = 0; ii < num_examples; ++ii)
  {
    //_check(examples.at(ii).nnz() > 0);
    target[ii] = targets.at(ii);
    convert(examples.at(ii), size_n, words);
    docs[ii] = create_example(ii,0,0,1.0,create_svector(words,comment,1.0));
  }		

  result = train(docs, totwords, totdoc, target, 1.0, parameters);

  free(words);

  return result;
}

MODEL const* SvmLightUtil::getModel(string const& svm_model_file)
{
	MODEL* svm_model = read_model(const_cast<char*>(svm_model_file.c_str()));

	// compute weight vector if a linear kernel was used
	if (svm_model->kernel_parm.kernel_type == SVM_LIGHT_LINEAR) 
	{ 
		add_weight_vector_to_linear_model(svm_model);
	}

	return svm_model;
}

void SvmLightUtil::deleteModel(MODEL const* svm_model) 
{
	free_model(const_cast<MODEL*>(svm_model),1);
}

// @TODO: make this function faster by removing all the evaluation code. also,
// the argument target is not needed
void SvmLightUtil::classify(MODEL const* svm_model, DOC **docs, long totwords, long totdoc, double* distances) const
{
	_check(svm_model->totwords == totwords);

	for(int ii = 0; ii < totdoc; ++ii)
	{
		DOC* doc = docs[ii];
		JWORD* words = doc->fvec->words;

		if(svm_model->kernel_parm.kernel_type == SVM_LIGHT_LINEAR) 
		{ 
			// For linear kernel
			for(int j=0;(words[j]).wnum != 0;j++) 
			{     
				// check if feature numbers are not larger than in svm_model
				if((words[j]).wnum>svm_model->totwords)     
				{
					printf("\ninvalid feature number: j = [%d], (words[j]).wnum = [%ld]", j, (words[j]).wnum);
					exit(0);
				}
			}                                   
		}

		if(svm_model->kernel_parm.kernel_type == SVM_LIGHT_LINEAR) 
		{ 
			// linear kernel
			distances[ii]=classify_example_linear(const_cast<MODEL*>(svm_model),doc);
		}
		else 
		{
			// non-linear kernel 
			distances[ii]=classify_example(const_cast<MODEL*>(svm_model),doc);
		}
	}  

	for(int i=0;i<totdoc;i++) 
		free_example(docs[i],1);
	free(docs);
}

double const* SvmLightUtil::classify(vector<SparseVector> const& examples, int size_n, string const& svm_model_file) const
{
	MODEL const* svm_model = getModel(svm_model_file);
	double const* result = classify(examples, size_n, svm_model);
	deleteModel(svm_model);

	return result;
}

double SvmLightUtil::classify(SparseVector const& example, int size_n, MODEL const* svm_model) const
{
	DOC** docs;
	long totwords,totdoc;

	int num_examples = 1;

	totwords = size_n;
	totdoc = num_examples;

	docs = (DOC **)my_malloc(sizeof(DOC *)*num_examples);   
	JWORD* words = allocateSparseVector(size_n);
	double distance;

	char* comment = NULL;

	convert(example, size_n, words);
	docs[0] = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));

	classify(svm_model, docs, totwords, totdoc, &distance);

	free(words);

	return distance;
}

double const* SvmLightUtil::classify(vector<SparseVector> const& examples, int size_n, MODEL const* svm_model) const
{
	DOC** docs;
	long totwords,totdoc;

	int num_examples = examples.size();

	totwords = size_n;
	totdoc = num_examples;

	docs = (DOC **)my_malloc(sizeof(DOC *)*num_examples);   
	JWORD* words = allocateSparseVector(size_n);
	double* distances = new double[num_examples];

	char* comment = NULL;

	for(int ii = 0; ii < num_examples; ++ii)
	{
		convert(examples.at(ii), size_n, words);
		docs[ii] = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));
	}		

	classify(svm_model, docs, totwords, totdoc, distances);

	free(words);

	return distances;
}

double const* SvmLightUtil::classify(vector<SparseVector> const& pos_examples, vector<SparseVector> const& neg_examples, int size_n, string const& svm_model_file) const
{
	MODEL const* svm_model = getModel(svm_model_file);
	double const* result = classify(pos_examples, neg_examples, size_n, svm_model);
	deleteModel(svm_model);

	return result;
}

double const* SvmLightUtil::classify(vector<SparseVector> const& pos_examples, vector<SparseVector> const& neg_examples, int size_n, MODEL const* svm_model) const
{
	vector<SparseVector> examples;

	examples.insert(examples.end(), pos_examples.begin(), pos_examples.end());
	examples.insert(examples.end(), neg_examples.begin(), neg_examples.end());

	return classify(examples, size_n, svm_model);
}

double const* SvmLightUtil::classify(string const& svm_data_file, string const& svm_model_file) const
{
	MODEL const* svm_model = getModel(svm_model_file);
	double const* result = classify(svm_data_file, svm_model);
	deleteModel(svm_model);

	return result;
}

double const* SvmLightUtil::classify(string const& svm_data_file, MODEL const* svm_model) const
{
	DOC **docs; // examples for classification 
	long totwords,totdoc;
	double *target;

	read_documents(const_cast<char*>(svm_data_file.c_str()),&docs,&target,&totwords,&totdoc);

	double* distances = new double[totdoc];

	classify(svm_model, docs, totwords, totdoc, distances);

	free(target);

	return distances;
}

double SvmLightUtil::classify(double const* example, int size_n, MODEL const* svm_model) const
{
  DOC** docs;
  long totwords,totdoc;

  int num_examples = 1;

  totwords = size_n;
  totdoc = num_examples;

  docs = (DOC **)my_malloc(sizeof(DOC *)*num_examples);   
  JWORD* words = allocateSparseVector(size_n);
  double distance;

  char* comment = NULL;

  convert(example, size_n, words);
  docs[0] = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));

  classify(svm_model, docs, totwords, totdoc, &distance);

  free(words);

  return distance;
}

void SvmLightUtil::convert(joachims::JWORD const* words, int size, double* dense_arr, int offset)
{
	for(int ii = 0; ii < size; ++ii) 
	{
		dense_arr[ii] = 0.0;
	}

	int wpos = 0;

	while(words[wpos].wnum)
	{
		_check(words[wpos].wnum > 0);
		_check(words[wpos].wnum <= size);
		dense_arr[words[wpos].wnum - offset] = words[wpos].weight;

		wpos++;
	}
}

void SvmLightUtil::convert(joachims::JWORD const* words, int size, SparseVector& sparse_vector, int offset)
{
	_check(sparse_vector.size() == size);
	// vector must be empty
	_check(sparse_vector.nnz() == 0);

	int wpos = 0;
	
	while(words[wpos].wnum)
	{
		_check(words[wpos].wnum > 0);
		_check(words[wpos].wnum <= size);
		sparse_vector[words[wpos].wnum - offset] = words[wpos].weight;

		wpos++;
	}
}

void SvmLightUtil::convert(SparseVector const& sparse_vector, int size, JWORD* words, int offset)
{
	_check(sparse_vector.size() == size);

	int wpos = 0;

	SparseVector::const_iterator it;
	for(it = sparse_vector.begin(); it != sparse_vector.end(); ++it)
	{
		// exmaples are stored on disc in svm light format, i.e. having indices in
		// range [1,size]. however, when creating dense examples from
		// stored svm light examples, we convert the index scheme back to
		// [0,size), so here we have to switch to svm light format 
		// once again
		(words[wpos]).wnum = it.index() + offset;
		_check((words[wpos]).wnum > 0);
		_check((words[wpos]).wnum <= size);
		(words[wpos]).weight = (FVAL)*it; 

		//printf("\n[%d]:[%f]", (words[wpos]).wnum, (words[wpos]).weight);

		++wpos;
	}

	_check(wpos == sparse_vector.nnz());

	(words[wpos]).wnum = 0;
	(words[wpos]).weight = 0.0;
}

void SvmLightUtil::convert(SparseVector const& sparse_vector, int size, JWORD* words)
{
	convert(sparse_vector, size, words, 1);
}

void SvmLightUtil::convert(double const* dense_arr, int size, joachims::JWORD* words, int offset)
{
	// consecutive index
	int kk = 0;

	for(int ii = 0; ii< size; ++ii) 
	{
		if (fabs(dense_arr[ii]) > 0)
		{
			words[kk].wnum = ii + offset;
			words[kk].weight = dense_arr[ii]; 
			kk++;
		}
	}

	// terminal word
	words[kk].wnum = 0;
	words[kk].weight = 0.0;
}

void SvmLightUtil::convert(double const* dense_arr, int size, joachims::JWORD* words)
{
	convert(dense_arr, size, words, 1);
}

void SvmLightUtil::convert(double const* dense_arr, SparseVector& sparse_vector, int size)
{
	_check(sparse_vector.nnz() == 0);
	_check(sparse_vector.size() == size);

	for(int ii = 0; ii < size; ++ii)
	{
		if (fabs(dense_arr[ii]) > 0)
		{
			sparse_vector[ii] = dense_arr[ii];
		}
	}
}

void SvmLightUtil::initialize(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm) const
{
	// set default 
	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0; // remove inconsistent examples and retrain
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=100;
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->svm_c=0.0;
	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0; // leave-one-out estimates
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	kernel_parm->kernel_type=SVM_LIGHT_LINEAR;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom, "empty");
	strcpy(learn_parm->predfile, "");
	strcpy(learn_parm->alphafile, "");

	if((learn_parm->skip_final_opt_check) 
		&& (kernel_parm->kernel_type == SVM_LIGHT_LINEAR)) {
			printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
			learn_parm->skip_final_opt_check=0;
	}    
	if((learn_parm->skip_final_opt_check) 
		&& (learn_parm->remove_inconsistent)) {
			printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
			exit(0);
	}    
	if((learn_parm->svm_maxqpsize<2)) {
		printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
		exit(0);
	}
	if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
		printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
		printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
		exit(0);
	}
	if(learn_parm->svm_iter_to_shrink<1) {
		printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
		exit(0);
	}
	if(learn_parm->svm_c<0) {
		printf("\nThe C parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->transduction_posratio>1) {
		printf("\nThe fraction of unlabeled examples to classify as positives must\n");
		printf("be less than 1.0 !!!\n\n");
		exit(0);
	}
	if(learn_parm->svm_costratio<=0) {
		printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->epsilon_crit<=0) {
		printf("\nThe epsilon parameter must be greater than zero!\n\n");
		exit(0);
	}
	if(learn_parm->rho<0) {
		printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
		printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
		printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
		exit(0);
	}
	if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
		printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
		printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
		printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
		exit(0);
	}
}

void SvmLightUtil::readFromFile(string const& file, vector<SparseVector>& examples, vector<double>& targets, int size_n) const
{
	DOC **docs; // training examples 
	long totwords,totdoc;
	double *target;

	printf("\nreading from file [%s]", file.c_str());

	read_documents(const_cast<char*>(file.c_str()),&docs,&target,&totwords,&totdoc);
	if (totdoc == 0)
	{
		//_warn("found no documents");
	}
	else
	{
		_check(totwords > 0);
		_check(totwords <= size_n);
	}

	for(int ii = 0; ii < totdoc; ++ii)
	{
		SparseVector example(size_n);

		JWORD* words = docs[ii]->fvec->words;
		for(int jj = 0; (words[jj]).wnum != 0; jj++)
		{
			// convert svm light index scheme to ours, i.e. [0, size_n)
			int word_id = (words[jj]).wnum - 1;
			_check(word_id >= 0);
			_check(word_id < size_n);
			example[word_id] = (words[jj]).weight;
		}

		examples.push_back(example);
		targets.push_back(target[ii]);
	}	

	for(int ii=0;ii<totdoc;ii++) 
		free_example(docs[ii],1);
	free(docs);
	free(target);
}

void SvmLightUtil::readFromFile(string const& file, vector<SparseVector>& examples, EXAMPLE_LABEL example_label, int size_n) const
{
	vector<double> targets;
	readFromFile(file, examples, targets, size_n);
	
	_check(examples.size() == targets.size());
	for(int ii = 0; ii < examples.size(); ++ii)
	{
		_check((int)targets.at(ii) == example_label);
	}
}

void SvmLightUtil::writeToFile(std::string const& file, std::vector<SparseVector> const& examples, std::vector<double> const& targets) const
{
	ofstream output_stream(file.c_str(), ofstream::trunc); 
	_check(output_stream.good(), "failed to write data file");

	for(int ii = 0; ii < examples.size(); ++ii)
	{
		writeToStream(output_stream, examples.at(ii), targets.at(ii));
	}

	output_stream.close();
}

void SvmLightUtil::writeToFile(string const& file, vector<SparseVector> const& examples, EXAMPLE_LABEL example_label) const
{
	// each example has the same label
	vector<double> targets(examples.size(), example_label);

	writeToFile(file, examples, targets);
}

void SvmLightUtil::writeToStream(ofstream& output_stream, SparseVector const& example, double target) const
{
	//<line> .=. <target> <feature>:<value> <feature>:<value> ... <feature>:<value> # <info>
	//<target> .=. +1 | -1 | 0 | <float> 
	//<feature> .=. <integer> | "qid"
	//<value> .=. <float>
	//<info> .=. <string>

	// @TODO: does it cause any problems when saving +1 or -1 as double?
	output_stream << target;

	SparseVector::const_iterator it;
	for(it = example.begin(); it != example.end(); ++it)
	{
		// words with id [0,size_n) are saved as indices in range [1,size_n]
		// @TODO: need to convert value to float, or does double precision work as well?
		output_stream << " " << it.index() + 1 << ":" << *it;			
	}

	output_stream << " # " << endl;
}
