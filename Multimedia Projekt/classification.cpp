#include "classification.h"
#include "image.h"
#include "extraction.h"
#include "helpers.h"
#include <opencv2/highgui/highgui.hpp>	// imread
#include <utility>	// pair, move
#include <iostream>	// cout, endl
#include <ctime>	// time
#include <iterator>	// back_inserter
#include <set>
#include <svm_light/svm.h>
using namespace mmp;
using namespace mmp;

namespace
{
	cv::RNG rng = std::time(nullptr);

	typedef std::pair<double, SparseVector> hog_pair;
	bool hog_pair_comp(const hog_pair& a, const hog_pair& b)
	{
		return a.first > b.first;
	}
}

classifier::classifier(const inria_cfg& c)
	: model(nullptr), cfg(c)
{		

}

classifier::classifier(classifier&& rhs)
	: cfg(rhs.cfg), model(rhs.model), negatives(std::move(rhs.negatives)), positives(std::move(rhs.positives))
{
	rhs.model = nullptr;
}

classifier::~classifier()
{
	if (model)
		free_model((MODEL *)model, 1);
}

void * classifier::mat_to_svector(const cv::Mat& mat)
{
	const auto channels = mat.channels();
	auto words = new WORD[mat.rows * mat.cols * channels + 1];
	unsigned i = 0;
	for (int y = 0; y < mat.rows; y++)
	{
		auto row = mat.ptr<float>(y);

		for (int x = 0; x < mat.cols; x++)
		{
			for (int c = 0; c < channels; c++)
			{
				if (row[c])
				{
					words[i].wnum = y * mat.cols + x * channels + c;
					words[i].weight = row[c];
					i++;
				}
			}

			row += channels;
		}
	}
	words[i].wnum = 0;

	auto vec = create_svector(words, "", 1);
	delete[] words;

	return vec;
}

void classifier::train()
{
	std::cout << "starting training at: ";
	mmp::print_time();
	std::cout << std::endl;

	const auto negative_filenames = files_in_folder(cfg.negative_train_path());
	const auto positive_filenames = files_in_folder(cfg.normalized_positive_train_path());
	const auto vec_size = (long)UocttiHOG::hog_size(cv::Rect(0, 0, mmp::sliding_window::width, mmp::sliding_window::height));
	const auto hogs_per_negative = cfg.random_windows_per_negative_training_sample();

	long processed = 0;
	MODEL * temp_model;

	//
	// positives
	//

	const unsigned sw_hog_length = sliding_window::width / UocttiHOG::hog_cellsize;
	const unsigned sw_hog_height = sliding_window::height / UocttiHOG::hog_cellsize;
	const unsigned y_offset = cfg.normalized_positive_training_y_offset(); // should be 16
	const unsigned x_offset = cfg.normalized_positive_training_x_offset(); // should be 16

#pragma omp parallel for schedule(static)
	for (long i = 0; i < positive_filenames.size(); i++)
	{
		auto& filename = positive_filenames[i];
		UocttiHOG hog(cv::imread(filename));
		auto vec = classifier::mat_to_svector(
			hog(cv::Rect(x_offset, y_offset, sliding_window::width, sliding_window::height))
		);

#pragma omp critical
		{
			positives.push_back(vec);

#pragma omp flush(processed)
			print_progress("positives processed", ++processed, positive_filenames.size(), filename);
		}
	}

	//
	// negatives
	//

	std::cout << std::endl;
	processed = 0;

#pragma omp parallel for schedule(dynamic, 100)
	for (long i = 0; i < negative_filenames.size(); i++)
	{
		auto& filename = negative_filenames[i];
		image img(cv::imread(filename));
		auto scaled = img.scaled_images();
		std::vector<void *> hogs;
		std::set<int> windows;

		// 10 windows per negative image
		while(hogs.size() < hogs_per_negative)
		{
			auto scaled_num = rng.uniform(0, (int)scaled.size());
			auto& scaled_img = scaled[scaled_num];
			auto sliding_windows = scaled_img.sliding_windows();
			auto sw_num = rng.uniform(0, (int)sliding_windows.size());

			// only add different windows into the hogs vector
			// a window is identified by: ij where 
			// i is the index of the random scaled image and 
			// j the index of one of sliding windows of image i
			auto id = scaled_num * 10 + sw_num;
			if (windows.find(id) == windows.end())
			{
				auto fvec = classifier::mat_to_svector(sliding_windows[sw_num].features());
				hogs.push_back(fvec);
				windows.insert(id);
			}
		}

#pragma omp critical
		{
			std::move(hogs.begin(), hogs.end(), std::back_inserter(negatives));

#pragma omp flush(processed)
			print_progress("negatives processed", ++processed, negative_filenames.size(), filename);
		}
	}

	//
	// train svm
	//

	std::cout << std::endl << std::endl << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
	temp_model = (MODEL *)train(positives, negatives, vec_size);
	save_model_to_file(temp_model, cfg.svm_file());
	//free_model(temp_model, 0);	// dont delete the support vectors
	std::cout << "done" << std::endl << std::endl;

	//
	// hard mining (false positives)
	//

	processed = 0;

#pragma omp parallel for schedule(dynamic, 5)
	for (long i = 0; i < negative_filenames.size(); i++)
	{
		auto& filename = negative_filenames[i];
		image img(cv::imread(filename));
		std::vector<void *> hogs;
		//std::vector<hog_pair> hogs;

		for (auto scaled : img.scaled_images())
		{
			for (auto window : scaled.sliding_windows())
			{
				void * vec;
				double a = classify(temp_model, window.features(), &vec);
				if (a > 0)
					//hogs.emplace_back(a, std::move(vec));
					hogs.push_back(vec);
				else
					free_svector((SVECTOR *)vec);
			}
		}

		//auto max = std::max_element(hogs.begin(), hogs.end(), hog_pair_comp);
#pragma omp critical
		{
			//if (max != hogs.end())
			//	false_positives.push_back(std::move(max->second));
			std::move(hogs.begin(), hogs.end(), std::back_inserter(negatives));

#pragma omp flush(processed)
			print_progress("false positives processed", ++processed, negative_filenames.size(), filename);
		}
	}

	//
	// hard train svm
	//

	std::cout << std::endl << std::endl << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
	temp_model = (MODEL *)train(positives, negatives, vec_size);
	save_model_to_file(temp_model, cfg.svm_file_hard());
	free_model(temp_model, 0);

	for (auto& vec : positives)
		free_svector((SVECTOR *)vec);

	for (auto& vec : negatives)
		free_svector((SVECTOR *)vec);

	std::cout << "done" << std::endl << "training finished at: ";
	mmp::print_time();
	std::cout << std::endl;
}

void classifier::load(bool hard)
{
	if (model)
		free_model((MODEL *)model, 1);

	model = load_model_from_file(hard ? cfg.svm_file_hard() : cfg.svm_file());
}

double classifier::classify(void * model, const cv::Mat& mat, void ** sparse_vec)
{
	auto vec = (SVECTOR *)mat_to_svector(mat);
	auto doc = create_example(0, 0, 0, 0, vec);		// no params except vec used in classify
	auto dist = classify_example_linear((MODEL *)model, doc);

	if (sparse_vec)
	{
		// caller wants to save the created sparse_vector
		*sparse_vec = vec;
		free_example(doc, 0);	// no deep delete (keep the sparse vector)
	}
	else
		free_example(doc, 1);	// we dont want to keep the sparse vector

	return dist;
}

double classifier::classify(const cv::Mat& mat) const
{
	if (!model)
		throw std::exception("not yet loaded");

	return classify(model, mat, nullptr);
}

void * classifier::load_model_from_file(const std::string& filename)
{
	return read_model(filename.c_str());
}

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

void * classifier::train(const std::deque<void *>& positives, const std::deque<void *>& negatives, long vec_size)
{
	std::vector<DOC *> docs(positives.size() + negatives.size());

	for (std::vector<DOC *>::size_type i = 0; i < positives.size(); i++)
		docs[i] = create_example(long(i), 0, 0, 1, (SVECTOR *)positives[i]);

	for (std::vector<DOC *>::size_type i = 0; i < negatives.size(); i++)
		docs[positives.size() + i] = create_example(long(positives.size() + i), 0, 0, 1, (SVECTOR *)negatives[i]);

	std::vector<double> targets;
	targets.reserve(positives.size() + negatives.size());
	targets.insert(targets.end(), positives.size(), 1);
	targets.insert(targets.end(), negatives.size(), -1);

	LEARN_PARM learn_param;
	KERNEL_PARM kernel_param;
	params_init(&learn_param, &kernel_param);
	learn_param.type = CLASSIFICATION;
	learn_param.svm_c = 0.01;
	learn_param.compute_loo = 0;
	learn_param.svm_costratio = 0;
	learn_param.biased_hyperplane = 1;
	kernel_param.kernel_type = LINEAR;

	auto model = new MODEL;
	svm_learn_classification(docs.data(), targets.data(), (long)docs.size(), vec_size, &learn_param, &kernel_param, nullptr, model, nullptr);
	add_weight_vector_to_linear_model(model);
	auto result = copy_model(model);
	
	free_model(model, 0);

	for (auto& doc : docs)
		free_example(doc, 0);	// the sparsevectors do not belong to us

	return result;
}

void classifier::save_model_to_file(void * model, const std::string& filename)
{
	write_model(filename.c_str(), (MODEL *)model);
}