#include "classification.h"
#include "image.h"
#include "extraction.h"
#include "helpers.h"
#include <opencv2/highgui/highgui.hpp>	// imread
#include <utility>		// pair, move
#include <iostream>		// cout, endl
#include <ctime>		// time
#include <iterator>		// back_inserter
#include <set>
#include <fstream>
using namespace mmp;

namespace
{
	cv::RNG rng = std::time(nullptr);

	typedef std::pair<double, svm::sparse_vector> hog_pair;
	bool hog_pair_comp(const hog_pair& a, const hog_pair& b)
	{
		return a.first > b.first;
	}
}

classifier::classifier(const inria_cfg& c)
	: cfg(c), model("R:/INRIAPerson/svm.dat")
{		

}

classifier::classifier(classifier&& rhs)
	: cfg(rhs.cfg), model(std::move(rhs.model)), negatives(std::move(rhs.negatives)), positives(std::move(rhs.positives))
{

}

classifier::~classifier()
{

}

void classifier::train()
{
	std::cout << "starting training at: " << time_string() << std::endl;

	const auto negative_filenames = files_in_folder(cfg.negative_train_path());
	const auto positive_filenames = files_in_folder(cfg.normalized_positive_train_path());
	const auto vec_size = (long)hog::hog_size(cv::Rect(0, 0, mmp::sliding_window::width, mmp::sliding_window::height));
	const auto hogs_per_negative = cfg.random_windows_per_negative_training_sample();
	long processed = 0;
	std::ofstream training_file;

	if (cfg.debug())
		training_file.open(cfg.training_file(), std::ios::app);
#if 0
	//
	// positives
	//
	
	const unsigned sw_hog_length = sliding_window::width / hog::hog_cellsize;
	const unsigned sw_hog_height = sliding_window::height / hog::hog_cellsize;
	const unsigned y_offset = cfg.normalized_positive_training_y_offset(); // should be 16
	const unsigned x_offset = cfg.normalized_positive_training_x_offset(); // should be 16

#pragma omp parallel for schedule(static)
	for (long i = 0; i < positive_filenames.size(); i++)
	{
		auto& filename = positive_filenames[i];
		hog hog(cv::imread(filename));
		svm::sparse_vector svec(
			hog(cv::Rect(x_offset, y_offset, sliding_window::width, sliding_window::height))
		);

#pragma omp critical
		{
			if (cfg.debug())
				training_file << "+1 " << svm::to_string(svec) << std::endl;

			positives.push_back(std::move(svec));

#pragma omp flush(processed)
			print_progress("positives processed", ++processed, positive_filenames.size(), filename);
		}
	}
	
	//
	// negatives
	//

	std::cout << std::endl;
	processed = 0;

#pragma omp parallel for schedule(dynamic, 50)
	for (long i = 0; i < negative_filenames.size(); i++)
	{
		auto& filename = negative_filenames[i];
		image img(cv::imread(filename));
		auto scaled = img.scaled_images();
		std::vector<svm::sparse_vector> hogs;
		std::set<int> windows;

		// 10 windows per negative image
		while(hogs.size() < hogs_per_negative)
		{
			auto scaled_num = rng.uniform(0, (int)scaled.size());
			auto& scaled_img = scaled[scaled_num];
			auto sliding_windows = scaled_img.sliding_windows();
			auto sw_num = rng.uniform(0, (int)sliding_windows.size());

			// only add distinct windows into the hogs vector
			// a window is identified by: ij where 
			// i is the index of the random scaled image and 
			// j the index of one of sliding windows of image i
			auto id = scaled_num * 10 + sw_num;
			if (windows.find(id) == windows.end())
			{
				svm::sparse_vector fvec(sliding_windows[sw_num].features());
				hogs.push_back(std::move(fvec));
				windows.insert(id);
			}
		}

#pragma omp critical
		{
			for (auto& svec : hogs)
				training_file << "-1 " << svm::to_string(svec) << std::endl;

			std::move(hogs.begin(), hogs.end(), std::back_inserter(negatives));

#pragma omp flush(processed)
			print_progress("negatives processed", ++processed, negative_filenames.size(), filename);
		}
	}

	//
	// train svm
	//

	svm::model normal_model(negatives, positives, vec_size);
	{
		std::cout << std::endl << std::endl << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
		normal_model.learn();
		std::cout << "done" << std::endl << std::endl;

		normal_model.save(cfg.svm_file());
	}
#endif
	//
	// hard mining (false positives)
	//

	processed = 0;

#pragma omp parallel for schedule(dynamic, 10)
	for (long i = 0; i < negative_filenames.size(); i++)
	{
		auto& filename = negative_filenames[i];
		image img(cv::imread(filename));

		std::vector<svm::sparse_vector> hogs;
		//std::vector<hog_pair> hogs;

		for (auto scaled : img.scaled_images())
		{
			for (auto window : scaled.sliding_windows())
			{
				svm::sparse_vector vec(window.features());
				double a = model.classify(vec);	// classify takes very long...
				if (a > 0)
					//hogs.emplace_back(a, std::move(vec));
					hogs.push_back(std::move(vec));
			}
		}

		//auto max = std::max_element(hogs.begin(), hogs.end(), hog_pair_comp);
#pragma omp critical
		{
			if (cfg.debug())
			{
				for (auto& hog : hogs)
					training_file << "-1 " << svm::to_string(hog) << std::endl;
			}

			/*if (max != hogs.end())
			{
				if (cfg.debug())
					training_file << "-1 " << svm::to_string(max->second) << std::endl;

				negatives.push_back(std::move(max->second));
			}*/
			//std::move(hogs.begin(), hogs.end(), std::back_inserter(negatives));

#pragma omp flush(processed)
			print_progress("false positives processed", ++processed, negative_filenames.size(), filename);
		}
	}

	//
	// hard train svm
	//

	//{
	//	svm::model hard_model(positives, negatives, vec_size);
	//	std::cout << std::endl << std::endl << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
	//	hard_model.learn();
	//	std::cout << "done" << std::endl << "training finished at: " << time_string() << std::endl;

	//	hard_model.save(cfg.svm_file_hard());
	//}

	training_file.close();
}

double classifier::classify(const cv::Mat& mat) const
{
	svm::sparse_vector svec(mat);
	return model.classify(svec);
}

void classifier::load(bool hard)
{
	model = svm::model(hard ? cfg.svm_file_hard() : cfg.svm_file());
}