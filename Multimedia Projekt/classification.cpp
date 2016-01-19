#include "classification.h"
#include "image.h"
#include "hog.h"
#include "helpers.h"
#include "log.h"
#include "inria.h"
#include <boost/bind.hpp>
#include <opencv2/highgui/highgui.hpp>	// imread
#include <utility>		// pair, move
#include <ctime>		// time
#include <iterator>		// back_inserter, advance, make_move_iterator
#include <set>
using namespace mmp;

namespace
{
	cv::RNG rng = std::time(nullptr);
}

classifier::classifier(const inria_cfg& c)
	: cfg(c)
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
	log << to::both << "starting training at: " << time_string() << std::endl;	
	unsigned long processed = 0;

	//
	// positives
	//
	const auto positive_filenames = files_in_folder(cfg.normalized_positive_train_path());
	const cv::Rect positive_roi(
		cfg.normalized_positive_training_x_offset(), cfg.normalized_positive_training_y_offset(), // both should be 16
		sliding_window::width, sliding_window::height
	);

#pragma omp parallel for schedule(static)
	for (long i = 0; i < positive_filenames.size(); i++)
	{
		hog hog(cv::imread(positive_filenames[i])(positive_roi));
		svm::sparse_vector svec(features_to_svector(hog()));

#pragma omp critical
		{
			positives.push_back(std::move(svec));

#pragma omp flush(processed)
			print_progress("positives processed", ++processed, positive_filenames.size(), positive_filenames[i]);
		}
	}
	
	//
	// negatives
	//
	processed = 0;
	const auto negative_filenames = files_in_folder(cfg.negative_train_path());
	const auto hogs_per_negative = cfg.random_windows_per_negative_training_sample();

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
				svm::sparse_vector fvec(features_to_svector(sliding_windows[sw_num].features()));
				hogs.push_back(std::move(fvec));
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
	const auto vec_size = (svm::sparse_vector::size_type)hog::hog_size(cv::Rect(0, 0, mmp::sliding_window::width, mmp::sliding_window::height));
	log << to::both << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
	model = new svm::linear_model(positives, negatives, vec_size, cfg.svm_c());
	model->save(cfg.svm_file());
	log << "done" << std::endl << std::endl;		
	
	//
	// hard mining (false positives)
	//
	processed = 0;
	typedef std::pair<double, svm::sparse_vector> weighted_svec;
	auto det_comp = [](const weighted_svec& a, const weighted_svec& b)
	{
		return a.first > b.first;
	};
	
	std::set<weighted_svec, bool(*)(const weighted_svec& a, const weighted_svec& b)> detections(det_comp);

#pragma omp parallel for schedule(dynamic, 10)
	for (long i = 0; i < negative_filenames.size(); i++)
	{
		auto& filename = negative_filenames[i];
		image img(cv::imread(filename));
		img.detect_all(*this);
		img.suppress_non_maximum();

		std::vector<weighted_svec> svecs;
		svecs.reserve(img.get_detections().size());
		for (auto& detection : img.get_detections())
			svecs.emplace_back(detection.first, features_to_svector(detection.second->features()));

#pragma omp critical
		{
			detections.insert(std::make_move_iterator(svecs.begin()), std::make_move_iterator(svecs.end()));

			// saves RAM
			if (detections.size() > cfg.num_hard_false_positive_retrain())
			{
				auto begin = detections.begin();
				std::advance(begin, cfg.num_hard_false_positive_retrain());
				detections.erase(begin, detections.end());
			}

#pragma omp flush(processed)
			print_progress("false positives processed", ++processed, negative_filenames.size(), filename);
		}
	}

	for (auto& detection : detections)
		negatives.push_back(std::move(const_cast<weighted_svec&>(detection).second));

	//
	// hard train svm
	//
	log << to::both << "training svm with " << positives.size() << " positives and " << negatives.size() << " negatives ... ";
	delete model;
	model = new svm::linear_model(positives, negatives, vec_size, cfg.svm_c());
	model->save(cfg.svm_file_hard());	
	log << "done" << std::endl << "training finished at: " << time_string() << std::endl;
}

double classifier::classify(const cv::Mat& mat) const
{
	if (!model) throw std::exception("classifier not loaded or trained");
	return model->classify(features_to_svector(mat));
}

void classifier::load(bool hard)
{
	model = new svm::linear_model(hard ? cfg.svm_file_hard() : cfg.svm_file());	
}

svm::sparse_vector classifier::features_to_svector(const cv::Mat& mat)
{
	assert(mat.channels() == hog::dimensions);
	struct mat_iter
	{
		cv::MatConstIterator_<hog::vector_type> iter;
		int c;

		mat_iter(const cv::MatConstIterator_<hog::vector_type>& _iter) : c(0), iter(_iter) { }

		float operator*() const { return (*iter)[c]; }
		void operator++() 
		{
			if (++c % hog::dimensions == 0) 
			{ 
				++iter; 
				c = 0; 
			}
		}

		bool operator!=(const mat_iter& rhs) { return iter != rhs.iter; }
	};

	return svm::sparse_vector(
		mat_iter(mat.begin<hog::vector_type>()), 
		mat_iter(mat.end<hog::vector_type>()),
		mat.rows * mat.cols * mat.channels()
	);
}