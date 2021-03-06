#include "evaulation.h"
#include "helpers.h"	// get_overlap, print_progress
#include "classifier.h"
#include "log.h"
#include "inria.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>	// RNG

#ifdef WITH_MATLAB
#include <engine.h>		// matlab
#include <cstring>		// memcpy
#endif

#include <algorithm>	// swap
#include <utility>		// move
#include <ctime>		// time
#include <iomanip>		// setprecision
#include <sstream>		// stringstream
#include <fstream>		// ofstream
#include <limits>		// numeric_limits

using namespace mmp;

namespace
{
	cv::RNG rng = std::time(nullptr);
}

annotated_image::annotated_image(const annotation::file& annotation, cv::Mat src)
	: image(src), annotation_file(annotation)
{

}

std::vector<cv::Rect> annotated_image::get_objects_boxes() const
{
	std::vector<cv::Rect> rects;
	for (auto& o : annotation_file.get_objects())
		rects.push_back(o.bounding_box);

	return rects;
}

bool annotated_image::is_valid_detection(const cv::Rect& rect) const
{
	for (auto& r : annotation_file.get_objects())
	{
		if (get_overlap(rect, r.bounding_box) >= 0.5f)
			return true;
	}

	return false;
}

quantitative_evaluator::quantitative_evaluator(const inria_cfg& cfg, const classifier& c)
	: svm(c)
{
	to target = log >> target;
	log << to::both << "starting evaluation at: " << time_string() << std::endl;

	const auto positive_roi = cv::Rect(
		cfg.normalized_positive_test_x_offset(), cfg.normalized_positive_test_y_offset(),
		sliding_window::width, sliding_window::height
	);
	const auto positives = files_in_folder(cfg.normalized_positive_test_path());
	//const auto positives = files_in_folder(cfg.test_annotation_path());
	const auto negatives = files_in_folder(cfg.negative_test_path());
	const auto detection_threshold = -std::numeric_limits<double>::infinity();
	unsigned long processed = 0;

	//
	// add positive detections
	//
#pragma omp parallel for schedule(static)
	for (long i = 0; i < positives.size(); i++)
	{
		hog hog(cv::imread(positives[i])(positive_roi));
		auto weight = c.classify(hog());

#pragma omp critical
		{
			labels.push_back(1);
			scores.push_back(weight);

#pragma omp flush(processed)
			print_progress("positives processed", ++processed, positives.size(), positives[i]);
		}
	}

	//
	// add negative detections
	//
	processed = 0;

#pragma omp parallel for schedule(dynamic)
	for (long i = 0; i < negatives.size(); i++)
	{
		image img(cv::imread(negatives[i]));
		img.detect_all(c, detection_threshold/*, 1.01f*/);

#pragma omp critical
		{
			for (auto& detection : img.get_detections())
			{
				labels.push_back(-1);
				scores.push_back(detection.first);
			}

#pragma omp flush(processed)
			print_progress("negatives processed", ++processed, negatives.size(), negatives[i]);
		}
	}

	log << to::both << "evaluation finished at: " << time_string() << std::endl;
	log << target;
}

mat_plot::mat_plot()
	: engine(nullptr), labels(nullptr), scores(nullptr)
{

}

mat_plot::mat_plot(const matlab_array& lbls, const matlab_array& scrs)
	: engine(nullptr), labels(nullptr), scores(nullptr), labels_data(lbls), scores_data(scrs)
{
#ifdef WITH_MATLAB
	int engine_error;
	engine = engOpenSingleUse(nullptr, nullptr, &engine_error);
	if (!engine)
		throw "could not open engine";

	// labels
	labels = mxCreateDoubleMatrix(1, labels_data.size(), mxREAL);
	std::memcpy(mxGetPr((mxArray *)labels), labels_data.data(), labels_data.size() * sizeof(double));
	if (engPutVariable((Engine *)engine, "labels", (mxArray *)labels))
	{
		engClose((Engine *)engine);
		throw "error putting labels\n";
	}

	// scores
	scores = mxCreateDoubleMatrix(1, scores_data.size(), mxREAL);
	std::memcpy(mxGetPr((mxArray *)scores), scores_data.data(), scores_data.size() * sizeof(double));
	if (engPutVariable((Engine *)engine, "scores", (mxArray *)scores))
	{
		mxDestroyArray((mxArray *)labels);
		engClose((Engine *)engine);
		throw "error putting scores\n";
	}
#endif
}

mat_plot::mat_plot(mat_plot&& rhs)
	: engine(rhs.engine), labels(rhs.labels), scores(rhs.scores), labels_data(std::move(rhs.labels_data)), scores_data(std::move(rhs.scores_data))
{
	rhs.engine = rhs.labels = rhs.scores = nullptr;
}

mat_plot& mat_plot::operator=(mat_plot&& rhs)
{
	this->~mat_plot();

	labels_data = std::move(rhs.labels_data);
	scores_data = std::move(rhs.scores_data);
	engine = rhs.engine;
	labels = rhs.labels;
	scores = rhs.scores;

	rhs.labels = rhs.scores = rhs.engine = nullptr;
	return *this;
}

mat_plot::~mat_plot()
{
#ifdef WITH_MATLAB
	if (labels)
		mxDestroyArray((mxArray *)labels);

	if (scores)
		mxDestroyArray((mxArray *)scores);

	if (engine)
		engClose((Engine *)engine);
#endif
}

void mat_plot::show(const std::string& title) const
{	
#ifdef WITH_MATLAB	
	// vl_det(labels, scores);
	if (engEvalString((Engine *)engine, "vl_det(labels, scores);"))
		throw "could not plot DET";

	engEvalString((Engine *)engine, "axis([10^-6 10^-1 0.01 0.5]);");

	if (!title.empty())
		engEvalString((Engine *)engine, ("title('" + title + "');").c_str());
#endif
}

void mat_plot::save(const std::string& filename) const
{
	std::ofstream skript_file(filename);
	std::ofstream label_file(filename + "_labels");
	std::ofstream score_file(filename + "_scores");

	// labels
	for (auto i = labels_data.begin(); i != labels_data.end();)
	{
		label_file << *i++;
		if (i != labels_data.end())
			label_file << ",";
	}

	// scores
	for (auto i = scores_data.begin(); i != scores_data.end();)
	{
		score_file << *i++;
		if (i != scores_data.end())
			score_file << ",";
	}

	// plot
	skript_file << "labels = dlmread('" << filename << "_labels');" << std::endl;
	skript_file << "scores = dlmread('" << filename << "_scores');" << std::endl;
	skript_file << "vl_det(labels, scores);" << std::endl;
	skript_file << "axis([10^-6 10^-1 0.01 0.5]);" << std::endl;
}

qualitative_evaluator::qualitative_evaluator(const mmp::inria_cfg& cfg, const classifier& c, const classifier& c_hard)
{
	auto positives = mmp::files_in_folder(cfg.test_annotation_path());
	to target = log >> target;
	log << to::console << "close windows to stop randomly selecting a image for qualitative evaluation" << std::endl;

	do
	{
		auto filename = positives[rng.uniform(0, (int)positives.size())];
		//std::string filename("C:/mmp/INRIAPerson/Test/annotations/person_132.txt");

		mmp::annotation::file annotation;
		auto parse_error = mmp::annotation::file::parse(filename, annotation);
		if (!parse_error)
		{
			auto img_path = cfg.root_path() + "/" + annotation.get_image_filename();
			if (path_exists(img_path))
			{
				auto img = cv::imread(img_path);
				mmp::qualitative_evaluator::show_detections(c, annotation, img.clone(), "normal classifier");
				mmp::qualitative_evaluator::show_detections(c_hard, annotation, img.clone(), "hard classifier");
			}
			else
				log << to::both << "error loading image: " << img_path << " (file does not exist)" << std::endl;
		}
		else
			log << to::both << "error parsing file: " << parse_error.error_msg() << std::endl;
	} while (cv::waitKey() != -1);
	log << target;
}

void qualitative_evaluator::show_detections(const classifier& c, annotation::file& ann, cv::Mat src, const std::string& windowname)
{
	auto img = mmp::annotated_image(ann, src);
	img.detect_all(c);
	img.suppress_non_maximum();

	for (auto& d : img.get_detections())
	{
		// determine maximum overlap with the ground truth boxes
		auto detection_window = d.second->rect();
		float max_overlap = 0;
		for (auto& g : img.get_objects_boxes())
		{
			auto temp = mmp::get_overlap(detection_window, g);

			if (temp > max_overlap)
				max_overlap = temp;
		}

		// generate distance and overlap strings
		std::stringstream ss;
		ss << "dist (" << std::setprecision(2) << d.first << ")";
		std::string dist_str = ss.str();
		ss.str(std::string());
		ss << "overlap(" << std::setprecision(2) << max_overlap << ")";
		std::string overlap_str = ss.str();

		// for readability put the strings on a monochromatic rectangle
		auto box_size = cv::getTextSize(dist_str.length() > overlap_str.length() ? dist_str : overlap_str, cv::FONT_HERSHEY_PLAIN, 0.7, 1, nullptr);
		cv::rectangle(src, cv::Rect(detection_window.x, detection_window.y, box_size.width + 2, 20), cv::Scalar(255, 255, 255), -1);
		
		cv::Scalar color;
		if (img.is_valid_detection(d.second->rect()))
			cv::rectangle(src, detection_window, color = cv::Scalar(0, 255, 0));
		else
			cv::rectangle(src, detection_window, color = cv::Scalar(255, 0, 0));
		
		cv::putText(src, dist_str, cv::Point(detection_window.x + 3, detection_window.y + 9), cv::FONT_HERSHEY_PLAIN, 0.7, color);
		cv::putText(src, overlap_str, cv::Point(detection_window.x + 3, detection_window.y + 18), cv::FONT_HERSHEY_PLAIN, 0.7, color);
	}

	for (auto& g : img.get_objects_boxes())
		cv::rectangle(src, g, cv::Scalar(0, 0, 255));

	cv::imshow(windowname.empty() ? ann.get_image_filename() : windowname, src);
}