#include "evaulation.h"
#include "helpers.h"	// get_overlap, print_progress
#include "classification.h"
#include "log.h"

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
	log << to::both << "starting evaluation at: " << time_string() << std::endl;

	const auto positive_roi = cv::Rect(
		cfg.normalized_positive_test_x_offset(), cfg.normalized_positive_test_y_offset(),
		sliding_window::width, sliding_window::height
	);
	const auto positives = files_in_folder(cfg.normalized_positive_test_path());
	const auto negatives = files_in_folder(cfg.negative_test_path());
	const auto positives_train = files_in_folder(cfg.normalized_positive_train_path());
	const auto negatives_train = files_in_folder(cfg.negative_train_path());
	unsigned long processed = 0;

	//
	// add positive detections
	//

#pragma omp parallel for schedule(static)
	for (long i = 0; i < positives.size(); i++)
	{
		image img(cv::imread(positives[i])(positive_roi));

		double a = 0;
		for (auto s : img.scaled_images())
		{
			for (auto sw : s.sliding_windows())
			{
				double a = c.classify(sw.features());
				if (a > 0)
					img.add_detection(sw.window(), a);
			}
		}

		img.suppress_non_maximum();

#pragma omp critical
		{
			for (auto detection : img.get_detections())
			{
				labels.push_back(1);
				scores.push_back(detection.second);
			}

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

		for (auto s : img.scaled_images())
		{
			for (auto sw : s.sliding_windows())
			{
				double a = c.classify(sw.features());
				if (a > 0)
					img.add_detection(sw.window(), a);
			}
		}

		img.suppress_non_maximum();

#pragma omp critical
		{
			for (auto detection : img.get_detections())
			{
				labels.push_back(-1);
				scores.push_back(detection.second);
			}

#pragma omp flush(processed)
			print_progress("negatives processed", ++processed, negatives.size(), negatives[i]);
		}
	}

	log << to::both << "evaluation finished at: " << time_string() << std::endl;
}

mat_plot::mat_plot()
	: engine(nullptr), labels(nullptr), scores(nullptr)
{

}

mat_plot::mat_plot(const std::vector<double>& lbls, const std::vector<double>& scrs)
	: engine(nullptr), labels(nullptr), scores(nullptr), labels_data(lbls), scores_data(scrs)
{
#ifdef WITH_MATLAB
	int engine_error;
	engine = engOpenSingleUse(nullptr, nullptr, &engine_error);
	if (!engine)
		throw std::exception("could not open engine");

	// labels
	labels = mxCreateDoubleMatrix(1, labels_data.size(), mxREAL);
	std::memcpy(mxGetPr((mxArray *)labels), labels_data.data(), labels_data.size() * sizeof(double));
	if (engPutVariable((Engine *)engine, "labels", (mxArray *)labels))
	{
		engClose((Engine *)engine);
		throw std::exception("error putting labels\n");
	}

	// scores
	scores = mxCreateDoubleMatrix(1, scores_data.size(), mxREAL);
	std::memcpy(mxGetPr((mxArray *)scores), scores_data.data(), scores_data.size() * sizeof(double));
	if (engPutVariable((Engine *)engine, "scores", (mxArray *)scores))
	{
		mxDestroyArray((mxArray *)labels);
		engClose((Engine *)engine);
		throw std::exception("error putting scores\n");
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
	labels = scores = engine = nullptr;
	std::swap(engine, rhs.engine);
	std::swap(labels, rhs.labels);
	std::swap(scores, rhs.scores);
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
		throw std::exception("could not plot DET");

	engEvalString((Engine *)engine, "axis([10^-6 10^-1 0.01 0.5]);");

	if (!title.empty())
		engEvalString((Engine *)engine, ("title('" + title + "');").c_str());
#endif
}

void mat_plot::save(const std::string& filename) const
{
	std::ofstream plot_file(filename);
	plot_file << "labels = [";
	for (auto& label : labels_data) plot_file << label << ", ";
	plot_file << "];" << std::endl;

	plot_file << "scores = [";
	for (auto& score : scores_data) plot_file << score << ", ";
	plot_file << "];" << std::endl;

	plot_file << "vl_det(labels, scores);" << std::endl;
	plot_file << "axis([10^-6 10^-1 0.01 0.5]);" << std::endl;
	plot_file.close();
}

qualitative_evaluator::qualitative_evaluator(const mmp::inria_cfg& cfg, const classifier& c, const classifier& c_hard)
{
	auto positives = mmp::files_in_folder(cfg.test_annotation_path());
	log << to::console << "close windows to stop randomly selecting a image for qualitative evaluation" << std::endl;

	do
	{
		auto filename = positives[rng.uniform(0, (int)positives.size())];

		mmp::annotation::file annotation;
		auto parse_error = mmp::annotation::file::parse(filename, annotation);
		if (!parse_error)
		{
			auto img = cv::imread(cfg.root_path() + annotation.get_image_filename());
			mmp::qualitative_evaluator::show_detections(c, annotation, img.clone(), "normal classifier");
			mmp::qualitative_evaluator::show_detections(c_hard, annotation, img.clone(), "hard classifier");
		}
		else
			log << to::both << "error parsing file: " << parse_error.error_msg() << std::endl;
	} while (cv::waitKey() != -1);
}

void qualitative_evaluator::show_detections(const classifier& c, annotation::file& ann, cv::Mat src, const std::string& windowname)
{
	auto img = mmp::annotated_image(ann, src);
	img.detect_all(c);
	img.suppress_non_maximum();

	for (auto& d : img.get_detections())
	{
		cv::rectangle(src, cv::Rect(d.first.x, d.first.y, 82, 18), cv::Scalar(255, 255, 255), -1);

		cv::Scalar color;
		if (img.is_valid_detection(d.first))
			cv::rectangle(src, d.first, color = cv::Scalar(0, 255, 0));
		else
			cv::rectangle(src, d.first, color = cv::Scalar(255, 0, 0));

		float max_overlap = 0;
		for (auto& g : img.get_objects_boxes())
		{
			auto temp = mmp::get_overlap(d.first, g);

			if (temp > max_overlap)
				max_overlap = temp;
		}

		std::stringstream ss;
		ss << "dist (" << std::setprecision(2) << d.second << ")";
		cv::putText(src, ss.str(), cv::Point(d.first.x + 3, d.first.y + 9), cv::FONT_HERSHEY_PLAIN, 0.7, color);

		ss.str(std::string());
		ss << "overlap(" << std::setprecision(2) << max_overlap << ")";
		cv::putText(src, ss.str(), cv::Point(d.first.x + 3, d.first.y + 18), cv::FONT_HERSHEY_PLAIN, 0.7, color);
	}

	for (auto& g : img.get_objects_boxes())
		cv::rectangle(src, g, cv::Scalar(0, 0, 255));

	cv::imshow(windowname.empty() ? ann.get_image_filename() : windowname, src);
}