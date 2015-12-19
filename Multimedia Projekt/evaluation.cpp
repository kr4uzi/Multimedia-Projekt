#include "evaulation.h"
#include "helpers.h"	// get_overlap, print_progress

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#ifdef WITH_MATLAB
#include <engine.h>		// matlab
#include <cstring>		// memcpy
#endif

#include <utility>		// move
#include <ctime>		// time
#include <iostream>		// cout, endl
#include <iomanip>		// setprecision
#include <sstream>		// stringstream
#include <fstream>		// ofstream

using namespace mmp;

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

void annotated_image::detect_all(classifier& c)
{
	for (auto s : scaled_images())
	{
		for (auto sw : s.sliding_windows())
		{
			double a = c.classify(sw.features());
			if (a > 0)
				add_detection(sw.window(), a);
		}
	}
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

void qualitative_evaluator::show_detections(classifier& c, annotation::file& ann, cv::Mat src, const std::string& windowname)
{
	auto img = mmp::annotated_image(ann, src);
	img.detect_all(c);
	img.suppress_non_maximum();

	for (auto& d : img.get_detections())
	{
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
		cv::putText(src, ss.str(), cv::Point(d.first.x + 6, d.first.y + 12), cv::FONT_HERSHEY_PLAIN, 1, color);

		ss.str(std::string());
		ss << "overlap(" << std::setprecision(2) << max_overlap << ")";
		cv::putText(src, ss.str(), cv::Point(d.first.x + 6, d.first.y + 24), cv::FONT_HERSHEY_PLAIN, 1, color);
	}

	for (auto& g : img.get_objects_boxes())
		cv::rectangle(src, g, cv::Scalar(0, 0, 255));

	cv::imshow(windowname.empty() ? ann.get_image_filename() : windowname, src);
}

quantitative_evaluator::quantitative_evaluator(inria_cfg& cfg, classifier& c)
	: svm(c)
{
	std::cout << "starting evaluation at: " << time_string() << std::endl;

	const auto positives = files_in_folder(cfg.normalized_positive_test_path());
	const auto negatives = files_in_folder(cfg.negative_test_path());
	const auto positives_train = files_in_folder(cfg.normalized_positive_train_path());
	const auto negatives_train = files_in_folder(cfg.negative_train_path());
	
	//
	// add positive detections (testing)
	//
	unsigned long processed = 0;

#pragma omp parallel for schedule(static)
	for (long i = 0; i < positives.size(); i++)
	{
		image img(cv::imread(positives[i]));

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

	std::cout << std::endl;
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

	std::cout << std::endl << "evaluation finished at: " << time_string() << std::endl;
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