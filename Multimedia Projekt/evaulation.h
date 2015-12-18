#pragma once
#include "image.h"					// image
#include "annotation.h"				// annotation::file
#include "inria.h"					// inria_cfg
#include "classification.h"			// classifier
#include <string>
#include <vector>
#include <opencv2/core/core.hpp>	// Rect, Mat

namespace mmp
{
	class annotated_image : public image
	{
	private:
		annotation::file annotation_file;

	public:
		annotated_image(const annotation::file& annotation, cv::Mat image);

		std::vector<cv::Rect> get_objects_boxes() const;
		bool is_valid_detection(const cv::Rect& rect) const;
		void detect_all(classifier& c);
	};

	class mat_plot
	{
	private:
		void * engine;
		void * labels;
		void * scores;

		std::vector<double> labels_data;
		std::vector<double> scores_data;
		
	public:
		mat_plot(mat_plot &&rhs);
		mat_plot(const std::vector<double>& labels, const std::vector<double>& scores);
		~mat_plot();

		void save(const std::string& filename) const;
		void show(const std::string& title = "") const;
	};

	class quantitative_evaluator
	{
	private:
		classifier& svm;
		std::vector<double> labels;
		std::vector<double> scores;

	public:
		quantitative_evaluator(inria_cfg& cfg, classifier& svm);

		const std::vector<double>& get_labels() const { return labels; }
		const std::vector<double>& get_scores() const { return scores; }
	};

	class qualitative_evaluator
	{
	public:
		static void show_detections(classifier& c, annotation::file& ann, cv::Mat img, const std::string& windowname = "");
	};
}
