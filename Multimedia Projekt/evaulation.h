#pragma once
#include "image.h"					// image
#include "annotation.h"				// annotation::file
#include <opencv2/core/core.hpp>	// Rect, Mat
#include <string>
#include <vector>

namespace mmp
{
	class classifier;
	class inria_cfg;

	class annotated_image : public image
	{
	private:
		annotation::file annotation_file;

	public:
		annotated_image(const annotation::file& annotation, cv::Mat image);

		std::vector<cv::Rect> get_objects_boxes() const;
		bool is_valid_detection(const cv::Rect& rect) const;
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
		mat_plot();
		mat_plot(mat_plot &&rhs);
		mat_plot& operator=(mat_plot&& rhs);
		mat_plot(const std::vector<double>& labels, const std::vector<double>& scores);
		~mat_plot();

		void save(const std::string& filename) const;
		void show(const std::string& title = "") const;
	};

	class quantitative_evaluator
	{
	private:
		const classifier& svm;
		std::vector<double> labels;
		std::vector<double> scores;

	public:
		quantitative_evaluator(const inria_cfg& cfg, const classifier& svm);

		std::vector<double> get_labels() const { return labels; }
		std::vector<double> get_scores() const { return scores; }
	};

	class qualitative_evaluator
	{
	public:
		qualitative_evaluator(const mmp::inria_cfg& cfg, const classifier& c_normal, const classifier& c_hard);

		static void show_detections(const classifier& c, annotation::file& ann, cv::Mat img, const std::string& windowname = "");
	};
}
