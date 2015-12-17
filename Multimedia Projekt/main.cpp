#include <opencv2/opencv.hpp>	// imread, waitKey, RNG
#include "inria.h"				// inria_cfg
#include "helpers.h"			// files_in_folder
#include "classification.h"		// classifier
#include "annotation.h"			// annotation::file
#include "evaulation.h"			// qualitative_evaluator, quantitative_evaluator, mat_plot
#include <ctime>				// time
#include <iostream>				// cout, endl

namespace
{
	cv::RNG rng = std::time(nullptr);
}

void qualitative_evaluation(mmp::inria_cfg& cfg, mmp::classifier& c, mmp::classifier& c_hard)
{
	auto positives = mmp::files_in_folder(cfg.test_annotation_path());
	std::cout << "close windows to stop randomly selecting a image for qualitative evaluation" << std::endl;

	do
	{
		auto filename = positives[rng.uniform(0, (int)positives.size())];

		mmp::annotation::file annotation;
		auto parse_error = mmp::annotation::file::parse(filename, annotation);
		if (!parse_error)
		{
			auto img = cv::imread(cfg.root_path() + annotation.get_image_filename());
			mmp::qualitative_evaluator::show_detections(c, annotation, img, "normal classifier");
			mmp::qualitative_evaluator::show_detections(c_hard, annotation, img, "hard classifier");
		}
		else
			std::cout << "error parsing file: " << parse_error.error_msg() << std::endl;
	} while (cv::waitKey() != -1);
}

int main()
{
	std::cout << "MMP Markus Kraus" << std::endl;
	std::cout << "starting at: "; 
	mmp::print_time();
	std::cout << std::endl << std::endl;


	mmp::inria_cfg cfg("R:/INRIAPerson/", "R:/INRIAPerson/svm.dat", "R:/INRIAPerson/svmhard.dat", "R:/INRIAPerson/evaluation.m", "R:/INRIAPerson/evaluation_hard.m");

	// train
	{
		mmp::classifier c(cfg);
		c.train();
	}

	std::cout << std::endl;

	mmp::classifier c_normal(cfg);
	mmp::classifier c_hard(cfg);
	c_normal.load();
	c_hard.load(true);

	// quantitative evaluation
	mmp::quantitative_evaluator eval(cfg, c_normal);
	mmp::mat_plot plot(eval.get_labels(), eval.get_scores());
	plot.show("evaluation");
	plot.save(cfg.evaluation_file());

	// quantitative evaluation with hard negative mined samples
	mmp::quantitative_evaluator eval_hard(cfg, c_hard);
	mmp::mat_plot plot_hard(eval_hard.get_labels(), eval_hard.get_scores());
	plot_hard.show("hard evaluation");
	plot_hard.save(cfg.evaluation_file_hard());

	// qualitative evaluation
	std::cout << std::endl << "qualitataive evaluation . . ." << std::endl;
	qualitative_evaluation(cfg, c_normal, c_hard);
	std::cout << "qualitiative evaluation terminated" << std::endl;


	std::cout << std::endl;
	std::cout << "finished at: ";
	mmp::print_time();
	std::cout << std::endl;

	std::cout << "press enter to exit" << std::endl;
	std::cin.get();
}