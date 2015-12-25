#include <opencv2/opencv.hpp>	// imread, waitKey
#include "inria.h"				// inria_cfg
#include "helpers.h"			// files_in_folder
#include "classification.h"		// classifier
#include "annotation.h"			// annotation::file
#include "evaulation.h"			// qualitative_evaluator, quantitative_evaluator, mat_plot
#include <ctime>				// time
#include <cstdlib>				// system
#include <iostream>				// cout, endl
#include <thread>
#include <condition_variable>

int main()
{
	std::cout << "MMP Markus Kraus" << std::endl;
	std::cout << "started at: " << mmp::time_string() << std::endl << std::endl;

	mmp::inria_cfg cfg(
		"R:/INRIAPerson/", 
		"R:/INRIAPerson/svm.dat", "R:/INRIAPerson/svm_hard.dat",
		"R:/INRIAPerson/evaluation.m", "R:/INRIAPerson/evaluation_hard.m",
		0.01
	);

	// train
	{
		mmp::classifier c(cfg);
		c.train();
	}

	std::cout << std::endl;

	mmp::classifier c_normal(cfg);
	mmp::classifier c_hard(cfg);

	std::cout << "loading svm files ..." << std::endl;

	std::thread thn([&c_normal]() { c_normal.load(); });
	std::thread thh([&c_hard]() { c_hard.load(true); });

	thn.join();
	std::cout << cfg.svm_file() << " loaded" << std::endl;
	thh.join();
	std::cout << cfg.svm_file_hard() << " loaded" << std::endl;

	// quantitative evaluation
	mmp::quantitative_evaluator eval(cfg, c_normal);
	mmp::mat_plot plot(eval.get_labels(), eval.get_scores());
	plot.show("evaluation");
	plot.save(cfg.evaluation_file());

	// quantitative evaluation with hard negative samples
	mmp::quantitative_evaluator eval_hard(cfg, c_hard);
	mmp::mat_plot plot_hard(eval_hard.get_labels(), eval_hard.get_scores());
	plot_hard.show("hard evaluation");
	plot_hard.save(cfg.evaluation_file_hard());

	// qualitative evaluation
	std::cout << std::endl << "qualitataive evaluation . . ." << std::endl;
	mmp::qualitative_evaluator(cfg, c_normal, c_hard);
	std::cout << "qualitiative evaluation terminated" << std::endl;


	std::cout << std::endl;
	std::cout << "finished at: " << mmp::time_string() << std::endl;
	std::cout << "press enter to exit" << std::endl;
	std::system("pause");
}