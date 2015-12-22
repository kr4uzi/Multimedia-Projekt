#include <opencv2/opencv.hpp>	// imread, waitKey, RNG
#include "inria.h"				// inria_cfg
#include "helpers.h"			// files_in_folder
#include "classification.h"		// classifier
#include "annotation.h"			// annotation::file
#include "evaulation.h"			// qualitative_evaluator, quantitative_evaluator, mat_plot
#include <ctime>				// time
#include <cstdlib>				// system
#include <iostream>				// cout, endl
#include <thread>
#include <vl/hog.h>

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
#include <fstream>
#include "hog_util.h"
int main()
{
#if 0
	auto img = cv::imread(R"(R:\INRIAPerson\70X134H96\Test\pos\crop_000001a.png)");
	{
		auto img_vl_order = mmp::hog::cvimg_to_vlarray(img);
		auto hog = vl_hog_new(VlHogVariantUoctti, 9, false);
		vl_hog_put_image(hog, img_vl_order.data(), img.cols, img.rows, img.channels(), 8);
		auto width = vl_hog_get_width(hog);
		auto height = vl_hog_get_height(hog);
		auto dimensions = vl_hog_get_dimension(hog);
		auto glyphs = vl_hog_get_glyph_size(hog);
		std::vector<float> hog_array(width * height * dimensions);
		vl_hog_extract(hog, hog_array.data());
		std::cout << hog_array.size() << std::endl;
		std::ofstream out;
		out.open("R:/original.txt");
		
		for (decltype(hog_array)::size_type i = 0; i < hog_array.size();)
		{
			out << hog_array[i] << " ";
			if ((++i % (width * height)) == 0)
				out << std::endl;
		}

		out.close();
	}

	{
		std::ofstream out;
		out.open("R:/self.txt");

		mmp::hog h(img);
		auto hh = h(cv::Rect(cv::Point(0, 0), cv::Point(img.cols, img.rows)));
		for (int y = 0; y < hh.rows; y++)
		{
			auto yptr = hh.ptr<float>(y);

			for (int x = 0; x < hh.cols; x++)
			{
				for (int c = 0; c < hh.channels(); c++)
				{
					out << yptr[c] << " ";
				}

				yptr += hh.channels();
				out << std::endl;
			}
		}

		std::ofstream outf;
		outf.open("R:/self_svector.txt");
		auto svec = mmp::svm::sparse_vector(hh);
		for (auto i = svec.begin(); i != svec.end(); ++i)
		{
			outf << *i << " ";
		}

		outf.close();

		out.close();
	}

	{
		auto img_vl_order = mmp::hog::cvimg_to_vlarray(img);
		auto hog = vl_hog_new(VlHogVariantUoctti, 9, false);
		vl_hog_put_image(hog, img_vl_order.data(), img.cols, img.rows, img.channels(), 8);
		auto width = vl_hog_get_width(hog);
		auto height = vl_hog_get_height(hog);
		auto dimensions = vl_hog_get_dimension(hog);
		auto glyphs = vl_hog_get_glyph_size(hog);
		std::vector<float> hog_array(width * height * dimensions);
		vl_hog_extract(hog, hog_array.data());
		auto hh = convert_hog_array(hog_array.data(), 9, width, height, width, height);
		std::ofstream out;
		out.open("R:/hog_util.txt");
		std::ofstream outf;
		outf.open("R:/hog_util_svector.txt");
		auto svec = mmp::svm::sparse_vector(hh);
		for (auto i = svec.begin(); i != svec.end(); ++i)
		{
			outf << *i << " ";
		}

		outf.close();

		for (int y = 0; y < hh.rows; y++)
		{
			auto yptr = hh.ptr<float>(y);

			for (int x = 0; x < hh.cols; x++)
			{
				for (int c = 0; c < hh.channels(); c++)
				{
					out << yptr[c] << " ";
				}

				yptr += hh.channels();
			}

			out << std::endl;
		}

		out.close();
	}





























	system("pause");
	exit(0);
#endif
	std::cout << "MMP Markus Kraus" << std::endl;
	std::cout << "started at: " << mmp::time_string() << std::endl << std::endl;

	mmp::inria_cfg cfg("R:/INRIAPerson/", 
		"R:/INRIAPerson/svm.dat", "R:/INRIAPerson/svm_hard.dat", 
		"R:/INRIAPerson/evaluation.m", "R:/INRIAPerson/evaluation_hard.m");

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
	qualitative_evaluation(cfg, c_normal, c_hard);
	std::cout << "qualitiative evaluation terminated" << std::endl;


	std::cout << std::endl;
	std::cout << "finished at: " << mmp::time_string() << std::endl;
	std::cout << "press enter to exit" << std::endl;
	std::system("pause");
}