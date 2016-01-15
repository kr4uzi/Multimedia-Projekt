#include "config.h"				// config
#include "inria.h"				// inria_cfg
#include "helpers.h"			// files_in_folder, path_exists
#include "classification.h"		// classifier
#include "annotation.h"			// annotation::file
#include "evaulation.h"			// qualitative_evaluator, quantitative_evaluator, mat_plot
#include "log.h"
#include <ctime>				// time
#include <cstdlib>				// system
#include <iostream>				// endl
#include <thread>

int main(int argc, char ** argv)
{
	mmp::inria_cfg cfg;
	std::string cfg_path = "mmp.cfg";
	if (argc > 1)
	{
		if (mmp::path_exists(argv[1]))
			cfg_path = argv[1];
		else
		{
			mmp::log << "config file [" << argv[1] << "] not existing!" << std::endl;
			mmp::log << "falling back to [mmp.cfg]" << std::endl;
		}
	}

	if (!mmp::path_exists(cfg_path))
	{
		mmp::log << "could not find a config file" << std::endl;
		return 1;
	}

	mmp::config raw_cfg;
	auto error = mmp::config::parse(cfg_path, raw_cfg);
	if (!!error)
	{
		mmp::log << "error parsing [" << cfg_path << "]: " << error.error_msg() << std::endl;
		return 1;
	}

	std::string required[] = { "root", "svm", "svm_hard" };
	for (auto& key : required)
	{
		if (!raw_cfg.exists(key))
		{
			mmp::log << "[" << key << "] is a required config key!" << std::endl;
			return 1;
		}
		else if (raw_cfg.get_bool("skip_training") && !mmp::path_exists(raw_cfg.get_string(key)))
		{
			mmp::log << "[" << key << "] = [" << raw_cfg.get_string(key) << "] invalid (path not existing)!" << std::endl;
			return 1;
		}
	}

	//
	// evaluation config
	//
	std::string evals[] = { "eval", "eval_hard" };
	for (auto& key : evals)
	{
		if (!raw_cfg.exists(key) && !raw_cfg.get_bool("skip_" + key))
		{
			mmp::log << "[" << key << "] is a required config key if skip_ << " << key << " false or not set!" << std::endl;
			return 1;
		}
	}

	cfg = mmp::inria_cfg(
		raw_cfg.get_string("root"),
		raw_cfg.get_string("svm"), raw_cfg.get_string("svm_hard"),
		raw_cfg.get_string("eval"), raw_cfg.get_string("eval_hard"),
		raw_cfg.get_double("svm_c", 0.01),
		raw_cfg.get_unsinged("randoms_per_negative", 10),
		raw_cfg.get_unsinged("num_false_positives", 1280)
	);

	mmp::log << "MMP Markus Kraus" << std::endl;
	mmp::log << "started at: " << mmp::time_string() << std::endl << std::endl;

	// train
	if (!raw_cfg.get_bool("skip_training"))
	{
		if (!mmp::path_exists(cfg.negative_train_path()) || !mmp::path_exists(cfg.normalized_positive_train_path()))
		{
			mmp::log << "invalid root folder specified (train path(s) not found)!" << std::endl;
			return 1;
		}

		mmp::classifier(cfg).train();
		mmp::log << mmp::to::both << std::endl;
	}

	mmp::classifier c_normal(cfg);
	mmp::classifier c_hard(cfg);

	// load svm_files
	{
		mmp::log << "loading svm files ..." << std::endl;
		std::thread thn, thh;
		if (!raw_cfg.get_bool("skip_eval") || !raw_cfg.get_bool("skip_eval_qual"))
			thn = std::thread([&c_normal]() { c_normal.load(); });
		if (!raw_cfg.get_bool("skip_eval_hard") || !raw_cfg.get_bool("skip_eval_qual"))
			thh = std::thread([&c_hard]() { c_hard.load(true); });

		if (!raw_cfg.get_bool("skip_eval") || !raw_cfg.get_bool("skip_eval_qual"))
		{
			thn.join();
			mmp::log << cfg.svm_file() << " loaded" << std::endl;
		}

		if (!raw_cfg.get_bool("skip_eval_hard") || !raw_cfg.get_bool("skip_eval_qual"))
		{
			thh.join();
			mmp::log << cfg.svm_file_hard() << " loaded" << std::endl;
		}
	}

	if (!raw_cfg.get_bool("skip_eval") || !raw_cfg.get_bool("skip_eval_hard"))
	{
		if (!mmp::path_exists(cfg.normalized_positive_test_path()) || !mmp::path_exists(cfg.negative_test_path()))
		{
			mmp::log << "invalid root folder specified (test folder(s) not found)!" << std::endl;
			return 1;
		}
	}

	// quantitative evaluation
	mmp::mat_plot plot;
	if (!raw_cfg.get_bool("skip_eval"))
	{
		mmp::quantitative_evaluator eval(cfg, c_normal);
		plot = mmp::mat_plot(eval.get_labels(), eval.get_scores());
		plot.show("evaluation");
		plot.save(cfg.evaluation_file());
	}

	// quantitative evaluation with hard negative samples
	mmp::mat_plot plot_hard;
	if (!raw_cfg.get_bool("skip_eval_hard"))
	{
		mmp::quantitative_evaluator eval_hard(cfg, c_hard);
		plot_hard = mmp::mat_plot(eval_hard.get_labels(), eval_hard.get_scores());
		plot_hard.show("hard evaluation");
		plot_hard.save(cfg.evaluation_file_hard());
	}

	mmp::log << mmp::to::both << std::endl;
	mmp::log << "finished at: " << mmp::time_string() << std::endl;

	// qualitative evaluation
	if (!raw_cfg.get_bool("skip_eval_qual"))
	{
		mmp::log << mmp::to::console << std::endl << "qualitataive evaluation . . ." << std::endl;
		mmp::qualitative_evaluator(cfg, c_normal, c_hard);
		mmp::log << mmp::to::console << "qualitiative evaluation terminated" << std::endl;
	}

#ifdef WITH_MATLAB
	// dont close the plots
	if (!raw_cfg.get_bool("skip_eval") || !raw_cfg.get_bool("skip_eval_hard"))
		std::system("pause");
#endif
}