#include "config.h"			// config
#include "inria.h"			// inria_cfg
#include "helpers.h"		// files_in_folder, path_exists, time_string
#include "classifier.h"		// classifier
#include "evaulation.h"		// qualitative_evaluator, quantitative_evaluator, mat_plot
#include "log.h"
#include <iostream>			// endl
#include <thread>

int main(int argc, char ** argv)
{
	//
	// parse config
	//
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

	//
	// validate config
	//
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
			// the svm files must be previously created in order to be able to evaluate
			mmp::log << "[" << key << "] = [" << raw_cfg.get_string(key) << "] invalid (path not existing)!" << std::endl;
			return 1;
		}
	}

	if (!raw_cfg.exists("eval") && !raw_cfg.get_bool("skip_eval"))
	{
		mmp::log << "[eval] is a required config key if [skip_eval] = [false]" << std::endl;
		return 1;
	}

	cfg = mmp::inria_cfg(
		raw_cfg.get_string("root"),
		raw_cfg.get_string("svm"), raw_cfg.get_string("svm_hard"),
		raw_cfg.get_string("eval"), raw_cfg.get_string("eval_hard"),
		raw_cfg.get_double("svm_c", 0.01),
		raw_cfg.get_unsinged("randoms_per_negative", 10),
		raw_cfg.get_unsinged("num_false_positives", 1218)
	);

	bool skip_training = raw_cfg.get_bool("skip_training");
	bool skip_eval = raw_cfg.get_bool("skip_eval");
	bool skip_eval_qual = raw_cfg.get_bool("skip_eval_qual");

	//
	// validate required paths and files
	//
	if (!skip_training)
	{
		if (!mmp::path_exists(cfg.normalized_positive_train_path()) || !mmp::path_exists(cfg.negative_train_path()))
		{
			mmp::log << "invalid root folder specified (train path(s) not found)!" << std::endl;
			return 1;
		}
	}

	if (!skip_eval || !skip_eval_qual)
	{
		// if we skip training (the both svm files are created in that process) and 
		// one of the required files are not existing we exit
		// the required paths are checked below because we only check for svm files here
		if (skip_training &&
			(!mmp::path_exists(cfg.svm_file()) || !mmp::path_exists(cfg.svm_file_hard())))
		{
			mmp::log << "svm file [" << cfg.svm_file() << "] or hard svm file [" << cfg.svm_file_hard() << "] not found!" << std::endl;
			return 1;
		}
	}

	if (!skip_eval)
	{
		// existence of svm files has been checked above
		if (!mmp::path_exists(cfg.normalized_positive_test_path()) || !mmp::path_exists(cfg.negative_test_path()))
		{
			mmp::log << "invalid root folder specified (test path(s) not found)!" << std::endl;
			return 1;
		}
	}

	if (!skip_eval_qual)
	{
		// existence of svm files has been checked above
		if (!mmp::path_exists(cfg.test_annotation_path()))
		{
			mmp::log << "invalid root folder specified (test path not found)!" << std::endl;
			return 1;
		}
	}

	//
	// actual program
	//
	mmp::classifier c_normal(cfg);
	mmp::classifier c_hard(cfg);

	mmp::log << "MMP Markus Kraus" << std::endl;
	mmp::log << "started at: " << mmp::time_string() << std::endl << std::endl;

	if (!skip_training)
	{
		mmp::log << "############ training ############" << std::endl;
		mmp::classifier(cfg).train();
		mmp::log << std::endl;
	}

	//
	// load and validate svm_files
	//
	if (!skip_eval || !skip_eval_qual)
	{
		mmp::log << "########### evaluation ###########" << std::endl;
		mmp::log << "loading svm files ..." << std::endl;
		std::thread thn = std::thread([&c_normal]() { c_normal.load(); });
		std::thread thh = std::thread([&c_hard]() { c_hard.load(true); });

		thn.join();
		mmp::log << cfg.svm_file() << " loaded" << std::endl;
		thh.join();
		mmp::log << cfg.svm_file_hard() << " loaded" << std::endl;
	}

	//
	// quantitative evaluation
	//
	mmp::mat_plot plot, plot_hard;
	if (!skip_eval)
	{
		mmp::quantitative_evaluator eval(cfg, c_normal);
		plot = mmp::mat_plot(eval.get_labels(), eval.get_scores());
		plot.show("evaluation");
		plot.save(cfg.evaluation_file());

		mmp::quantitative_evaluator eval_hard(cfg, c_hard);
		plot_hard = mmp::mat_plot(eval_hard.get_labels(), eval_hard.get_scores());
		plot_hard.show("hard evaluation");
		plot_hard.save(cfg.evaluation_file_hard());
	}

	mmp::log << std::endl << "finished at: " << mmp::time_string() << std::endl;

	//
	// qualitative evaluation
	//
	if (!skip_eval_qual)
	{
		mmp::log << mmp::to::console << std::endl << "qualitataive evaluation . . ." << std::endl;
		mmp::qualitative_evaluator(cfg, c_normal, c_hard);
		mmp::log << "qualitiative evaluation terminated" << std::endl;
	}

#ifdef WITH_MATLAB
	// let the user work with the plots
	if (!skip_eval)
	{
		mmp::log << mmp::to::console << "Press enter to exit ...";
		std::cin.get();
	}
#endif
}