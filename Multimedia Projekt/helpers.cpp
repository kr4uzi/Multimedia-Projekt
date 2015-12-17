#include "helpers.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <chrono>	// now, to_time_t
#include <ctime>	// localtime
#include <iostream>	// cout
#include <iomanip>	// put_time

std::vector<std::string> mmp::files_in_folder(const std::string& folder)
{
	std::vector<std::string> files;

	auto path = boost::filesystem::path(folder);
	for (boost::filesystem::directory_iterator i(path), end; i != end; i++)
	{
		if (boost::filesystem::is_regular_file(*i))
		{
			std::string fpath(i->path().string());
			boost::replace_all(fpath, "\\", "/");
			boost::replace_all(fpath, "//", "/");
			files.emplace_back(fpath);
		}
	}

	return files;
}

float mmp::get_overlap(const cv::Rect& rect, const cv::Rect& groud_truth)
{
	auto inter = (groud_truth & rect).area();
	auto _union = groud_truth.area() + rect.area() - inter;
	return _union ? (float(inter) / _union) : 0;
}

void mmp::print_progress(const std::string& info, unsigned long value, std::size_t max, std::string filename)
{
	const std::string::size_type line_length = 80;

	std::string message(info);
	message += ": ";
	message.append(std::to_string((double(value) / max) * 100), 0, 5);
	message += "%";

	// do not add filename to the message if we are done
	if (value != max)
	{
		auto pos = filename.find_last_of('/');
		if (pos != std::string::npos)
			filename = filename.substr(pos + 1, std::string::npos);

		if (message.size() + filename.size() + 3 >= line_length - 1)	// "<message> (<filename>)"
		{
			filename.resize(line_length - 1 - message.size() - 6);			// "<message> (<file...>)"
			filename += "...";
		}

		message += " (";
		message += filename;
		message += ")";
	}

	// fill with whitespaces to overwrite previous long message (carriage return)
	message.resize(line_length - 1, ' ');

	std::cout << message << "\r";
}

void mmp::print_time()
{
	using namespace std::chrono;
	auto now = system_clock::now();
	auto now_t = std::chrono::system_clock::to_time_t(now);
	std::cout << std::put_time(std::localtime(&now_t), "%Y-%m-%d %X");
}
