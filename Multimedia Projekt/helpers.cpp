#include "helpers.h"
#include "log.h"

#include <boost/filesystem.hpp>
#include <boost/algorithm/string.hpp>

#include <chrono>	// now, to_time_t
#include <ctime>	// localtime
#include <iomanip>	// put_time
#include <sstream>	// stringstream

bool mmp::path_exists(const std::string& path)
{
	return boost::filesystem::exists(path);
}

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
			filename.resize(line_length - 1 - message.size() - 6);		// "<message> (<file...>)"
			filename += "...";
		}

		message += " (";
		message += filename;
		message += ")";
	}

	// fill with whitespaces to overwrite previous long message (carriage return)
	message.resize(line_length - 1, ' ');

	if (value == max)
		log << to::both << message << std::endl;
	else
		log << to::console << message << "\r";
}

std::string mmp::time_string()
{
	using namespace std::chrono;
	auto now = system_clock::now();
	auto now_t = std::chrono::system_clock::to_time_t(now);
	std::stringstream ss;
	ss << std::put_time(std::localtime(&now_t), "%F %X");
	return ss.str();
}
