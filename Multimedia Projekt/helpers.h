#pragma once
#include <vector>
#include <string>

#include <opencv2/core/core.hpp>	// Rect

namespace mmp
{
	bool path_exists(const std::string& path);
	std::vector<std::string> files_in_folder(const std::string& folder);
	float get_overlap(const cv::Rect& rect, const cv::Rect& groud_truth);
	void print_progress(const std::string& info, unsigned long value, std::size_t max, std::string filename);
	std::string time_string();
}