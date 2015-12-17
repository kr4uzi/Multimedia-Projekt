#include "annotation.h"
#include <fstream>		// ifstream
#include <algorithm>	// find_if
#include <regex>		// regex_match

using namespace mmp::annotation;

file::file()
{

}

file::file(const std::string& img_filename, const cv::Vec3i img_size, const std::string& database)
	: image_filename(img_filename), image_size(img_size), database(database)
{

}

void file::add_object(const element& element)
{
	objects.push_back(element);
}

parse_error file::parse(const std::string& filename, file& annotation)
{
	std::ifstream in;
	in.open(filename.c_str());

	if (in.fail())
		return parse_error("could not open file!");

	std::string line;
	std::getline(in, line);
	
	if (line != "# PASCAL Annotation Version 1.00")
	{
		in.close();
		return parse_error("invalid annotation file!");
	}

	while (!in.eof())
	{
		std::getline(in, line);

		// ignore blank lines and comments
		if (line.empty() || line[0] == '#')
			continue;

		// remove newlines and partial newlines to help fix issues with Windows formatted config files on Linux systems
		line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
		line.erase(std::remove(line.begin(), line.end(), '\n'), line.end());

		// todotodo: remove newlines to fix issues with windows formated files on linux
		static const std::regex filename_pattern("^Image filename : \"(.*)\"$");
		static const std::regex size_pattern("^Image size \\(X x Y x C\\) : (\\d+) x (\\d+) x (\\d)$");
		static const std::regex database_pattern("^Database : \"(.*)\"");
		static const std::regex label_pattern("^Original label for object (\\d+) \"(.*)\" : \"(.*)\"$");
		static const std::regex center_pattern("^Center point on object (\\d+) \"(.*)\" \\(X, Y\\) : \\((\\d+), (\\d+)\\)");
		static const std::regex box_pattern("^Bounding box for object (\\d+) \"(.*)\" \\(Xmin, Ymin\\) - \\(Xmax, Ymax\\) : \\((\\d+), (\\d+)\\) - \\((\\d+), (\\d+)\\)");

		std::smatch match;
		if (std::regex_match(line, match, filename_pattern))
			annotation.image_filename = match[1];
		else if (std::regex_match(line, match, size_pattern))
			annotation.image_size = cv::Vec3i(std::stoul(match[1]), std::stoul(match[2]), std::stoul(match[3]));
		else if (std::regex_match(line, match, database_pattern))
			annotation.database = match[1];
		else if (std::regex_match(line, match, label_pattern))
		{
			element element;
			element.id = std::stoul(match[1]);
			element.label = match[2];
			element.original_label = match[3];
			annotation.add_object(element);
		}
		else if (std::regex_match(line, match, center_pattern))
		{
			unsigned id = std::stoul(match[1]);

			auto& iter = std::find_if(annotation.objects.begin(), annotation.objects.end(), [id](element& el) { return el.id == id; });
			if (iter != annotation.objects.end())
				iter->center = cv::Point(std::stoul(match[3]), std::stoul(match[4]));
			else
				return parse_error("parse mismatch");
		}
		else if (std::regex_match(line, match, box_pattern))
		{
			unsigned id = std::stoul(match[1]);

			auto& iter = std::find_if(annotation.objects.begin(), annotation.objects.end(), [id](element& el) { return el.id == id; });
			if (iter != annotation.objects.end())
				iter->bounding_box = cv::Rect(cv::Point(std::stoul(match[3]), std::stoul(match[4])), cv::Point(std::stoul(match[5]), std::stoul(match[6])));
			else
				return parse_error("parse mismatch");
		}
	}

	return parse_error("");
}