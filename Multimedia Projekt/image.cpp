#include "image.h"
#include "helpers.h"
#include "extraction.h"
#include "scale_cache.h"
#include <opencv2/imgproc/imgproc.hpp>	// resize
#include <algorithm>					// sort, remove_if
using namespace mmp;

sliding_window::sliding_window(std::shared_ptr<const UocttiHOG> h, int x, int y, float scale)
	: hog(h), _scale(scale), x(x), y(y)
{

}

cv::Mat sliding_window::features() const
{
	return (*hog)(cv::Rect(x, y, width, height));
}

cv::Rect sliding_window::window() const
{
	return cv::Rect(int(x * _scale), int(y * _scale), int(width * _scale), int(height * _scale));
}

scaled_image::scaled_image(const scaled_image& rhs)
	: scale(rhs.scale), windows(rhs.windows), hog(rhs.hog)
{

}

scaled_image::scaled_image(scaled_image&& rhs)
	: scale(rhs.scale), windows(std::move(rhs.windows)), hog(std::move(rhs.hog))
{

}

scaled_image::scaled_image(cv::Mat src, float scale)
	: scale(scale), hog(std::make_shared<UocttiHOG>(src))
{
	// sliding windows for current scale
	for (int y = 0; y <= src.rows - sliding_window::height; y += UocttiHOG::hog_cellsize)
	{
		for (int x = 0; x <= src.cols - sliding_window::width; x += UocttiHOG::hog_cellsize)
			windows.emplace_back(std::const_pointer_cast<const UocttiHOG>(hog), x, y, scale);
	}
}

image::image(const image& rhs)
	: images(rhs.images), detections(rhs.detections)
{

}


image::image(image&& rhs)
	: images(std::move(rhs.images)), detections(std::move(rhs.detections))
{

}

image::image(cv::Mat src)
{
	static scale_cache scales(scales_per_octave);

	cv::Mat work = src;
	float scale = 1;
	for (unsigned i = 1; work.rows >= sliding_window::height && work.cols >= sliding_window::width; i++)
	{
		images.emplace_back(scaled_image(work, scale));

		scale = scales[i];
		auto mod = i % scales_per_octave;
		if (mod == 0)
			cv::resize(src, src, cv::Size(), 0.5f, 0.5f);
		
		cv::resize(src, work, cv::Size(), 1 / scales[mod], 1 / scales[mod]);
	}
}

void image::add_detection(const cv::Rect& rect, double weight)
{
	detections.emplace_back(rect, weight);
}

void image::non_maximum_suppression(float min_overlap)
{
	std::sort(detections.begin(), detections.end(), [](const detection& a, const detection& b) { return a.second > b.second; });

	for (auto i = detections.begin(); i != detections.end(); i++)
	{
		for (auto j = i + 1; j != detections.end(); j++)
		{
			if (get_overlap(i->first, j->first) >= min_overlap)
				j->second = 0; // mark entry for deletion
		}
	}

	auto marked_for_deletion = [](const detection& d)
	{
		return d.second == 0;
	};

	detections.erase(std::remove_if(detections.begin(), detections.end(), marked_for_deletion), detections.end());
}