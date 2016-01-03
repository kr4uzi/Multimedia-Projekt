#include "image.h"
#include "helpers.h"
#include "extraction.h"
#include "scale_cache.h"
#include "classification.h"
#include <opencv2/imgproc/imgproc.hpp>	// resize
#include <algorithm>					// sort, remove_if
#include <boost/bind.hpp>
using namespace mmp;

sliding_window::sliding_window(std::shared_ptr<const hog> h, int x, int y, float scale)
	: _hog(h), _scale(scale), x(x), y(y)
{

}

cv::Mat sliding_window::features() const
{
	return (*_hog)(cv::Rect(x, y, width, height));
}

cv::Rect sliding_window::rect() const
{
	return cv::Rect(int(x * _scale), int(y * _scale), int(width * _scale), int(height * _scale));
}

scaled_image::scaled_image(cv::Mat src, float scale)
	: scale(scale), _hog(std::make_shared<hog>(src))
{
	// sliding windows for current scale
	for (int y = 0; y <= src.rows - sliding_window::height; y += hog::cellsize)
	{
		for (int x = 0; x <= src.cols - sliding_window::width; x += hog::cellsize)
			windows.emplace_back(std::const_pointer_cast<const hog>(_hog), x, y, scale);
	}
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

void image::add_detection(detection det)
{
	bool overlapped = false;
	for (unsigned i = 0; i < detections.size(); i++)
	{
		if (get_overlap(detections[i].second->rect(), det.second->rect()) >= 0.2f)
		{
			overlapped = true;

			if (detections[i].first < det.first)
			{
				detections[i] = std::move(det);
				return;
			}
		}
	}

	if (!overlapped)
		detections.push_back(std::move(det));
}

void image::suppress_non_maximum(float min_overlap)
{
	std::sort(detections.begin(), detections.end(), boost::bind(&detection::first, _1) > boost::bind(&detection::first, _2));
	
	for (auto i = detections.begin(); i != detections.end(); ++i)
	{
		for (auto j = i + 1; j != detections.end(); ++j)
		{
			if (j->first == 0) continue;

			if (get_overlap(i->second->rect(), j->second->rect()) >= min_overlap)
				j->first = 0; // mark entry for deletion
		}
	}

	auto marked_for_deletion = [](const detection& d)
	{
		return d.first == 0;
	};

	detections.erase(std::remove_if(detections.begin(), detections.end(), marked_for_deletion), detections.end());
}

void image::detect_all(const classifier& c)
{
	for (auto& s : scaled_images())
	{
		for (auto& sw : s.sliding_windows())
		{
			double a = c.classify(sw.features());
			if (a > 0)
				add_detection(std::make_pair(a, &sw));
		}
	}
}