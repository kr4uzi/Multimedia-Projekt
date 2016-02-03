#pragma once
#include <opencv2/core/core.hpp>	// Rect, Mat
#include <vector>
#include <utility>		// pair
#include <memory>		// shared_ptr, const_pointer_cast
#include "hog.h"

namespace mmp
{
	class sliding_window
	{
	public:
		static const int width = 64;
		static const int height = 128;

	private:
		std::shared_ptr<const hog> _hog;
		float _scale;
		int x;
		int y;

	public:
		sliding_window(std::shared_ptr<const hog> _hog, int x, int y, float scale);

		cv::Mat features() const;
		cv::Rect rect() const;
		float scale() const		{ return _scale; }
	};

	class scaled_image
	{
	private:
		float scale;
		std::vector<sliding_window> windows;
		std::shared_ptr<hog> _hog;

	public:
		scaled_image(cv::Mat src, float scale);

		const std::vector<sliding_window>& sliding_windows() const { return windows; }
		float get_scale() const { return scale; }
		std::shared_ptr<const hog> get_hog() const { return std::const_pointer_cast<const hog>(_hog); }
	};

	class classifier;
	class image
	{
	public:
		static const unsigned scales_per_octave = 5;
		typedef std::pair<double, const sliding_window *> detection;

	private:
		std::vector<scaled_image> images;
		std::vector<detection> detections;

	private:
		void add_detection(detection det/*, float max_overlap*/);

	public:
		image(cv::Mat img);

		const std::vector<detection>& get_detections() const { return detections; }
		void detect_all(const classifier& c, double detection_threshold = 0/*, float max_overlap = 0.2f*/);
		void suppress_non_maximum(float min_overlap = 0.2f);		

		const std::vector<scaled_image>& scaled_images() const { return images; }
	};
}