#pragma once

#include <opencv2/core/core.hpp>	// Mat, Rect
#include <vector>

namespace mmp
{
	class UocttiHOG
	{
	public:
		static const unsigned hog_cellsize = 8;
		static const unsigned orientation_bins = 9;
		typedef std::vector<float> array_type;

	private:
		void * hog;
		std::vector<float> hog_converted_data;
		cv::Mat hog_converted;	// view on hog_converted_data

		array_type::size_type hog_width;
		array_type::size_type hog_height;
		array_type::size_type hog_dimensions;
		array_type::size_type hog_glyph_size;

	public:
		//
		// 111 222 333    123 123 123
		// 111 222 333 => 123 123 123
		// 111 222 333    123 123 123
		//
		static array_type vlarray_to_cvstylevec(const array_type& hogarray, array_type::size_type height, array_type::size_type width, array_type::size_type dimensions);

		//
		// 123 123 123    111 222 333
		// 123 123 123 => 111 222 333
		// 123 123 123    111 222 333
		//
		static array_type cvmat_to_vlarray(const cv::Mat& mat);	// convert a float cv::Mat to float vlarray
		static array_type cvimg_to_vlarray(const cv::Mat& mat); // like above but used to convert a (uchar) RGB image to float RGB vlarray

	public:
		static std::size_t hog_size(const cv::Rect& roi);

		UocttiHOG(const cv::Mat& src);
		~UocttiHOG();

		cv::Mat operator()(const cv::Rect& roi) const;
		cv::Mat render() const;
		cv::Mat render(const cv::Mat& mat) const;
	};
}