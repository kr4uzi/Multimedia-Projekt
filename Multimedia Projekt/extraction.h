#pragma once

#include <opencv2/core/core.hpp>	// Mat, Rect
#include <vector>

namespace mmp
{
	class hog
	{
	public:
		static const unsigned hog_cellsize = 8;
		static const unsigned orientation_bins = 9;
		typedef std::vector<float> array_type;

		static const enum
		{
			DalalTriggs,
			Uoctti
		} hog_variant = Uoctti;

	private:
		void * _hog;							// vl_hog
		std::vector<float> hog_converted_data;	// hogarray converted to cv-order
		cv::Mat hog_converted;					// cv::Mat view on this converted array

		array_type::size_type hog_width;
		array_type::size_type hog_height;
		array_type::size_type hog_dimensions;
		array_type::size_type hog_glyph_size;
		int cells_per_row;
		int cols;

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

		hog(const cv::Mat& src);
		~hog();

		cv::Mat operator()(const cv::Rect& roi) const;
		cv::Mat render() const;

		// render a certain feature return by operator()
		cv::Mat render(const cv::Mat& mat) const;
	};
}