#pragma once

#include <opencv2/core/core.hpp>	// Mat, Rect
#include <vector>

namespace mmp
{
	class hog
	{
	public:
		enum hog_variant
		{
			DalalTriggs,
			UoCCTi
		};

		static const hog_variant variant = UoCCTi;
		static const unsigned cellsize = 8;
		static const unsigned orientations = 9;
		static const unsigned dimensions = (variant == UoCCTi) ? (4 + 3 * orientations) : (4 * orientations);

		typedef std::vector<float> array_type;		
		typedef cv::Vec<float, dimensions> vector_type;

	private:
		void * _hog;					// vl_hog
		array_type hog_converted_data;	// hogarray converted to cv-order
		cv::Mat hog_converted;	// cv::Mat view on this converted array

		array_type::size_type hog_width;
		array_type::size_type hog_height;
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

		hog(const cv::Mat& src);
		~hog();

		const cv::Mat operator()() const { return hog_converted; }
		const cv::Mat operator()(const cv::Rect& roi) const;
		
		// renders whole hog
		cv::Mat render() const;
		// render a certain feature returned by operator()
		cv::Mat render(const cv::Mat& mat) const;
	};
}