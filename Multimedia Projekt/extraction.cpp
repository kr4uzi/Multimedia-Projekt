#include "extraction.h"
#include <vl/hog.h>
using namespace mmp;

UocttiHOG::array_type UocttiHOG::vlarray_to_cvstylevec(const array_type& hog_array, array_type::size_type height, array_type::size_type width, array_type::size_type dimensions)
{
	std::vector<float> hog_converted(height * width * dimensions);
	for (unsigned y = 0; y < height; y++)
	{
		for (unsigned x = 0; x < width; x++)
		{
			for (unsigned c = 0; c < dimensions; c++)
				hog_converted[y * width * dimensions + x * dimensions + c] = hog_array[c * width * height + y * width + x];
		}
	}

	return hog_converted;
}

UocttiHOG::array_type UocttiHOG::cvmat_to_vlarray(const cv::Mat& mat)
{
	const int channels = mat.channels();

	std::vector<float> vlarray(channels * mat.rows * mat.cols);
	auto dataptr = vlarray.data();

	for (int c = 0; c < channels; c++)
	{
		for (int y = 0; y < mat.rows; y++)
		{
			auto yptr = mat.ptr<float>(y) + c;
			for (int x = 0; x < mat.cols; x++)
			{
				*dataptr++ = *yptr;
				yptr += channels;
			}
		}
	}

	return vlarray;
}

UocttiHOG::array_type UocttiHOG::cvimg_to_vlarray(const cv::Mat& mat)
{
	const int channels = mat.channels();

	std::vector<float> vlarray(channels * mat.rows * mat.cols);
	auto dataptr = vlarray.data();

	for (int c = 0; c < channels; c++)
	{
		for (int y = 0; y < mat.rows; y++)
		{
			auto yptr = mat.ptr<uchar>(y) + c;
			for (int x = 0; x < mat.cols; x++)
			{
				*dataptr++ = *yptr / 255.f;
				yptr += channels;
			}
		}
	}

	return vlarray;
}

UocttiHOG::UocttiHOG(const cv::Mat& src)
	: hog(vl_hog_new(VlHogVariant::VlHogVariantUoctti, orientation_bins, VL_FALSE))
{
	// RGB RGB RGB -> RRR GGG BBB
	auto img_converted = cvimg_to_vlarray(src);

	vl_hog_put_image((VlHog *)hog, img_converted.data(), src.cols, src.rows, src.channels(), hog_cellsize);
	hog_width = vl_hog_get_width((VlHog *)hog);
	hog_height = vl_hog_get_height((VlHog *)hog);
	hog_dimensions = vl_hog_get_dimension((VlHog *)hog);

	std::vector<float> hog_array(hog_width * hog_height * hog_dimensions);
	vl_hog_extract((VlHog *)hog, hog_array.data());
	hog_glyph_size = vl_hog_get_glyph_size((VlHog *)hog);

	hog_converted_data = vlarray_to_cvstylevec(hog_array, hog_height, hog_width, hog_dimensions);
	// create cv::Mat view on hog_converted_data
	hog_converted = cv::Mat((int)hog_height, (int)hog_width, CV_32FC(int(hog_dimensions)), hog_converted_data.data());
}

UocttiHOG::~UocttiHOG()
{
	vl_hog_delete((VlHog *)hog);
}

cv::Mat UocttiHOG::render() const
{
	std::vector<float> hog_array = cvmat_to_vlarray(hog_converted);
	std::vector<float> img(hog_height * hog_glyph_size * hog_width * hog_glyph_size);
	vl_hog_render((VlHog *)hog, img.data(), hog_array.data(), hog_width, hog_height);
	auto image = cv::Mat(int(hog_glyph_size * hog_height), int(hog_glyph_size * hog_width), CV_32FC1, img.data());
	return image.clone();
}

cv::Mat UocttiHOG::render(const cv::Mat& mat) const
{
	std::vector<float> hog_array = cvmat_to_vlarray(mat);
	std::vector<float> img(mat.cols * hog_glyph_size * mat.rows * hog_glyph_size);
	vl_hog_render((VlHog *)hog, img.data(), hog_array.data(), mat.cols, mat.rows);
	auto image = cv::Mat(int(hog_glyph_size * mat.rows), int(hog_glyph_size * mat.cols), CV_32FC1, img.data());
	return image.clone();
}

cv::Mat UocttiHOG::operator()(const cv::Rect& roi) const
{
	const int sub_x = roi.x / hog_cellsize;
	const int sub_y = roi.y / hog_cellsize;
	const int sub_height = roi.height / hog_cellsize;
	const int sub_width = roi.width / hog_cellsize;
	return hog_converted(cv::Rect(sub_x, sub_y, sub_width, sub_height));
}

std::size_t UocttiHOG::hog_size(const cv::Rect& roi)
{
	return 31 * (roi.height / hog_cellsize) * (roi.width / hog_cellsize);
}