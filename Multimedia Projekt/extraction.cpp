#include "extraction.h"
#include <vl/hog.h>
#include "hog_util.h"
using namespace mmp;

hog::array_type hog::vlarray_to_cvstylevec(const array_type& vlarray, array_type::size_type height, array_type::size_type width, array_type::size_type dimensions)
{
	std::vector<float> cstylevec(height * width * dimensions);
	for (array_type::size_type y = 0; y < height; y++)
	{
		for (array_type::size_type x = 0; x < width; x++)
		{
			for (array_type::size_type c = 0; c < dimensions; c++)
				cstylevec[y * width * dimensions + x * dimensions + c] = vlarray[c * width * height + y * width + x];
		}
	}

	return cstylevec;
}

hog::array_type hog::cvmat_to_vlarray(const cv::Mat& mat)
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

hog::array_type hog::cvimg_to_vlarray(const cv::Mat& mat)
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

hog::hog(const cv::Mat& src)
	: _hog(vl_hog_new(VlHogVariant::VlHogVariantUoctti, orientation_bins, VL_FALSE))
{
	// RGB RGB RGB -> RRR GGG BBB
	auto img_converted = cvimg_to_vlarray(src);

	vl_hog_put_image((VlHog *)_hog, img_converted.data(), src.cols, src.rows, src.channels(), hog_cellsize);
	hog_width = vl_hog_get_width((VlHog *)_hog);
	hog_height = vl_hog_get_height((VlHog *)_hog);
	hog_dimensions = vl_hog_get_dimension((VlHog *)_hog);

	std::vector<float> hog_array(hog_width * hog_height * hog_dimensions);
	vl_hog_extract((VlHog *)_hog, hog_array.data());
	hog_glyph_size = vl_hog_get_glyph_size((VlHog *)_hog);

	hog_converted_data = vlarray_to_cvstylevec(hog_array, hog_height, hog_width, hog_dimensions);
	// create cv::Mat view on hog_converted_data
	hog_converted = cv::Mat((int)hog_height, (int)hog_width, CV_32FC(int(hog_dimensions)), hog_converted_data.data());
	//hog_converted = convert_hog_array(hog_array.data(), 9, hog_width, hog_height, hog_width, hog_height);
	//cells_per_row = src.cols / 8;
	//cols = src.cols;
}

hog::~hog()
{
	vl_hog_delete((VlHog *)_hog);
}

cv::Mat hog::render() const
{
	/*
	std::vector<float> hog_array = cvmat_to_vlarray(hog_converted);
	std::vector<float> img(hog_height * hog_glyph_size * hog_width * hog_glyph_size);
	vl_hog_render((VlHog *)hog, img.data(), hog_array.data(), hog_width, hog_height);
	auto image = cv::Mat(int(hog_glyph_size * hog_height), int(hog_glyph_size * hog_width), CV_32FC1, img.data());
	return image.clone();
	*/
	return render(hog_converted);
}

cv::Mat hog::render(const cv::Mat& mat) const
{
	std::vector<float> hog_array = cvmat_to_vlarray(mat);
	std::vector<float> img(mat.cols * hog_glyph_size * mat.rows * hog_glyph_size);
	vl_hog_render((VlHog *)_hog, img.data(), hog_array.data(), mat.cols, mat.rows);
	auto image = cv::Mat(int(hog_glyph_size * mat.rows), int(hog_glyph_size * mat.cols), CV_32FC1, img.data());
	return image.clone();
}

cv::Mat hog::operator()(const cv::Rect& roi) const
{
	const int sub_x = (roi.x + hog_cellsize / 2) / hog_cellsize;
	const int sub_y = (roi.y + hog_cellsize / 2) / hog_cellsize;
	const int sub_height = (roi.height + hog_cellsize / 2) / hog_cellsize;
	const int sub_width = (roi.width + hog_cellsize / 2) / hog_cellsize;
	return hog_converted(cv::Rect(sub_x, sub_y, sub_width, sub_height));
	//const int roi_hog_length = roi.width / hog_cellsize;
	//const int roi_hog_height = roi.height / hog_cellsize;
	//const int roi_offset = (roi.y / hog_cellsize) * cells_per_row + (roi.x / hog_cellsize);
	//cv::Mat result(roi_hog_length * roi_hog_height, 31, CV_32FC1);
	//for (int y = 0; y < 16; y++)
	//{
	//	for (int x = 0; x < 8; x++)
	//		hog_converted.row(roi_offset + y * cells_per_row + x).copyTo(result.row(y * roi_hog_length + x));
	//}

	//return result;
}

std::size_t hog::hog_size(const cv::Rect& roi)
{
	const int height = (roi.height + hog_cellsize / 2) / hog_cellsize;
	const int width = (roi.width + hog_cellsize / 2) / hog_cellsize;

	if (hog_variant == Uoctti)
		return (4 + 3 * orientation_bins) * height * width;
	else
		return (4 * orientation_bins) * height * width;
}