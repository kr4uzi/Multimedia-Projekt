#include "extraction.h"
#include <vl/hog.h>
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
	assert((mat.type() == CV_8UC1 || mat.type() == CV_8UC3) && "input image type has to be unsigned char");
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
				*dataptr++ = *yptr / 255.0f;
				yptr += channels;
			}
		}
	}

	return vlarray;
}

hog::hog(const cv::Mat& src)
	: _hog(vl_hog_new(VlHogVariant::VlHogVariantUoctti, orientations, VL_FALSE))
{
	auto img_converted = cvimg_to_vlarray(src);
	vl_hog_put_image((VlHog *)_hog, img_converted.data(), src.cols, src.rows, src.channels(), cellsize);
	hog_width = vl_hog_get_width((VlHog *)_hog);
	hog_height = vl_hog_get_height((VlHog *)_hog);
	assert(dimensions == vl_hog_get_dimension((VlHog *)_hog));

	std::vector<float> hog_array(hog_width * hog_height * dimensions);
	vl_hog_extract((VlHog *)_hog, hog_array.data());
	hog_glyph_size = vl_hog_get_glyph_size((VlHog *)_hog);

	hog_converted_data = vlarray_to_cvstylevec(hog_array, hog_height, hog_width, dimensions);
	hog_converted = cv::Mat((int)hog_height, (int)hog_width, CV_32FC(int(dimensions)), hog_converted_data.data());
}

hog::~hog()
{
	vl_hog_delete((VlHog *)_hog);
}

cv::Mat hog::render() const
{
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

const cv::Mat hog::operator()(const cv::Rect& roi) const
{
	const int from_x = (roi.x + cellsize / 2) / cellsize;
	const int from_y = (roi.y + cellsize / 2) / cellsize;
	const int to_x = ((roi.x + roi.width) + cellsize / 2) / cellsize;
	const int to_y = ((roi.y + roi.height) + cellsize / 2) / cellsize;
	return hog_converted(cv::Rect(cv::Point(from_x, from_y), cv::Point(to_x, to_y)));
}

std::size_t hog::hog_size(const cv::Rect& roi)
{
	const int height = (roi.height + cellsize / 2) / cellsize;
	const int width = (roi.width + cellsize / 2) / cellsize;

	return dimensions * height * width;
}