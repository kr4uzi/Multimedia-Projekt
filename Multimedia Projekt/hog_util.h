#pragma once
#ifndef __HOG_UTIL_H_
#define __HOG_UTIL_H_

#include <opencv/cv.h>

/**
* Most of the code is taken from hog.c (VLFEAT v0.9.16)
*/

// Input: HOG array, computed with vl_hog_extract(), which stores 
// [hogArrayWidth x hogArrayHeight] many cell descriptors
//
// Output: cv::Mat with [hogArrayWidth x hogArrayHeight] many rows and 31 cols 
// (only works with variant UoCCTi).
// Each row stores one cell descriptor, e.g. the descriptor at grid position
// (r,c) is stored in row (r * hogArrayWidth) + c
// 
// Note: cols_needed and rows_needed must be less or equal than hogArrayWidth 
// and hogArrayHeight respectively; since VLFeat extracts features for cells 
// that are not fully contained in the image, one can use these parameters to 
// make sure that only cells fully contained in the image are stored in the 
// resulting feature matrix.
cv::Mat convert_hog_array(float* features, int numOrientations,
	int hogArrayWidth, int hogArrayHeight, int cols_needed, int rows_needed);
void hog_render(float* image, int num_channels, float const* descriptor,
	int glyph_size, int hog_array_width, int hog_array_height);

#endif
