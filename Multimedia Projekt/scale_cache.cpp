#include "scale_cache.h"
#include <cmath>	// pow
using namespace mmp;

scale_cache::scale_cache(unsigned x)
	: lambda(x), step(std::pow(2.0f, 1.0f / x))
{
	cache.push_back(1);
	push_octave();
}

void scale_cache::push_octave()
{
#pragma omp critical
	{
		auto size = cache.size();
		cache.resize(size + lambda);
		for (unsigned i = 1; i <= lambda; i++)
			cache[size - 1 + i] = cache[size + i - 2] * step;
	}
}

float scale_cache::operator[](cache_type::size_type i)
{
	if (i >= cache.size())
		push_octave();

	return cache[i];
}