#pragma once

#include <vector>

namespace mmp
{
	class scale_cache
	{
	private:
		typedef std::vector<float> cache_type;

		cache_type cache;
		unsigned lambda;
		float step;

	private:
		void push_octave();

	public:
		scale_cache(unsigned scales_per_octave);

		float operator[](cache_type::size_type i);
	};
}
