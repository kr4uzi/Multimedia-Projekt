#pragma once
#include <fstream>
#include <string>
#include <iostream>
#include "helpers.h"

namespace mmp
{
	enum class to
	{
		console,
		file,
		both
	};

	class logger
	{
	private:
		std::ofstream log_file;
		to target;

	public:
		logger(const std::string& filename);
		~logger();

		logger& operator<<(const to& val)
		{
			target = val;
			return *this;
		}

		template<class T>
		logger& operator<<(T val)
		{
			if (target == to::console || target == to::both)
				std::cout << val;

			if (target == to::file || target == to::both)
				log_file << val;

			return *this;
		}

		logger& operator<<(std::ostream& (*func) (std::ostream&))
		{
			if (target == to::console || target == to::both)
				func(std::cout);

			if (target == to::file || target == to::both)
			{
				func(log_file);
				log_file << time_string() << " ";
			}

			return *this;
		}
	};

	extern logger log;
}
