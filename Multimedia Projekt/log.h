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

	class Logger
	{
	private:
		std::ofstream log_file;
		to target;

	public:
		Logger(const std::string& filename);
		~Logger();

		Logger& operator<<(const to& val)
		{
			target = val;
			return *this;
		}

		template<class T>
		Logger& operator<<(T val)
		{
			if (target == to::console || target == to::both)
				std::cout << val;

			if (target == to::file || target == to::both)
				log_file << val;

			return *this;
		}

		Logger& operator<<(std::ostream& (*func) (std::ostream&))
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

	extern Logger log;
}
