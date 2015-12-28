#include "log.h"
#include "helpers.h"
using namespace mmp;

Logger::Logger(const std::string& filename)
	: target(to::both)
{
	log_file.open(filename, std::ios::app);
	if (log_file.is_open())
	{
		log_file << time_string() << " ";
	}
	else
		std::cout << "could not open logfile [" << filename << "]" << std::endl;
}

Logger::~Logger()
{
	if (log_file.is_open())
	{
		log_file << std::endl;
		log_file.close();
	}
}

Logger mmp::log("mmp.log");